"""The Tailor — first LLM node. One Claude call: resume + brief + evidence
-> rewritten resume JSON + a change log that cites atoms.

Roadmap invariants 1, 2 and 3 meet here:
  1. The model writes *data*. It returns the whole `Resume` (not a patch,
     not markup), so the same Pydantic gate that protects the renderer
     protects the agent's output.
  2. Hard limits are code, not judgment. The API's structured outputs
     guarantee the JSON *shape* (constrained decoding against our
     schema); Pydantic re-validates the *constraints* a grammar can't
     express (date patterns, lengths); `verify()` checks *grounding*
     (cited atoms exist, land in the right section, no new numbers).
     The model is never asked "is this valid?" or "did you make that up?".
  3. Every claim traces to an atom. The prompt says so; `verify()`
     enforces the checkable part; the Judge (Phase 6) audits the rest.

The retry loop is deliberately visible: two attempts, and the second
one carries the first response plus a list of concrete problems back
into the conversation. Both failure kinds — a ValidationError's `loc`
paths and `verify()`'s grounding problems — go through the same path,
so the model sees "what was wrong" in one format regardless of which
tool caught it.

Model routing lives in one env var for now (`TAILOR_MODEL`); Phase 6
moves it to `src/config.py` alongside the Judge's model.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anthropic import transform_schema
from pydantic import Field, ValidationError

from src.atoms import Atom
from src.brief import JobBrief
from src.models import NonEmptyStr, Resume, StrictModel
from src.tools.evidence import BriefHit, tokenize

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = os.environ.get("TAILOR_MODEL", "claude-sonnet-5")
# Claude 5 models think adaptively by default, and thinking tokens count
# against max_tokens. The JSON answer needs ~5k tokens; the rest is
# headroom so a long think never truncates the answer (that failure mode
# is stop_reason="max_tokens" with only a thinking block returned).
MAX_TOKENS = 32000
# Adaptive-thinking effort. "high" is the API default and was overkill
# for a rewrite; "medium" keeps reasoning on at a fraction of the tokens.
DEFAULT_EFFORT = os.environ.get("TAILOR_EFFORT", "medium")
MAX_ATTEMPTS = 2   # roadmap: retry once


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

class Change(StrictModel):
    """One entry in the change log — the "why" half of the git diff."""
    where: NonEmptyStr        # short path, e.g. "experience[Target Corporation].bullets[0]", "summary"
    atom_id: str | None       # cited evidence; null only for rephrasing that adds no fact
    why: NonEmptyStr          # which brief requirement this serves


class TailoredResume(StrictModel):
    resume: Resume
    changes: list[Change] = Field(min_length=1)


def output_schema() -> dict[str, Any]:
    """JSON schema the API constrains decoding to. `transform_schema` strips
    what the grammar can't express (patterns, max lengths) and folds them
    into descriptions; Pydantic re-applies them on the way back in."""
    return transform_schema(TailoredResume.model_json_schema())


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_TEMPLATE = """\
You are the Tailor in an automated resume pipeline. You receive a candidate's \
current resume as JSON, a job brief, and a ranked list of evidence atoms: \
verified facts about the candidate, each tagged with the brief requirements it \
covers. You return the complete resume JSON, rewritten to fit the brief, plus a \
change log.

Grounding rules. Violations are rejected by code, not by judgment:
1. Assert only what is in the current resume or in an evidence atom. Never \
invent numbers, dates, titles, employers, tools, or outcomes. Quote figures \
exactly as they appear in an atom's `metrics` or the current resume.
2. Insert evidence only from the atoms provided. Every insertion, and every \
rewrite that adds a fact, cites the atom's `id` in the change log. Rephrasing \
existing content without adding facts cites null.
3. Placement follows the atom. section "experience": a bullet under the \
experience entry whose `company` equals the atom's `source`. "project": a \
project entry. "publication": a publication entry. Never move content between \
sections. Education atoms are never inserted.
4. Leave `contact` and `education` exactly as given.
5. The brief lists must-haves with no supporting evidence. Do not add any \
claim about them: no new bullet, sentence, or skill item. Existing resume \
content is left as it is.
6. Every bullet states an achievement or deliverable: what was built, \
analysed, or decided, and its effect. A bullet whose only content is \
collaboration, communication, or attitude is not a bullet. Soft skills from \
the brief are conveyed by phrasing inside achievement bullets ("...with \
product and engineering partners", "...presented to leadership") and never \
become their own bullet, sentence, or skill item, even when an atom describes \
them.
7. Skills: add an item only if the fact behind it appears in an offered atom \
or on the current resume. Keep the existing categories.

Tailoring goals, in priority order:
- Lead with what the brief values. Reorder bullets, projects and skill items so \
the strongest matches for this job come first. Rewrite the summary for this \
role, in the brief's own vocabulary where truthful.
- Use the brief's words inside claims, not appended to them. "Analyzed 60M \
orders in SQL to find..." is good; "..., using SQL and Python" tacked onto a \
finished sentence is not.
- Add atoms that cover must-haves the resume does not yet show. Drop or \
shorten bullets that serve this job least. The finished resume must fit \
{pages} A4 page(s); a separate tool measures that and will ask you to cut if \
needed, so prefer fewer, stronger bullets over many.
- One achievement per bullet, metric-led when an atom provides one, under 500 \
characters.
- Links: use only URLs present on the current resume or in an atom's `links`. \
Map an atom's `github` link to type "github" and a medium.com `article` link to \
type "medium"; omit others.

Output the complete resume (not a patch), then the change log with one entry \
per change: `where` (a short path such as "summary", \
"experience[Target Corporation].bullets[0]", "projects[1]", "skills[Technical]"), \
`atom_id`, and `why` (the brief requirement it serves).\
"""

CUT_INSTRUCTION = (
    "The rendered resume is over the page budget at the smallest readable font. "
    "Remove exactly one item — the bullet or project that serves this brief "
    "least — and return the complete resume. Do not shorten or rewrite anything "
    "else. The change log has one entry: `where` names the removed item, "
    "`atom_id` is null, `why` says why it was the weakest for this brief."
)


def system_prompt(max_pages: int = 1) -> str:
    return SYSTEM_TEMPLATE.format(pages=max_pages)


def _evidence_block(hits: list[BriefHit]) -> list[dict[str, Any]]:
    """Atoms as the model should see them: facts plus the retriever's
    reason for offering each one. Retrieval scores are dropped — the
    rank order carries that information and a number invites the model
    to reason about the scorer instead of the job."""
    out = []
    for h in hits:
        a = h.atom.model_dump(mode="json", exclude_none=True)
        a["links"] = {k: v for k, v in a.get("links", {}).items() if v}
        a["covers"] = list(h.covers)
        a["supports"] = list(h.supports)
        a.pop("keywords", None)  # retrieval tags, not content
        out.append(a)
    return out


def uncovered_must_haves(brief: JobBrief, hits: list[BriefHit]) -> tuple[str, ...]:
    covered = {req for h in hits for req in h.covers}
    return tuple(m for m in brief.must_haves if m not in covered)


def build_prompt(resume: Resume, brief: JobBrief, hits: list[BriefHit]) -> str:
    """The user turn. Pure function of its inputs, so tests can assert on it."""
    uncovered = uncovered_must_haves(brief, hits)
    parts = [
        "# Job brief",
        json.dumps(brief.model_dump(mode="json"), indent=1),
        "",
        "# Must-haves with no supporting evidence. Do not claim these.",
        "\n".join(f"- {m}" for m in uncovered) if uncovered else "(none)",
        "",
        "# Evidence atoms, ranked by relevance to the brief",
        json.dumps(_evidence_block(hits), indent=1),
        "",
        "# Current resume",
        json.dumps(resume.model_dump(mode="json"), indent=1),
        "",
        "Return the tailored resume and the change log.",
    ]
    return "\n".join(parts)


def _feedback(problems: list[str]) -> str:
    return (
        "Your previous output was rejected. Fix every problem below and return "
        "the complete corrected resume and change log again.\n\n"
        + "\n".join(f"- {p}" for p in problems)
    )


# ---------------------------------------------------------------------------
# Deterministic grounding check
# ---------------------------------------------------------------------------

# Figures as they appear in prose: $8M, 25,000, 20%, 90M, 2.5, 150. A trailing
# "+" is stripped so "$8M+" and "$8M" are the same figure.
_NUM = re.compile(r"\$?\d[\d,]*(?:\.\d+)?[MKkBx%]*")


def _numbers(text: str) -> set[str]:
    return {m.rstrip("+") for m in _NUM.findall(text)}


def verify(out: TailoredResume, original: Resume, hits: list[BriefHit]) -> list[str]:
    """Grounding problems a string comparison can catch. Empty list = pass.

    This is the cheap, exact half of invariant #3. It cannot tell whether
    a sentence is *true*; it can tell whether a number, an atom id, or a
    placement has any basis in the inputs. The Judge does the rest.
    """
    offered: dict[str, Atom] = {h.atom.id: h.atom for h in hits}
    problems: list[str] = []

    companies = {e.company for e in out.resume.experience}
    for c in out.changes:
        if c.atom_id is None:
            continue
        atom = offered.get(c.atom_id)
        if atom is None:
            problems.append(f"change at {c.where!r} cites atom {c.atom_id!r}, which was not offered")
            continue
        if atom.section == "education":
            problems.append(f"atom {atom.id!r} is an education atom and may not be inserted")
        elif atom.section == "experience" and atom.source not in companies:
            problems.append(
                f"atom {atom.id!r} belongs under company {atom.source!r}, "
                f"which is not an experience entry in the output"
            )

    if out.resume.contact != original.contact:
        problems.append("`contact` must be returned unchanged")
    if out.resume.education != original.education:
        problems.append("`education` must be returned unchanged")

    allowed = _numbers(original.model_dump_json())
    for atom in offered.values():
        allowed |= _numbers(atom.model_dump_json())
    unknown = _numbers(out.resume.model_dump_json()) - allowed
    if unknown:
        problems.append(
            "these figures appear in neither the current resume nor any offered "
            f"atom and must be removed: {sorted(unknown)}"
        )

    # New skill items must have a basis somewhere in the inputs. Compared
    # as normalised token strings (same tokenizer as retrieval), so
    # "Knowledge Graphs" is grounded by an atom that says "knowledge graph".
    corpus = _normalise(original.model_dump_json() + " " + " ".join(a.model_dump_json() for a in offered.values()))
    existing = {_normalise(i) for g in original.skills for i in g.items}
    for group in out.resume.skills:
        for item in group.items:
            norm = _normalise(item)
            if norm and norm not in existing and norm not in corpus:
                problems.append(
                    f"skill item {item!r} in {group.category!r} has no basis in the "
                    "current resume or any offered atom; remove it"
                )
    return problems


def _normalise(text: str) -> str:
    return " ".join(tokenize(text))


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

class TailorError(RuntimeError):
    """Both attempts failed. `.problems` holds the last rejection reasons."""

    def __init__(self, problems: list[str]) -> None:
        super().__init__("Tailor output rejected after retry:\n" + "\n".join(f"- {p}" for p in problems))
        self.problems = problems


@dataclass(frozen=True)
class TailorResult:
    resume: Resume
    changes: list[Change]
    uncovered: tuple[str, ...]  # must-haves the resume could not honestly claim
    attempts: int
    usage: dict[str, int]       # summed over attempts; Phase 6 turns this into cost


def _response_text(response: Any) -> str | None:
    """The JSON text, or None if the response carries no text at all."""
    for block in response.content:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            return block.text
    return None


def _default_client():
    from anthropic import Anthropic  # reads ANTHROPIC_API_KEY
    return Anthropic()


def _run(
    *,
    messages: list[dict[str, str]],
    original: Resume,
    brief: JobBrief,
    hits: list[BriefHit],
    client: Any,
    model: str,
    effort: str,
    max_pages: int,
) -> TailorResult:
    """The validate -> verify -> retry loop shared by tailor() and revise()."""
    client = client or _default_client()
    usage = {"input_tokens": 0, "output_tokens": 0}
    schema = output_schema()
    system = system_prompt(max_pages)

    problems: list[str] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        # Streamed, not because we want tokens as they arrive, but because
        # the SDK refuses non-streaming requests whose max_tokens could
        # take >10 min. get_final_message() returns the same Message
        # object create() would.
        with client.messages.stream(
            model=model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            output_config={
                "effort": effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        ) as stream:
            response = stream.get_final_message()
        usage["input_tokens"] += response.usage.input_tokens
        usage["output_tokens"] += response.usage.output_tokens
        text = _response_text(response)

        if text is None:
            # Nothing to validate and nothing to carry forward. A refusal or
            # an empty response is not something a retry turn can fix.
            raise TailorError([
                f"response contained no text block "
                f"(stop_reason={response.stop_reason!r}, "
                f"blocks={[getattr(b, 'type', '?') for b in response.content]})"
            ])
        if response.stop_reason == "max_tokens":
            problems = ["output was cut off at the token limit; return a shorter resume"]
        elif response.stop_reason == "refusal":
            raise TailorError([f"model refused (stop_reason='refusal'): {text[:300]}"])
        else:
            try:
                out = TailoredResume.model_validate_json(text)
            except ValidationError as e:
                # `loc` is the path into the JSON — the retry signal from models.py
                problems = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()]
            else:
                problems = verify(out, original, hits)
                if not problems:
                    return TailorResult(
                        resume=out.resume,
                        changes=out.changes,
                        uncovered=uncovered_must_haves(brief, hits),
                        attempts=attempt,
                        usage=usage,
                    )

        # Carry the rejected turn forward so the model corrects, not restarts.
        messages += [
            {"role": "assistant", "content": text},
            {"role": "user", "content": _feedback(problems)},
        ]

    raise TailorError(problems)


def tailor(
    resume: Resume,
    brief: JobBrief,
    hits: list[BriefHit],
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_pages: int = 1,
) -> TailorResult:
    """resume + brief + evidence -> validated, grounded TailorResult.

    `client` is injectable so tests run the whole validate/verify/retry
    path against a stub without spending credits.
    """
    return _run(
        messages=[{"role": "user", "content": build_prompt(resume, brief, hits)}],
        original=resume, brief=brief, hits=hits,
        client=client, model=model, effort=effort, max_pages=max_pages,
    )


def revise(
    resume: Resume,
    brief: JobBrief,
    hits: list[BriefHit],
    instruction: str,
    *,
    client: Any = None,
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    max_pages: int = 1,
) -> TailorResult:
    """A follow-up edit to an already-tailored resume, under the same rules.

    Used by the fit loop (CUT_INSTRUCTION). `resume` here is the tailored
    one, and it is also the grounding baseline: verify() runs against it,
    so a revision can remove and rephrase but cannot introduce figures or
    skills the tailored resume and offered atoms don't already contain.
    """
    prompt = build_prompt(resume, brief, hits) + "\n\n# Revision requested\n" + instruction
    return _run(
        messages=[{"role": "user", "content": prompt}],
        original=resume, brief=brief, hits=hits,
        client=client, model=model, effort=effort, max_pages=max_pages,
    )


# ---------------------------------------------------------------------------
# CLI — Phase 3 acceptance
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # python -m src.agents.tailor --brief data/sample_jd.brief.yaml --render
    import argparse

    from dotenv import load_dotenv

    from src.atoms import Profile
    from src.tools.evidence import retrieve_for_brief

    load_dotenv()

    ap = argparse.ArgumentParser(description="Tailor resume.json to a job brief.")
    ap.add_argument("--brief", required=True, help="path to a *.brief.yaml")
    ap.add_argument("--resume", default=ROOT / "assets" / "resume.json")
    ap.add_argument("--profile", default=ROOT / "data" / "profile.yaml")
    ap.add_argument("--out", default=ROOT / "output" / "tailored" / "resume.json")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--effort", default=DEFAULT_EFFORT, help="adaptive-thinking effort: low|medium|high")
    ap.add_argument("--pages", type=int, default=1, help="page budget stated to the model")
    ap.add_argument("--render", action="store_true", help="also render the PDF and report pages")
    args = ap.parse_args()

    brief = JobBrief.load(args.brief)
    resume = Resume.load(args.resume)
    profile = Profile.load(args.profile)
    hits = retrieve_for_brief(brief, k=len(profile.atoms), profile=profile)

    print(f"brief: {brief.title} @ {brief.company} — {len(hits)} atoms offered, "
          f"model {args.model}, effort {args.effort}")
    result = tailor(resume, brief, hits, model=args.model, effort=args.effort, max_pages=args.pages)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(result.resume.model_dump_json(indent=2), encoding="utf-8")
    out.with_name("changes.json").write_text(
        json.dumps([c.model_dump() for c in result.changes], indent=2), encoding="utf-8"
    )

    print(f"attempts: {result.attempts}   tokens in/out: "
          f"{result.usage['input_tokens']}/{result.usage['output_tokens']}")
    if result.uncovered:
        print(f"uncovered must-haves (not claimed): {', '.join(result.uncovered)}")
    print("changes:")
    for c in result.changes:
        print(f"  {c.where:<48} {c.atom_id or '-':<32} {c.why}")
    print(f"wrote {out} and {out.with_name('changes.json')}")

    if args.render:
        from src.tools.render import Renderer
        with Renderer() as r:
            res = r.render(result.resume, out_path=out.with_suffix(".pdf"))
        print(f"rendered {res.pdf_path} — {res.page_count} page(s), fill {res.fill:.2f} "
              f"(fit loop: python -m src.tools.fit --brief ...)")
