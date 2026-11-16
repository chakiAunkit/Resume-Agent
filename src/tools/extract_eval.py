"""Extractor eval: how close is the agent's brief to the hand-written one,
and does retrieval still rank the same atoms?

Every `data/<jd>.brief.yaml` with a `data/<jd>.txt` beside it is a case.
For each, the Extractor runs on the raw JD and the result is compared to
the human brief two ways:

  1. Per-bucket precision / recall on normalised items. Items are
     matched by their retrieval tokens (`tokenize`), so "LLMs" == "LLM"
     and "e-commerce" == "eCommerce", with spacing ignored (`_key`), so
     "problem solving" == "problem-solving". A hit in the wrong bucket is
     reported separately from a miss — bucketing errors and extraction
     errors have different fixes (prompt rule vs. prompt example).
  2. Retrieval. `retrieve_for_brief` on both briefs; the top-k atom sets
     are compared. This is the number that matters downstream: the
     Tailor sees atoms, not requirement strings. Two briefs that differ
     in wording but pull the same atoms in the same order are, for the
     pipeline, the same brief.

Each case is extracted several times (`--runs`, default 3) because the
Extractor is not deterministic: identical inputs at temperature 0 have
produced must-have recall between 0.50 and 0.82 on one case, so a single
run cannot tell a prompt change from sampling noise. Scores are reported
as mean (min-max), and every expected item carries the number of runs it
was extracted in — an item that appears in 2 runs of 3 is a rule the
model follows sometimes, which is the signal for moving that rule out of
the prompt and into `verify()`, where the retry loop enforces it.

Repeats are an *evaluation* device. Production extracts once: nothing
here changes `extract()` or the pipeline.

Extracted briefs are written to `data/evals/extracted/<jd>.<model>.brief.yaml`
(one file per run when there are several) so a run is a diffable artifact,
not a terminal scrollback. Paste the summary table into
`data/evals/README.md` — that file is the log of record, this tool only
produces rows for it.

`--rescore` skips the API entirely: it re-scores the briefs already saved
under `data/evals/extracted/` (the `.r1`-`.rN` files of a multi-run eval)
against the current reference briefs, with the current tokenizer and
profile. Scoring is deterministic, so a rescore isolates a change to the
*scorer* (tokenizer, `_key()`, a reference brief) from sampling noise:
run it before and after the change and the whole difference is the
change. It does not see extraction-side effects — `repair()` and
`verify()` also use the tokenizer — which only a fresh run shows.

    python -m src.tools.extract_eval                      # all cases, 3 runs each
    python -m src.tools.extract_eval --runs 1             # single pass, old behaviour
    python -m src.tools.extract_eval --model <id>         # override
    python -m src.tools.extract_eval --case sample_jd     # one case
    python -m src.tools.extract_eval --rescore            # saved runs, no API calls

Model comparison. Each model runs with its own settings from
`config.MODEL_SETTINGS` (temperature, reasoning effort, ...), shown in the
report header. `--tag` names a batch, so its files never overwrite an
earlier batch's (the M7/M8 briefs are the log's record of the accepted
Extractor); `--reasoning-effort` overrides the table and is added to the
file names too.

    python -m src.tools.extract_eval --runs 6 --tag m9                     # gpt-oss, fresh baseline
    python -m src.tools.extract_eval --runs 6 --tag m9 --model qwen/qwen3.8-27b
    python -m src.tools.extract_eval --rescore --model qwen/qwen3.8-27b    # that model's saved batches
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.brief import JobBrief
from src.tools.evidence import tokenize

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROFILE = DATA / "profile.yaml"
OUT_DIR = DATA / "evals" / "extracted"   # beside the retrieval log in data/evals/README.md

BUCKETS = ("must_haves", "nice_to_haves", "soft_skills")


# "retrieval-augmented generation (RAG)" — a trailing acronym in brackets.
_ACRONYM = re.compile(r"\s*\(([A-Za-z][A-Za-z0-9]{1,7})\)\s*$")


def _key(item: str) -> str:
    """Matching key: retrieval tokens joined *without spaces*, with a
    trailing parenthesised acronym dropped when it spells the initials of
    the words before it.

    The prompt asks for "spelled-out term (ACRONYM)" when the JD gives both;
    a model that returns only the spelled-out half has written the same
    requirement, not a different one, and scoring it as one miss plus one
    extra double-counts a spelling difference. The initials test keeps this
    narrow: "Data Scientist (Analytics)" is untouched.

    Joining without spaces makes word boundaries irrelevant, so "problem
    solving" == "problem-solving" (which tokenize() makes "problemsolving").
    This lives here and not in tokenize() because retrieval needs word
    tokens; the eval only needs to know whether two items are the same
    string. Audited on 2026-09-23 against every item in the references and
    saved runs: the only merge it makes is that one."""
    m = _ACRONYM.search(item)
    if m:
        stem = item[: m.start()]
        initials = "".join(w[0] for w in re.findall(r"[A-Za-z]+", stem)).lower()
        acronym = m.group(1).lower()
        if initials == acronym or initials.rstrip("s") == acronym.rstrip("s"):
            item = stem
    return "".join(tokenize(item))


@dataclass
class BucketScore:
    bucket: str
    expected: int
    hit: list[str] = field(default_factory=list)        # in the right bucket
    misplaced: list[str] = field(default_factory=list)  # extracted, but in another bucket
    missed: list[str] = field(default_factory=list)     # not extracted anywhere
    extra: list[str] = field(default_factory=list)      # extracted here, not expected anywhere

    @property
    def precision(self) -> float:
        got = len(self.hit) + len(self.extra)
        return len(self.hit) / got if got else 1.0

    @property
    def recall(self) -> float:
        return len(self.hit) / self.expected if self.expected else 1.0


@dataclass
class Comparison:
    buckets: dict[str, BucketScore]
    title_match: bool
    company_match: bool
    seniority_match: bool
    years_match: bool

    @property
    def scalar_fields(self) -> str:
        flags = [("title", self.title_match), ("company", self.company_match),
                 ("seniority", self.seniority_match), ("years", self.years_match)]
        return " ".join(f"{name}:{'ok' if ok else 'X'}" for name, ok in flags)


def compare(expected: JobBrief, got: JobBrief) -> Comparison:
    """Pure function of two briefs, so it's testable on fixtures."""
    exp_where = {_key(i): b for b in BUCKETS for i in getattr(expected, b)}
    got_where = {_key(i): b for b in BUCKETS for i in getattr(got, b)}
    scores: dict[str, BucketScore] = {}
    for b in BUCKETS:
        score = BucketScore(bucket=b, expected=len(getattr(expected, b)))
        for item in getattr(expected, b):
            k = _key(item)
            if k not in got_where:
                score.missed.append(item)
            elif got_where[k] == b:
                score.hit.append(item)
            else:
                score.misplaced.append(f"{item} -> {got_where[k]}")
        for item in getattr(got, b):
            if _key(item) not in exp_where:
                score.extra.append(item)
        scores[b] = score
    return Comparison(
        buckets=scores,
        title_match=_key(expected.title) == _key(got.title),
        company_match=_key(expected.company) == _key(got.company),
        seniority_match=expected.seniority == got.seniority,
        years_match=expected.years_experience == got.years_experience,
    )


def retrieval_overlap(expected: JobBrief, got: JobBrief, k: int = 5,
                      profile: Path = PROFILE) -> tuple[list[str], list[str], float]:
    """Top-k atom ids for each brief and their Jaccard overlap. Needs the
    real profile.yaml; callers skip this when it's absent."""
    from src.tools.evidence import retrieve_for_brief
    a = [h.atom.id for h in retrieve_for_brief(expected, k=k, profile=profile)]
    b = [h.atom.id for h in retrieve_for_brief(got, k=k, profile=profile)]
    union = set(a) | set(b)
    return a, b, (len(set(a) & set(b)) / len(union) if union else 1.0)


def _slug(model: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")


def run_path(stem: str, slug: str, run: int, runs: int, out_dir: Path = OUT_DIR) -> Path:
    """Where run `run` of a batch is saved. A single-run eval has no `.rN`."""
    return out_dir / f"{stem}.{slug}{f'.r{run}' if runs > 1 else ''}.brief.yaml"


# Groq: "... on tokens per minute (TPM): Limit 8000, Requested 11053, ..."
_LIMIT = re.compile(r"on ([^:]+?): Limit (\d+), Requested (\d+)")


def describe_provider_error(exc: Exception) -> str:
    """One line for a provider failure. Groq's "Request too large" is not a
    passing rate limit — the same request fails every time — so it gets the
    limit it broke, with its unit, instead of a truncated message."""
    text = str(exc)
    if "request too large" in text.lower():
        m = _LIMIT.search(text)
        detail = f"on {m.group(1)}: limit {m.group(2)}, requested {m.group(3)}" if m else text[:300]
        return (f"request too large ({detail}). Lower this model's max_tokens in "
                "config.MODEL_SETTINGS, or run it on another provider (--provider)")
    return f"{type(exc).__name__}: {text[:200]}"


def batch_slug(model: str, tag: str | None = None, reasoning_effort: str | None = None) -> str:
    """The name a batch's files are saved under: the model slug, then the
    tag and any reasoning-effort override as dot-separated parts, so two
    batches of one model never share file names:
    'openai-gpt-oss-120b', 'qwen-qwen3-8-27b.m9', '....m9.effort-high'."""
    parts = [_slug(model)]
    if tag:
        parts.append(_slug(tag))
    if reasoning_effort:
        parts.append(f"effort-{_slug(reasoning_effort)}")
    return ".".join(parts)


def cases() -> list[tuple[str, Path, Path]]:
    out = []
    for brief_path in sorted(DATA.glob("*.brief.yaml")):
        stem = brief_path.name[: -len(".brief.yaml")]
        jd_path = DATA / f"{stem}.txt"
        if jd_path.exists():
            out.append((stem, jd_path, brief_path))
    return out


def report(stem: str, cmp: Comparison, retrieval: tuple[list[str], list[str], float] | None, meta: dict[str, Any]) -> str:
    lines = [f"## {stem} — {meta['model']} ({meta['format']}, {meta['attempts']} attempt(s), "
             f"tokens {meta['prompt_tokens']}/{meta['completion_tokens']})",
             f"   {cmp.scalar_fields}"]
    for b, s in cmp.buckets.items():
        lines.append(f"   {b:<14} P {s.precision:.2f}  R {s.recall:.2f}   "
                     f"hit {len(s.hit)}/{s.expected}  misplaced {len(s.misplaced)}  extra {len(s.extra)}")
        for m in s.missed:
            lines.append(f"       missed:    {m}")
        for m in s.misplaced:
            lines.append(f"       misplaced: {m}")
        for e in s.extra:
            lines.append(f"       extra:     {e}")
    if retrieval:
        a, b, j = retrieval
        lines.append(f"   retrieval top-{len(a)}: overlap {j:.2f}")
        for i, (x, y) in enumerate(zip(a, b), 1):
            mark = " " if x == y else "*"
            lines.append(f"     {i}. {mark} {x:<40} {y}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Several runs of the same case
# ---------------------------------------------------------------------------

@dataclass
class RunOutcome:
    """One extraction. `comparison` is None when the run failed outright —
    a failure is data about the prompt, not a reason to lose the other runs."""
    comparison: Comparison | None = None
    retrieval: tuple[list[str], list[str], float] | None = None
    usage: dict[str, int] = field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0})
    attempts: int = 0
    format: str = "-"
    error: str | None = None
    rejected: tuple[str, ...] = ()   # why earlier attempts were retried (see ExtractResult)
    repaired: tuple[str, ...] = ()   # mechanical fixes repair() applied (see ExtractResult)


def spread(values: list[float]) -> tuple[float, float, float]:
    """mean, min, max. Empty input is (0, 0, 0) — callers check first."""
    if not values:
        return 0.0, 0.0, 0.0
    return sum(values) / len(values), min(values), max(values)


def fmt_spread(values: list[float]) -> str:
    if not values:
        return "n/a"
    mean, lo, hi = spread(values)
    return f"{mean:.2f}" if lo == hi else f"{mean:.2f} ({lo:.2f}-{hi:.2f})"


@dataclass
class ItemStability:
    """How often each item survived, across runs of one case and bucket."""
    runs: int
    expected: list[str]
    hit: Counter            # expected item -> runs where it landed in this bucket
    misplaced: Counter      # expected item -> runs where it landed elsewhere
    extra: Counter          # produced item -> runs where it appeared unexpectedly

    def always(self) -> list[str]:
        return [i for i in self.expected if self.hit[i] == self.runs]

    def sometimes(self) -> list[str]:
        """The interesting ones: the model knows the rule but not reliably."""
        return [i for i in self.expected if 0 < self.hit[i] < self.runs]

    def never(self) -> list[str]:
        return [i for i in self.expected if self.hit[i] == 0]


def stability(comparisons: list[Comparison], bucket: str) -> ItemStability:
    runs = len(comparisons)
    first = comparisons[0].buckets[bucket]
    expected = first.hit + first.missed + [m.split(" -> ")[0] for m in first.misplaced]
    st = ItemStability(runs=runs, expected=sorted(expected), hit=Counter(),
                       misplaced=Counter(), extra=Counter())
    for cmp in comparisons:
        score = cmp.buckets[bucket]
        st.hit.update(score.hit)
        st.misplaced.update(m.split(" -> ")[0] for m in score.misplaced)
        st.extra.update(score.extra)
    return st


def _counted(counter: Counter, runs: int, items: list[str] | None = None) -> str:
    keys = items if items is not None else [k for k, _ in counter.most_common()]
    return ", ".join(f"{k} {counter[k]}/{runs}" for k in keys)


def report_runs(stem: str, model: str, outcomes: list[RunOutcome], rescored: bool = False) -> str:
    """The multi-run report: scores as mean (min-max), items as counts.
    `rescored` marks outcomes built from saved briefs: there were no
    attempts or tokens to report, so the header says so instead."""
    ok = [o for o in outcomes if o.comparison is not None]
    runs = len(outcomes)
    tokens_in = sum(o.usage["prompt_tokens"] for o in outcomes)
    tokens_out = sum(o.usage["completion_tokens"] for o in outcomes)
    if rescored:
        lines = [f"## {stem} — {model} (rescored {len(ok)}/{runs} saved runs, no API calls)"]
    else:
        lines = [f"## {stem} — {model} ({len(ok)}/{runs} runs ok, "
                 f"attempts {'/'.join(str(o.attempts) if o.comparison else 'x' for o in outcomes)}, "
                 f"tokens {tokens_in}/{tokens_out})"]
    for o in outcomes:
        if o.error:
            lines.append(f"   FAILED: {o.error}")
    # A retry that succeeded is invisible in the scores; its cause is the
    # clearest pointer to a rule the model keeps breaking.
    reasons = Counter(r for o in outcomes for r in o.rejected)
    if reasons:
        lines.append("   retried because:")
        for reason, n in reasons.most_common(6):
            lines.append(f"       {n}x {reason[:110]}")
    # Repairs cost nothing and never fail, but each one is a rule the model
    # broke. The count is how compliance is still measured.
    repairs = Counter(r for o in outcomes for r in o.repaired)
    if repairs:
        lines.append("   repaired:")
        for fix, n in repairs.most_common(8):
            lines.append(f"       {n}x {fix[:110]}")
    if not ok:
        return "\n".join(lines)

    comparisons = [o.comparison for o in ok]
    for field_name in ("title", "company", "seniority", "years"):
        agreed = sum(getattr(c, f"{field_name}_match") for c in comparisons)
        if agreed < len(ok):
            lines.append(f"   {field_name}: matched in {agreed}/{len(ok)} runs")

    for bucket in BUCKETS:
        st = stability(comparisons, bucket)
        lines.append(
            f"   {bucket:<14} P {fmt_spread([c.buckets[bucket].precision for c in comparisons])}"
            f"   R {fmt_spread([c.buckets[bucket].recall for c in comparisons])}"
            f"   expected {len(st.expected)}"
        )
        if st.always():
            lines.append(f"       always:    {len(st.always())} of {len(st.expected)}")
        if st.sometimes():
            lines.append(f"       sometimes: {_counted(st.hit, st.runs, st.sometimes())}")
        if st.never():
            lines.append(f"       never:     {', '.join(st.never())}")
        if st.misplaced:
            lines.append(f"       misplaced: {_counted(st.misplaced, st.runs)}")
        if st.extra:
            lines.append(f"       extra:     {_counted(st.extra, st.runs)}")

    overlaps = [o.retrieval[2] for o in ok if o.retrieval]
    if overlaps:
        lines.append(f"   retrieval overlap: {fmt_spread(overlaps)}")
        atoms = Counter(a for o in ok if o.retrieval for a in o.retrieval[1])
        lines.append(f"       extracted brief pulled: {_counted(atoms, len(overlaps))}")
        lines.append(f"       reference pulled:       {', '.join(ok[0].retrieval[0])}")
    return "\n".join(lines)


def summary_row(stem: str, model: str, outcomes: list[RunOutcome], rescored: bool = False) -> str:
    ok = [o for o in outcomes if o.comparison is not None]
    if not ok:
        return f"| {stem} | {model} | {len(outcomes)} | all runs failed |"
    mh = [o.comparison.buckets["must_haves"] for o in ok]
    overlaps = [o.retrieval[2] for o in ok if o.retrieval]
    # Mean over the runs that produced a brief; a failed run reports no usage.
    tokens = (sum(o.usage["prompt_tokens"] for o in ok) // len(ok),
              sum(o.usage["completion_tokens"] for o in ok) // len(ok))
    return (f"| {stem} | {model} | {len(ok)}/{len(outcomes)} | "
            f"{fmt_spread([s.precision for s in mh])} | {fmt_spread([s.recall for s in mh])} | "
            f"{fmt_spread(overlaps)} | {'rescored' if rescored else f'{tokens[0]}/{tokens[1]}'} |")


# ---------------------------------------------------------------------------
# Rescoring saved runs
# ---------------------------------------------------------------------------

def saved_runs(stem: str, out_dir: Path = OUT_DIR) -> dict[str, list[Path]]:
    """The saved multi-run briefs of one case, grouped by model slug and
    sorted by run number: {"openai-gpt-oss-120b": [r1, r2, ...]}.

    Only `<stem>.<slug>.r<N>.brief.yaml` counts. A single-run file (no
    `.rN`) is an older, unrepeated artifact and is left out, so every
    group is one multi-run eval. Runs sort numerically (r10 after r9).
    """
    pattern = re.compile(rf"^{re.escape(stem)}\.(.+)\.r(\d+)\.brief\.yaml$")
    groups: dict[str, list[tuple[int, Path]]] = defaultdict(list)
    for path in out_dir.glob(f"{stem}.*.brief.yaml"):
        m = pattern.match(path.name)
        if m:
            groups[m.group(1)].append((int(m.group(2)), path))
    return {slug: [p for _, p in sorted(runs)] for slug, runs in sorted(groups.items())}


def rescore(expected: JobBrief, paths: list[Path], k: int = 5,
            profile: Path | None = PROFILE) -> list[RunOutcome]:
    """Score saved briefs against a reference: no client, no tokens.

    Retrieval overlap is computed when `profile` exists (pass None to
    skip it). The briefs are scored as saved — `repair()` is not
    re-applied — because the point is to measure the scorer on a fixed
    set of Extractor outputs.
    """
    use_profile = profile is not None and Path(profile).exists()
    outcomes = []
    for path in paths:
        got = JobBrief.load(path)
        outcomes.append(RunOutcome(
            comparison=compare(expected, got),
            retrieval=retrieval_overlap(expected, got, k=k, profile=profile) if use_profile else None,
            format="saved",
        ))
    return outcomes


if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Score the Extractor against the hand-written briefs.")
    ap.add_argument("--case", default=None, help="run one case by stem, e.g. sample_jd")
    ap.add_argument("--model", default=None,
                    help="override EXTRACTOR_MODEL; with --rescore, only that model's saved runs")
    ap.add_argument("--runs", type=int, default=3,
                    help="extractions per case (default 3; the Extractor is not deterministic). "
                         "Use 1 for a quick single pass.")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score the saved .rN briefs in data/evals/extracted/; no API calls")
    ap.add_argument("-k", type=int, default=5, help="top-k for the retrieval overlap")
    ap.add_argument("--tag", default=None,
                    help="batch name added to the saved file names (e.g. m9), so an earlier batch is kept")
    ap.add_argument("--provider", default=None,
                    help="groq | cerebras | openrouter | huggingface (default: the model's own, else .env)")
    ap.add_argument("--reasoning-effort", default=None,
                    help="override the model's reasoning effort from config.MODEL_SETTINGS "
                         "(gpt-oss: low/medium/high; qwen3.8: none/default/low/medium/high)")
    args = ap.parse_args()

    selected = [c for c in cases() if not args.case or c[0] == args.case]
    if not selected:
        ap.error("no cases: need data/<jd>.txt beside data/<jd>.brief.yaml")

    header = ("| case | model | runs ok | must-have P | must-have R | retrieval overlap@k "
              "| tokens in/out (mean) |\n|---|---|---|---|---|---|---|")

    if args.rescore:
        # Deterministic and offline: no .env, no SDK, no credits.
        summary = []
        for stem, _jd_path, brief_path in selected:
            expected = JobBrief.load(brief_path)
            groups = saved_runs(stem)
            if args.model:     # every batch of that model: the bare slug and any tagged ones
                groups = {s: p for s, p in groups.items()
                          if s == _slug(args.model) or s.startswith(_slug(args.model) + ".")}
            if not groups:
                print(f"## {stem} — no saved .rN briefs in {OUT_DIR}\n")
                continue
            for slug, paths in groups.items():
                outcomes = rescore(expected, paths, k=args.k)
                print(report_runs(stem, slug, outcomes, rescored=True) + "\n")
                summary.append(summary_row(stem, slug, outcomes, rescored=True))
        if summary:
            print("Rows for data/evals/README.md:")
            print(header)
            print("\n".join(summary))
        sys.exit(0)

    from dotenv import load_dotenv
    from openai import APIError

    from src.agents.extractor import ExtractorError, brief_to_yaml, extract, extractor_route
    from src.config import model_settings, route_for

    load_dotenv()

    default = extractor_route()
    route = route_for(args.model or default.model, default, args.provider)
    model = route.model
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    settings = model_settings(model, args.reasoning_effort)
    label = f"{model} [{settings.label}]"          # what the reports and rows show
    slug = batch_slug(model, args.tag, args.reasoning_effort)
    print(f"extractor: {label} via {route.base_url}; saving as "
          f"{OUT_DIR.name}/<case>.{slug}{'.rN' if args.runs > 1 else ''}.brief.yaml\n")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for stem, jd_path, brief_path in selected:
        expected = JobBrief.load(brief_path)
        jd_text = jd_path.read_text(encoding="utf-8")
        outcomes: list[RunOutcome] = []

        for run in range(1, args.runs + 1):
            if args.runs > 1:
                print(f"{stem}: run {run}/{args.runs} ...", flush=True)
            try:
                result = extract(jd_text, route=route, model=model, settings=settings)
            except ExtractorError as e:
                # Two rejected attempts is a result about the prompt; keep the
                # other runs and report the failure rate.
                outcomes.append(RunOutcome(error=f"run {run}: {e.problems[0] if e.problems else e}"))
                run_path(stem, slug, run, args.runs).unlink(missing_ok=True)   # no stale file from a past batch
                continue
            except APIError as e:
                # The provider gave up (rate limit, outage) even after the SDK's
                # retries. Says nothing about the prompt, but losing the runs
                # already done would waste them: record it and carry on.
                outcomes.append(RunOutcome(error=f"run {run}: provider error, not scored "
                                                 f"({describe_provider_error(e)})"))
                run_path(stem, slug, run, args.runs).unlink(missing_ok=True)
                continue
            out = run_path(stem, slug, run, args.runs)
            out.write_text(
                brief_to_yaml(result.brief, header=f"# extracted from {jd_path.name} by {model} "
                                                   f"(run {run}; {settings.label})"),
                encoding="utf-8")
            outcomes.append(RunOutcome(
                comparison=compare(expected, result.brief),
                retrieval=retrieval_overlap(expected, result.brief, k=args.k) if PROFILE.exists() else None,
                usage=dict(result.usage), attempts=result.attempts, format=result.format,
                rejected=result.rejected, repaired=result.repaired,
            ))

        if args.runs == 1 and outcomes[0].comparison is not None:
            o = outcomes[0]
            meta = {"model": label, "format": o.format, "attempts": o.attempts, **o.usage}
            print(report(stem, o.comparison, o.retrieval, meta))
            for reason in o.rejected:
                print(f"   retried because: {reason}")
            for fix in o.repaired:
                print(f"   repaired: {fix}")
        else:
            print(report_runs(stem, label, outcomes))
        print(f"   wrote {OUT_DIR}\n")
        summary.append(summary_row(stem, label, outcomes))

    print("Rows for data/evals/README.md:")
    print(header)
    print("\n".join(summary))
