"""The Extractor — raw JD text -> `JobBrief`. Second LLM node, first on
an open model.

It replaces the human who wrote the Phase 3 briefs by hand and nothing
downstream notices: `retrieve_for_brief`, the Tailor and (later) the
Judge all read a `JobBrief`, whoever wrote it. That is the interface-first
payoff — and it also makes the hand-written briefs this agent's eval
set (`src/tools/extract_eval.py`).

Same three-layer contract as the Tailor, on a different SDK:
  1. Shape by constrained decoding — `response_format=json_schema`
     (Groq / OpenAI / Ollama all accept it; a provider that rejects the
     schema for a given model gets one automatic downgrade to plain JSON
     mode, and the gate below still holds).
  2. Constraints by Pydantic — `JobBrief` re-validates lengths (an item
     over 80 chars is a pasted sentence), the seniority Literal, the
     non-empty must_haves. A `ValidationError`'s `loc` paths are the
     retry signal.
  3. Grounding by string checks — `verify()`: every requirement has a
     basis in the JD text, the company is named in it, the title is
     spelled as it is, the years figure appears in it. Cheap, exact, and the thing a model can't be trusted
     to self-report.

Cost routing (roadmap Phase 4 learning): this is the mechanical step, so
it runs on a cheap open model via Groq by default. The provider is a
`.env` switch (`src/config.py`); the model is injectable for tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.brief import JobBrief
from src.config import (DEFAULT_MAX_TOKENS, ModelRoute, ModelSettings, extractor_route, model_settings,
                        route_for)
from src.tools.evidence import tokenize

ROOT = Path(__file__).resolve().parents[2]
MAX_ATTEMPTS = 2        # roadmap: retry once
# Two different kinds of retry, deliberately kept apart:
#   MAX_ATTEMPTS           — *content* retries: the answer was wrong, so the
#                            model gets it back with a list of problems.
#   MAX_PROVIDER_RETRIES   — *transport* retries: rate limits (429), 5xx,
#                            dropped connections. Same request resent; the SDK
#                            waits as long as the provider's retry-after says.
# Groq's free tier allows 8,000 tokens per minute and one extraction is
# ~5,000, so back-to-back calls (eval repeats, several JDs in Phase 5) hit
# 429 routinely. The SDK default of 2 retries gave up mid-eval.
MAX_PROVIDER_RETRIES = 6
# A brief is a few hundred tokens, but reasoning models (gpt-oss, Qwen
# thinking variants) spend the same budget on their reasoning channel
# first — the Phase 3 lesson about Claude 5's thinking tokens, on another
# SDK. At 2048 gpt-oss-120b exhausted the budget and returned an empty
# answer, which the provider reports as a 400 json_validate_failed with
# an empty failed_generation. Keep this generous; it is a ceiling, not a
# target, and unused tokens cost nothing — except on providers that check
# prompt + max_tokens against a per-minute limit up front (Groq), which is
# why each model can set its own (config.ModelSettings.max_tokens).
MAX_TOKENS = DEFAULT_MAX_TOKENS
# Temperature, top_p and reasoning effort are per model: src/config.py
# MODEL_SETTINGS. gpt-oss runs at temperature 0 (extraction, not
# generation); Qwen's thinking mode is not meant to run greedy.


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

# The few-shot example is synthetic on purpose. Using data/sample_jd.txt
# here would make eval case 001 a memorisation test instead of a
# generalisation test.
_EXAMPLE_JD = """\
Analytics Engineer — Northwind Traders (Remote, EU)
We're a fast-growing wholesale marketplace. You'll own our metrics layer.
Requirements: 3-5 years in analytics engineering or data engineering; expert
SQL and dbt for modelling and testing; Python for pipeline automation;
experience with a cloud warehouse (BigQuery or Snowflake). Strong written
communication — you'll document models for the whole company.
What you'll do: build wholesale pricing reports; own monitoring, alerting and
lineage for our pipelines; forecast demand with the planning team.
Nice to have: Airflow or Dagster; Looker; exposure to experimentation
platforms. Bonus points for open-source contributions.
We offer flexible hours and a learning budget. Northwind is an equal
opportunity employer."""

_EXAMPLE_BRIEF = {
    "title": "Analytics Engineer",
    "company": "Northwind Traders",
    "seniority": "mid",
    "years_experience": 3,
    "must_haves": ["SQL", "dbt", "data modelling", "data testing", "Python",
                   "pipeline automation", "cloud warehouse", "BigQuery", "Snowflake",
                   "wholesale", "pricing", "reports", "monitoring", "alerting",
                   "lineage", "demand forecasting"],
    "nice_to_haves": ["Airflow", "Dagster", "Looker", "experimentation platforms",
                      "open-source contributions"],
    "soft_skills": ["written communication", "documentation"],
}

# Every example below is chosen to be absent from the eval cases' briefs:
# test_prompt_contains_no_eval_case_requirements enforces it. An example
# that happens to be an eval answer turns that case into a lookup.
SYSTEM_PROMPT = f"""\
You are the Extractor in an automated resume pipeline. You read one raw job \
description and return a job brief as JSON: the requirements that matter, \
classified into three lists, plus title, company, seniority and years.

Fields:
- title: the job title exactly as the JD writes it — same words, same \
spelling — without location suffixes or requisition ids.
- company: the hiring company as named in the JD. If a staffing agency posts \
for an unnamed client, use the agency.
- seniority: one of junior | mid | senior | lead | unspecified. A level word in \
the title decides: junior, associate, entry -> junior; senior, sr -> senior; \
lead, principal, staff -> lead. Otherwise use the minimum years asked: 0-2 -> \
junior, 3-7 -> mid, 8 or more -> senior. "unspecified" only when the JD gives \
neither a level word nor a number.
- years_experience: the minimum years asked, as an integer ("5+ years" -> 5, \
"3-5 years" -> 3). null when the JD does not state a number.
- must_haves: every technical requirement the JD names, wherever it appears — \
under requirements or qualifications, and equally inside the responsibilities \
("what you'll do"). A thing the job does is a thing the candidate needs.
- nice_to_haves: ONLY requirements the JD explicitly marks as optional with a \
word like preferred, nice to have, bonus, a plus, ideally, desirable, or \
"would be great". The marker covers everything named in its sentence or its \
list, and nothing outside it. If the JD marks nothing, return an empty list — \
an empty nice_to_haves is a correct answer, and softening an unmarked \
requirement into this list is an error.
- soft_skills: communication, collaboration, attitude and working-style \
requirements. These never go in must_haves or nice_to_haves.

Why the item rules matter: each item is later matched against the candidate's \
evidence token by token, and every token must match. A bundled item matches \
nothing ("warehouse search tooling" matches neither "warehouse" nor "search"), \
a missing item is a requirement nobody checks, and a vague item matches \
everything.

Rules for every item in the three lists:
1. One concept per item — the single most important rule. Split on and, &, \
or, commas and "for", so every thing named becomes its own item:
   "expert SQL and dbt for modelling and testing"
     -> "SQL", "dbt", "data modelling", "data testing"
   "experience with a cloud warehouse (BigQuery or Snowflake)"
     -> "cloud warehouse", "BigQuery", "Snowflake"
   "build and monitor ranking models"
     -> "ranking", "model monitoring"
   A domain, industry or data type in front of a capability is its own \
requirement — split it off even when the phrase reads naturally:
   "retail forecasting" -> "retail", "forecasting"
   "insurance pricing models" -> "insurance", "pricing"
   Keep words together only when they name one technique or tool ("vector \
search", "feature store", "gradient boosting", "time series"). Test: if each \
word could stand alone as a requirement, split it.
2. Every member of a list is an item. When the JD enumerates — "X, Y, and Z", \
"X or Y", "such as X and Y" — each member becomes its own item, in the \
requirements and the responsibilities alike. Never keep only the first \
member, only the most general one, or a summary of the list:
   "own monitoring, alerting and lineage for our pipelines"
     -> "monitoring", "alerting", "lineage"
   A slash inside one term is not a list ("CI/CD" stays whole).
3. Tag-sized: one to three words, under 40 characters. Nouns, not verbs: \
"model monitoring", not "monitor models"; "forecast demand" -> "demand \
forecasting". Never copy a sentence. A requirement phrased as something the \
person will do still counts, including inside a heading or a working-style \
sentence: "Move quickly – draft forecasts and stress-test them" -> \
"forecasting", "stress testing".
4. No generic words on their own. A bare metrics, models, data, tools, \
frameworks, solutions, systems, applications or analytics is not an item; keep \
such a word only with the qualifier that makes it specific ("latency metrics", \
"sensor data"). Drop a trailing generic noun that adds nothing: "anomaly \
detection models" -> "anomaly detection", "forecasting techniques" -> \
"forecasting", "startup environment" -> "startup", "pricing use cases" -> \
"pricing", "agile methodologies" -> "agile". Umbrella labels and headings ("AI/ML", "Data & BI skills") are \
not items when the text names the specifics under them.
5. The JD's own vocabulary, spelled as the JD spells it. Keep a spelled-out \
term together with its acronym when the JD gives both ("continuous \
integration (CI)").
6. Only what the JD asks of the candidate — never the company's products, \
mission, benefits, culture statements or equal-opportunity text. Areas of the \
product and outcomes the business wants are context, not requirements: in \
"own reporting across onboarding, checkout and returns" or "improve \
activation and loyalty", only "reporting" is an item. Measurement is the \
exception: defining metrics, targets or thresholds is the person's work and \
so a requirement, even though its subject is an outcome — "define north-star \
metrics and alert thresholds for launches" -> "north-star metrics", "alert \
thresholds".
7. Years of experience belong in years_experience, never in a list — and so \
do the role names on the years line: "5+ years in data engineering or \
analytics roles" gives years_experience 5 and no items.
8. No duplicates across the three lists, and no near-duplicates within one: \
"forecasting" and "demand forecasting" are one item (keep the more specific), \
and so are "X" and "X models". Something listed as both required and \
preferred is a must-have.
9. Keep the JD's order: the requirements section first, then requirements \
implied by the responsibilities.

Before you answer, check your lists against the JD once more:
- every member of every enumeration is present;
- no item still joins two requirements (an "and", a list, or a domain in \
front of a capability);
- no bare generic word or umbrella label;
- no product areas, journey stages or business outcomes;
- no item is a longer or shorter form of another;
- the title is spelled exactly as in the JD, and seniority follows the \
level word or the years.

Example JD:
{_EXAMPLE_JD}

Example brief:
{json.dumps(_EXAMPLE_BRIEF, indent=1)}

Return only the JSON object.\
"""


def output_schema() -> dict[str, Any]:
    """JobBrief's JSON schema in the form constrained decoders want:
    every property required (defaults are a Pydantic notion), and the
    constraints a grammar can't express (lengths, minimums) stripped —
    Pydantic re-applies them on the way back in. Same division of labour
    as `transform_schema` in the Tailor."""
    schema = JobBrief.model_json_schema()
    schema["required"] = list(schema["properties"])
    return _strip_constraints(schema)


_GRAMMAR_UNFRIENDLY = {"minLength", "maxLength", "minItems", "maxItems",
                       "minimum", "maximum", "pattern", "format", "default"}


def _strip_constraints(node: Any) -> Any:
    if isinstance(node, dict):
        return {k: _strip_constraints(v) for k, v in node.items() if k not in _GRAMMAR_UNFRIENDLY}
    if isinstance(node, list):
        return [_strip_constraints(v) for v in node]
    return node


def build_messages(jd_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "# Job description\n\n" + jd_text.strip()},
    ]


def _feedback(problems: list[str]) -> str:
    return (
        "Your previous output was rejected. Fix every problem below and return "
        "the complete corrected brief as JSON.\n\n" + "\n".join(f"- {p}" for p in problems)
    )


# ---------------------------------------------------------------------------
# Deterministic grounding check
# ---------------------------------------------------------------------------

_NUM = re.compile(r"\d+")

# Rule 4 as code: repair(), not verify(). The prompt states these rules, but
# six runs showed them followed only sometimes ("analytics" alone in 5/6,
# "...environment" in 6/6). v3 made them verify() rejections and the retry
# failed twice: once the model ignored "write 'early-stage product'", once
# "keep one" made it keep the wrong item. Each fix is mechanical and
# *subtractive* — it removes or shortens an item, never adds one — so code
# applies it and cannot introduce a claim. Retries are kept for problems whose
# fix needs judgement (grounding, title, years). Applied to must_haves and
# nice_to_haves only: soft skills are phrasing ("fast-paced environments").
#
# A one-word item that is only one of these is too vague to match evidence.
# Multi-word items are fine ("data analytics", "success metrics").
_GENERIC_WORDS = frozenset(tokenize(
    "analytics metrics models data tools frameworks solutions systems "
    "applications insights methods techniques approaches"))
# Tails that never carry the requirement: dropping them leaves the concept
# intact ("early-stage product environment" -> "early-stage product"). Kept
# deliberately short — "methods" or "frameworks" can be the concept
# ("quasi-experimental methods", "measurement frameworks").
_DROPPABLE_TAIL = re.compile(r"^(.+?)[\s-]+(methodolog(?:y|ies)|environments?|use cases?)$", re.I)
# "X" plus "X <one of these>" is the same requirement written twice
# ("recommendations" and "recommendation models").
_REPEAT_HEADS = frozenset(tokenize(
    "models frameworks solutions systems applications techniques methods "
    "methodologies development environment capabilities platforms tools"))


def _letters(text: str) -> str:
    """Lower-case alphanumerics only, so 'e-commerce' == 'eCommerce' and
    'high-agency' == 'high agency'."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _same_stem(a: str, b: str) -> bool:
    """'experimentation' ~ 'experiment', 'communication' ~ 'communicate',
    'clear' ~ 'clearly'. No stemmer: two tokens match when they share a
    prefix of at least 5 characters covering 70% of the shorter one.
    Crude, symmetric, and lenient in the right direction for a check
    whose job is catching invention, not paraphrase."""
    if a == b:
        return True
    shorter = min(len(a), len(b))
    if shorter < 5:
        return False
    common = 0
    for x, y in zip(a, b):
        if x != y:
            break
        common += 1
    return common >= 5 and common >= 0.7 * shorter


def grounded(item: str, jd_text: str, jd_tokens: set[str] | None = None) -> bool:
    """An item has a basis in the JD if it appears as a phrase (punctuation
    and spacing ignored) or shares at least one substantive token with it,
    allowing for inflection. Lenient by design: this catches invention,
    not paraphrase — 'product catalog' is grounded by a JD that says
    'catalog data'."""
    if _letters(item) and _letters(item) in _letters(jd_text):
        return True
    tokens = jd_tokens if jd_tokens is not None else set(tokenize(jd_text))
    for t in tokenize(item):
        if len(t) < 3:
            continue
        if t in tokens or any(_same_stem(t, j) for j in tokens):
            return True
    return False


def repair(brief: JobBrief) -> tuple[JobBrief, tuple[str, ...]]:
    """Apply rule 4's mechanical fixes; return the repaired brief and a log.

    Subtractive only: drop a bare generic word, strip a droppable tail, drop
    "X <generic head>" when "X" is also an item, then drop any duplicate that
    stripping created (first occurrence wins, so must_haves beat
    nice_to_haves). A hand-written brief that follows the rules comes back
    unchanged — the tests pin that.
    """
    log: list[str] = []
    fixed: dict[str, list[str]] = {}
    for bucket in ("must_haves", "nice_to_haves"):
        items = []
        for item in getattr(brief, bucket):
            tokens = tokenize(item)
            if len(tokens) == 1 and tokens[0] in _GENERIC_WORDS:
                log.append(f"dropped bare generic word {item!r} ({bucket})")
                continue
            tail = _DROPPABLE_TAIL.match(item.strip())
            if tail:
                short = tail.group(1).strip()
                log.append(f"stripped generic tail: {item!r} -> {short!r} ({bucket})")
                item = short
            items.append(item)
        fixed[bucket] = items

    keys = {" ".join(tokenize(i)) for b in fixed for i in fixed[b]}
    seen: set[str] = set()
    for bucket in ("must_haves", "nice_to_haves"):
        kept = []
        for item in fixed[bucket]:
            tokens = tokenize(item)
            key = " ".join(tokens)
            base = " ".join(tokens[:-1])
            if len(tokens) >= 2 and tokens[-1] in _REPEAT_HEADS and base in keys:
                log.append(f"dropped {item!r}: repeats another item plus a generic word ({bucket})")
                continue
            if key in seen:
                log.append(f"dropped {item!r}: duplicate after repair ({bucket})")
                continue
            seen.add(key)
            kept.append(item)
        fixed[bucket] = kept

    if not log or not fixed["must_haves"]:
        # Nothing to do — or repair would empty must_haves, which the schema
        # forbids; leave that brief for verify() and the retry to handle.
        return brief, ()
    return brief.model_copy(update=fixed), tuple(log)


def verify(brief: JobBrief, jd_text: str) -> list[str]:
    """Grounding problems a string comparison can catch. Empty = pass."""
    problems: list[str] = []
    jd_tokens = set(tokenize(jd_text))

    if not grounded(brief.company, jd_text, jd_tokens):
        problems.append(f"company {brief.company!r} is not named in the JD")

    # Title: strict on spelling, lenient on order and punctuation. Exact
    # token match on purpose — grounded()'s _same_stem would accept
    # "Scientst" for "Scientist", which is precisely the error this catches.
    # Tokens under 3 chars ("Sr", "II", "ML") are skipped as too ambiguous.
    misspelled = [t for t in tokenize(brief.title) if len(t) >= 3 and t not in jd_tokens]
    if misspelled:
        problems.append(
            f"title {brief.title!r} is not written as in the JD ({', '.join(misspelled)} "
            "not found); copy the job title exactly as posted"
        )

    if brief.years_experience is not None and str(brief.years_experience) not in _NUM.findall(jd_text):
        problems.append(
            f"years_experience={brief.years_experience} does not appear as a number "
            "in the JD; use the number the JD states, or null"
        )

    for bucket in ("must_haves", "nice_to_haves", "soft_skills"):
        for item in getattr(brief, bucket):
            if not grounded(item, jd_text, jd_tokens):
                problems.append(f"{bucket} item {item!r} has no basis in the JD text; remove it")

    seen: dict[str, str] = {}
    for bucket in ("must_haves", "nice_to_haves", "soft_skills"):
        for item in getattr(brief, bucket):
            key = " ".join(tokenize(item))
            if key in seen and seen[key] != bucket:
                problems.append(f"{item!r} appears in both {seen[key]} and {bucket}; keep one")
            seen.setdefault(key, bucket)
    return problems


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------

class ExtractorError(RuntimeError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("Extractor output rejected after retry:\n" + "\n".join(f"- {p}" for p in problems))
        self.problems = problems


@dataclass(frozen=True)
class ExtractResult:
    brief: JobBrief
    attempts: int
    usage: dict[str, int]   # prompt/completion tokens summed over attempts
    model: str
    format: str             # "json_schema" or "json_object" — which mode the provider accepted
    # Problems that caused a retry, in order. Empty when the first attempt
    # passed. A retry that succeeds is otherwise invisible except as extra
    # tokens; this is what the eval reports so a rule that keeps firing shows.
    rejected: tuple[str, ...] = ()
    # Mechanical fixes repair() applied to the returned brief. Not failures —
    # a count of how often the model broke a rule that code enforces anyway.
    repaired: tuple[str, ...] = ()
    # Temperature / reasoning settings the requests were sent with.
    settings: ModelSettings = ModelSettings()


def _default_client(route: ModelRoute):
    from openai import OpenAI
    return OpenAI(base_url=route.base_url, api_key=route.api_key(), max_retries=MAX_PROVIDER_RETRIES)


def _response_format(mode: str, schema: dict[str, Any]) -> dict[str, Any]:
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "job_brief", "schema": schema}}
    return {"type": "json_object"}


def _is_schema_rejection(exc: Exception) -> bool:
    """True for the two 400s that mean "constrained decoding did not work
    here", both answerable by falling back to plain JSON mode:

      * the model does not support response_format / json_schema at all;
      * `json_validate_failed` — the provider's validator rejected what the
        model produced against our schema. With a non-empty
        `failed_generation` that is a real schema mismatch; with an empty
        one the model returned nothing at all, usually because a reasoning
        model spent max_tokens before answering.

    Anything else (401, 404 model not found, 429) is a real error and
    propagates untouched.
    """
    text = str(exc).lower()
    return ("response_format" in text or "json_schema" in text
            or "json schema" in text or "json_validate_failed" in text)


def _complete(client: Any, model: str, messages: list[dict[str, str]], mode: str, schema: dict[str, Any],
              settings: ModelSettings | None = None) -> tuple[Any, str]:
    """One chat completion. Returns (response, mode actually used).
    `settings` (temperature, max_tokens, reasoning effort, ...) go on every
    request, the JSON-mode fallback included; default: model_settings(model)."""
    extra = (settings or model_settings(model)).request_kwargs()
    try:
        return client.chat.completions.create(
            model=model, messages=messages,
            response_format=_response_format(mode, schema), **extra,
        ), mode
    except Exception as e:  # openai.BadRequestError, but keep the SDK out of the signature
        if mode == "json_schema" and _is_schema_rejection(e):
            return client.chat.completions.create(
                model=model, messages=messages,
                response_format=_response_format("json_object", schema), **extra,
            ), "json_object"
        raise


def extract(
    jd_text: str,
    *,
    client: Any = None,
    route: ModelRoute | None = None,
    model: str | None = None,
    settings: ModelSettings | None = None,
) -> ExtractResult:
    """Raw JD text -> validated, grounded JobBrief.

    `client` is any object with `chat.completions.create(...)` — the
    openai SDK pointed at Groq/OpenRouter/Ollama, or a stub in tests.
    `settings` defaults to the model's entry in config.MODEL_SETTINGS.
    """
    default = extractor_route()
    model = model or (route or default).model
    route = route or route_for(model, default)       # the model's own provider, if it has one
    settings = settings or model_settings(model)
    client = client or _default_client(route)
    schema = output_schema()
    messages = build_messages(jd_text)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    mode = "json_schema"
    problems: list[str] = []
    rejected: list[str] = []

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response, mode = _complete(client, model, messages, mode, schema, settings)
        if response.usage is not None:
            usage["prompt_tokens"] += response.usage.prompt_tokens or 0
            usage["completion_tokens"] += response.usage.completion_tokens or 0
        choice = response.choices[0]
        text = choice.message.content or ""

        if not text.strip():
            raise ExtractorError([
                f"empty response (finish_reason={choice.finish_reason!r}, "
                f"max_tokens={settings.max_tokens}). A reasoning model may have spent the "
                f"whole budget before answering: raise its max_tokens in "
                f"config.MODEL_SETTINGS, lower its reasoning effort, or use an instruct model."
            ])
        if choice.finish_reason == "length":
            problems = ["output was cut off at the token limit; return a shorter brief"]
        else:
            try:
                brief = JobBrief.model_validate_json(_strip_fences(text))
            except ValidationError as e:
                # Pydantic reports unparsable text as a `json_invalid` error
                # with an empty loc; everything else is a real path into the brief.
                problems = [
                    f"output was not valid JSON: {err['msg']}" if err["type"] == "json_invalid"
                    else f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                    for err in e.errors()
                ]
            else:
                brief, repaired = repair(brief)
                problems = verify(brief, jd_text)
                if not problems:
                    return ExtractResult(brief=brief, attempts=attempt, usage=usage, model=model,
                                         format=mode, rejected=tuple(rejected), repaired=repaired,
                                         settings=settings)

        rejected.extend(problems)
        messages = messages + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": _feedback(problems)},
        ]

    raise ExtractorError(problems)


def _strip_fences(text: str) -> str:
    """JSON mode never fences, but a downgraded json_object call on a
    chatty model sometimes does."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        text = text.rsplit("```", 1)[0]
    return text.strip()


def brief_to_yaml(brief: JobBrief, header: str = "") -> str:
    import yaml
    body = yaml.safe_dump(brief.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    return (header + "\n" if header else "") + body


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # python -m src.agents.extractor --list-models
    # python -m src.agents.extractor data/sample_jd.txt
    # python -m src.agents.extractor data/sample_jd.txt --out data/eval/extracted/sample_jd.brief.yaml
    import argparse
    import sys

    from dotenv import load_dotenv

    load_dotenv()

    ap = argparse.ArgumentParser(description="Extract a JobBrief from raw JD text.")
    ap.add_argument("jd", nargs="?", help="path to a .txt JD (omit with --list-models)")
    ap.add_argument("--out", default=None, help="write the brief as YAML here")
    ap.add_argument("--model", default=None, help="override EXTRACTOR_MODEL")
    ap.add_argument("--list-models", action="store_true", help="print the model ids the provider offers, then exit")
    ap.add_argument("--provider", default=None,
                    help="groq | cerebras | openrouter | huggingface (default: the model's own, else .env)")
    args = ap.parse_args()

    default = extractor_route()
    route = route_for(args.model or default.model, default, args.provider)
    if args.list_models:
        client = _default_client(route)
        ids = sorted(m.id for m in client.models.list().data)
        print(f"{len(ids)} model(s) at {route.base_url}:")
        for i in ids:
            print(f"  {i}")
        print(f"\ncurrent EXTRACTOR_MODEL: {route.model}")
        sys.exit(0)
    if not args.jd:
        ap.error("give a JD path or --list-models")

    jd_text = Path(args.jd).read_text(encoding="utf-8")
    model = route.model
    print(f"extracting {args.jd} with {model} ({model_settings(model).label}) via {route.base_url}")
    result = extract(jd_text, route=route, model=model)
    print(f"attempts: {result.attempts}   format: {result.format}   "
          f"tokens in/out: {result.usage['prompt_tokens']}/{result.usage['completion_tokens']}\n")
    yaml_text = brief_to_yaml(result.brief, header=f"# extracted from {args.jd} by {model}")
    print(yaml_text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(yaml_text, encoding="utf-8")
        print(f"wrote {out}")
