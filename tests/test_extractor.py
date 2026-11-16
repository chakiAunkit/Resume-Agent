"""Phase 4 Extractor tests. No API calls: a stub with the slice of the
OpenAI SDK surface the agent touches (`chat.completions.create`) returns
canned JSON, so validate -> verify -> retry and the json_schema ->
json_object downgrade are pinned deterministically.

Run from the repo root:  pytest tests/test_extractor.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.extractor import (
    SYSTEM_PROMPT,
    ExtractorError,
    build_messages,
    extract,
    grounded,
    output_schema,
    verify,
)
from src.brief import JobBrief
from src.config import ModelRoute

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Every hand-written brief with its raw JD beside it is an eval case.
EVAL_CASES = [b for b in sorted(DATA.glob("*.brief.yaml"))
              if (DATA / (b.name[: -len(".brief.yaml")] + ".txt")).exists()]

JD = """\
Senior Data Scientist — Acme Payments (Bangalore)
Requirements: 5+ years in data science; strong SQL and Python; experience
with fraud detection or risk models; familiarity with A/B testing.
Preferred: PySpark; knowledge of graph analytics.
You communicate clearly with product and engineering partners.
Acme is an equal opportunity employer."""

GOOD = {
    "title": "Senior Data Scientist",
    "company": "Acme Payments",
    "seniority": "senior",
    "years_experience": 5,
    "must_haves": ["SQL", "Python", "fraud detection", "risk models", "A/B testing"],
    "nice_to_haves": ["PySpark", "graph analytics"],
    "soft_skills": ["clear communication"],
}

ROUTE = ModelRoute(base_url="http://stub", key_env="STUB_KEY", model="stub-model")


def _response(text: str, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40),
    )


def stub_client(*texts: str, finish_reason: str = "stop", reject_schema: bool = False):
    """`client.chat.completions.create(**kw)`; records each call's kwargs.
    With reject_schema=True the first json_schema request raises the way
    a provider without constrained decoding would."""
    calls: list[dict] = []
    answers = list(texts)

    def create(**kw):
        calls.append(kw)
        if reject_schema and kw["response_format"]["type"] == "json_schema":
            raise RuntimeError("400: response_format json_schema is not supported for this model")
        return _response(answers.pop(0), finish_reason)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.calls = calls
    return client


# -------------------------------------------------------------- schema

def test_output_schema_is_decoder_ready():
    schema = output_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])     # no optional keys for the grammar
    assert "maxLength" not in json.dumps(schema) and "minItems" not in json.dumps(schema)
    assert schema["properties"]["seniority"]["enum"] == ["junior", "mid", "senior", "lead", "unspecified"]


def test_few_shot_is_synthetic_not_an_eval_case():
    """The eval set must test generalisation: no real JD in the prompt."""
    assert "eBay" not in SYSTEM_PROMPT and "Northwind" in SYSTEM_PROMPT
    assert build_messages("hello")[1]["content"].endswith("hello")


@pytest.mark.skipif(not list(DATA.glob("*.brief.yaml")), reason="no hand-written briefs present")
def test_prompt_contains_no_eval_case_requirements():
    """A rule example that happens to be an eval answer turns that case into
    a lookup. Multi-word items only: single words like "SQL" are
    legitimately generic and appear in the synthetic example."""
    prompt = SYSTEM_PROMPT.lower()
    leaks = []
    for path in sorted(DATA.glob("*.brief.yaml")):
        brief = JobBrief.load(path)
        for item in brief.must_haves + brief.nice_to_haves + brief.soft_skills:
            if len(item.split()) >= 2 and item.lower() in prompt:
                leaks.append(f"{path.name}: {item!r}")
    assert not leaks, f"eval-case requirements quoted in SYSTEM_PROMPT: {leaks}"


# -------------------------------------------------------------- verify

@pytest.mark.parametrize("brief_path", EVAL_CASES, ids=lambda p: p.name[: -len(".brief.yaml")])
def test_hand_written_brief_passes_verify(brief_path):
    """Every hand-written brief is ground truth; verify() must not reject it.
    (If this fails after editing verify(), verify() got too strict — or the
    brief and its JD have drifted apart.)"""
    jd = brief_path.with_name(brief_path.name[: -len(".brief.yaml")] + ".txt")
    assert verify(JobBrief.load(brief_path), jd.read_text(encoding="utf-8")) == []


def test_grounded_ignores_punctuation_and_case():
    assert grounded("e-commerce", "a global eCommerce leader")
    assert grounded("high agency", "a high-agency, startup-style role")
    assert grounded("product catalog", "optimize taxonomy and catalog data")   # token fallback
    assert grounded("experimentation", "build quick experiments")             # inflection
    assert grounded("clear communication", "you communicate clearly")
    assert not grounded("Kubernetes", "we use SQL and Python")
    assert not grounded("data", "we use SQL and Python")                      # short tokens need an exact hit


def test_verify_catches_misspelled_title():
    """The model once returned "Data Scientst". grounded()'s stem matching
    would forgive that, so the title gets an exact-token check."""
    problems = verify(JobBrief.model_validate({**GOOD, "title": "Senior Data Scientst"}), JD)
    assert len(problems) == 1 and "scientst" in problems[0]
    # order and punctuation are not spelling
    assert verify(JobBrief.model_validate({**GOOD, "title": "Data Scientist, Senior"}), JD) == []


RULE4_JD = """Data Scientist - Acme Payments. 5+ years. Experience with analytics,
fraud detection models, fraud detection, quasi-experimental methods, measurement
frameworks, data analytics, success metrics and a startup environment."""
RULE4_BASE = {"title": "Data Scientist", "company": "Acme Payments", "years_experience": 5,
              "seniority": "mid", "soft_skills": []}


def test_repair_applies_rule_4_mechanically():
    """Six runs showed rule 4 followed only sometimes, and v3's retries on it
    failed twice. The fixes need no judgement, so code applies them."""
    from src.agents.extractor import repair
    brief = JobBrief.model_validate({**RULE4_BASE,
        "must_haves": ["analytics", "fraud detection models", "fraud detection"],
        "nice_to_haves": ["startup environment"]})
    fixed, log = repair(brief)
    assert fixed.must_haves == ["fraud detection"]          # the bare word and the repeat are gone
    assert fixed.nice_to_haves == ["startup"]                # the tail is stripped
    assert len(log) == 3 and verify(fixed, RULE4_JD) == []


def test_repair_keeps_the_shorter_item_of_a_repeat():
    """v3's "keep one" made the model keep "recommendation models" and drop
    "recommendations". The repair always drops the generic-headed one."""
    from src.agents.extractor import repair
    brief = JobBrief.model_validate({**RULE4_BASE,
        "must_haves": ["fraud detection models", "SQL"], "nice_to_haves": ["fraud detection"]})
    fixed, _ = repair(brief)
    assert fixed.must_haves == ["SQL"] and fixed.nice_to_haves == ["fraud detection"]


def test_repair_removes_duplicates_it_creates():
    from src.agents.extractor import repair
    brief = JobBrief.model_validate({**RULE4_BASE,
        "must_haves": ["startup"], "nice_to_haves": ["startup environment"]})
    fixed, log = repair(brief)
    assert fixed.must_haves == ["startup"] and fixed.nice_to_haves == []
    assert any("duplicate after repair" in entry for entry in log)


def test_repair_leaves_real_requirements_alone():
    """The word lists are short on purpose: a generic word can be the concept.
    These are all items in the hand-written briefs."""
    from src.agents.extractor import repair
    brief = JobBrief.model_validate({**RULE4_BASE,
        "must_haves": ["quasi-experimental methods", "measurement frameworks",
                       "data analytics", "success metrics", "fraud detection"],
        "soft_skills": ["startup environment"]})    # soft skills are phrasing: not repaired
    fixed, log = repair(brief)
    assert log == () and fixed == brief


@pytest.mark.parametrize("brief_path", EVAL_CASES, ids=lambda p: p.name[: -len(".brief.yaml")])
def test_repair_is_a_no_op_on_hand_written_briefs(brief_path):
    """The references follow the rules, so repair must not change them."""
    from src.agents.extractor import repair
    brief = JobBrief.load(brief_path)
    assert repair(brief) == (brief, ())


def test_repair_costs_no_retry():
    """A repairable brief passes on the first attempt; the fix is logged."""
    client = stub_client(json.dumps({**GOOD, "must_haves": GOOD["must_haves"] + ["models"]}))
    result = extract(JD, client=client, route=ROUTE)
    assert result.attempts == 1 and len(client.calls) == 1
    assert "models" not in result.brief.must_haves
    assert result.repaired and "'models'" in result.repaired[0]


def test_verify_catches_invention_and_duplicates():
    brief = JobBrief.model_validate({**GOOD, "must_haves": GOOD["must_haves"] + ["Kubernetes"],
                                     "nice_to_haves": ["PySpark", "SQL"], "years_experience": 7,
                                     "company": "Globex"})
    problems = verify(brief, JD)
    assert any("Kubernetes" in p for p in problems)
    assert any("years_experience=7" in p for p in problems)
    assert any("Globex" in p for p in problems)
    assert any("'SQL' appears in both must_haves and nice_to_haves" in p for p in problems)
    assert len(problems) == 4


# ------------------------------------------------------------- extract

def test_happy_path_one_attempt():
    client = stub_client(json.dumps(GOOD))
    result = extract(JD, client=client, route=ROUTE)
    assert result.attempts == 1 and result.format == "json_schema"
    assert result.brief.company == "Acme Payments" and result.brief.years_experience == 5
    assert result.usage == {"prompt_tokens": 100, "completion_tokens": 40}
    call = client.calls[0]
    assert call["model"] == "stub-model" and call["temperature"] == 0.0
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["schema"]["required"]
    assert [m["role"] for m in call["messages"]] == ["system", "user"]


def test_schema_failure_retries_with_loc_path():
    bad = {**GOOD, "must_haves": ["Proficiency in SQL and Python for data wrangling, analytics, and automation across the org"]}
    client = stub_client(json.dumps(bad), json.dumps(GOOD))
    result = extract(JD, client=client, route=ROUTE)
    assert result.attempts == 2
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert "must_haves.0" in second[-1]["content"]


def test_grounding_failure_retries():
    bad = {**GOOD, "must_haves": GOOD["must_haves"] + ["Kubernetes"]}
    client = stub_client(json.dumps(bad), json.dumps(GOOD))
    result = extract(JD, client=client, route=ROUTE)
    assert result.attempts == 2
    assert "Kubernetes" in client.calls[1]["messages"][-1]["content"]
    # the reason survives into the result, so the eval can report it
    assert len(result.rejected) == 1 and "Kubernetes" in result.rejected[0]


def test_first_attempt_pass_records_no_rejections():
    result = extract(JD, client=stub_client(json.dumps(GOOD)), route=ROUTE)
    assert result.rejected == ()


def test_two_failures_raise():
    bad = json.dumps({**GOOD, "company": "Globex"})
    client = stub_client(bad, bad)
    with pytest.raises(ExtractorError) as exc:
        extract(JD, client=client, route=ROUTE)
    assert len(client.calls) == 2 and "Globex" in exc.value.problems[0]


def test_non_json_is_retried_then_fenced_json_accepted():
    client = stub_client("Sure! Here is the brief:", "```json\n" + json.dumps(GOOD) + "\n```")
    result = extract(JD, client=client, route=ROUTE)
    assert result.attempts == 2
    assert "not valid JSON" in client.calls[1]["messages"][-1]["content"]


def test_truncation_is_a_retryable_problem():
    client = stub_client("{", "{", finish_reason="length")
    with pytest.raises(ExtractorError, match="token limit"):
        extract(JD, client=client, route=ROUTE)
    assert len(client.calls) == 2


def test_empty_response_is_not_retried():
    client = stub_client("")
    with pytest.raises(ExtractorError, match="empty response"):
        extract(JD, client=client, route=ROUTE)
    assert len(client.calls) == 1


def test_downgrades_to_json_object_when_schema_rejected():
    client = stub_client(json.dumps(GOOD), reject_schema=True)
    result = extract(JD, client=client, route=ROUTE)
    assert result.format == "json_object" and result.attempts == 1
    assert [c["response_format"]["type"] for c in client.calls] == ["json_schema", "json_object"]


def test_unrelated_provider_error_propagates():
    def create(**kw):
        raise RuntimeError("401: invalid api key")
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    with pytest.raises(RuntimeError, match="invalid api key"):
        extract(JD, client=client, route=ROUTE)


# ----------------------------------------------------------------- eval

def test_compare_separates_misses_misplacements_and_extras():
    from src.tools.extract_eval import compare
    expected = JobBrief.model_validate(GOOD)
    got = JobBrief.model_validate({**GOOD,
        "must_haves": ["SQL", "Python", "fraud detection", "PySpark", "Kubernetes"],   # A/B testing, risk models missing; PySpark misplaced; Kubernetes extra
        "nice_to_haves": ["graph analytics"],
        "years_experience": 3})
    cmp = compare(expected, got)
    mh = cmp.buckets["must_haves"]
    assert set(mh.hit) == {"SQL", "Python", "fraud detection"}
    assert set(mh.missed) == {"risk models", "A/B testing"}
    assert mh.extra == ["Kubernetes"]
    assert mh.precision == pytest.approx(3 / 4) and mh.recall == pytest.approx(3 / 5)
    nh = cmp.buckets["nice_to_haves"]
    assert nh.misplaced == ["PySpark -> must_haves"] and nh.hit == ["graph analytics"]
    assert cmp.years_match is False and cmp.company_match is True


def test_compare_matches_items_by_normalised_tokens():
    from src.tools.extract_eval import compare
    expected = JobBrief.model_validate({**GOOD, "must_haves": ["LLMs", "e-commerce"]})
    got = JobBrief.model_validate({**GOOD, "must_haves": ["LLM", "E-Commerce"]})
    assert compare(expected, got).buckets["must_haves"].recall == 1.0


def test_compare_treats_a_spelled_out_acronym_as_one_item():
    """"retrieval-augmented generation (RAG)" vs "retrieval-augmented
    generation" is one requirement written two ways, not a miss + an extra.
    A bracketed word that is not the initials stays part of the item."""
    from src.tools.extract_eval import _key, compare
    expected = JobBrief.model_validate({**GOOD, "nice_to_haves": ["retrieval-augmented generation (RAG)"]})
    got = JobBrief.model_validate({**GOOD, "nice_to_haves": ["retrieval-augmented generation"]})
    nh = compare(expected, got).buckets["nice_to_haves"]
    assert nh.recall == 1.0 and nh.extra == []
    assert _key("large language models (LLMs)") == _key("large language models")
    assert _key("Data Scientist (Analytics)") != _key("Data Scientist")


def test_compare_ignores_spacing_and_hyphens():
    """Case 002 lost "analytical problem-solving" to "analytical problem
    solving" in 5 of 6 runs. Keyword v6 drops the hyphen, which alone still
    leaves "problemsolving" vs "problem solving"; the eval key ignores the
    space as well. Retrieval keeps word tokens — this is eval-only."""
    from src.tools.extract_eval import _key, compare
    expected = JobBrief.model_validate({**GOOD, "soft_skills": ["analytical problem-solving"],
                                        "must_haves": ["eCommerce", "A/B testing"]})
    got = JobBrief.model_validate({**GOOD, "soft_skills": ["analytical problem solving"],
                                   "must_haves": ["e-commerce", "AB testing"]})
    cmp = compare(expected, got)
    assert cmp.buckets["soft_skills"].recall == 1.0 and cmp.buckets["soft_skills"].extra == []
    assert cmp.buckets["must_haves"].recall == 1.0
    assert _key("AI/ML") != _key("AIML models")        # still two different items


def _save_brief(path: Path, **over) -> Path:
    from src.agents.extractor import brief_to_yaml
    path.write_text(brief_to_yaml(JobBrief.model_validate({**GOOD, **over}), header="# saved"),
                    encoding="utf-8")
    return path


def test_saved_runs_groups_by_model_and_sorts_numerically(tmp_path):
    for name in ("case.model-a.r2.brief.yaml", "case.model-a.r10.brief.yaml",
                 "case.model-a.r1.brief.yaml", "case.model-b.r1.brief.yaml",
                 "case.model-a.brief.yaml",               # single run: not part of a multi-run eval
                 "other.model-a.r1.brief.yaml"):          # another case
        _save_brief(tmp_path / name)
    from src.tools.extract_eval import saved_runs
    groups = saved_runs("case", out_dir=tmp_path)
    assert list(groups) == ["model-a", "model-b"]
    assert [p.name for p in groups["model-a"]] == [
        "case.model-a.r1.brief.yaml", "case.model-a.r2.brief.yaml", "case.model-a.r10.brief.yaml"]


def test_rescore_scores_saved_briefs_without_a_client(tmp_path):
    """Deterministic: same files in, same numbers out, no API involved."""
    from src.tools.extract_eval import report_runs, rescore, summary_row
    paths = [_save_brief(tmp_path / "c.m.r1.brief.yaml"),
             _save_brief(tmp_path / "c.m.r2.brief.yaml", must_haves=["SQL", "Kubernetes"])]
    expected = JobBrief.model_validate(GOOD)
    outcomes = rescore(expected, paths, profile=None)
    assert [o.comparison.buckets["must_haves"].recall for o in outcomes] == [1.0, pytest.approx(1 / 5)]
    assert all(o.retrieval is None for o in outcomes)          # profile=None skips retrieval
    assert rescore(expected, paths, profile=None)[1].comparison.buckets["must_haves"].extra == ["Kubernetes"]
    text = report_runs("c", "m", outcomes, rescored=True)
    assert "rescored 2/2 saved runs, no API calls" in text and "attempts" not in text
    assert summary_row("c", "m", outcomes, rescored=True).endswith("| rescored |")


def test_stability_counts_how_often_each_item_survives():
    """Repeats exist to separate "the model follows this rule" from "the
    model follows this rule sometimes". 2/3 is the signal to move a rule
    into verify(); 0/3 is a prompt or reference problem."""
    from src.tools.extract_eval import compare, stability
    expected = JobBrief.model_validate(GOOD)
    got = [
        JobBrief.model_validate(GOOD),                                        # perfect
        JobBrief.model_validate({**GOOD, "must_haves": ["SQL", "Python", "fraud detection",
                                                        "risk models", "Kubernetes"]}),
        JobBrief.model_validate({**GOOD, "must_haves": ["SQL", "Python", "fraud detection",
                                                        "risk models"],
                                 "nice_to_haves": ["PySpark", "graph analytics", "A/B testing"]}),
    ]
    st = stability([compare(expected, g) for g in got], "must_haves")
    assert st.runs == 3
    assert set(st.always()) == {"SQL", "Python", "fraud detection", "risk models"}
    assert st.sometimes() == ["A/B testing"] and st.hit["A/B testing"] == 1
    assert st.misplaced["A/B testing"] == 1      # run 3 filed it under nice_to_haves
    assert st.extra["Kubernetes"] == 1
    assert st.never() == []


def test_spread_reports_mean_and_range():
    from src.tools.extract_eval import fmt_spread, spread
    assert spread([0.5, 0.8, 0.8]) == (pytest.approx(0.7), 0.5, 0.8)
    assert fmt_spread([0.5, 0.8, 0.8]) == "0.70 (0.50-0.80)"
    assert fmt_spread([0.8, 0.8]) == "0.80"      # no range when it never moved
    assert fmt_spread([]) == "n/a"


def test_report_survives_a_failed_run():
    """One rejected extraction must not lose the other runs' numbers."""
    from src.tools.extract_eval import RunOutcome, compare, report_runs, summary_row
    expected = JobBrief.model_validate(GOOD)
    ok = RunOutcome(comparison=compare(expected, expected),
                    usage={"prompt_tokens": 100, "completion_tokens": 40})
    outcomes = [ok, RunOutcome(error="run 2: company 'Globex' is not named in the JD")]
    text = report_runs("case", "stub-model", outcomes)
    assert "1/2 runs ok" in text and "FAILED" in text and "must_haves" in text
    assert "attempts 0/x" in text   # failed runs show as x
    assert summary_row("case", "stub-model", outcomes).startswith("| case | stub-model | 1/2 |")


def test_report_shows_attempts_and_retry_reasons():
    from src.tools.extract_eval import RunOutcome, compare, report_runs
    expected = JobBrief.model_validate(GOOD)
    cmp = compare(expected, expected)
    why = "must_haves item 'analytics' is a bare generic word; name the specific requirement"
    outcomes = [RunOutcome(comparison=cmp, attempts=1),
                RunOutcome(comparison=cmp, attempts=2, rejected=(why,)),
                RunOutcome(comparison=cmp, attempts=2, rejected=(why,))]
    text = report_runs("case", "stub-model", outcomes)
    assert "attempts 1/2/2" in text
    assert "retried because:" in text and "2x must_haves item 'analytics'" in text

    fix = "dropped bare generic word 'metrics' (must_haves)"
    text = report_runs("case", "stub-model", [RunOutcome(comparison=cmp, attempts=1, repaired=(fix,))] * 3)
    assert "repaired:" in text and "3x dropped bare generic word 'metrics'" in text


def test_json_validate_failed_downgrades_to_json_object():
    """Groq's constrained decoder returns this 400 when the generation
    doesn't validate against the schema (including when a reasoning model
    returns nothing). Plain JSON mode plus the Pydantic gate still works."""
    calls: list[dict] = []

    def create(**kw):
        calls.append(kw)
        if kw["response_format"]["type"] == "json_schema":
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': 'Failed to validate JSON.', "
                "'code': 'json_validate_failed', 'failed_generation': ''}}")
        return _response(json.dumps(GOOD))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    result = extract(JD, client=client, route=ROUTE)
    assert result.format == "json_object" and result.attempts == 1


def test_client_retries_rate_limits_beyond_the_sdk_default(monkeypatch):
    """Groq's free tier is 8k tokens/minute and an extraction is ~5k, so a
    second call inside a minute gets a 429. The SDK waits out retry-after
    on its own, but its default of 2 retries gave up mid-eval."""
    pytest.importorskip("openai")
    from src.agents.extractor import MAX_PROVIDER_RETRIES, _default_client
    monkeypatch.setenv("STUB_KEY", "x")
    assert _default_client(ROUTE).max_retries == MAX_PROVIDER_RETRIES >= 5


def test_token_budget_is_generous_enough_for_a_reasoning_model():
    from src.agents.extractor import MAX_TOKENS
    assert MAX_TOKENS >= 8192


# ------------------------------------------- per-model settings (Phase 4, model comparison)

def test_gpt_oss_requests_state_their_reasoning_effort():
    """Groq's default for gpt-oss is already medium: sending it explicitly
    changes nothing for the accepted runs, it just stops depending on it."""
    client = stub_client(json.dumps(GOOD))
    result = extract(JD, client=client, route=ROUTE, model="openai/gpt-oss-120b")
    call = client.calls[0]
    assert call["reasoning_effort"] == "medium" and call["temperature"] == 0.0
    assert "extra_body" not in call and "top_p" not in call          # gpt-oss rejects reasoning_format
    assert result.settings.label == "effort medium, T 0"


def test_qwen_requests_turn_reasoning_on_and_keep_it_out_of_the_json():
    """Groq's default for qwen3.8-27b is reasoning_effort "none" (no thinking)
    and reasoning_format "raw" (thinking inside the answer, refused in JSON
    mode). Both must be set, on the fallback request too."""
    client = stub_client(json.dumps(GOOD), reject_schema=True)
    result = extract(JD, client=client, route=ROUTE, model="qwen/qwen3.8-27b")
    assert len(client.calls) == 2 and result.format == "json_object"
    for call in client.calls:
        assert call["reasoning_effort"] == "medium"
        assert call["extra_body"] == {"reasoning_format": "hidden"}
        assert call["temperature"] == 1.0 and call["top_p"] == 0.95


def test_unlisted_model_sends_temperature_only():
    client = stub_client(json.dumps(GOOD))
    extract(JD, client=client, route=ROUTE)                            # "stub-model": not in the table
    assert client.calls[0]["temperature"] == 0.0
    assert not {"reasoning_effort", "extra_body", "top_p"} & set(client.calls[0])


def test_reasoning_effort_override_keeps_the_rest_of_the_settings():
    from src.config import MODEL_SETTINGS, model_settings
    high = model_settings("qwen/qwen3.8-27b", reasoning_effort="high")
    assert high.reasoning_effort == "high" and high.reasoning_format == "hidden" and high.temperature == 1.0
    assert model_settings("qwen/qwen3.8-27b") == MODEL_SETTINGS["qwen/qwen3.8-27b"]   # the table is untouched
    assert model_settings("some/new-model", reasoning_effort="low").request_kwargs() == \
        {"temperature": 0.0, "max_tokens": 8192, "reasoning_effort": "low"}


def test_explicit_settings_win_over_the_table():
    from src.config import ModelSettings
    client = stub_client(json.dumps(GOOD))
    extract(JD, client=client, route=ROUTE, model="openai/gpt-oss-120b",
            settings=ModelSettings(temperature=0.0, reasoning_effort="high"))
    assert client.calls[0]["reasoning_effort"] == "high"


def test_batches_are_saved_under_distinct_names(tmp_path):
    """A new batch must never overwrite M7/M8, the saved record of the
    accepted Extractor; rescore still groups each batch on its own."""
    from src.tools.extract_eval import batch_slug, saved_runs
    assert batch_slug("openai/gpt-oss-120b") == "openai-gpt-oss-120b"             # M7/M8's names
    assert batch_slug("openai/gpt-oss-120b", tag="m9") == "openai-gpt-oss-120b.m9"
    assert batch_slug("qwen/qwen3.8-27b", "m9", "high") == "qwen-qwen3-8-27b.m9.effort-high"
    for slug in ("openai-gpt-oss-120b", "openai-gpt-oss-120b.m9", "qwen-qwen3-8-27b.m9"):
        for run in (1, 2):
            (tmp_path / f"case.{slug}.r{run}.brief.yaml").write_text("x", encoding="utf-8")
    groups = saved_runs("case", tmp_path)
    assert sorted(groups) == ["openai-gpt-oss-120b", "openai-gpt-oss-120b.m9", "qwen-qwen3-8-27b.m9"]
    assert all(len(paths) == 2 for paths in groups.values())


def test_max_tokens_is_per_model():
    """Groq refuses a request whose prompt + max_tokens exceeds the model's
    tokens-per-minute limit, before running it. Qwen's free-tier limit is 8K,
    so 8192 reserved on a ~2.9k-token prompt failed every time (m9)."""
    for model, budget in (("openai/gpt-oss-120b", 8192), ("qwen/qwen3.8-27b", 4096), ("stub-model", 8192)):
        client = stub_client(json.dumps(GOOD))
        extract(JD, client=client, route=ROUTE, model=model)
        assert client.calls[0]["max_tokens"] == budget
    from src.config import model_settings
    assert model_settings("qwen/qwen3.8-27b").label.endswith("max_tokens 4096")
    assert "max_tokens" not in model_settings("openai/gpt-oss-120b").label        # the default isn't shown


def test_request_too_large_is_explained_with_its_numbers():
    from src.tools.extract_eval import describe_provider_error
    groq = RuntimeError(
        "Error code: 429 - {'error': {'message': 'Request too large for model `qwen/qwen3.8-27b` in "
        "organization `org_x` service tier `on_demand` on tokens per minute (TPM): Limit 8000, "
        "Requested 11053, please reduce your message size and try again.', 'type': 'tokens', "
        "'code': 'rate_limit_exceeded'}}")
    line = describe_provider_error(groq)
    assert "on tokens per minute (TPM): limit 8000, requested 11053" in line   # the unit is kept
    assert "max_tokens" in line and "--provider" in line
    assert describe_provider_error(RuntimeError("boom")) == "RuntimeError: boom"


def test_run_paths_match_what_rescore_reads(tmp_path):
    from src.tools.extract_eval import run_path, saved_runs
    path = run_path("case", "qwen-qwen3-8-27b.m9", 3, runs=6, out_dir=tmp_path)
    assert path.name == "case.qwen-qwen3-8-27b.m9.r3.brief.yaml"
    path.write_text("x", encoding="utf-8")
    assert saved_runs("case", tmp_path) == {"qwen-qwen3-8-27b.m9": [path]}
    assert run_path("case", "m", 1, runs=1, out_dir=tmp_path).name == "case.m.brief.yaml"


# ------------------------------------------------------------- providers

def test_a_model_pinned_to_a_provider_routes_there(monkeypatch):
    """`qwen-3.8-27b` exists on Cerebras only, so it goes there whatever
    .env says; an unpinned id follows the .env route; a flag beats both."""
    from src.config import ConfigError, route_for
    env = ModelRoute(base_url="https://api.groq.com/openai/v1", key_env="GROQ_API_KEY", model="openai/gpt-oss-120b")
    cerebras = route_for("qwen-3.8-27b", env)
    assert cerebras.base_url == "https://api.cerebras.ai/v1" and cerebras.key_env == "CEREBRAS_API_KEY"
    assert cerebras.model == "qwen-3.8-27b"
    assert route_for("openai/gpt-oss-120b", env) == env                          # not pinned: .env route
    assert route_for("some/new-model", env).base_url == env.base_url
    hf = route_for("Qwen/Qwen3.8-27B:novita", env, provider="huggingface")
    assert hf.base_url == "https://router.huggingface.co/v1" and hf.key_env == "HF_TOKEN"
    assert route_for("qwen-3.8-27b", env, provider="groq").base_url == env.base_url   # the flag wins
    with pytest.raises(ConfigError, match="unknown provider"):
        route_for("x", env, provider="grok")


def test_extract_opens_the_client_on_the_models_provider(monkeypatch):
    import src.agents.extractor as ex
    opened = []

    def fake_client(route):
        opened.append(route)
        return stub_client(json.dumps(GOOD))

    monkeypatch.setattr(ex, "_default_client", fake_client)
    monkeypatch.delenv("EXTRACTOR_BASE_URL", raising=False)
    result = ex.extract(JD, model="qwen-3.8-27b")
    assert opened[0].base_url == "https://api.cerebras.ai/v1" and result.model == "qwen-3.8-27b"
    ex.extract(JD, model="openai/gpt-oss-120b")
    assert opened[1].base_url.startswith("https://api.groq.com")               # the .env default (Groq)


def test_cerebras_settings_match_the_groq_plan():
    """Cerebras's defaults differ (Qwen reasons at "high" unless told), so
    both models state the same settings they get on Groq."""
    from src.config import model_settings
    oss, qwen = model_settings("gpt-oss-120b"), model_settings("qwen-3.8-27b")
    assert oss.request_kwargs() == {"temperature": 0.0, "max_tokens": 8192, "reasoning_effort": "medium"}
    assert qwen.request_kwargs() == {"temperature": 1.0, "max_tokens": 8192, "top_p": 0.95,
                                     "reasoning_effort": "medium", "extra_body": {"reasoning_format": "parsed"}}
