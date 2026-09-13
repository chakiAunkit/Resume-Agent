"""Phase 3 Tailor tests. No API calls: a stub client returns canned JSON,
so the validate -> verify -> retry mechanics are pinned deterministically.

Run from the repo root:  pytest tests/test_tailor.py -v
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.agents.tailor import (
    CUT_INSTRUCTION,
    TailorError,
    TailoredResume,
    build_prompt,
    output_schema,
    revise,
    system_prompt,
    tailor,
    uncovered_must_haves,
    verify,
)
from src.atoms import Profile
from src.brief import JobBrief
from src.models import Resume
from src.tools.evidence import BriefHit

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data" / "profile.example.yaml"

# A minimal but complete resume; assets/resume.json is gitignored.
RESUME = {
    "contact": {"name": "Test Person", "phone": "+1 555 0100", "email": "t@example.com", "links": []},
    "summary": "Data scientist with 4 years of experience.",
    "experience": [{
        "company": "Target Corporation", "title": "Analyst", "location": "Bangalore",
        "start": "09/2022", "end": "05/2025",
        "bullets": ["Ran analyses on 12 datasets."],
    }],
    "skills": [{"category": "Technical", "items": ["SQL"]}],
    "education": [{"degree": "B.Tech", "institution": "U", "location": "City",
                   "start": "07/2017", "end": "04/2021"}],
}

BRIEF = JobBrief(title="Data Scientist", company="Acme",
                 must_haves=["SQL", "e-commerce", "taxonomy"], soft_skills=["collaboration"])


@pytest.fixture
def resume() -> Resume:
    return Resume.model_validate(RESUME)


@pytest.fixture
def hits() -> list[BriefHit]:
    p = Profile.load(EXAMPLE)
    return [
        BriefHit(p.by_id("target-goa-dmo-mismatch"), 10.0, ("SQL", "e-commerce"), ()),
        BriefHit(p.by_id("self-text2sql-memory-agent"), 3.0, ("SQL",), ()),
    ]


def _good_output(**over) -> str:
    """A valid, grounded tailored resume: one atom inserted, one rephrase."""
    r = json.loads(json.dumps(RESUME))
    r["summary"] = "Data scientist with 4 years of experience in e-commerce analytics."
    r["experience"][0]["bullets"].append(
        "Analyzed 90M+ parcels and 60M+ customer orders using HiveQL to uncover a 20% "
        "mismatch in shipping service selection, unlocking $8M in annual cost savings."
    )
    doc = {"resume": r, "changes": [
        {"where": "summary", "atom_id": None, "why": "e-commerce vocabulary"},
        {"where": "experience[Target Corporation].bullets[1]",
         "atom_id": "target-goa-dmo-mismatch", "why": "e-commerce, SQL"},
    ]}
    doc.update(over)
    return json.dumps(doc)


def _stub_client(responder):
    """Stub with the slice of the SDK surface the Tailor touches:
    `client.messages.stream(**kw)` as a context manager whose
    `get_final_message()` returns a Message-like object."""
    calls: list[dict] = []

    @contextmanager
    def stream(**kw):
        calls.append(kw)
        yield SimpleNamespace(get_final_message=lambda: responder(len(calls)))

    client = SimpleNamespace(messages=SimpleNamespace(stream=stream))
    client.calls = calls
    return client


def fake_client(*texts: str, stop_reason: str = "end_turn"):
    def responder(n: int):
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=texts[n - 1])],
            stop_reason=stop_reason,
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
        )
    return _stub_client(responder)


# ------------------------------------------------------------- prompt

def test_prompt_lists_uncovered_must_haves_and_evidence(resume, hits):
    assert uncovered_must_haves(BRIEF, hits) == ("taxonomy",)
    prompt = build_prompt(resume, BRIEF, hits)
    assert "- taxonomy" in prompt
    assert "target-goa-dmo-mismatch" in prompt
    assert '"covers"' in prompt and "collaboration" in prompt
    assert '"keywords"' not in prompt  # retrieval tags are not content


def test_output_schema_is_api_ready():
    schema = output_schema()
    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"resume", "changes"}


# -------------------------------------------------------------- verify

def test_verify_passes_grounded_output(resume, hits):
    out = TailoredResume.model_validate_json(_good_output())
    assert verify(out, resume, hits) == []


def test_verify_catches_unknown_atom(resume, hits):
    out = TailoredResume.model_validate_json(_good_output(changes=[
        {"where": "summary", "atom_id": "nope-atom", "why": "x"}]))
    problems = verify(out, resume, hits)
    assert len(problems) == 1 and "nope-atom" in problems[0]


def test_verify_catches_invented_number(resume, hits):
    r = json.loads(_good_output())
    r["resume"]["experience"][0]["bullets"][0] = "Saved $50M by running 12 analyses."
    problems = verify(TailoredResume.model_validate(r), resume, hits)
    assert len(problems) == 1 and "$50M" in problems[0] and "12" not in problems[0]


def test_verify_catches_experience_atom_under_wrong_company(resume, hits):
    r = json.loads(_good_output())
    r["resume"]["experience"][0]["company"] = "Target Corp"  # renamed: join key broken
    problems = verify(TailoredResume.model_validate(r), resume, hits)
    assert any("Target Corporation" in p for p in problems)


def test_verify_catches_contact_edit(resume, hits):
    r = json.loads(_good_output())
    r["resume"]["contact"]["email"] = "other@example.com"
    problems = verify(TailoredResume.model_validate(r), resume, hits)
    assert any("contact" in p for p in problems)


def test_verify_catches_ungrounded_skill_item(resume, hits):
    r = json.loads(_good_output())
    r["resume"]["skills"][0]["items"] += ["Knowledge Graphs", "HiveQL", "Kubernetes"]
    problems = verify(TailoredResume.model_validate(r), resume, hits)
    # HiveQL is in the offered atom's skills; "Knowledge Graphs" is nowhere
    # in this fixture's inputs; Kubernetes is invented.
    assert len(problems) == 2
    assert any("Knowledge Graphs" in p for p in problems)
    assert any("Kubernetes" in p for p in problems)


def test_verify_skill_check_is_normalised(resume, hits):
    r = json.loads(_good_output())
    r["resume"]["skills"][0]["items"] += ["Vector Databases", "ROOT-CAUSE ANALYSIS"]
    # atom keywords say "vector database"; atom skills say "Root-cause analysis".
    # (Irregular plurals — "analyses" — are outside the singulariser's remit.)
    assert verify(TailoredResume.model_validate(r), resume, hits) == []


def test_system_prompt_carries_page_budget():
    assert "fit 1 A4 page(s)" in system_prompt(1)
    assert "fit 2 A4 page(s)" in system_prompt(2)


# --------------------------------------------------------------- tailor

def test_happy_path_one_attempt(resume, hits):
    client = fake_client(_good_output())
    result = tailor(resume, BRIEF, hits, client=client)
    assert result.attempts == 1
    assert len(result.resume.experience[0].bullets) == 2
    assert result.uncovered == ("taxonomy",)
    assert result.usage == {"input_tokens": 100, "output_tokens": 50}
    call = client.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["output_config"]["effort"] == "medium"
    assert call["max_tokens"] >= 16000  # thinking shares the budget with the answer
    assert call["messages"][0]["role"] == "user"


def test_schema_failure_retries_with_loc_path(resume, hits):
    bad = json.loads(_good_output())
    bad["resume"]["experience"][0]["start"] = "Sept 2022"
    client = fake_client(json.dumps(bad), _good_output())
    result = tailor(resume, BRIEF, hits, client=client)
    assert result.attempts == 2
    second = client.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assert "resume.experience.0.start" in second[-1]["content"]


def test_grounding_failure_retries_with_problem(resume, hits):
    bad = _good_output(changes=[{"where": "summary", "atom_id": "nope-atom", "why": "x"}])
    client = fake_client(bad, _good_output())
    result = tailor(resume, BRIEF, hits, client=client)
    assert result.attempts == 2
    assert "nope-atom" in client.calls[1]["messages"][-1]["content"]


def test_two_failures_raise(resume, hits):
    bad = _good_output(changes=[{"where": "summary", "atom_id": "nope-atom", "why": "x"}])
    client = fake_client(bad, bad)
    with pytest.raises(TailorError) as exc:
        tailor(resume, BRIEF, hits, client=client)
    assert len(client.calls) == 2
    assert "nope-atom" in exc.value.problems[0]


def test_empty_response_reports_stop_reason(resume, hits):
    client = _stub_client(lambda n: SimpleNamespace(
        content=[], stop_reason="refusal",
        usage=SimpleNamespace(input_tokens=1, output_tokens=0)))
    with pytest.raises(TailorError, match="stop_reason='refusal'"):
        tailor(resume, BRIEF, hits, client=client)
    assert len(client.calls) == 1  # nothing to retry with


def test_revise_appends_instruction_and_grounds_on_current(resume, hits):
    tailored = TailoredResume.model_validate_json(_good_output()).resume
    cut = tailored.model_copy(deep=True)
    cut.experience[0].bullets.pop(0)
    reply = json.dumps({"resume": cut.model_dump(mode="json"),
                        "changes": [{"where": "experience[Target Corporation].bullets[0]",
                                     "atom_id": None, "why": "least relevant"}]})
    client = fake_client(reply)
    result = revise(tailored, BRIEF, hits, CUT_INSTRUCTION, client=client)
    assert len(result.resume.experience[0].bullets) == 1
    prompt = client.calls[0]["messages"][0]["content"]
    assert prompt.rstrip().endswith(CUT_INSTRUCTION)
    assert client.calls[0]["system"].count("A4 page(s)") == 1


def test_truncation_is_a_retryable_problem(resume, hits):
    client = fake_client("{", "{", stop_reason="max_tokens")
    with pytest.raises(TailorError, match="token limit"):
        tailor(resume, BRIEF, hits, client=client)
    assert len(client.calls) == 2  # truncation is retried, then given up on
