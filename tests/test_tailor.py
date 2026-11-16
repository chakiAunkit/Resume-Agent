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
    expand_instruction,
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
    assert verify(TailoredResume.model_validate(r), resume, hits) == []
    # "analyses" reduces to the same token as "analysis" (evidence._singular)
    r["resume"]["skills"][0]["items"] += ["Root-cause analyses"]
    assert verify(TailoredResume.model_validate(r), resume, hits) == []


def test_verify_skill_check_follows_the_tokenizer(resume, hits):
    """_normalise() is " ".join(tokenize(...)), so keyword v6 reaches the
    Tailor too: a skill spelled the JD's way ("eCommerce") is grounded by an
    atom tagged "e-commerce". Before v6 this was rejected as invented."""
    r = json.loads(_good_output())
    r["resume"]["skills"][0]["items"] += ["eCommerce"]
    assert verify(TailoredResume.model_validate(r), resume, hits) == []


def test_verify_skill_check_matches_whole_tokens(resume, hits):
    """A short skill is not grounded by letters inside other words: every
    input contains an "r", a "c" and a "go" somewhere ("Bangalore", "cost",
    "category"). The inputs must contain the token itself."""
    r = json.loads(_good_output())
    r["resume"]["skills"][0]["items"] += ["R", "Go", "C", "Postgres", "pgvector"]
    problems = verify(TailoredResume.model_validate(r), resume, hits)
    # pgvector is a skill of the offered project atom; "Postgres" is only a
    # prefix of its "PostgreSQL".
    assert len(problems) == 4
    for item in ("'R'", "'Go'", "'C'", "'Postgres'"):
        assert any(f"skill item {item}" in p for p in problems)


# ------------------------------------------------ verify: figures

def test_numbers_do_not_swallow_a_trailing_comma():
    from src.agents.tailor import _numbers
    assert _numbers("In 2024, cut cost by 20%, across 25,000 stores; 1,00,000 rows, $8M+.") == {
        "2024", "20%", "25,000", "1,00,000", "$8M"}


def test_verify_accepts_an_allowed_figure_followed_by_a_comma(resume, hits):
    """Before the fix "20%," and "12," read as the figures '20%' and '12,',
    and the second was rejected as fabricated."""
    r = json.loads(_good_output())
    r["resume"]["experience"][0]["bullets"][0] = "Across 12, then 60M+, datasets, found a 20% mismatch, worth $8M."
    assert verify(TailoredResume.model_validate(r), resume, hits) == []


# ------------------------------- verify: facts that are not the Tailor's

@pytest.mark.parametrize("field, value", [
    ("title", "Principal Data Scientist"),
    ("start", "09/2021"),            # "2021" is on the resume (education), so the figure check passes it
    ("end", "Present"),
    ("location", "London"),
])
def test_verify_catches_an_edited_experience_fact(resume, hits, field, value):
    r = json.loads(_good_output())
    r["resume"]["experience"][0][field] = value
    problems = verify(TailoredResume.model_validate(r), resume, hits)
    assert len(problems) == 1 and "`experience` entries" in problems[0]


def test_verify_catches_added_removed_and_reordered_experience(hits):
    two = json.loads(json.dumps(RESUME))
    two["experience"].append({"company": "Keka HR", "title": "Engineer", "location": "Hyderabad",
                              "start": "02/2021", "end": "08/2022", "bullets": []})
    original = Resume.model_validate(two)
    changes = [{"where": "summary", "atom_id": None, "why": "x"}]

    def check(experience):
        doc = {"resume": {**two, "experience": experience}, "changes": changes}
        return verify(TailoredResume.model_validate(doc), original, hits)

    assert check(two["experience"]) == []
    assert any("`experience` entries" in p for p in check(two["experience"][:1]))        # a job dropped
    assert any("`experience` entries" in p for p in check(two["experience"][::-1]))      # reordered
    invented = {**two["experience"][1], "company": "Acme"}
    assert any("`experience` entries" in p for p in check(two["experience"] + [invented]))


def _with_project(**project) -> TailoredResume:
    # The example atom states no year, so any year already on the resume passes.
    r = json.loads(_good_output())
    r["resume"]["projects"] = [{
        "name": "Text2SQL Agent with Long-Term Memory", "year": "2022",
        "links": [{"type": "github", "label": "GitHub",
                   "url": "https://github.com/chakiAunkit/Text2SQL-Memory-Agent"}],
        "description": "Built a fully local Text2SQL agent with long-term memory.", **project}]
    r["changes"].append({"where": "projects[0]", "atom_id": "self-text2sql-memory-agent", "why": "SQL"})
    return TailoredResume.model_validate(r)


def test_verify_accepts_a_project_from_an_offered_atom(resume, hits):
    assert verify(_with_project(), resume, hits) == []
    # the atom's title is accepted as a name too
    assert verify(_with_project(name="Local Text2SQL agent with long-term memory"), resume, hits) == []


def test_verify_catches_an_invented_project(resume, hits):
    """No figure and no atom id to trip the other checks."""
    out = _with_project(name="Kubernetes Autoscaler", links=[],
                        description="Built an autoscaler for LLM serving.")
    problems = verify(out, resume, hits)
    assert len(problems) == 1 and "Kubernetes Autoscaler" in problems[0]


def test_verify_catches_an_invented_url(resume, hits):
    out = _with_project(links=[{"type": "github", "label": "GitHub",
                                "url": "https://github.com/someone-else/repo"}])
    problems = verify(out, resume, hits)
    assert len(problems) == 1 and "https://github.com/someone-else/repo" in problems[0]


def test_verify_catches_a_changed_project_year(hits):
    base = json.loads(json.dumps(RESUME))
    base["projects"] = [{"name": "Attrition Model", "year": "2021", "links": [], "description": "A model."}]
    out = json.loads(json.dumps(base))
    out["projects"][0]["year"] = "2022"                       # "2022" is on the resume: the figure check passes it
    doc = {"resume": out, "changes": [{"where": "summary", "atom_id": None, "why": "x"}]}
    problems = verify(TailoredResume.model_validate(doc), Resume.model_validate(base), hits)
    assert len(problems) == 1 and "year must be '2021'" in problems[0]


def test_verify_holds_a_new_project_to_its_atoms_year(resume, hits):
    atom = Profile.model_validate({"atoms": [{
        "id": "self-attrition", "section": "project", "source": "Attrition Prediction Modelling",
        "title": "Employee attrition prediction model", "summary": "Predicted attrition.", "year": "2021"}]}).atoms[0]

    def check(year):
        r = json.loads(json.dumps(RESUME))
        r["projects"] = [{"name": "Attrition Prediction Modelling", "year": year, "links": [],
                          "description": "Predicted attrition."}]
        doc = {"resume": r, "changes": [{"where": "projects[0]", "atom_id": "self-attrition", "why": "x"}]}
        return verify(TailoredResume.model_validate(doc), resume, hits + [BriefHit(atom, 1.0, (), ())])

    assert check("2021") == []
    assert check("2022") == ["project 'Attrition Prediction Modelling': year must be '2021', not '2022'"]


def test_model_and_effort_come_from_the_environment_at_call_time(resume, hits, monkeypatch):
    """They used to be read at import, before the CLIs' load_dotenv(), so a
    TAILOR_MODEL in .env was ignored."""
    monkeypatch.setenv("TAILOR_MODEL", "claude-opus-5")
    monkeypatch.setenv("TAILOR_EFFORT", "high")
    client = fake_client(_good_output(), _good_output(), _good_output())
    tailor(resume, BRIEF, hits, client=client)
    assert client.calls[0]["model"] == "claude-opus-5"
    assert client.calls[0]["output_config"]["effort"] == "high"
    tailor(resume, BRIEF, hits, client=client, model="claude-sonnet-5", effort="low")   # an argument wins
    assert client.calls[1]["model"] == "claude-sonnet-5"
    assert client.calls[1]["output_config"]["effort"] == "low"
    monkeypatch.setenv("TAILOR_MODEL", "")                    # an empty value is not a model id
    tailor(resume, BRIEF, hits, client=client)
    assert client.calls[2]["model"] == "claude-sonnet-5"


def test_system_prompt_carries_page_budget():
    assert "fit 1 A4 page(s)" in system_prompt(1)
    assert "fit 2 A4 page(s)" in system_prompt(2)


# --------------------------------------------------------------- tailor

def test_happy_path_one_attempt(resume, hits, monkeypatch):
    monkeypatch.delenv("TAILOR_MODEL", raising=False)       # the defaults, whatever the shell has set
    monkeypatch.delenv("TAILOR_EFFORT", raising=False)
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
    # what a caller needs to record the failure: the rejected answer and its cost
    assert exc.value.last_output == bad
    assert exc.value.usage == {"input_tokens": 200, "output_tokens": 100}


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


# ----------------------------------------------- expand step (Oct 2026)

PUB_ATOM = {"id": "pub-unet", "section": "publication", "source": "Journal of Imaging",
            "title": "Improved U-Net", "year": "2021",
            "summary": "U-Net with a VGG-16 encoder reaching 0.9975 pixel accuracy on 3,929 MRI images.",
            "metrics": ["0.9975 pixel accuracy", "3,929 images"]}


@pytest.fixture
def with_pub():
    resume = json.loads(json.dumps(RESUME))
    resume["publications"] = [{"title": "Improved U-Net", "venue": "Journal of Imaging",
                               "citations": 151, "url": "https://example.com/unet"}]
    atom = Profile.model_validate({"atoms": [PUB_ATOM]}).atoms[0]
    return Resume.model_validate(resume), BriefHit(atom, 0.0, (), ())


def _with_summary(base: Resume, summary: str) -> str:
    r = base.model_dump(mode="json")
    r["publications"][0]["summary"] = summary
    return json.dumps({"resume": r, "changes": [
        {"where": "publications[0].summary", "atom_id": "pub-unet", "why": "fills leftover room"}]})


def test_expand_instruction_states_the_room_and_the_limits():
    text = expand_instruction(0.10, 6)
    assert "about 10%" in text and "roughly 6 lines" in text
    assert "at most 5 lines (about 500 characters)" in text           # one line of slack
    assert "Do not remove, shorten, reorder or rewrite" in text and '"none"' in text
    assert "publication" in text and "experience atom not yet on the resume" in text


def test_system_prompt_leaves_publication_summaries_to_the_fit_step():
    assert "8. Publications: leave each publication's `summary` as it is" in system_prompt(1)


def test_expand_adds_a_grounded_publication_summary(with_pub, hits):
    resume, pub_hit = with_pub
    summary = "U-Net with a VGG-16 encoder for brain MRI segmentation, reaching 0.9975 pixel accuracy on 3,929 images."
    client = fake_client(_with_summary(resume, summary))
    result = revise(resume, BRIEF, hits + [pub_hit], expand_instruction(0.1, 6), client=client)
    assert result.resume.publications[0].summary == summary and result.attempts == 1
    prompt = client.calls[0]["messages"][0]["content"]
    assert prompt.rstrip().endswith('whose `where` is "none".')
    assert '"pub-unet"' in prompt                                     # the atom is offered


def test_expand_cannot_invent_a_figure_in_a_publication_summary(with_pub, hits):
    resume, pub_hit = with_pub
    out = TailoredResume.model_validate_json(_with_summary(resume, "Reached 0.9991 pixel accuracy."))
    problems = verify(out, resume, hits + [pub_hit])
    assert any("0.9991" in p for p in problems)


# ------------------------------------- adapters for the fit loop (Phase 5)

def test_cutter_revises_with_the_cut_instruction_and_logs_it(resume, hits):
    from src.agents.tailor import RevisionLog, cutter
    tailored = TailoredResume.model_validate_json(_good_output()).resume
    cut_resume = tailored.model_copy(deep=True)
    cut_resume.experience[0].bullets.pop(0)
    reply = json.dumps({"resume": cut_resume.model_dump(mode="json"),
                        "changes": [{"where": "experience[Target Corporation].bullets[0]",
                                     "atom_id": None, "why": "least relevant"}]})
    client, log = fake_client(reply), RevisionLog()
    out = cutter(BRIEF, hits, log, client=client)(tailored)
    assert len(out.experience[0].bullets) == 1
    assert client.calls[0]["messages"][0]["content"].rstrip().endswith(CUT_INSTRUCTION)
    assert log.usage == {"input_tokens": 100, "output_tokens": 50} and len(log.cuts) == 1


def test_expander_logs_a_kept_expansion_and_survives_a_rejected_one(with_pub, hits):
    from src.agents.tailor import RevisionLog, expander, expansion_hits
    resume, pub_hit = with_pub
    offered = expansion_hits(hits, [pub_hit.atom, hits[0].atom])
    assert [h.atom.id for h in offered] == [h.atom.id for h in hits] + ["pub-unet"]   # added once, scored 0
    assert offered[-1].score == 0.0 and offered[-1].covers == ()

    good = _with_summary(resume, "U-Net with a VGG-16 encoder reaching 0.9975 pixel accuracy on 3,929 images.")
    log = RevisionLog()
    out = expander(BRIEF, offered, log, client=fake_client(good))(resume, 0.1, 6)
    assert out.publications[0].summary.startswith("U-Net") and len(log.expansions) == 1
    assert log.kept_expansions(1)[0].atom_id == "pub-unet" and log.kept_expansions(0) == []

    bad = _with_summary(resume, "Reached 0.9991 pixel accuracy.")                    # an invented figure, twice
    log = RevisionLog()
    assert expander(BRIEF, offered, log, client=fake_client(bad, bad))(resume, 0.1, 6) is None
    assert log.expansions == [] and "0.9991" in log.notes[0]
    assert log.usage == {"input_tokens": 200, "output_tokens": 100}                 # the failed calls still cost


# --------------------------------------- verify: publication facts

@pytest.mark.parametrize("field, value", [
    ("venue", "Nature"),
    ("citations", 3929),             # a figure the atom does state, about something else
    ("url", "https://example.com/other"),
])
def test_verify_catches_an_edited_publication_fact(with_pub, hits, field, value):
    resume, pub_hit = with_pub
    r = resume.model_dump(mode="json")
    r["publications"][0][field] = value
    doc = {"resume": r, "changes": [{"where": "publications[0]", "atom_id": None, "why": "x"}]}
    problems = verify(TailoredResume.model_validate(doc), resume, hits + [pub_hit])
    assert any("venue, citations and url must be returned exactly as given" in p for p in problems)


def test_verify_checks_a_new_publication_against_its_atom(resume, hits):
    atom = Profile.model_validate({"atoms": [{
        **PUB_ATOM, "metrics": PUB_ATOM["metrics"] + ["151 citations"],
        "links": {"paper": "https://example.com/unet"}}]}).atoms[0]
    offered = hits + [BriefHit(atom, 1.0, (), ())]

    def check(**over):
        r = json.loads(json.dumps(RESUME))
        r["publications"] = [{"title": "Improved U-Net", "venue": "Journal of Imaging",
                              "citations": 151, "url": "https://example.com/unet", **over}]
        doc = {"resume": r, "changes": [{"where": "publications[0]", "atom_id": "pub-unet", "why": "x"}]}
        return verify(TailoredResume.model_validate(doc), resume, offered)

    assert check() == []
    assert any("citations=3929" in p for p in check(citations=3929))
    assert any("venue must be the atom's `source`" in p for p in check(venue="Nature"))
    assert any("neither on the current resume nor an offered publication atom" in p
               for p in check(title="A Paper Nobody Wrote"))
