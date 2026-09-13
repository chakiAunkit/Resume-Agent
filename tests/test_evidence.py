"""Phase 2 + 3 retrieval tests.

Run from the repo root:  pytest tests/test_evidence.py -v

Two layers: synthetic atoms pin the scoring rules (always run); the
roadmap acceptance queries run against the real profile.yaml when it is
present and skip cleanly where it isn't (it's gitignored).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.atoms import Profile
from src.brief import JobBrief
from src.tools.evidence import load_evidence, retrieve_for_brief, search, tokenize

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "profile.yaml"
SAMPLE_BRIEF = ROOT / "data" / "sample_jd.brief.yaml"


def _atom(id: str, **over) -> dict:
    base = {
        "id": id,
        "section": "project",
        "source": "src",
        "title": "untitled",
        "summary": "nothing here",
        "year": "2024",
    }
    return {**base, **over}


@pytest.fixture
def synthetic() -> Profile:
    return Profile.model_validate({"atoms": [
        _atom("tagged", keywords=["fraud detection"]),
        _atom("prose-only", summary="did some fraud work and some detection work"),
        _atom("split-tags", keywords=["fraud", "detection"]),
        _atom("unrelated", keywords=["gardening"]),
    ]})


# ---------------------------------------------------------------- scoring

def test_tokenize_drops_stopwords_and_keeps_compound_tokens():
    assert tokenize("Built the 1D-CNN with scikit-learn.") == ["built", "1d-cnn", "scikit-learn"]


def test_tokenize_normalises_plurals_symmetrically():
    assert tokenize("LLMs models recommendations") == ["llm", "model", "recommendation"]
    # short tokens and -ss words are untouched
    assert tokenize("aws k8s class") == ["aws", "k8s", "class"]
    # both sides get the same treatment, so plural tag vs singular query matches
    prof = Profile.model_validate({"atoms": [_atom("plural", keywords=["recommendations"])]})
    assert search("recommendation", profile=prof)[0].atom.id == "plural"


def test_curated_tags_outrank_prose(synthetic):
    ids = [h.atom.id for h in search("fraud detection", k=10, profile=synthetic)]
    assert ids.index("tagged") < ids.index("prose-only")
    assert ids.index("split-tags") < ids.index("prose-only")


def test_phrase_bonus_prefers_exact_tag(synthetic):
    ids = [h.atom.id for h in search("fraud detection", k=10, profile=synthetic)]
    assert ids[0] == "tagged"


def test_zero_score_atoms_excluded_and_k_respected(synthetic):
    hits = search("fraud detection", k=10, profile=synthetic)
    assert "unrelated" not in {h.atom.id for h in hits}
    assert len(search("fraud detection", k=1, profile=synthetic)) == 1
    assert search("quantum", profile=synthetic) == []


def test_hits_are_explainable(synthetic):
    top = search("fraud detection", k=1, profile=synthetic)[0]
    assert top.matched == ("detection", "fraud")


# -------------------------------------------------------- brief retrieval

@pytest.fixture
def brief_profile() -> Profile:
    return Profile.model_validate({"atoms": [
        _atom("graph", keywords=["knowledge graph"]),
        _atom("generic", summary="data data data everywhere", keywords=["data"]),
        _atom("nice-only", keywords=["quantum"]),
        _atom("must-only", keywords=["gardening"]),
        _atom("silent"),
    ]})


def _brief(must, nice=()):
    return JobBrief(title="t", company="c", must_haves=list(must), nice_to_haves=list(nice))


def test_brief_hit_records_which_requirements_it_covers(brief_profile):
    hits = retrieve_for_brief(_brief(["knowledge graph", "gardening"]), profile=brief_profile)
    by_id = {h.atom.id: h for h in hits}
    assert by_id["graph"].covers == ("knowledge graph",)
    assert by_id["must-only"].covers == ("gardening",)
    assert "silent" not in by_id


def test_nice_to_have_scores_below_equivalent_must_have(brief_profile):
    hits = retrieve_for_brief(
        _brief(must=["gardening"], nice=["quantum"]), profile=brief_profile
    )
    by_id = {h.atom.id: h for h in hits}
    assert by_id["must-only"].score > by_id["nice-only"].score
    assert by_id["nice-only"].covers == ()
    assert by_id["nice-only"].supports == ("quantum",)


def test_phrase_match_outranks_generic_term_spread(brief_profile):
    """The reason for per-requirement querying: one exact requirement
    match beats an atom that merely repeats a common word."""
    hits = retrieve_for_brief(
        _brief(["knowledge graph", "data analytics"]), profile=brief_profile
    )
    assert hits[0].atom.id == "graph"


def test_partial_match_ranks_but_does_not_cover(brief_profile):
    """Coverage is strict, ranking is soft: "generic" matches only
    "data" out of "data analytics" — it appears, scaled down, but the
    requirement is not listed as covered."""
    hits = retrieve_for_brief(_brief(["data analytics"]), profile=brief_profile)
    by_id = {h.atom.id: h for h in hits}
    assert "generic" in by_id
    assert by_id["generic"].covers == ()
    full = search("data analytics", profile=brief_profile)[0].score
    assert by_id["generic"].score == pytest.approx(full / 2)


def test_soft_skills_do_not_affect_retrieval(brief_profile):
    base = retrieve_for_brief(_brief(["knowledge graph"]), profile=brief_profile)
    soft = JobBrief(title="t", company="c", must_haves=["knowledge graph"],
                    soft_skills=["gardening"])  # would match "must-only" if retrieved
    with_soft = retrieve_for_brief(soft, profile=brief_profile)
    assert [(h.atom.id, h.score) for h in with_soft] == [(h.atom.id, h.score) for h in base]


def test_brief_k_respected(brief_profile):
    assert len(retrieve_for_brief(_brief(["knowledge graph", "gardening", "data"]),
                                  k=1, profile=brief_profile)) == 1


# ---------------------------------------------------------- acceptance

needs_real = pytest.mark.skipif(not REAL.exists(), reason="data/profile.yaml not present")


@needs_real
def test_fraud_detection_returns_paypal_first():
    ids = [a.id for a in load_evidence("fraud detection", k=3)]
    assert ids[0].startswith("paypal-")
    assert "paypal-rule-governance-analytics" in ids


@needs_real
def test_rag_returns_insight_store_and_text2sql():
    ids = [a.id for a in load_evidence("RAG", k=3)]
    assert ids[0] == "paypal-fra-insight-store"
    assert "self-text2sql-memory-agent" in ids


@needs_real
def test_supply_chain_returns_target():
    ids = [a.id for a in load_evidence("supply chain logistics", k=3)]
    assert all(i.startswith("target-") for i in ids)
    assert "target-goa-dmo-mismatch" in ids


@needs_real
def test_sample_brief_surfaces_insight_store():
    """Eval case 001 (data/eval/README.md): the knowledge-graph RAG atom
    must be in the top 3 for the eBay Product Knowledge brief."""
    hits = retrieve_for_brief(JobBrief.load(SAMPLE_BRIEF), k=8)
    ids = [h.atom.id for h in hits]
    assert "paypal-fra-insight-store" in ids[:3]
    assert "knowledge graph" in dict((h.atom.id, h.covers) for h in hits)["paypal-fra-insight-store"]
