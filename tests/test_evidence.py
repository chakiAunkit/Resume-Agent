"""Phase 2 retrieval tests.

Run from the repo root:  pytest tests/test_evidence.py -v

Two layers: synthetic atoms pin the scoring rules (always run); the
roadmap acceptance queries run against the real profile.yaml when it is
present and skip cleanly where it isn't (it's gitignored).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.atoms import Profile
from src.tools.evidence import load_evidence, search, tokenize

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "data" / "profile.yaml"


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
