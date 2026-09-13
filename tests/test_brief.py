"""Phase 3 brief tests: a malformed brief must fail before the Tailor runs.

Run from the repo root:  pytest tests/test_brief.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.brief import JobBrief

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample_jd.brief.yaml"

GOOD = {
    "title": "Data Scientist",
    "company": "Acme",
    "must_haves": ["SQL", "Python"],
}


@pytest.mark.parametrize("path", sorted((ROOT / "data").glob("*.brief.yaml")), ids=lambda p: p.stem)
def test_every_brief_in_data_loads(path):
    """Every committed brief is an Extractor eval case; all must validate."""
    JobBrief.load(path)


def test_sample_brief_loads():
    brief = JobBrief.load(SAMPLE)
    assert brief.company == "eBay"
    assert "knowledge graph" in brief.must_haves
    assert brief.years_experience == 4


def test_sample_brief_routes_soft_skills_out_of_must_haves():
    brief = JobBrief.load(SAMPLE)
    assert "cross-functional collaboration" in brief.soft_skills
    assert "cross-functional collaboration" not in brief.must_haves


def test_minimal_brief_uses_defaults():
    brief = JobBrief.model_validate(GOOD)
    assert brief.nice_to_haves == []
    assert brief.soft_skills == []
    assert brief.seniority == "unspecified"
    assert brief.years_experience is None


def test_empty_must_haves_rejected():
    with pytest.raises(ValidationError):
        JobBrief.model_validate({**GOOD, "must_haves": []})


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        JobBrief.model_validate({**GOOD, "salary": "lots"})


def test_free_text_seniority_rejected():
    with pytest.raises(ValidationError):
        JobBrief.model_validate({**GOOD, "seniority": "Mid-level (4+ yrs)"})


def test_sentence_length_requirement_rejected():
    long = "Proficiency in SQL & Python for data wrangling, analytics, and automation, plus more"
    with pytest.raises(ValidationError):
        JobBrief.model_validate({**GOOD, "must_haves": [long]})
