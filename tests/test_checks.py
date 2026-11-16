"""Deterministic Judge checks (Phase 5). No PDF and no browser: the PDF
text is injected.

Run from the repo root:  pytest tests/test_checks.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.atoms import Profile
from src.brief import JobBrief
from src.models import Resume
from src.tools.checks import JudgeReport, check_resume
from src.tools.evidence import retrieve_for_brief

ROOT = Path(__file__).resolve().parents[1]

RESUME = Resume.model_validate({
    "contact": {"name": "Test Person", "phone": "+1 555 0100", "email": "t@example.com", "links": []},
    "summary": "Data scientist working on e-commerce logistics with product and engineering partners.",
    "experience": [{
        "company": "Target Corporation", "title": "Senior Data Scientist", "location": "Bangalore",
        "start": "09/2022", "end": "05/2025",
        "bullets": ["Analyzed 60M+ customer orders in HiveQL to find a 20% mismatch, unlocking $8M."],
    }],
    "skills": [{"category": "Technical", "items": ["SQL", "HiveQL"]}],
    "education": [{"degree": "B.Tech", "institution": "U", "location": "City",
                   "start": "07/2017", "end": "04/2021"}],
})

BRIEF = JobBrief(title="Data Scientist", company="Acme",
                 must_haves=["SQL", "supply chain", "taxonomy"],
                 soft_skills=["collaboration", "engineering partners"])
HITS = retrieve_for_brief(BRIEF, k=5, profile=Profile.load(ROOT / "data" / "profile.example.yaml"))

# What a healthy PDF yields: every string, with a title wrapped across lines.
GOOD_TEXT = ("TEST PERSON\nt@example.com\nTarget Corporation\nSenior Data\nScientist\nTechnical: SQL, HiveQL\n"
             "B.Tech\n" + RESUME.summary + " " + RESUME.experience[0].bullets[0])


def check(text: str = GOOD_TEXT, pages: int = 1, **kw) -> JudgeReport:
    return check_resume(RESUME, "unused.pdf", pages, BRIEF, HITS, read_pdf=lambda _path: text, **kw)


def test_a_fitted_extractable_resume_passes():
    report = check()
    assert report.passed and report.fixes == [] and report.defects == []


def test_an_over_length_pdf_is_a_fix_for_the_tailor():
    report = check(pages=2)
    assert not report.passed and len(report.fixes) == 1 and "2 pages and the budget is 1" in report.fixes[0]
    assert check(pages=2, max_pages=2).passed


def test_text_missing_from_the_pdf_is_a_defect_not_a_fix():
    """Rewriting cannot repair a render that hides text, so it must not be
    sent back to the Tailor."""
    report = check(GOOD_TEXT.replace("Target Corporation", ""))
    assert not report.passed and report.fixes == []
    assert "company 'Target Corporation'" in report.defects[0]
    blank = check("")
    assert any("exposes only 0%" in d for d in blank.defects)


def test_gaps_are_reported_and_never_fail_the_resume():
    report = check()
    assert report.passed
    # SQL is covered and on the resume: no gap.
    assert not any("'SQL'" in g for g in report.gaps)
    # "supply chain" has an atom behind it but the resume never says it.
    assert any(g.startswith("must-have 'supply chain': supported by target-goa-dmo-mismatch") for g in report.gaps)
    # "taxonomy" has no atom at all.
    assert "must-have 'taxonomy': no evidence in the profile" in report.gaps
    # one soft skill is conveyed by phrasing, the other is not
    assert any("'collaboration'" in g for g in report.gaps)
    assert not any("'engineering partners'" in g for g in report.gaps)


def test_default_reader_extracts_text_from_a_real_pdf(tmp_path):
    """The injected reader's default: pypdf on the rendered file."""
    pytest.importorskip("playwright")
    from src.tools.checks import pdf_text
    from src.tools.render import Renderer
    with Renderer() as renderer:
        result = renderer.render(RESUME, out_path=tmp_path / "cv.pdf")
    report = check_resume(RESUME, result.pdf_path, result.page_count, BRIEF, HITS)
    assert report.passed, report.defects
    assert "target corporation" in pdf_text(result.pdf_path).lower()
