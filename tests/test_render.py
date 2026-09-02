"""Phase 1 acceptance tests: real resume.json -> PDF, deterministically.

Run from the repo root:  pytest tests/test_render.py -v

One Chromium session is shared across the whole module (module-scoped
fixtures) so the suite stays fast: two renders total.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError
from pypdf import PdfReader

from src.models import Resume
from src.tools.render import Renderer, build_html

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "assets" / "resume.json"


# ---------------------------------------------------------------------------
# Shared fixtures: one browser, one baseline render
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def resume() -> Resume:
    return Resume.load(DATA)


@pytest.fixture(scope="module")
def renderer():
    with Renderer() as r:
        yield r


@pytest.fixture(scope="module")
def baseline(renderer, resume, tmp_path_factory):
    out = tmp_path_factory.mktemp("render") / "cv.pdf"
    return renderer.render(resume, out_path=out)


def _pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() for page in reader.pages).lower()


# ---------------------------------------------------------------------------
# Fast checks: no browser needed, fail in milliseconds
# ---------------------------------------------------------------------------

def test_garbage_data_rejected_before_browser():
    with pytest.raises(ValidationError):
        build_html({"contact": {}})


def test_scale_out_of_bounds_rejected(resume):
    with pytest.raises(ValueError):
        build_html(resume, scale=0.3)


# ---------------------------------------------------------------------------
# Render checks: the real artifact
# ---------------------------------------------------------------------------

def test_pdf_exists_within_page_budget(baseline):
    assert baseline.pdf_path.exists()
    assert baseline.html_path.exists()  # debug artifact written alongside
    assert 1 <= baseline.page_count <= 2


def test_content_survives_round_trip(baseline, resume):
    """Every load-bearing fact must be extractable from the PDF text.

    This doubles as the ATS-safety floor the Judge will later enforce:
    if pypdf can't read it, an applicant tracking system can't either.
    """
    text = _pdf_text(baseline.pdf_path)

    assert resume.contact.name.lower() in text
    assert resume.contact.email.lower() in text
    for job in resume.experience:
        assert job.company.lower() in text, f"missing company: {job.company}"
    for group in resume.skills:
        assert group.category.lower() in text
    # metrics render intact (short tokens: safe from hyphenation)
    assert "$8m" in text
    assert "25,000+" in text


def test_scale_knob_actually_scales(renderer, resume, baseline, tmp_path_factory):
    """Rendering smaller must never *increase* the page count, and the
    content must be unchanged — scale is presentation-only."""
    out = tmp_path_factory.mktemp("scaled") / "cv_small.pdf"
    result = renderer.render(resume, scale=0.85, out_path=out)

    assert result.page_count <= baseline.page_count
    assert resume.contact.name.lower() in _pdf_text(result.pdf_path)
