"""Phase 3 fit-loop tests. No browser: a fake renderer reports page count
and fill from a simple content model, so the loop's decisions are pinned.

Run from the repo root:  pytest tests/test_fit.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.models import Resume
from src.tools.fit import FILL_MIN, MAX_ITERATIONS, FitError, estimate_scale, fit
from src.tools.render import SCALE_MAX, SCALE_MIN, RenderResult

RESUME = {
    "contact": {"name": "T", "phone": "1", "email": "t@example.com"},
    "summary": "s",
    "experience": [{"company": "C", "title": "t", "location": "l", "start": "01/2020", "end": "Present",
                    "bullets": ["b1", "b2", "b3", "b4"]}],
    "skills": [{"category": "Technical", "items": ["SQL"]}],
    "education": [{"degree": "d", "institution": "i", "location": "l", "start": "01/2015", "end": "01/2019"}],
    "projects": [{"name": "p", "year": "2024", "description": "d"}],
}


class FakeRenderer:
    """Content height = units_per_item * items * scale^2 — the same square
    law the loop assumes, so estimates land; `pages` is ceil(fill)."""

    def __init__(self, units_per_item: float):
        self.units = units_per_item
        self.calls: list[tuple[int, float]] = []

    def render(self, data: Resume, scale: float = 1.0, out_path=None) -> RenderResult:
        items = sum(len(e.bullets) for e in data.experience) + len(data.projects)
        fill = self.units * items * scale ** 2
        self.calls.append((items, scale))
        return RenderResult(pdf_path=Path("x.pdf"), html_path=Path("x.html"),
                            page_count=max(1, math.ceil(fill - 1e-9)), fill=fill, scale=scale)


def cut_last(resume: Resume) -> Resume:
    r = resume.model_copy(deep=True)
    r.experience[0].bullets.pop()
    return r


@pytest.fixture
def resume() -> Resume:
    return Resume.model_validate(RESUME)


def test_estimate_uses_square_law():
    # 1.10 pages of content -> scale ~ 1/sqrt(1.1) * margin
    assert estimate_scale(1.0, 1.10, 1) == pytest.approx(1 / math.sqrt(1.10) * 0.985, abs=1e-3)
    assert estimate_scale(1.0, 3.0, 1) == SCALE_MIN      # clamped
    assert estimate_scale(1.0, 0.3, 1) == SCALE_MAX      # clamped


def test_first_render_fits_no_steps(resume):
    r = FakeRenderer(units_per_item=0.19)  # 5 items -> 0.95 page
    out = fit(resume, r, max_pages=1)
    assert out.iterations == 0 and out.page_count == 1 and out.scale == 1.0
    assert len(r.calls) == 1


def test_slightly_over_scales_down_once(resume):
    r = FakeRenderer(units_per_item=0.21)  # 5 items -> 1.05 pages: the run-001 case
    out = fit(resume, r, max_pages=1)
    assert out.page_count == 1
    assert out.iterations == 1
    assert SCALE_MIN < out.scale < 1.0
    assert out.actions[0].startswith("over (2p")


def test_far_over_cuts_after_hitting_scale_min(resume):
    r = FakeRenderer(units_per_item=0.40)  # 5 items -> 2.0 pages; at 0.8 still 1.28
    out = fit(resume, r, cutter=cut_last, max_pages=1)
    assert out.page_count == 1
    assert any("cut" in a for a in out.actions)
    assert any("scale 1.000 -> 0.8" in a for a in out.actions)
    assert len(out.resume.experience[0].bullets) < 4
    assert out.iterations <= MAX_ITERATIONS


def test_over_without_cutter_raises(resume):
    r = FakeRenderer(units_per_item=0.40)
    with pytest.raises(FitError, match="no cutter"):
        fit(resume, r, max_pages=1)


def test_gives_up_after_cap(resume):
    r = FakeRenderer(units_per_item=2.0)  # hopeless: each item is 2 pages
    with pytest.raises(FitError, match=f"after {MAX_ITERATIONS} steps"):
        fit(resume, r, cutter=cut_last, max_pages=1)


def test_under_filled_scales_up(resume):
    r = FakeRenderer(units_per_item=0.12)  # 5 items -> 0.60 page
    out = fit(resume, r, max_pages=1)
    assert out.scale > 1.0
    assert out.page_count == 1
    assert out.fill >= FILL_MIN or out.scale == SCALE_MAX


def test_two_page_budget(resume):
    r = FakeRenderer(units_per_item=0.42)  # 2.1 pages -> one scale step to 2
    out = fit(resume, r, max_pages=2)
    assert out.page_count == 2 and out.iterations == 1
