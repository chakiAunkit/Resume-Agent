"""Phase 3 fit-loop tests. No browser: a fake renderer reports page count
and fill from a simple content model, so the loop's decisions are pinned.

Run from the repo root:  pytest tests/test_fit.py -v
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.models import Resume
from src.tools.fit import (FILL_MIN, FILL_TARGET, MAX_EXPANSIONS, MAX_ITERATIONS, FitError,
                          estimate_scale, fit, free_lines)
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


def test_a_failed_run_is_recorded_in_the_output_folder(tmp_path, resume):
    """A failed run used to leave an empty folder and no reason."""
    import json

    from src.agents.tailor import Change, TailorError
    from src.tools.fit import FAILURE_FILES, record_failure

    # Tailor stage: the rejected answer and the reasons are kept.
    rejected = TailorError(["these figures ... must be removed: ['$50M']"], last_output='{"resume": "..."}',
                           usage={"input_tokens": 200, "output_tokens": 90})
    path = record_failure(tmp_path / "a", "tailor", rejected, {"input_tokens": 200, "output_tokens": 90})
    error = json.loads(path.read_text(encoding="utf-8"))
    assert path.name == "error.json" and error["stage"] == "tailor" and error["error"] == "TailorError"
    assert error["problems"] == ["these figures ... must be removed: ['$50M']"]
    assert error["tokens"] == {"input_tokens": 200, "output_tokens": 90} and error["failed_at"]
    assert (tmp_path / "a" / "rejected_output.txt").read_text(encoding="utf-8") == '{"resume": "..."}'
    assert not (tmp_path / "a" / "resume.unfitted.json").exists()          # nothing was accepted yet

    # Fit stage: the Tailor's accepted resume is kept, and still loads.
    changes = [Change(where="summary", atom_id=None, why="x")]
    path = record_failure(tmp_path / "b", "fit", FitError("still 2 pages after 3 steps"), {"input_tokens": 1},
                          tailored=resume, changes=changes)
    error = json.loads(path.read_text(encoding="utf-8"))
    assert error["stage"] == "fit" and error["problems"] == [] and "still 2 pages" in error["message"]
    assert Resume.load(tmp_path / "b" / "resume.unfitted.json") == resume
    assert json.loads((tmp_path / "b" / "changes.unfitted.json").read_text(encoding="utf-8"))[0]["where"] == "summary"
    assert {p.name for p in (tmp_path / "b").iterdir()} <= set(FAILURE_FILES)


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


# ------------------------------------------------- expand step (Oct 2026)

def add_bullet(resume: Resume, room: float, lines: int) -> Resume:
    r = resume.model_copy(deep=True)
    r.experience[0].bullets.append("added")
    return r


def test_free_lines_converts_room_to_lines_of_body_text():
    assert free_lines(0.75, 1, 1.0) == 15        # 0.25 x 1055.8 px / 17.5 px per line
    assert free_lines(1.02, 1, 1.0) == 0         # no room when already over
    assert free_lines(0.8, 1, 0.9) > free_lines(0.8, 1, 1.0)   # smaller type, more lines


def test_expands_while_there_is_room_and_reverts_an_overflow(resume):
    """0.15 per item: 5 -> 0.75 (room), 6 -> 0.90 (kept, still room), 7 -> 1.05 (overflow)."""
    r = FakeRenderer(units_per_item=0.15)
    asked: list[tuple[float, int]] = []

    def expander(res, room, lines):
        asked.append((round(room, 2), lines))
        return add_bullet(res, room, lines)

    out = fit(resume, r, expander=expander, max_pages=1)
    assert out.expanded == 1 and len(out.resume.experience[0].bullets) == 5
    assert out.page_count == 1 and out.fill == pytest.approx(0.90)
    assert any("expanded -> fill 0.90" in a for a in out.actions)
    assert any("overflowed to 2p, reverted" in a for a in out.actions)
    assert r.calls[-1] == (6, 1.0)               # the PDF left on disk is the kept version
    assert asked[0] == (0.25, free_lines(0.75, 1, 1.0)) and len(asked) == MAX_EXPANSIONS


def test_content_comes_before_type_size(resume):
    """0.12 per item: 0.60 -> 0.72 -> 0.84 by content, then the last bit by scale."""
    r = FakeRenderer(units_per_item=0.12)
    out = fit(resume, r, expander=add_bullet, max_pages=1)
    assert out.expanded == 2
    kinds = ["expanded" if "expanded" in a else "scale" if "scale" in a else a for a in out.actions]
    assert kinds == ["expanded", "expanded", "scale"]
    assert out.scale > 1.0 and out.page_count == 1


def test_nothing_to_add_falls_back_to_scaling_up(resume):
    r = FakeRenderer(units_per_item=0.12)
    out = fit(resume, r, expander=lambda res, room, lines: None, max_pages=1)
    assert out.expanded == 0 and "nothing to add" in out.actions[0]
    assert out.scale > 1.0                       # same outcome as with no expander


def test_a_full_page_is_not_expanded(resume):
    r = FakeRenderer(units_per_item=0.19)        # 0.95: at FILL_TARGET already
    called: list = []
    out = fit(resume, r, expander=lambda *a: called.append(a), max_pages=1)
    assert called == [] and out.iterations == 0 and 0.95 >= FILL_TARGET


def test_expansion_happens_after_an_over_page_is_fitted(resume):
    """Over at first: scale down to fit, then use any room left for content."""
    r = FakeRenderer(units_per_item=0.21)        # 1.05 pages -> one scale step -> ~0.96
    out = fit(resume, r, expander=add_bullet, max_pages=1)
    assert out.actions[0].startswith("over (2p")
    assert out.page_count == 1
