"""resume.json must agree with profile.yaml — one source of truth.

The roadmap says profile.yaml is a superset of the CV. Nothing enforced
it, and the two drifted (citation counts updated in atoms, not on the
resume; the Tailor faithfully propagated the stale figure). These tests
make drift a red pytest instead of a stale PDF.

Both files are gitignored, so everything here skips where they're absent.

Run from the repo root:  pytest tests/test_consistency.py -v
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from src.agents.tailor import _numbers
from src.atoms import Profile
from src.models import Resume
from src.tools.evidence import tokenize

ROOT = Path(__file__).resolve().parents[1]
RESUME = ROOT / "assets" / "resume.json"
PROFILE = ROOT / "data" / "profile.yaml"

pytestmark = pytest.mark.skipif(
    not (RESUME.exists() and PROFILE.exists()),
    reason="assets/resume.json and data/profile.yaml are gitignored",
)


def _norm(text: str) -> str:
    return " ".join(tokenize(text))


@pytest.fixture(scope="module")
def resume() -> Resume:
    return Resume.load(RESUME)


@pytest.fixture(scope="module")
def profile() -> Profile:
    return Profile.load(PROFILE)


def test_every_publication_has_an_atom_with_matching_figures(resume, profile):
    pubs = [a for a in profile.atoms if a.section == "publication"]
    for pub in resume.publications:
        atom = next((a for a in pubs if _norm(a.title) == _norm(pub.title)), None)
        assert atom is not None, f"no publication atom titled {pub.title!r}"
        assert any(_norm(m) == _norm(f"{pub.citations} citations") for m in atom.metrics), (
            f"{pub.title!r}: resume says {pub.citations} citations, "
            f"atom {atom.id!r} metrics say {atom.metrics}"
        )
        assert _norm(pub.venue) in _norm(atom.source) or _norm(atom.source) in _norm(pub.venue), (
            f"{pub.title!r}: venue {pub.venue!r} vs atom source {atom.source!r}"
        )


def test_every_project_has_an_atom(resume, profile):
    projects = {_norm(a.source) for a in profile.atoms if a.section == "project"}
    for project in resume.projects:
        assert _norm(project.name) in projects, f"no project atom for {project.name!r}"


def test_every_figure_on_the_resume_exists_in_some_atom(resume, profile):
    """The fabrication check the Tailor runs, applied to the base resume:
    bullets, project descriptions and publication figures must all be
    quotable from profile.yaml. The summary is checked separately — its
    figures are aggregates, not atom facts. Contact and education are
    skipped (phone numbers and dates aren't achievements)."""
    allowed: set[str] = set()
    for atom in profile.atoms:
        allowed |= _numbers(atom.model_dump_json())

    claims = [b for e in resume.experience for b in e.bullets]
    claims += [p.description for p in resume.projects]
    claims += [str(p.citations) for p in resume.publications]

    missing = {n for text in claims for n in _numbers(text)} - allowed
    assert not missing, f"figures on resume.json with no atom behind them: {sorted(missing)}"


def _claimed(summary: str, unit: str) -> list[int]:
    """'4+ years' -> [4]; '150+ citations' -> [150]. A '+' means a floor."""
    return [int(n) for n in re.findall(rf"(\d[\d,]*)\+?\s*{unit}", summary)]


def test_summary_aggregates_are_derivable(resume, profile):
    """'N+ years' and 'N+ citations' in the summary are derived claims:
    each must be at or below the value the atoms actually support."""
    starts = [a.start for a in profile.atoms if a.section == "experience" and a.start]
    earliest = min(date(int(s[3:]), int(s[:2]), 1) for s in starts)
    years_supported = (date.today() - earliest).days / 365.25
    for claimed in _claimed(resume.summary, "years"):
        assert claimed <= years_supported, (
            f"summary claims {claimed}+ years; atoms support {years_supported:.1f}"
        )

    citations_supported = sum(
        int(m.split()[0].replace(",", ""))
        for a in profile.atoms if a.section == "publication"
        for m in a.metrics if m.endswith("citations")
    )
    for claimed in _claimed(resume.summary, "citations"):
        assert claimed <= citations_supported, (
            f"summary claims {claimed}+ citations; atoms sum to {citations_supported}"
        )