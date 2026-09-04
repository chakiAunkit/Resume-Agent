"""Phase 2 schema tests: profile.yaml must fail loudly on bad data.

Run from the repo root:  pytest tests/test_atoms.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.atoms import Atom, Profile

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data" / "profile.example.yaml"

GOOD = {
    "id": "x-one",
    "section": "project",
    "source": "Side project",
    "title": "Thing",
    "summary": "Did a thing.",
}


def test_example_profile_loads():
    profile = Profile.load(EXAMPLE)
    assert len(profile.atoms) == 2
    atom = profile.by_id("target-goa-dmo-mismatch")
    assert "$8M annual cost savings" in atom.metrics


def test_duplicate_ids_rejected():
    with pytest.raises(ValidationError, match="duplicate atom ids"):
        Profile.model_validate({"atoms": [GOOD, GOOD]})


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        Atom.model_validate({**GOOD, "certifications": ["AWS"]})


@pytest.mark.parametrize("bad_id", ["Target-GOA", "target_goa", "-target", "target-"])
def test_id_must_be_kebab_case(bad_id):
    with pytest.raises(ValidationError):
        Atom.model_validate({**GOOD, "id": bad_id})


def test_bad_date_rejected():
    with pytest.raises(ValidationError):
        Atom.model_validate({**GOOD, "start": "Sept 2022", "end": "Present"})


def test_start_without_end_rejected():
    with pytest.raises(ValidationError, match="given together"):
        Atom.model_validate({**GOOD, "start": "09/2022"})


def test_missing_by_id_raises():
    with pytest.raises(KeyError):
        Profile.load(EXAMPLE).by_id("nope")


def test_real_profile_loads_if_present():
    """profile.yaml is gitignored; skip cleanly where it doesn't exist."""
    real = ROOT / "data" / "profile.yaml"
    if not real.exists():
        pytest.skip("data/profile.yaml not present")
    profile = Profile.load(real)
    assert len(profile.atoms) >= 10
    for atom in profile.atoms:
        if atom.section in ("experience", "education"):
            assert atom.start is not None, f"{atom.id} needs start/end"
        else:
            assert atom.year is not None, f"{atom.id} needs year"