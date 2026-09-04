"""Evidence atoms — the ground truth the Tailor is allowed to assert.

Roadmap invariant #3 lives here: every claim on a tailored CV must trace
back to an atom in `data/profile.yaml`. This module is the *schema* for
that store. Retrieval (`load_evidence(query, k)`) lives in
`src/tools/evidence.py`; keeping the two apart mirrors the
`models.py` / `render.py` split — the Tailor and Judge can import the
schema without dragging in the retrieval machinery.

Design intent:
  * `profile.yaml` is a *superset* of the CV: everything true about the
    candidate, including things not currently on the resume. Mining old
    resumes into atoms is how the superset gets built.
  * One atom = one project, achievement, or metric. Its `id` is the
    handle the Tailor cites and the Judge verifies against.
  * Facts the Tailor may reuse verbatim (metrics, dates, links) are
    stored as literal strings, never paraphrased, so "what is allowed"
    is unambiguous when the Judge checks for fabrication.
  * `section` says which part of `resume.json` an atom may populate.
    Without it the Tailor could file a publication under Work Experience
    and the schema in `models.py` would happily accept it.
  * Same strictness as `models.py`: unknown keys, malformed dates, and
    duplicate ids fail loudly at load time, not at prompt time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, StringConstraints, model_validator

from src.models import (
    HttpUrlStr,
    MonthYear,
    MonthYearOrPresent,
    NonEmptyStr,
    StrictModel,
    Year,
)

# ---------------------------------------------------------------------------
# Constrained types
# ---------------------------------------------------------------------------

# kebab-case slug: stable, greppable, safe to echo through prompts and
# JSON. e.g. "target-goa-dmo-mismatch".
AtomId = Annotated[str, StringConstraints(pattern=r"^[a-z0-9]+(-[a-z0-9]+)*$")]

# Short retrieval tags. The ceiling stops a "tag" from becoming a sentence
# that would dominate keyword matching.
Tag = Annotated[str, StringConstraints(min_length=1, max_length=60)]

# Mirrors the Bullet ceiling in models.py: an atom summary should already
# be roughly bullet-sized so the Tailor edits it rather than compressing it.
Summary = Annotated[str, StringConstraints(min_length=1, max_length=700)]

Section = Literal["experience", "project", "publication", "education"]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class Links(StrictModel):
    github: HttpUrlStr | None = None
    article: HttpUrlStr | None = None
    paper: HttpUrlStr | None = None


class Atom(StrictModel):
    id: AtomId
    section: Section
    # The org / project / venue this belongs to. For `experience` atoms
    # this must match the company name used in resume.json so the Tailor
    # can attach the bullet to the right entry.
    source: NonEmptyStr
    title: NonEmptyStr
    summary: Summary
    # experience / education atoms carry a month range; project /
    # publication atoms carry a year — mirrors the two shapes in models.py.
    start: MonthYear | None = None
    end: MonthYearOrPresent | None = None
    year: Year | None = None
    skills: list[Tag] = Field(default_factory=list)     # tools & methods
    keywords: list[Tag] = Field(default_factory=list)   # domain / JD vocabulary
    metrics: list[NonEmptyStr] = Field(default_factory=list)  # verbatim-allowed figures
    links: Links = Field(default_factory=Links)

    @model_validator(mode="after")
    def _dates_consistent(self) -> "Atom":
        if (self.start is None) != (self.end is None):
            raise ValueError(f"atom {self.id!r}: start and end must be given together")
        if self.year is not None and self.start is not None:
            raise ValueError(f"atom {self.id!r}: use either year or start/end, not both")
        return self


class Profile(StrictModel):
    atoms: list[Atom] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> "Profile":
        seen: set[str] = set()
        dups = sorted({a.id for a in self.atoms if a.id in seen or seen.add(a.id)})
        if dups:
            raise ValueError(f"duplicate atom ids: {dups}")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        """Parse and validate a profile.yaml file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)

    def by_id(self, atom_id: str) -> Atom:
        for atom in self.atoms:
            if atom.id == atom_id:
                return atom
        raise KeyError(atom_id)