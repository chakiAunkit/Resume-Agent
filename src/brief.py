"""Job brief — the Tailor's view of a job description.

The Tailor never reads a raw JD. It reads a *brief*: the handful of
requirements that matter, already extracted. In Phase 3 the brief is
hand-written (`data/sample_jd.brief.yaml`); in Phase 4 the Extractor
agent writes it from raw JD text. Same shape either way, so the Tailor
is agnostic to who did the extracting — the same interface-first move
as `load_evidence()` in `src/tools/evidence.py`.

Why a brief and not the JD:
  * Retrieval. The keyword scorer's known limit (roadmap, Phase 2): a
    400-word JD flattens every atom's score. `must_haves` is the query.
  * Judging. The Judge (Phase 6) checks "must-have keywords covered"
    against *this list*, so Tailor and Judge agree on what the job
    asked for instead of each re-deriving it from the JD.
  * Eval. A hand-written brief is ground truth for the Phase 4
    Extractor: its output on the same JD can be diffed against this.

Sits beside `models.py` (resume schema) and `atoms.py` (evidence
schema) as the third data contract; like them, it is schema only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import Field, StringConstraints

from src.models import NonEmptyStr, StrictModel

# A requirement is tag-sized, like Atom.keywords: it's a retrieval term
# and a checklist item, not a sentence. The ceiling stops an extractor
# from pasting whole JD bullets into must_haves.
Requirement = Annotated[str, StringConstraints(min_length=1, max_length=80)]

# Literal, not free text: the Ranker (Phase 4) will compare this against
# the candidate's level, and "Mid-level (4+ yrs)" vs "mid" would break
# that silently. Years live in their own field.
Seniority = Literal["junior", "mid", "senior", "lead", "unspecified"]


class JobBrief(StrictModel):
    title: NonEmptyStr
    company: NonEmptyStr
    must_haves: list[Requirement] = Field(min_length=1)
    nice_to_haves: list[Requirement] = Field(default_factory=list)
    # Soft skills are routed, not filtered. They are *phrasing*
    # requirements ("does the CV convey this?"), not evidence requirements
    # ("which atom proves this?"), so retrieval ignores them, the Tailor
    # weaves them into summary/bullets where truthful, and the Judge
    # checks for a mention rather than for a covering atom.
    soft_skills: list[Requirement] = Field(default_factory=list)
    seniority: Seniority = "unspecified"
    years_experience: int | None = Field(default=None, ge=0)

    @classmethod
    def load(cls, path: str | Path) -> "JobBrief":
        """Parse and validate a *.brief.yaml file."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls.model_validate(raw)
