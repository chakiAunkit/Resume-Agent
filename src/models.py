"""Schema for resume.json — the contract every agent edit must satisfy.

Design intent (roadmap invariants 1 and 2):
  * The Tailor writes structured data, never markup. This schema is the
    gate: a hallucinated field, malformed date, or empty bullet becomes a
    loud ValidationError instead of a silently broken PDF.
  * `extra="forbid"` everywhere — unknown keys are rejected, so an agent
    that invents `"certifications": [...]` fails instead of having its
    output silently dropped by the template.
  * Constraints are tripwires, not straitjackets: generous max lengths
    catch runaway generation; date patterns stop format drift
    ("Sept 2025" vs "09/2025"). Loosen them consciously if they ever
    block a legitimate edit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

# ---------------------------------------------------------------------------
# Reusable constrained string types
# ---------------------------------------------------------------------------

NonEmptyStr = Annotated[str, StringConstraints(min_length=1)]

# Bullets: non-empty, with a generous ceiling as a runaway-generation tripwire.
Bullet = Annotated[str, StringConstraints(min_length=1, max_length=700)]

# "09/2025" — zero-padded month, four-digit year.
MonthYear = Annotated[
    str, StringConstraints(pattern=r"^(0[1-9]|1[0-2])/\d{4}$")
]

# "05/2025" or the literal "Present".
MonthYearOrPresent = Annotated[
    str, StringConstraints(pattern=r"^(0[1-9]|1[0-2])/\d{4}$|^Present$")
]

Year = Annotated[str, StringConstraints(pattern=r"^\d{4}$")]

# Plain str with a pattern instead of pydantic's HttpUrl: HttpUrl dumps as
# a Url object in python mode, which complicates the Jinja context. A str
# stays a str all the way to `href="{{ link.url }}"`.
HttpUrlStr = Annotated[str, StringConstraints(pattern=r"^https?://\S+$")]


class StrictModel(BaseModel):
    """Base for all resume models: unknown keys are validation errors."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

class Link(StrictModel):
    # Literal, not str: the template maps `type` to an icon, so schema and
    # template must evolve together. A new link type fails here first —
    # loudly — instead of rendering with a missing icon.
    type: Literal["linkedin", "github", "medium"]
    label: NonEmptyStr
    url: HttpUrlStr


class Contact(StrictModel):
    name: NonEmptyStr
    phone: NonEmptyStr
    email: Annotated[str, StringConstraints(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")]
    links: list[Link] = Field(default_factory=list)


class Experience(StrictModel):
    company: NonEmptyStr
    title: NonEmptyStr
    location: NonEmptyStr
    start: MonthYear
    end: MonthYearOrPresent
    bullets: list[Bullet] = Field(default_factory=list)  # [] is valid (Keka)


class SkillGroup(StrictModel):
    category: NonEmptyStr
    items: list[NonEmptyStr] = Field(min_length=1)


class Education(StrictModel):
    degree: NonEmptyStr
    institution: NonEmptyStr
    location: NonEmptyStr
    start: MonthYear
    end: MonthYearOrPresent


class Project(StrictModel):
    name: NonEmptyStr
    year: Year
    links: list[Link] = Field(default_factory=list)
    description: NonEmptyStr


class Publication(StrictModel):
    title: NonEmptyStr
    venue: NonEmptyStr
    citations: int = Field(ge=0)
    url: HttpUrlStr


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

class Resume(StrictModel):
    contact: Contact
    summary: NonEmptyStr
    experience: list[Experience] = Field(min_length=1)
    skills: list[SkillGroup] = Field(min_length=1)
    education: list[Education] = Field(min_length=1)
    projects: list[Project] = Field(default_factory=list)
    publications: list[Publication] = Field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Resume":
        """Parse and validate a resume.json file."""
        return cls.model_validate_json(
            Path(path).read_text(encoding="utf-8")
        )

    def render_context(self) -> dict:
        """Plain-primitives dict for the Jinja template.

        mode="json" guarantees everything is str/int/list/dict — no
        model instances leak into the template layer.
        """
        return self.model_dump(mode="json")
