"""Deterministic checks on a fitted resume: the Judge's tool-read half.

Roadmap invariant 2 says hard constraints belong to tools, and the Phase 6
rubric marks two items "read from the tool, not judged by the model": the
page count and ATS safety. Those two live here, so the Phase 5 graph has a
real Judge node to route on before the LLM rubric exists. Phase 6 adds
`src/agents/judge.py` behind the same `JudgeReport`; these checks stay.

A report sorts what it finds into three kinds, because they have three
different consequences in the graph:

    fixes    the Tailor can repair by rewriting (the page budget). The graph
             sends these back to the Tailor, at most MAX_JUDGE_RETRIES times.
    defects  rewriting cannot repair (text the PDF does not expose to an
             applicant tracking system). The posting is delivered, flagged.
    gaps     true but unflattering: a must-have with no evidence, a
             must-have with evidence that the resume never mentions, a soft
             skill not conveyed. Reported, never a failure: closing a gap
             is a judgement about what belongs on one page, which is the LLM
             Judge's job in Phase 6, or needs a new atom in profile.yaml.

Nothing here calls a model, and nothing here decides whether a sentence is
true: verify() in the Tailor already checked grounding before the fit loop.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from pydantic import Field
from pypdf import PdfReader

from src.brief import JobBrief
from src.models import Resume, StrictModel
from src.tools.evidence import BriefHit, tokenize

# The PDF must expose at least this share of the resume's characters as
# text. A healthy render extracts ~100%; the floor catches a blank or
# image-only page without tripping on hyphenation or ligatures.
MIN_TEXT_SHARE = 0.6


class JudgeReport(StrictModel):
    passed: bool
    fixes: list[str] = Field(default_factory=list)
    defects: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


def pdf_text(path: Path | str) -> str:
    """All text pypdf can extract: what an applicant tracking system sees."""
    return "\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)


def _flat(text: str) -> str:
    """Lower-case with all whitespace removed, so a line wrap inside a job
    title or a publication title cannot hide it."""
    return re.sub(r"\s+", "", text.lower())


def _resume_prose(resume: Resume) -> str:
    """Everything a reader would call the resume's claims."""
    parts = [resume.summary]
    parts += [b for e in resume.experience for b in e.bullets]
    parts += [i for g in resume.skills for i in g.items]
    parts += [p.description for p in resume.projects]
    parts += [p.summary for p in resume.publications if p.summary]
    return " ".join(parts)


def _mentions(requirement: str, resume_tokens: set[str]) -> bool:
    """Every token of the requirement appears on the resume (the same strict
    rule retrieval uses for coverage)."""
    tokens = set(tokenize(requirement))
    return bool(tokens) and tokens <= resume_tokens


def check_resume(
    resume: Resume,
    pdf_path: Path | str,
    page_count: int,
    brief: JobBrief,
    hits: list[BriefHit],
    *,
    max_pages: int = 1,
    read_pdf: Callable[[Path | str], str] = pdf_text,
) -> JudgeReport:
    """The deterministic Judge. `read_pdf` is injectable so tests need no PDF."""
    fixes: list[str] = []
    defects: list[str] = []
    gaps: list[str] = []

    # 1. Page budget: the number comes from pypdf (render.py), never a guess.
    if page_count > max_pages:
        fixes.append(
            f"the rendered resume is {page_count} pages and the budget is {max_pages}: remove "
            "the bullet or project that serves this brief least")

    # 2. ATS safety: the structural facts must be in the PDF's text layer.
    text = read_pdf(pdf_path)
    flat = _flat(text)
    expected = [("name", resume.contact.name), ("email", resume.contact.email)]
    expected += [("company", e.company) for e in resume.experience]
    expected += [("job title", e.title) for e in resume.experience]
    expected += [("skill category", g.category) for g in resume.skills]
    expected += [("degree", ed.degree) for ed in resume.education]
    expected += [("project", p.name) for p in resume.projects]
    missing = [f"{kind} {value!r}" for kind, value in expected if _flat(value) not in flat]
    if missing:
        defects.append("not extractable from the PDF text (an ATS would not see them): " + ", ".join(missing))
    share = len(flat) / max(len(_flat(_resume_prose(resume))), 1)
    if share < MIN_TEXT_SHARE:
        defects.append(f"the PDF exposes only {share:.0%} of the resume's text; it may be blank or image-only")

    # 3. Coverage, reported as gaps.
    resume_tokens = set(tokenize(_resume_prose(resume)))
    evidence = {req: [h.atom.id for h in hits if req in h.covers] for req in brief.must_haves}
    for req in brief.must_haves:
        if not evidence[req]:
            gaps.append(f"must-have {req!r}: no evidence in the profile")
        elif not _mentions(req, resume_tokens):
            gaps.append(f"must-have {req!r}: supported by {', '.join(evidence[req])} but not mentioned on the resume")
    for skill in brief.soft_skills:
        if not any(len(t) >= 4 and t in resume_tokens for t in tokenize(skill)):
            gaps.append(f"soft skill {skill!r}: not conveyed in the summary or bullets")

    return JudgeReport(passed=not fixes and not defects, fixes=fixes, defects=defects, gaps=gaps)
