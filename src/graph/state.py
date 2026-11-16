"""Typed shared state for the Phase 5 graphs.

Two graphs, two states:

    RunState      one per run. Sourcer and Ranker write it; `results` is
                  where every posting's outcome lands.
    PostingState  one per shortlisted posting. The posting graph's nodes
                  each read what earlier nodes wrote and add their own.

A node never returns the whole state. It returns a dict of the fields it
changed and LangGraph merges that in. For most fields "merge" means
replace. `RunState.results` is the exception: several postings write it,
so it carries a reducer (`operator.add`) that appends instead.

No `from __future__ import annotations` here on purpose: LangGraph reads
these annotations at graph-build time to find the reducers.
"""

import operator
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.agents.tailor import Change
from src.brief import JobBrief
from src.models import Resume
from src.tools.checks import JudgeReport
from src.tools.evidence import BriefHit
from src.tools.jobs import JobPost

# Tokens by node name, each in its SDK's own units. ranker and extract run
# on the open model (prompt/completion tokens); tailor and fit are Claude
# (input/output tokens; fit's are the cuts and expansions it asked for):
#   {"ranker": {"prompt_tokens": ..., "completion_tokens": ...},
#    "tailor": {"input_tokens": ..., "output_tokens": ...}}
Usage = dict[str, dict[str, int]]


class ShortlistEntry(BaseModel):
    """One ranked posting: a row of output/<keyword>/shortlist.json."""

    model_config = ConfigDict(extra="ignore")      # shortlist_json() also carries the stage-1 matches

    rank: int
    ref: str                          # the 8-character id used in folder names and labels
    id: str                           # JobPost.id, the key back into RunState.jobs
    title: str
    company: str
    url: str | None = None
    score: int | None = None          # stage-2 rating, None if not reranked
    reason: str | None = None
    years_asked: int | None = None
    demotion: str | None = None       # why the level gate moved it down
    similarity: float = 0.0


class PostingState(BaseModel):
    """Everything known about one posting as it moves through the graph."""

    rank: int
    post: JobPost
    # written by: extract
    brief: JobBrief | None = None
    # retrieve
    hits: list[BriefHit] = Field(default_factory=list)
    # tailor, then fit (which may cut or expand it)
    resume: Resume | None = None
    changes: list[Change] = Field(default_factory=list)
    uncovered: list[str] = Field(default_factory=list)
    # fit
    pdf_path: Path | None = None
    page_count: int | None = None
    fill: float | None = None
    scale: float | None = None
    fit_actions: list[str] = Field(default_factory=list)
    # judge
    judge_report: JudgeReport | None = None
    attempts: int = 0                 # Tailor revisions the Judge has asked for so far
    # any node
    usage: Usage = Field(default_factory=dict)
    error: str | None = None          # set by the node that failed; routes straight to finish
    failed_stage: str | None = None

    @property
    def ref(self) -> str:
        return self.post.slug[-8:]


class PostingResult(BaseModel):
    """What the run keeps of one posting: a row of summary.json."""

    rank: int
    ref: str
    title: str
    company: str
    url: str | None = None
    status: Literal["ok", "flagged", "failed"]   # flagged: delivered, but the Judge has open findings
    folder: str
    pdf_path: str | None = None
    page_count: int | None = None
    fill: float | None = None
    scale: float | None = None
    must_haves: int = 0
    uncovered: int = 0
    judge_retries: int = 0
    findings: list[str] = Field(default_factory=list)   # Judge fixes and defects still open
    gaps: list[str] = Field(default_factory=list)
    failed_stage: str | None = None
    error: str | None = None
    usage: Usage = Field(default_factory=dict)


class RunState(BaseModel):
    """One run: keyword in, a result per shortlisted posting out."""

    keyword: str
    location: str | None = None
    pages: int = 1                    # pages of postings to source
    top: int = 3                      # postings to tailor a CV for
    # sourcer
    jobs: list[JobPost] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    # ranker
    shortlist: list[ShortlistEntry] = Field(default_factory=list)
    usage: Usage = Field(default_factory=dict)
    # posting (one write per shortlisted posting, appended by the reducer)
    results: Annotated[list[PostingResult], operator.add] = Field(default_factory=list)


def add_usage(usage: Usage, node: str, spent: dict[str, Any] | None) -> Usage:
    """A copy of `usage` with `spent` added under `node`."""
    out = {name: dict(tokens) for name, tokens in usage.items()}
    bucket = out.setdefault(node, {})
    for key, n in (spent or {}).items():
        bucket[key] = bucket.get(key, 0) + int(n)
    return out
