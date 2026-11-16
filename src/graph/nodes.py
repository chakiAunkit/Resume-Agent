"""The graph's nodes: thin wrappers around the functions Phases 1-4 built.

A node is a function `(state, deps) -> dict of the fields it changed`. It
holds no logic of its own beyond calling one existing function, saving that
function's artifact, and saying what happened. The logic stays where its
tests are: `fetch_jobs`, `rank`, `extract`, `retrieve_for_brief`,
`tailor` / `revise`, `fit`, `check_resume`.

Two rules every posting node follows:

  * It never raises. LangGraph stops the whole run when a node raises, so
    one bad posting would take the other two with it. A node that fails
    records the failure (`record_failure`, the same error.json the fit CLI
    writes), sets `error` on the state, and the graph routes that posting
    straight to `finish`.
  * It takes its collaborators from `Deps`, never from a global. Tests hand
    in fakes for the job board, the embedder, all three LLM clients and the
    renderer, and the whole graph runs with no network and no browser.

The Sourcer and Ranker nodes do raise: with no postings or no ranking there
is nothing to continue with.
"""

from __future__ import annotations

import functools
import json
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager

from langgraph.types import Send

from src.agents.extractor import brief_to_yaml, extract as extract_brief
from src.agents.ranker import (DiskCachedEmbedder, Embedder, Stage1Options, default_options,
                               make_embedder, rank, shortlist_json)
from src.agents.tailor import RevisionLog, cutter, expander, expansion_hits, revise, tailor as tailor_resume
from src.atoms import Profile
from src.graph.state import PostingResult, PostingState, RunState, ShortlistEntry, Usage, add_usage
from src.models import Resume
from src.tools.checks import JudgeReport, check_resume
from src.tools.evidence import retrieve_for_brief
from src.tools.fit import FAILURE_FILES, RendererLike, fit as fit_resume, record_failure
from src.tools.jobs import DEFAULT_CACHE_DIR, STALE_AFTER_DAYS, Provider, fetch_jobs, jd_text
from src.tools.render import Renderer

# Roadmap: the Judge may send a resume back to the Tailor at most twice. An
# uncapped judge <-> tailor cycle is how a run burns credits.
MAX_JUDGE_RETRIES = 2


@dataclass
class Deps:
    """Everything a node needs that is not state. None means "the real one"."""

    profile: Profile
    resume: Resume                              # the base CV every posting starts from
    out_dir: Path                               # output/<keyword-slug>/
    max_pages: int = 1
    expand: bool = True                         # let the fit loop fill leftover room (extra Claude calls)
    refresh: bool = False                       # re-fetch postings instead of reading the cache
    stale_after_days: float | None = STALE_AFTER_DAYS
    cache_dir: Path = DEFAULT_CACHE_DIR
    today: date | None = None                   # for the level gate; None = today
    provider: Provider | None = None            # job board; None = JSearch
    embedder: Embedder | None = None            # None = Qwen3-0.6B with the on-disk vector cache
    options: Stage1Options | None = None        # None = ranker.default_options()
    ranker_client: Any = None                   # None = the OpenAI SDK pointed at the ranker's route
    extractor_client: Any = None
    tailor_client: Any = None                   # None = the Anthropic SDK
    # A new browser per posting, opened inside the node: Playwright's sync
    # API is bound to the thread that started it, and LangGraph runs nodes
    # on worker threads.
    renderer_factory: Callable[[], ContextManager[RendererLike]] = Renderer
    judge: Callable[..., JudgeReport] = check_resume
    log: Callable[[str], None] = print


def posting_dir(deps: Deps, state: PostingState) -> Path:
    return deps.out_dir / f"{state.rank}-{state.ref}"


def _say(deps: Deps, state: PostingState, text: str) -> None:
    deps.log(f"  [#{state.rank} {state.ref}] {text}")


def _flat_tokens(usage: Usage) -> dict[str, int]:
    return {f"{node}.{key}": n for node, tokens in usage.items() for key, n in tokens.items()}


def _failed(state: PostingState, deps: Deps, stage: str, exc: Exception) -> dict:
    """Record a node's failure and return the state update that ends this posting."""
    spent = getattr(exc, "usage", None)             # TailorError says what the failed calls cost
    usage = add_usage(state.usage, stage, spent) if isinstance(spent, dict) else state.usage
    path = record_failure(posting_dir(deps, state), stage, exc, _flat_tokens(usage),
                          tailored=state.resume, changes=state.changes)
    reason = (getattr(exc, "problems", None) or [str(exc)])[0]
    _say(deps, state, f"{stage} FAILED ({type(exc).__name__}): {str(reason)[:160]} -> {path.name}")
    return {"error": f"{type(exc).__name__}: {str(reason)[:300]}", "failed_stage": stage, "usage": usage}


def _guarded(stage: str):
    """Turn any exception in a posting node into a recorded failure."""
    def wrap(node: Callable[[PostingState, Deps], dict]) -> Callable[[PostingState, Deps], dict]:
        @functools.wraps(node)
        def run(state: PostingState, deps: Deps) -> dict:
            try:
                return node(state, deps)
            except Exception as exc:        # noqa: BLE001 — isolating the posting is the point
                return _failed(state, deps, stage, exc)
        return run
    return wrap


# ---------------------------------------------------------------------------
# Run graph
# ---------------------------------------------------------------------------

def sourcer(state: RunState, deps: Deps) -> dict:
    """keyword -> postings (cached). Warnings (a stale cache, a failed later
    page) travel in the state so `present` can repeat them at the end."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        posts = fetch_jobs(state.keyword, state.location, provider=deps.provider, pages=state.pages,
                           cache_dir=deps.cache_dir, refresh=deps.refresh,
                           stale_after_days=deps.stale_after_days)
    notes = [str(w.message) for w in caught]
    for note in notes:
        deps.log(f"warning: {note}")
    if not posts:
        raise RuntimeError(f"no postings for {state.keyword!r}"
                           f"{' in ' + state.location if state.location else ''}")
    deps.log(f"sourcer: {len(posts)} posting(s) for {state.keyword!r}"
             f"{' in ' + state.location if state.location else ''}")
    return {"jobs": posts, "warnings": state.warnings + notes}


def ranker(state: RunState, deps: Deps) -> dict:
    """postings -> ranked shortlist. If the stage-2 rerank fails (a rate
    limit, a rejected answer), the shortlist falls back to stage 1's order
    rather than ending the run: stage 1 is local and already paid for."""
    embedder = deps.embedder or DiskCachedEmbedder(make_embedder())
    options = deps.options or default_options()
    notes: list[str] = []
    common = dict(embedder=embedder, options=options, today=deps.today)
    deps.log(f"ranker: stage 1 ({embedder.name}; {options.label}), then rerank")
    try:
        result = rank(state.jobs, deps.profile, deps.resume, client=deps.ranker_client, **common)
    except Exception as exc:                # noqa: BLE001
        notes.append(f"stage-2 rerank failed ({type(exc).__name__}: {str(exc)[:160]}); "
                     "the shortlist is in stage 1's order")
        deps.log(f"warning: {notes[-1]}")
        result = rank(state.jobs, deps.profile, deps.resume, rerank_top=0, **common)

    rows = shortlist_json(result)
    deps.out_dir.mkdir(parents=True, exist_ok=True)
    (deps.out_dir / "shortlist.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    shortlist = [ShortlistEntry(**row) for row in rows]
    for entry in shortlist[:state.top]:
        rating = "-" if entry.score is None else f"{entry.score}/10"
        deps.log(f"  {entry.rank}. {entry.ref}  {rating:>5}  {entry.title[:50]} — {entry.company[:28]}"
                 f"{'  [' + entry.demotion + ']' if entry.demotion else ''}")
    return {"shortlist": shortlist, "warnings": state.warnings + notes,
            "usage": add_usage(state.usage, "ranker", result.usage)}


def fan_out(state: RunState) -> list[Send] | str:
    """The ranker's outgoing edge: one `posting` task per shortlisted posting.
    Each Send carries that posting's own starting state."""
    by_id = {p.id: p for p in state.jobs}
    sends = [Send("posting", {"rank": entry.rank, "post": by_id[entry.id]})
             for entry in state.shortlist[:state.top]]
    return sends or "present"


def present(state: RunState, deps: Deps) -> dict:
    """Write summary.json and print the outcome table."""
    results = sorted(state.results, key=lambda r: r.rank)
    deps.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "keyword": state.keyword,
        "location": state.location,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "postings_sourced": len(state.jobs),
        "warnings": state.warnings,
        "tokens": token_totals(state.usage, results),
        "postings": [r.model_dump(mode="json") for r in results],
    }
    path = deps.out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    deps.log(summary_text(results, summary["tokens"], state.warnings))
    deps.log(f"wrote {path}")
    return {}


def token_totals(run_usage: Usage, results: list[PostingResult]) -> dict[str, dict[str, int]]:
    """Tokens per node name, summed over the run."""
    total: Usage = {}
    for usage in [run_usage, *(r.usage for r in results)]:
        for node, tokens in usage.items():
            total = add_usage(total, node, tokens)
    return total


def summary_text(results: list[PostingResult], tokens: Usage, notes: list[str]) -> str:
    lines = ["", f"{'#':>2}  {'ref':<8}  {'status':<8} {'pages':>5} {'fill':>5}  {'gaps':>4}  title — company"]
    for r in results:
        pages = "-" if r.page_count is None else str(r.page_count)
        fill = "-" if r.fill is None else f"{r.fill:.2f}"
        lines.append(f"{r.rank:>2}  {r.ref:<8}  {r.status:<8} {pages:>5} {fill:>5}  {len(r.gaps):>4}  "
                     f"{r.title[:44]} — {r.company[:24]}")
        if r.status == "failed":
            lines.append(f"      failed at {r.failed_stage}: {r.error}")
            lines.append(f"      details: {Path(r.folder) / 'error.json'}")
        else:
            lines.append(f"      {r.pdf_path}")
            lines += [f"      open finding: {f}" for f in r.findings]
    ok = sum(r.status != "failed" for r in results)
    lines.append(f"{ok} of {len(results)} resume(s) delivered")
    for node, spent in tokens.items():
        lines.append(f"  tokens  {node:<10} " + "  ".join(f"{k} {v}" for k, v in spent.items()))
    lines += [f"  warning: {n}" for n in notes]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Posting graph
# ---------------------------------------------------------------------------

@_guarded("extract")
def extract(state: PostingState, deps: Deps) -> dict:
    """posting -> JobBrief. The JD text carries a title/company header:
    the Extractor's verify() looks for both, and descriptions often name
    neither (see jobs.jd_text)."""
    result = extract_brief(jd_text(state.post), client=deps.extractor_client)
    folder = posting_dir(deps, state)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "brief.yaml").write_text(
        brief_to_yaml(result.brief, header=f"# extracted from {state.post.id} by {result.model}"),
        encoding="utf-8")
    brief = result.brief
    _say(deps, state, f"extract: {len(brief.must_haves)} must-haves, {len(brief.nice_to_haves)} nice, "
                      f"{len(brief.soft_skills)} soft ({result.attempts} attempt(s))")
    return {"brief": brief, "usage": add_usage(state.usage, "extract", result.usage)}


@_guarded("retrieve")
def retrieve(state: PostingState, deps: Deps) -> dict:
    """brief -> evidence atoms with coverage. Every non-zero atom is offered:
    at this profile size the retriever annotates, it does not filter."""
    hits = retrieve_for_brief(state.brief, k=len(deps.profile.atoms), profile=deps.profile)
    covered = {req for h in hits for req in h.covers}
    _say(deps, state, f"retrieve: {len(hits)} atom(s), {len(covered)} of {len(state.brief.must_haves)} "
                      "must-haves covered")
    return {"hits": hits}


def judge_instruction(fixes: list[str]) -> str:
    return ("A check of the rendered resume found the problems below. Fix every one and return "
            "the complete resume. Change nothing else. The change log has one entry per fix.\n"
            + "\n".join(f"- {f}" for f in fixes))


@_guarded("tailor")
def tailor(state: PostingState, deps: Deps) -> dict:
    """First visit: tailor the base CV to the brief. A later visit means the
    Judge sent the resume back: revise the fitted resume with its fixes."""
    kw = dict(client=deps.tailor_client, max_pages=deps.max_pages)
    retry = state.judge_report is not None and bool(state.judge_report.fixes) and state.resume is not None
    if retry:
        r = revise(state.resume, state.brief, state.hits, judge_instruction(state.judge_report.fixes), **kw)
        changes = state.changes + r.changes
        _say(deps, state, f"tailor: revision {state.attempts + 1} for the Judge, {len(r.changes)} change(s)")
    else:
        r = tailor_resume(deps.resume, state.brief, state.hits, **kw)
        changes = r.changes
        _say(deps, state, f"tailor: {len(r.changes)} change(s) in {r.attempts} attempt(s), "
                          f"{len(r.uncovered)} must-have(s) uncovered")
    return {"resume": r.resume, "changes": changes, "uncovered": list(r.uncovered),
            "attempts": state.attempts + (1 if retry else 0),
            "usage": add_usage(state.usage, "tailor", r.usage)}


def fit(state: PostingState, deps: Deps) -> dict:
    """Render, measure, scale or cut until it fits; fill leftover room on the
    first pass only (a Judge retry is a repair, not another chance to add).
    Guarded by hand, not by @_guarded: the revisions' tokens live in `log`
    and must reach the failure record."""
    log = RevisionLog()
    kw = dict(client=deps.tailor_client, max_pages=deps.max_pages)
    grow = deps.expand and state.attempts == 0
    folder = posting_dir(deps, state)
    try:
        with deps.renderer_factory() as renderer:
            f = fit_resume(
                state.resume, renderer,
                cutter=cutter(state.brief, state.hits, log, **kw),
                expander=(expander(state.brief, expansion_hits(state.hits, deps.profile.atoms), log, **kw)
                          if grow else None),
                max_pages=deps.max_pages, out_path=folder / "resume.pdf")
        changes = state.changes + log.cuts + log.kept_expansions(f.expanded)
        (folder / "resume.json").write_text(f.resume.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
        (folder / "changes.json").write_text(json.dumps([c.model_dump() for c in changes], indent=2),
                                             encoding="utf-8")
    except Exception as exc:                # noqa: BLE001
        return _failed(state.model_copy(update={"usage": add_usage(state.usage, "fit", log.usage)}),
                       deps, "fit", exc)
    _say(deps, state, f"fit: {f.page_count} page(s), scale {f.scale:.3f}, fill {f.fill:.2f}, "
                      f"{f.iterations} step(s), {f.expanded} expansion(s) kept")
    return {"resume": f.resume, "changes": changes, "pdf_path": f.pdf_path, "page_count": f.page_count,
            "fill": f.fill, "scale": f.scale, "fit_actions": list(f.actions) + log.notes,
            "usage": add_usage(state.usage, "fit", log.usage)}


@_guarded("judge")
def judge(state: PostingState, deps: Deps) -> dict:
    """Check the fitted resume. Deterministic in Phase 5 (tools/checks.py);
    Phase 6 adds the LLM rubric behind the same JudgeReport."""
    report = deps.judge(state.resume, state.pdf_path, state.page_count, state.brief, state.hits,
                        max_pages=deps.max_pages)
    (posting_dir(deps, state) / "judge.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    verdict = "pass" if report.passed else f"{len(report.fixes)} fix(es), {len(report.defects)} defect(s)"
    _say(deps, state, f"judge: {verdict}; {len(report.gaps)} gap(s) reported")
    return {"judge_report": report}


def route_after_judge(state: PostingState) -> str:
    """The loop's one decision. Back to the Tailor only for something the
    Tailor can fix, and only while the retry budget lasts."""
    if state.error or state.judge_report is None:
        return "finish"
    if state.judge_report.fixes and state.attempts < MAX_JUDGE_RETRIES:
        return "tailor"
    return "finish"


def finish(state: PostingState, deps: Deps) -> dict:
    """Where every posting ends, delivered or failed."""
    if not state.error:
        for name in FAILURE_FILES:          # a delivered posting clears an earlier run's failure record
            (posting_dir(deps, state) / name).unlink(missing_ok=True)
    return {}


def to_result(state: PostingState, deps: Deps) -> PostingResult:
    """The row this posting contributes to the run's results."""
    report = state.judge_report
    if state.error:
        status = "failed"
    elif report is not None and not report.passed:
        status = "flagged"
    else:
        status = "ok"
    return PostingResult(
        rank=state.rank, ref=state.ref, title=state.post.title, company=state.post.company,
        url=state.post.url, status=status, folder=str(posting_dir(deps, state)),
        pdf_path=None if state.error or state.pdf_path is None else str(state.pdf_path),
        page_count=state.page_count, fill=state.fill, scale=state.scale,
        must_haves=len(state.brief.must_haves) if state.brief else 0,
        uncovered=len(state.uncovered), judge_retries=state.attempts,
        findings=(report.fixes + report.defects) if report else [],
        gaps=report.gaps if report else [],
        failed_stage=state.failed_stage, error=state.error, usage=state.usage)


def crashed_result(rank: int, post: Any, deps: Deps, exc: Exception) -> PostingResult:
    """A posting whose graph itself raised: a bug, not a node failure. It
    still becomes a row, so the run reports it instead of dying."""
    ref = post.slug[-8:]
    return PostingResult(rank=rank, ref=ref, title=post.title, company=post.company, url=post.url,
                         status="failed", folder=str(deps.out_dir / f"{rank}-{ref}"),
                         failed_stage="graph", error=f"{type(exc).__name__}: {str(exc)[:300]}")
