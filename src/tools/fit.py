"""The fit loop: make the tailored resume land on exactly `max_pages`.

Roadmap Phase 3, tool-in-the-loop pattern. The loop owns the hard limit;
a model is consulted only to decide *what* to cut or add, never *whether*
it fits. Each iteration:

    render -> page_count (pypdf, hard) + fill (layout, soft)
      -> done          if page_count <= max_pages and fill is high enough
      -> scale down    if over and the readability floor allows it
      -> cut           if over and already at SCALE_MIN: ask the cutter
                       (the Tailor's revise()) to drop the weakest bullet
      -> scale up      if under-filled and SCALE_MAX allows it
      -> accept        if under-filled and already at SCALE_MAX

The expand step (added after the first dry run, Oct 2026) is the cut
step's mirror. A fitted page with room left (fill under FILL_TARGET) asks
the expander — the Tailor's revise() with `expand_instruction()` — to add
grounded detail worth the measured room, stated in lines. The loop
re-renders and keeps the addition only if it still fits; an overflow is
reverted, not cut. Content comes before type size: with an expander the
scale-up waits until expansion is done, so leftover room becomes another
bullet or a paper's one-line summary rather than a bigger font.

    fit:     render, then scale down / cut until within budget
    fill:    while fill < FILL_TARGET (at most MAX_EXPANSIONS): expand,
             re-render, keep if it fits, else revert and stop
    finish:  scale up if still under FILL_MIN, as before

Step size comes from the measurement, not from a fixed decrement: page
height scales roughly with the square of the font scale (smaller type
means both shorter lines *and* more characters per line), so the scale
that would fit is about `scale / sqrt(fill / max_pages)`. One good step
usually replaces three guesses.

The cutter is injected as a callable so this tool does not import the
agent: `fit()` knows nothing about Claude; the CLI wires
`tailor.revise` in. Tests inject a fake renderer and a fake cutter.
"""

from __future__ import annotations

import json
import math
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from src.models import Resume
from src.tools.render import CONTENT_HEIGHT_PX, SCALE_MAX, SCALE_MIN, RenderResult

MAX_ITERATIONS = 3     # corrective steps after the first render (roadmap cap)
FILL_MIN = 0.85        # below this fraction of the last page, scale up
STEP_MARGIN = 0.985    # aim slightly under the estimate so one step lands
FILL_TARGET = 0.95     # below this, an expander is asked to add detail
MAX_EXPANSIONS = 2     # expand calls per fit: each is one Tailor call
# One line of body text, from templates/print.css: 10.5pt at --scale 1,
# line-height 1.25. Converts measured free height into lines for the prompt.
LINE_PX = 10.5 * 96 / 72 * 1.25


class RendererLike(Protocol):
    def render(self, data: Resume, scale: float = ..., out_path: Path | str | None = ...) -> RenderResult: ...


Cutter = Callable[[Resume], Resume]
# (resume, free fraction of the last page, free lines) -> the resume with
# detail added, or None when there is nothing worth adding.
Expander = Callable[[Resume, float, int], "Resume | None"]


class FitError(RuntimeError):
    """Still over the page budget after MAX_ITERATIONS corrective steps."""


@dataclass(frozen=True)
class FitResult:
    resume: Resume          # possibly cut
    scale: float
    page_count: int
    fill: float
    iterations: int         # corrective steps taken (0 = first render fit)
    actions: tuple[str, ...]  # what each step did — the loop's change log
    pdf_path: Path
    html_path: Path
    expanded: int = 0       # expansions kept; always the first ones made (the loop stops at an overflow)


def _clamp(scale: float) -> float:
    return max(SCALE_MIN, min(SCALE_MAX, round(scale, 3)))


def estimate_scale(scale: float, fill: float, max_pages: int) -> float:
    """Scale that should make `fill` pages of content fit `max_pages`."""
    if fill <= 0:
        return scale
    return _clamp(scale * math.sqrt(max_pages / fill) * STEP_MARGIN)


def free_lines(fill: float, max_pages: int, scale: float) -> int:
    """Lines of body text that would still fit on the last page."""
    free_px = max(max_pages - fill, 0.0) * CONTENT_HEIGHT_PX
    return int(free_px // (LINE_PX * scale))


def fit(
    resume: Resume,
    renderer: RendererLike,
    *,
    cutter: Cutter | None = None,
    expander: Expander | None = None,
    max_pages: int = 1,
    out_path: Path | str | None = None,
    scale: float = 1.0,
) -> FitResult:
    actions: list[str] = []
    result = renderer.render(resume, scale=scale, out_path=out_path)
    expanded = 0

    if expander is None:
        resume, result, scale = _settle(resume, result, scale, renderer, cutter, max_pages, out_path,
                                        actions, scale_up=True)
    else:
        # Fit first without growing the type, then fill with content, then
        # give whatever room is left to the type size, as before.
        resume, result, scale = _settle(resume, result, scale, renderer, cutter, max_pages, out_path,
                                        actions, scale_up=False)
        resume, result, expanded = _expand(resume, result, scale, renderer, expander, max_pages,
                                           out_path, actions)
        resume, result, scale = _settle(resume, result, scale, renderer, cutter, max_pages, out_path,
                                        actions, scale_up=True)

    return FitResult(
        resume=resume,
        scale=scale,
        page_count=result.page_count,
        fill=result.fill,
        iterations=len(actions),
        actions=tuple(actions),
        pdf_path=result.pdf_path,
        html_path=result.html_path,
        expanded=expanded,
    )


def _expand(resume, result, scale, renderer, expander, max_pages, out_path, actions):
    """Add detail while the page has room. Returns (resume, result, kept)."""
    kept = 0
    for _ in range(MAX_EXPANSIONS):
        if result.page_count > max_pages or result.fill / max_pages >= FILL_TARGET:
            break
        room, lines = max_pages - result.fill, free_lines(result.fill, max_pages, scale)
        if lines < 2:                       # not enough room for a sentence
            break
        candidate = expander(resume, room, lines)
        if candidate is None or candidate == resume:
            actions.append(f"room (fill {result.fill:.2f}, ~{lines} lines): nothing to add")
            break
        trial = renderer.render(candidate, scale=scale, out_path=out_path)
        if trial.page_count > max_pages:
            actions.append(f"room (fill {result.fill:.2f}, ~{lines} lines): expansion overflowed "
                           f"to {trial.page_count}p, reverted")
            result = renderer.render(resume, scale=scale, out_path=out_path)   # the PDF on disk is the kept one
            break
        actions.append(f"room (fill {result.fill:.2f}, ~{lines} lines): expanded -> fill {trial.fill:.2f}")
        resume, result, kept = candidate, trial, kept + 1
    return resume, result, kept


def _settle(resume, result, scale, renderer, cutter, max_pages, out_path, actions, *, scale_up):
    """Scale down or cut until within budget; with `scale_up`, also grow
    the type when under-filled. Returns (resume, result, scale)."""
    for step in range(1, MAX_ITERATIONS + 1):
        over = result.page_count > max_pages
        under = scale_up and not over and result.fill / max_pages < FILL_MIN

        if not over and not under:
            break

        if over:
            target = estimate_scale(scale, result.fill, max_pages)
            if target < scale - 1e-6:
                actions.append(f"over ({result.page_count}p, fill {result.fill:.2f}): scale {scale:.3f} -> {target:.3f}")
                scale = target
            elif cutter is None:
                raise FitError(
                    f"{result.page_count} pages at SCALE_MIN={SCALE_MIN} and no cutter given"
                )
            else:
                before = _bullet_count(resume)
                resume = cutter(resume)
                actions.append(
                    f"over at SCALE_MIN: cut ({before} -> {_bullet_count(resume)} bullets/projects)"
                )
        else:  # under-filled
            target = estimate_scale(scale, result.fill, max_pages)
            if target > scale + 1e-6:
                actions.append(f"under (fill {result.fill:.2f}): scale {scale:.3f} -> {target:.3f}")
                scale = target
            else:
                actions.append(f"under (fill {result.fill:.2f}) at SCALE_MAX: accepted")
                break

        result = renderer.render(resume, scale=scale, out_path=out_path)

    if result.page_count > max_pages:
        raise FitError(
            f"still {result.page_count} pages after {MAX_ITERATIONS} steps: " + "; ".join(actions)
        )
    return resume, result, scale


def _bullet_count(resume: Resume) -> int:
    return sum(len(e.bullets) for e in resume.experience) + len(resume.projects)


# Files a failed run leaves beside where the PDF would have gone. A later
# successful run removes them, so their presence always means "the last run
# for this brief failed".
FAILURE_FILES = ("error.json", "rejected_output.txt", "resume.unfitted.json", "changes.unfitted.json")


def record_failure(out_dir: Path, stage: str, exc: BaseException, usage: dict[str, int], *,
                   tailored: Resume | None = None, changes: list | None = None) -> Path:
    """Write what is known about a failed run to `out_dir`; return error.json.

    A failed run used to leave an empty folder (the Tailor failed before any
    render) or a stray over-length PDF (the fit loop gave up), and nothing
    saying which or why. This records:
      error.json             the stage ("tailor" or "fit"), the error, its
                             list of problems if it has one, tokens spent
      rejected_output.txt    the model's last rejected answer, when the error
                             carries one (TailorError.last_output)
      resume.unfitted.json   the Tailor's accepted resume, when the failure
      changes.unfitted.json  came after tailoring: the work is not lost
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    error = out_dir / "error.json"
    error.write_text(json.dumps({
        "stage": stage,
        "error": type(exc).__name__,
        "message": str(exc),
        "problems": list(getattr(exc, "problems", []) or []),
        "tokens": dict(usage),
        "failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # Where it was raised. Matters for the errors that are not results
        # (an API error, a bug): the Phase 5 graph catches those too, so
        # this file is the only place their traceback survives.
        "traceback": traceback.format_exception(exc),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    last_output = getattr(exc, "last_output", None)
    if last_output:
        (out_dir / "rejected_output.txt").write_text(last_output, encoding="utf-8")
    if tailored is not None:
        (out_dir / "resume.unfitted.json").write_text(
            tailored.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
        (out_dir / "changes.unfitted.json").write_text(
            json.dumps([c.model_dump() for c in changes or []], indent=2), encoding="utf-8")
    return error


if __name__ == "__main__":
    # Phase 3 acceptance — one command, JD brief -> tailored, fitted PDF:
    #   python -m src.tools.fit --brief data/sample_jd.brief.yaml
    import argparse
    import sys
    from typing import NoReturn

    from dotenv import load_dotenv

    from src.agents import tailor as tailor_agent
    from src.atoms import Profile
    from src.brief import JobBrief
    from src.tools.evidence import retrieve_for_brief
    from src.tools.render import ROOT, Renderer

    load_dotenv()

    ap = argparse.ArgumentParser(description="Tailor resume.json to a brief and fit it to the page budget.")
    ap.add_argument("--brief", required=True)
    ap.add_argument("--resume", default=ROOT / "assets" / "resume.json")
    ap.add_argument("--profile", default=ROOT / "data" / "profile.yaml")
    ap.add_argument("--out", default=None, help="default: output/<brief stem>/resume.pdf")
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--model", default=None,
                    help=f"default: TAILOR_MODEL, else {tailor_agent.DEFAULT_MODEL}")
    ap.add_argument("--effort", default=None,
                    help=f"default: TAILOR_EFFORT, else {tailor_agent.DEFAULT_EFFORT}")
    ap.add_argument("--no-expand", action="store_true",
                    help="don't ask the Tailor to fill leftover room (saves up to 2 Claude calls)")
    args = ap.parse_args()
    # Resolved after load_dotenv(), so a TAILOR_MODEL / TAILOR_EFFORT in .env counts.
    args.model = args.model or tailor_agent.default_model()
    args.effort = args.effort or tailor_agent.default_effort()

    brief = JobBrief.load(args.brief)
    base = Resume.load(args.resume)
    profile = Profile.load(args.profile)
    hits = retrieve_for_brief(brief, k=len(profile.atoms), profile=profile)
    out = Path(args.out) if args.out else ROOT / "output" / Path(args.brief).stem.replace(".brief", "") / "resume.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    usage = {"input_tokens": 0, "output_tokens": 0}

    def fail(stage: str, exc: Exception, tailored: Resume | None = None, changes: list | None = None) -> NoReturn:
        """Record the failure in the output folder, then stop. The two errors
        that are results — the Tailor's answer rejected twice, the page budget
        not met — exit with their message; anything else (an API error, a
        bug) keeps its traceback."""
        spent = dict(usage)
        lost = getattr(exc, "usage", None)          # tokens the failed call itself used
        if isinstance(lost, dict):
            for key, n in lost.items():
                spent[key] = spent.get(key, 0) + n
        path = record_failure(out.parent, stage, exc, spent, tailored=tailored, changes=changes)
        print(f"FAILED at the {stage} stage ({type(exc).__name__}); details in {path}")
        if isinstance(exc, (tailor_agent.TailorError, FitError)):
            print(exc)
            sys.exit(1)
        raise exc

    print(f"brief: {brief.title} @ {brief.company} — {len(hits)} atoms, target {args.pages} page(s)")
    try:
        t = tailor_agent.tailor(base, brief, hits, model=args.model, effort=args.effort, max_pages=args.pages)
    except Exception as e:
        fail("tailor", e)
    print(f"tailor: {t.attempts} attempt(s), tokens {t.usage['input_tokens']}/{t.usage['output_tokens']}, "
          f"{len(t.changes)} changes; uncovered: {', '.join(t.uncovered) or 'none'}")
    usage.update(t.usage)

    # The cutter and expander are the Tailor's revise() behind the fit loop's
    # two callables; `log` collects what they spend and change. `usage` is the
    # same dict, so fail() and the total below see the revisions' tokens too.
    log = tailor_agent.RevisionLog(usage=usage)
    settings = dict(model=args.model, effort=args.effort, max_pages=args.pages)
    cut = tailor_agent.cutter(brief, hits, log, **settings)
    expand = tailor_agent.expander(brief, tailor_agent.expansion_hits(hits, profile.atoms), log, **settings)

    try:
        with Renderer() as renderer:
            f = fit(t.resume, renderer, cutter=cut, expander=None if args.no_expand else expand,
                    max_pages=args.pages, out_path=out)
    except Exception as e:
        # The Tailor's answer was accepted; keep it beside the error.
        fail("fit", e, tailored=t.resume, changes=t.changes)

    for name in FAILURE_FILES:                                       # this run succeeded: clear an old failure
        out.with_name(name).unlink(missing_ok=True)
    cut_log, added = log.cuts, log.kept_expansions(f.expanded)       # kept expansions only
    for note in log.notes:
        print(f"  {note}")
    out.with_suffix(".json").write_text(f.resume.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    out.with_name("changes.json").write_text(
        json.dumps([c.model_dump() for c in t.changes + cut_log + added], indent=2), encoding="utf-8")

    print(f"fit: {f.iterations} step(s) -> {f.page_count} page(s), scale {f.scale:.3f}, fill {f.fill:.2f}")
    for a in f.actions:
        print(f"  - {a}")
    for c in cut_log:
        print(f"  cut   {c.where:<44} {c.why}")
    for c in added:
        print(f"  added {c.where:<44} {c.atom_id}")
    print(f"total tokens in/out: {usage['input_tokens']}/{usage['output_tokens']}")
    print(f"wrote {f.pdf_path}, {out.with_suffix('.json')}, {out.with_name('changes.json')}")
