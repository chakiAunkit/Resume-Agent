"""The fit loop: make the tailored resume land on exactly `max_pages`.

Roadmap Phase 3, tool-in-the-loop pattern. The loop owns the hard limit;
a model is consulted only to decide *what* to cut, never *whether* it
fits. Each iteration:

    render -> page_count (pypdf, hard) + fill (layout, soft)
      -> done          if page_count <= max_pages and fill is high enough
      -> scale down    if over and the readability floor allows it
      -> cut           if over and already at SCALE_MIN: ask the cutter
                       (the Tailor's revise()) to drop the weakest bullet
      -> scale up      if under-filled and SCALE_MAX allows it
      -> accept        if under-filled and already at SCALE_MAX

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

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from src.models import Resume
from src.tools.render import SCALE_MAX, SCALE_MIN, RenderResult

MAX_ITERATIONS = 3     # corrective steps after the first render (roadmap cap)
FILL_MIN = 0.85        # below this fraction of the last page, scale up
STEP_MARGIN = 0.985    # aim slightly under the estimate so one step lands


class RendererLike(Protocol):
    def render(self, data: Resume, scale: float = ..., out_path: Path | str | None = ...) -> RenderResult: ...


Cutter = Callable[[Resume], Resume]


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


def _clamp(scale: float) -> float:
    return max(SCALE_MIN, min(SCALE_MAX, round(scale, 3)))


def estimate_scale(scale: float, fill: float, max_pages: int) -> float:
    """Scale that should make `fill` pages of content fit `max_pages`."""
    if fill <= 0:
        return scale
    return _clamp(scale * math.sqrt(max_pages / fill) * STEP_MARGIN)


def fit(
    resume: Resume,
    renderer: RendererLike,
    *,
    cutter: Cutter | None = None,
    max_pages: int = 1,
    out_path: Path | str | None = None,
    scale: float = 1.0,
) -> FitResult:
    actions: list[str] = []
    result = renderer.render(resume, scale=scale, out_path=out_path)

    for step in range(1, MAX_ITERATIONS + 1):
        over = result.page_count > max_pages
        under = not over and result.fill / max_pages < FILL_MIN

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
    else:
        step = MAX_ITERATIONS

    if result.page_count > max_pages:
        raise FitError(
            f"still {result.page_count} pages after {MAX_ITERATIONS} steps: " + "; ".join(actions)
        )

    return FitResult(
        resume=resume,
        scale=scale,
        page_count=result.page_count,
        fill=result.fill,
        iterations=len(actions),
        actions=tuple(actions),
        pdf_path=result.pdf_path,
        html_path=result.html_path,
    )


def _bullet_count(resume: Resume) -> int:
    return sum(len(e.bullets) for e in resume.experience) + len(resume.projects)


if __name__ == "__main__":
    # Phase 3 acceptance — one command, JD brief -> tailored, fitted PDF:
    #   python -m src.tools.fit --brief data/sample_jd.brief.yaml
    import argparse
    import json

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
    ap.add_argument("--model", default=tailor_agent.DEFAULT_MODEL)
    ap.add_argument("--effort", default=tailor_agent.DEFAULT_EFFORT)
    args = ap.parse_args()

    brief = JobBrief.load(args.brief)
    base = Resume.load(args.resume)
    profile = Profile.load(args.profile)
    hits = retrieve_for_brief(brief, k=len(profile.atoms), profile=profile)
    out = Path(args.out) if args.out else ROOT / "output" / Path(args.brief).stem.replace(".brief", "") / "resume.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"brief: {brief.title} @ {brief.company} — {len(hits)} atoms, target {args.pages} page(s)")
    t = tailor_agent.tailor(base, brief, hits, model=args.model, effort=args.effort, max_pages=args.pages)
    print(f"tailor: {t.attempts} attempt(s), tokens {t.usage['input_tokens']}/{t.usage['output_tokens']}, "
          f"{len(t.changes)} changes; uncovered: {', '.join(t.uncovered) or 'none'}")
    usage = dict(t.usage)
    cut_log: list = []

    def cut(current: Resume) -> Resume:
        r = tailor_agent.revise(current, brief, hits, tailor_agent.CUT_INSTRUCTION,
                                model=args.model, effort=args.effort, max_pages=args.pages)
        usage["input_tokens"] += r.usage["input_tokens"]
        usage["output_tokens"] += r.usage["output_tokens"]
        cut_log.extend(r.changes)
        return r.resume

    with Renderer() as renderer:
        f = fit(t.resume, renderer, cutter=cut, max_pages=args.pages, out_path=out)

    out.with_suffix(".json").write_text(f.resume.model_dump_json(indent=2), encoding="utf-8")
    out.with_name("changes.json").write_text(
        json.dumps([c.model_dump() for c in t.changes + cut_log], indent=2), encoding="utf-8")

    print(f"fit: {f.iterations} step(s) -> {f.page_count} page(s), scale {f.scale:.3f}, fill {f.fill:.2f}")
    for a in f.actions:
        print(f"  - {a}")
    for c in cut_log:
        print(f"  cut {c.where:<44} {c.why}")
    print(f"total tokens in/out: {usage['input_tokens']}/{usage['output_tokens']}")
    print(f"wrote {f.pdf_path}, {out.with_suffix('.json')}, {out.with_name('changes.json')}")
