"""One command: keyword -> tailored PDFs. Phase 5's acceptance.

    python run.py --keyword "data scientist" --location Bangalore --top 3
    python run.py --graph                    # print both graphs as Mermaid; no calls

What runs, in order (src/graph/build.py is the wiring, src/graph/nodes.py
the steps):

    sourcer   postings for the keyword, from data/jobs_cache when cached
    ranker    stage 1 local embeddings + level gate, stage 2 open-model rerank
    then, for each of the top N postings:
      extract   posting -> JobBrief                      (open model)
      retrieve  brief -> evidence atoms with coverage    (local)
      tailor    base CV + brief + evidence -> resume     (Claude)
      fit       render, measure, scale / cut / expand    (Chromium; Claude for cuts and expansions)
      judge     page budget, ATS text, coverage gaps     (local checks)
    present   summary table and output/<keyword>/summary.json

Cost of one run: one JSearch request per page not already cached; the
rerank and N extractions on the open model; per posting, one Claude call to
tailor plus up to two more if the page has room to fill (--no-expand turns
those off).

Exit code: 0 if every shortlisted posting got a resume, 1 if any failed
(each failure is recorded in its folder's error.json), 2 if the run could
not start (no postings, no API key).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from src.tools.jobs import STALE_AFTER_DAYS

    ap = argparse.ArgumentParser(description="Source, rank and tailor: keyword -> one tailored PDF per top posting.")
    ap.add_argument("--keyword", help='job search keyword, e.g. "data scientist"')
    ap.add_argument("--location", default=None)
    ap.add_argument("--top", type=int, default=3, help="postings to tailor a resume for (default 3)")
    ap.add_argument("--pages", type=int, default=3,
                    help="pages of postings to rank, 10 per page (default 3); one JSearch request per "
                         "page that is not in data/jobs_cache")
    ap.add_argument("--max-pages", type=int, default=1, help="page budget of each resume (default 1)")
    ap.add_argument("--refresh", action="store_true",
                    help="re-fetch the postings; overwrites the cached pool for this search")
    ap.add_argument("--stale-after", type=float, default=STALE_AFTER_DAYS, metavar="DAYS",
                    help=f"warn when cached postings are older than this (default {STALE_AFTER_DAYS})")
    ap.add_argument("--no-expand", action="store_true",
                    help="don't ask the Tailor to fill leftover room (saves up to 2 Claude calls per resume)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="postings worked on at once (default 1: the open model's free tier is rate-limited)")
    ap.add_argument("--out", default=None, help="output folder (default: output/<keyword>[-<location>])")
    ap.add_argument("--profile", default=ROOT / "data" / "profile.yaml")
    ap.add_argument("--resume", default=ROOT / "assets" / "resume.json")
    ap.add_argument("--graph", action="store_true", help="print the two graphs as Mermaid and exit")
    args = ap.parse_args(argv)
    if not args.graph and not args.keyword:
        ap.error("--keyword is required (or --graph to print the graphs)")
    if args.top < 1 or args.pages < 1 or args.parallel < 1:
        ap.error("--top, --pages and --parallel must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    # Posting titles and company names are arbitrary Unicode; a Windows
    # console or pipe in a legacy code page would raise on the first one.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)

    from src.graph.build import build_graph, build_posting_graph
    from src.graph.nodes import Deps

    if args.graph:
        deps = Deps(profile=None, resume=None, out_dir=ROOT / "output")     # building a graph runs no node
        print("%% run graph\n" + build_graph(deps).get_graph().draw_mermaid())
        print("%% posting graph (runs once inside each `posting` task)\n"
              + build_posting_graph(deps).get_graph().draw_mermaid())
        return 0

    from src.atoms import Profile
    from src.models import Resume
    from src.tools.jobs import ProviderError

    out_dir = Path(args.out) if args.out else ROOT / "output" / _slug(f"{args.keyword} {args.location or ''}")
    deps = Deps(profile=Profile.load(args.profile), resume=Resume.load(args.resume), out_dir=out_dir,
                max_pages=args.max_pages, expand=not args.no_expand, refresh=args.refresh,
                stale_after_days=args.stale_after)
    print(f"run: {args.keyword!r}{' in ' + args.location if args.location else ''}, top {args.top} of up to "
          f"{args.pages * 10} postings -> {out_dir}")
    try:
        final = build_graph(deps).invoke(
            {"keyword": args.keyword, "location": args.location, "pages": args.pages, "top": args.top},
            config={"max_concurrency": args.parallel})
    except (ProviderError, RuntimeError) as exc:
        # The run could not start: no postings, no key, the job board down.
        print(f"run stopped: {type(exc).__name__}: {exc}")
        return 2
    results = final.get("results", [])
    return 0 if results and all(r.status != "failed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
