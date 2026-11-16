"""Ranker eval: precision@k and stage-1 recall against your own labels.

The Phase 4 acceptance test is "a shortlist whose top 5 I would genuinely
consider". Labelling the postings once turns that into numbers:

  * precision@k (strict)  — share of the top k labelled `yes`
  * precision@k (lenient) — share labelled `yes` or `maybe`
  * stage-1 recall@15     — share of the `yes` postings stage 1 keeps in its
                            top 15. Stage 2 only ever sees those fifteen, so a
                            wanted posting cut here is lost however good the
                            rerank is. Stage 1 is a filter; this is its metric.
  * best possible / random — what a perfect ranking and a random pick score,
                            so the numbers above have something to stand against

Labels: yes | maybe | no | skip. `skip` means "couldn't judge it" (a posting
that wouldn't open, a JD you couldn't read). Skipped postings are still
ranked — the Ranker sees the whole pool — but dropped from the ranking before
scoring, so they neither help nor hurt.

Stage 1 is deterministic and runs once. Stage 2 is an LLM and is not, so it
runs `--runs` times (default 3) and reports mean (min-max) plus how many runs
put each posting in the top k — the same discipline as the Extractor eval.

`--compare` answers "which stage-1 setup should we use": every listed
embedder x every stage-1 variant, stage 1 only, one table. No API calls.
The rows run from the 2026-09-29 table (so its numbers reproduce) plus
two with the level gate, each adding one change to a row above it.

A normal run uses the chosen setup (ranker.default_options(): Qwen3-0.6B,
titles, RANKER_EXCLUDE_ATOMS left out, level gate on) and lists what the
gate demoted, next to your label for each.

Postings come from `data/jobs_cache/` only. This tool never calls JSearch:
a page missing from the cache is an error, not a spent credit.

    python -m src.tools.rank_eval <labels>                         # stage 1 + 3 stage-2 runs
    python -m src.tools.rank_eval <labels> --stage1-only
    python -m src.tools.rank_eval <labels> --compare minilm,bge-base,qwen3-0.6b
    python -m src.tools.rank_eval <labels> --no-gate               # the gate's ablation
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.tools.jobs import JobPost, JobQuery, JSearchProvider, ProviderError

ROOT = Path(__file__).resolve().parents[2]
LABELS = ("yes", "maybe", "no", "skip")
CUT = 15   # stage 2 sees stage 1's top 15 (ranker.RERANK_TOP)


def load_labels(path: Path | str) -> tuple[dict[str, Any], dict[str, str]]:
    """-> (query block, {ref: label}). Every label must be yes / maybe / no /
    skip: a half-labelled file would score a ranking against a guess."""
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    labels: dict[str, str] = {}
    missing = []
    for entry in doc["postings"]:
        raw = entry.get("label")
        # YAML 1.1 reads a bare yes / no as True / False; map them back.
        if isinstance(raw, bool):
            raw = "yes" if raw else "no"
        label = str(raw or "").strip().lower()
        if label not in LABELS:
            missing.append(f"{entry['ref']} ({entry.get('title', '')[:40]})")
            continue
        labels[str(entry["ref"])] = label
    if missing:
        raise ValueError(f"{len(missing)} posting(s) have no yes/maybe/no/skip label: " + "; ".join(missing))
    return doc.get("query", {}), labels


def judged(refs: list[str], labels: dict[str, str]) -> list[str]:
    """The ranking with skipped (and unlabelled) postings taken out."""
    return [r for r in refs if labels.get(r) in ("yes", "maybe", "no")]


def precision_at_k(refs: list[str], labels: dict[str, str], k: int = 5,
                   accept: tuple[str, ...] = ("yes",)) -> float:
    top = judged(refs, labels)[:k]
    return sum(labels[r] in accept for r in top) / len(top) if top else 0.0


def recall_at_k(refs: list[str], labels: dict[str, str], k: int = CUT,
                accept: tuple[str, ...] = ("yes",)) -> float:
    """Share of the accepted postings that sit in the top k of the raw
    ranking. Raw, not condensed: a skipped posting still takes one of the
    fifteen seats stage 2 sees."""
    wanted = [r for r, lab in labels.items() if lab in accept]
    return sum(r in refs[:k] for r in wanted) / len(wanted) if wanted else 0.0


def best_possible(labels: dict[str, str], k: int = 5, accept: tuple[str, ...] = ("yes",)) -> float:
    n = sum(v in ("yes", "maybe", "no") for v in labels.values())
    return min(sum(v in accept for v in labels.values()), k) / min(k, n) if n else 0.0


def random_expected(labels: dict[str, str], accept: tuple[str, ...] = ("yes",)) -> float:
    """Expected precision of a random ranking: the base rate."""
    pool = [v for v in labels.values() if v in ("yes", "maybe", "no")]
    return sum(v in accept for v in pool) / len(pool) if pool else 0.0


class CacheOnlyProvider:
    """A JSearch provider that can read the cache but never the network."""

    name = JSearchProvider.name

    def __init__(self) -> None:
        self._parser = JSearchProvider(api_key="cache-only")

    def fetch_raw(self, query: JobQuery) -> dict[str, Any]:
        raise ProviderError(f"page not in data/jobs_cache (cursor={query.cursor!r}); "
                            "fetch it with python -m src.tools.jobs first")

    def parse(self, raw: dict[str, Any]) -> list[JobPost]:
        return self._parser.parse(raw)

    def next_cursor(self, raw: dict[str, Any]) -> str | None:
        return JSearchProvider.next_cursor(raw)


def cached_posts(query: dict[str, Any], pages: int) -> list[JobPost]:
    import warnings

    from src.tools.jobs import fetch_jobs
    with warnings.catch_warnings():
        warnings.simplefilter("error")          # a missing later page is an error here too
        return fetch_jobs(query["keyword"], query.get("location"), pages=pages,
                          provider=CacheOnlyProvider())


def _fmt(values: list[float]) -> str:
    lo, hi, mean = min(values), max(values), sum(values) / len(values)
    return f"{mean:.2f}" if lo == hi else f"{mean:.2f} ({lo:.2f}-{hi:.2f})"


def report_stage(name: str, rankings: list[list[str]], labels: dict[str, str],
                 titles: dict[str, str], k: int, stage1: bool = False) -> list[str]:
    strict = [precision_at_k(r, labels, k) for r in rankings]
    lenient = [precision_at_k(r, labels, k, ("yes", "maybe")) for r in rankings]
    runs = len(rankings)
    lines = [f"## {name} — {runs} run(s)",
             f"   P@{k} strict {_fmt(strict)}   lenient {_fmt(lenient)}"]
    if stage1:
        kept = [r for r, lab in labels.items() if lab == "yes" and r in rankings[0][:CUT]]
        n_yes = sum(lab == "yes" for lab in labels.values())
        lines.append(f"   recall@{CUT} (yes kept for stage 2): {len(kept)}/{n_yes} = "
                     f"{recall_at_k(rankings[0], labels):.2f}")
    in_top = Counter(ref for r in rankings for ref in judged(r, labels)[:k])
    for ref, n in in_top.most_common():
        lines.append(f"     {n}/{runs}  {labels[ref]:<5}  {ref}  {titles[ref][:60]}")
    missed = [ref for ref, lab in labels.items() if lab == "yes" and ref not in in_top]
    if missed:
        lines.append(f"   yes-labelled, never in top {k}: " +
                     ", ".join(f"{ref} {titles[ref][:30]}" for ref in missed))
    return lines


def gate_report(demoted: list[tuple[str, str]], labels: dict[str, str], titles: dict[str, str]) -> list[str]:
    """demoted: (ref, demotion) in rank order -> what the level gate moved
    down, by label. The gate should demote no's and never a yes."""
    counts = Counter(labels[ref] for ref, _ in demoted)
    lines = [f"## level gate — demoted {len(demoted)}: " +
             ", ".join(f"{counts[lab]} {lab}" for lab in LABELS if counts[lab])
             if demoted else "## level gate — demoted nothing"]
    lines += [f"     {labels[ref]:<5}  {ref}  {why:<26}  {titles[ref][:50]}" for ref, why in demoted]
    return lines


def compare_table(rows: list[tuple[str, str, list[str]]], labels: dict[str, str], k: int) -> list[str]:
    """rows: (embedder, variant, stage-1 ranking) -> the comparison table."""
    lines = [f"| embedder | stage-1 variant | recall@{CUT} (yes) | P@{k} strict | P@{k} lenient |",
             "|---|---|---|---|---|"]
    for embedder, variant, ranking in rows:
        lines.append(f"| {embedder} | {variant} | {recall_at_k(ranking, labels):.2f} | "
                     f"{precision_at_k(ranking, labels, k):.2f} | "
                     f"{precision_at_k(ranking, labels, k, ('yes', 'maybe')):.2f} |")
    return lines


if __name__ == "__main__":
    import argparse
    import time

    from dotenv import load_dotenv

    from src.agents.ranker import (DEFAULT_EMBEDDER, LEVEL_SLACK, RERANK_TOP, CachedEmbedder,
                                   Stage1Options, experience_years, make_embedder, rank)
    from src.atoms import Profile
    from src.config import ranker_exclude_atoms
    from src.models import Resume

    load_dotenv()

    ap = argparse.ArgumentParser(description="Score the Ranker against hand labels.")
    ap.add_argument("labels", help="a *.labels.yaml file")
    ap.add_argument("--pages", type=int, default=None,
                    help="cached pages to rank (default: query.pages in the labels file, else 3)")
    ap.add_argument("--runs", type=int, default=3, help="stage-2 runs (it is not deterministic)")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--stage1-only", action="store_true", help="no API calls")
    ap.add_argument("--embed", default=DEFAULT_EMBEDDER, help="stage-1 embedder for a normal run")
    ap.add_argument("--no-titles", action="store_true")
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--exclude-atoms", default=None,
                    help='comma-separated atom ids to leave out of stage 1 (default: RANKER_EXCLUDE_ATOMS; "" for none)')
    ap.add_argument("--no-gate", action="store_true", help="turn the level gate off")
    ap.add_argument("--gate-slack", type=float, default=LEVEL_SLACK,
                    help=f"years a posting may ask beyond yours before it is demoted (default {LEVEL_SLACK})")
    ap.add_argument("--compare", default=None,
                    help="comma-separated embedders: compare stage-1 variants on each, no API calls")
    ap.add_argument("--profile", default=ROOT / "data" / "profile.yaml")
    ap.add_argument("--resume", default=ROOT / "assets" / "resume.json")
    args = ap.parse_args()
    exclude = ranker_exclude_atoms(args.exclude_atoms)

    query, labels = load_labels(args.labels)
    posts = cached_posts(query, args.pages or int(query.get("pages", 3)))
    refs = {p.slug[-8:] for p in posts}
    unlabelled = sorted(refs - set(labels))
    if unlabelled:
        ap.error(f"postings in the cache but not in the labels file: {unlabelled}")
    posts = [p for p in posts if p.slug[-8:] in labels]
    titles = {p.slug[-8:]: f"{p.title} — {p.company}" for p in posts}
    counts = Counter(labels.values())
    profile, resume = Profile.load(args.profile), Resume.load(args.resume)

    print(f"{len(posts)} postings: {counts['yes']} yes, {counts['maybe']} maybe, {counts['no']} no, "
          f"{counts['skip']} skip.")
    print(f"P@{args.k} best possible: strict {best_possible(labels, args.k):.2f}, lenient "
          f"{best_possible(labels, args.k, ('yes', 'maybe')):.2f}. Random pick: strict "
          f"{random_expected(labels):.2f}, lenient {random_expected(labels, ('yes', 'maybe')):.2f}, "
          f"recall@{CUT} {min(CUT / len(posts), 1):.2f}.\n")

    if args.compare:
        # The variants, each adding one change to a row above it, so every
        # row isolates one effect. Exclusion rows only run when ids are given.
        gate = args.gate_slack
        variants = [Stage1Options(titles=False), Stage1Options(titles=True)]
        if exclude:
            variants.append(Stage1Options(titles=True, exclude=exclude))
        variants.append(Stage1Options(titles=True, center=True))
        if exclude:
            variants.append(Stage1Options(titles=True, exclude=exclude, center=True))
        variants.append(Stage1Options(titles=True, level_slack=gate))
        if exclude:
            variants.append(Stage1Options(titles=True, exclude=exclude, level_slack=gate))
        print(f"level gate rows: you have {experience_years(resume):.1f} years; postings asking more than "
              f"{experience_years(resume) + gate:.1f} are demoted.\n")
        rows = []
        for spec in (s.strip() for s in args.compare.split(",") if s.strip()):
            start = time.perf_counter()
            try:
                embedder = CachedEmbedder(make_embedder(spec))
                for options in variants:
                    result = rank(posts, profile, resume, embedder=embedder, options=options, rerank_top=0)
                    rows.append((spec, options.label, [r.ref for r in result.ranked]))
            except Exception as e:                 # one unavailable model must not lose the others
                print(f"{spec}: skipped — {type(e).__name__}: {str(e)[:160]}")
                continue
            print(f"{spec}: {time.perf_counter() - start:.0f}s")
        print("\n" + "\n".join(compare_table(rows, labels, args.k)))
        raise SystemExit(0)

    # With no flags this is ranker.default_options(): the flags' defaults are its values.
    options = Stage1Options(titles=not args.no_titles, center=args.center, exclude=exclude,
                            level_slack=None if args.no_gate else args.gate_slack)
    embedder = CachedEmbedder(make_embedder(args.embed))    # one load; stage 1 repeats every run
    stage1 = rank(posts, profile, resume, embedder=embedder, options=options, rerank_top=0)
    print("\n".join(report_stage(f"stage 1 ({args.embed}; {options.label})",
                                 [[r.ref for r in stage1.ranked]], labels, titles, args.k,
                                 stage1=True)) + "\n")
    if options.level_slack is not None:
        print("\n".join(gate_report([(r.ref, r.stage1.demotion) for r in stage1.ranked if r.stage1.demotion],
                                    labels, titles)) + "\n")

    if not args.stage1_only:
        rankings, tokens = [], [0, 0]
        for run in range(1, args.runs + 1):
            print(f"stage 2: run {run}/{args.runs} ...", flush=True)
            result = rank(posts, profile, resume, embedder=embedder, options=options, rerank_top=RERANK_TOP)
            rankings.append([r.ref for r in result.ranked])
            tokens[0] += result.usage["prompt_tokens"]
            tokens[1] += result.usage["completion_tokens"]
        print("\n".join(report_stage(f"stage 1 + 2 ({result.model})", rankings, labels, titles, args.k)))
        print(f"   tokens in/out per run (mean): {tokens[0] // args.runs}/{tokens[1] // args.runs}")
