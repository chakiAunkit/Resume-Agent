"""Keyword retrieval over the evidence store: query -> ranked atoms.

This is the Retriever node's tool (roadmap Phase 2 / 5). Given a query —
a JD must-have, a list of them, or a bare keyword — it returns the atoms
whose tags and text best match, so the Tailor sees a *bounded* evidence
set instead of the whole profile.

Why keyword matching, not embeddings (yet):
  * The corpus is ~15 documents. IDF, BM25, or a vector index buy
    nothing at this size; a weighted overlap count is fully explainable.
  * Explainability matters more than recall here: `Hit.matched` shows
    exactly which query terms pulled an atom in, so a bad ranking is a
    bad tag, which you fix in profile.yaml, not in the algorithm.
  * The interface (`search` / `load_evidence`) is what the graph depends
    on. Swapping the scorer for embeddings later touches this file only.

Scoring rules (deliberately simple):
  * Query and fields are tokenised the same way; stopwords dropped;
    trailing plural "s" stripped on both sides (models == model,
    LLMs == LLM) — symmetric, so it never creates a mismatch.
  * A query term counts once per field it appears in (presence, not
    frequency) so long summaries can't out-score precise tags.
  * Fields are weighted: curated tags (`keywords`, `skills`) dominate,
    prose (`summary`) is a weak tie-breaker.
  * Multi-word queries that appear verbatim as a tag get a phrase bonus,
    so "fraud detection" prefers an atom tagged `fraud detection rules`
    over one that merely mentions fraud and detection separately.
  * Atoms with score 0 are never returned — no evidence is better than
    irrelevant evidence handed to a model that will try to use it.

Two entry points sit on the same scorer:
  * `search(query, k)` — one query string. The primitive.
  * `retrieve_for_brief(brief, k)` — one `search()` per requirement in
    a `JobBrief`, scores merged per atom. This is what the Tailor calls.
    Querying per requirement (not one joined string) matters: the
    phrase bonus fires per requirement, and a generic term like "data"
    can only contribute within the one requirement that contains it
    instead of once per token across the whole JD. Each result carries
    `covers` — the must-haves the atom evidences — which is both the
    Tailor's reason-to-include and, later, the Judge's coverage check.
    Coverage is strict, ranking is soft: a requirement is *covered* only
    when every one of its tokens matched; a partial match still adds to
    the score, scaled by the fraction matched, so "data analytics"
    matching via "data" alone nudges the ranking but claims nothing.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.atoms import Atom, Profile
from src.brief import JobBrief

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = ROOT / "data" / "profile.yaml"

# Tokens keep inner + . # - so "1d-cnn", "c++", "scikit-learn", "qwen2.5"
# survive; leading/trailing punctuation is stripped.
_TOKEN = re.compile(r"[a-z0-9][a-z0-9+.#-]*[a-z0-9+#]|[a-z0-9]")

STOPWORDS = frozenset(
    "a an and are as at be by for from in into is it of on or that the this "
    "to with using via our your we you will".split()
)

FIELD_WEIGHTS: dict[str, float] = {
    "keywords": 3.0,
    "skills": 3.0,
    "title": 2.0,
    "source": 1.5,
    "summary": 1.0,
    "metrics": 0.5,
}
PHRASE_BONUS = 2.0

# A nice-to-have is worth half a must-have. Enough to break ties between
# atoms that cover the same must-haves; not enough to promote an atom
# that covers none of them.
NICE_TO_HAVE_WEIGHT = 0.5


def _singular(token: str) -> str:
    """Strip a trailing plural 's' (models -> model, llms -> llm).

    Deliberately crude: no stemmer, no irregulars. It is applied to both
    the query and the fields, so "analytics" -> "analytic" on both sides
    still matches — symmetry is what makes crude safe. Three-letter
    tokens ("aws", "k8s") and "-ss" words ("class") are left alone.
    """
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [
        _singular(t)
        for t in _TOKEN.findall(text.lower())
        if t not in STOPWORDS
    ]


@dataclass(frozen=True)
class Hit:
    atom: Atom
    score: float
    matched: tuple[str, ...]  # query terms that contributed, for explainability


def _field_text(atom: Atom) -> dict[str, str]:
    return {
        "keywords": " ".join(atom.keywords),
        "skills": " ".join(atom.skills),
        "title": atom.title,
        "source": atom.source,
        "summary": atom.summary,
        "metrics": " ".join(atom.metrics),
    }


def score_atom(atom: Atom, query: str) -> Hit:
    q_terms = tokenize(query)
    q_set = set(q_terms)
    score = 0.0
    matched: set[str] = set()

    for field, weight in FIELD_WEIGHTS.items():
        field_terms = set(tokenize(_field_text(atom)[field]))
        for term in q_set & field_terms:
            score += weight
            matched.add(term)

    if len(q_terms) > 1:
        phrase = " ".join(q_terms)
        for tag in atom.keywords + atom.skills:
            if phrase in " ".join(tokenize(tag)):
                score += PHRASE_BONUS
                break

    return Hit(atom=atom, score=score, matched=tuple(sorted(matched)))


def _resolve(profile: Profile | Path | str | None) -> Profile:
    if isinstance(profile, Profile):
        return profile
    return Profile.load(profile or DEFAULT_PROFILE)


def search(
    query: str,
    k: int = 5,
    profile: Profile | Path | str | None = None,
) -> list[Hit]:
    """Top-k atoms by score, highest first. Ties keep profile.yaml order
    (which is newest-first by convention), so recency breaks ties."""
    hits = [score_atom(a, query) for a in _resolve(profile).atoms]
    hits = [h for h in hits if h.score > 0]
    hits.sort(key=lambda h: h.score, reverse=True)  # sort is stable
    return hits[:k]


def load_evidence(
    query: str,
    k: int = 5,
    profile: Profile | Path | str | None = None,
) -> list[Atom]:
    """The roadmap signature: ranked atoms only, scores dropped."""
    return [h.atom for h in search(query, k, profile)]


# ---------------------------------------------------------------------------
# Brief-level retrieval: what the Tailor actually calls
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BriefHit:
    atom: Atom
    score: float
    covers: tuple[str, ...]    # must-haves this atom evidences
    supports: tuple[str, ...]  # nice-to-haves this atom evidences


def retrieve_for_brief(
    brief: JobBrief,
    k: int = 8,
    profile: Profile | Path | str | None = None,
) -> list[BriefHit]:
    """One `search()` per requirement, merged per atom.

    Per requirement, an atom contributes `hit.score * fraction`, where
    fraction = matched query tokens / query tokens. Only a full match
    (fraction == 1) lands the requirement in `covers` / `supports`, so
    coverage never overclaims. Nice-to-haves are scaled by
    NICE_TO_HAVE_WEIGHT. Atoms matching nothing are never returned.
    Ties keep profile order (recency).
    """
    prof = _resolve(profile)
    n = len(prof.atoms)  # ask for everything: merging needs every non-zero hit

    scores: dict[str, float] = defaultdict(float)
    covers: dict[str, list[str]] = defaultdict(list)
    supports: dict[str, list[str]] = defaultdict(list)

    def _merge(reqs: list[str], weight: float, bucket: dict[str, list[str]]) -> None:
        for req in reqs:
            n_terms = len(set(tokenize(req)))
            if n_terms == 0:  # a requirement made only of stopwords
                continue
            for hit in search(req, k=n, profile=prof):
                fraction = len(hit.matched) / n_terms
                scores[hit.atom.id] += hit.score * fraction * weight
                if fraction == 1.0:
                    bucket[hit.atom.id].append(req)

    _merge(brief.must_haves, 1.0, covers)
    _merge(brief.nice_to_haves, NICE_TO_HAVE_WEIGHT, supports)

    # Iterate prof.atoms (not the dict) so the pre-sort order is profile
    # order and the stable sort keeps recency as the tie-break.
    hits = [
        BriefHit(
            atom=a,
            score=scores[a.id],
            covers=tuple(covers[a.id]),
            supports=tuple(supports[a.id]),
        )
        for a in prof.atoms
        if a.id in scores
    ]
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[:k]


if __name__ == "__main__":
    # Phase 2:  python -m src.tools.evidence "fraud detection"
    # Phase 3:  python -m src.tools.evidence --brief data/sample_jd.brief.yaml -k 8
    import argparse

    ap = argparse.ArgumentParser(description="Query the evidence store.")
    ap.add_argument("query", nargs="?", help="free-text query (omit with --brief)")
    ap.add_argument("--brief", default=None, help="path to a *.brief.yaml")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--profile", default=None, help="path to profile.yaml")
    args = ap.parse_args()

    if args.brief:
        brief = JobBrief.load(args.brief)
        for rank, h in enumerate(retrieve_for_brief(brief, args.k, args.profile), 1):
            print(f"{rank}. {h.score:5.1f}  {h.atom.id}")
            print(f"        covers:   {', '.join(h.covers) or '-'}")
            if h.supports:
                print(f"        supports: {', '.join(h.supports)}")
    elif args.query:
        for rank, hit in enumerate(search(args.query, args.k, args.profile), 1):
            print(f"{rank}. {hit.score:5.1f}  {hit.atom.id:<40} [{', '.join(hit.matched)}]")
    else:
        ap.error("give a query or --brief")
