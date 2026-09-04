"""Keyword retrieval over the evidence store: query -> ranked atoms.

This is the Retriever node's tool (roadmap Phase 2 / 5). Given a query —
a JD, a list of must-haves, or a bare keyword — it returns the atoms
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
  * Query and fields are tokenised the same way; stopwords dropped.
  * A query term counts once per field it appears in (presence, not
    frequency) so long summaries can't out-score precise tags.
  * Fields are weighted: curated tags (`keywords`, `skills`) dominate,
    prose (`summary`) is a weak tie-breaker.
  * Multi-word queries that appear verbatim as a tag get a phrase bonus,
    so "fraud detection" prefers an atom tagged `fraud detection rules`
    over one that merely mentions fraud and detection separately.
  * Atoms with score 0 are never returned — no evidence is better than
    irrelevant evidence handed to a model that will try to use it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from src.atoms import Atom, Profile

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


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


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


if __name__ == "__main__":
    # Phase 2 acceptance:  python -m src.tools.evidence "fraud detection"
    import argparse

    ap = argparse.ArgumentParser(description="Query the evidence store.")
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--profile", default=None, help="path to profile.yaml")
    args = ap.parse_args()

    for rank, hit in enumerate(search(args.query, args.k, args.profile), 1):
        print(f"{rank}. {hit.score:5.1f}  {hit.atom.id:<40} [{', '.join(hit.matched)}]")
