"""The Ranker — job postings -> a ranked shortlist. Phase 4 deliverable 3.

Two stages, cheap before expensive (Phase 4's learning focus):

  1. Local embeddings: free, deterministic, no API. Each posting is split
     into sentences, and each profile atom finds its best-matching
     sentence. A posting scores the mean similarity of its TOP_ATOMS
     best-matched atoms.
       * Why sentences, not the whole posting: MiniLM reads only the first
         256 tokens (~1,000 characters) and silently drops the rest. The
         median posting is ~3,400 characters and most open with company
         boilerplate, so a whole-posting embedding mostly compares
         "About us" sections.
       * Why not an LLM-written brief: that is the Extractor, at ~5k tokens
         per posting — ~140k tokens to rank 28 postings, spent deciding
         which ones deserve attention. Best-sentence matching condenses
         for free: boilerplate never wins a match, so it never counts.
       * Why the top few atoms, not all of them: no single job wants fraud
         analytics *and* warehouse slotting *and* medical imaging. A
         posting that strongly matches three of your atoms is a fit; an
         average over all thirteen would punish focused postings.
     Every score carries its explanation: which atom matched which sentence.
  2. An open-model rerank of the top RERANK_TOP, in batches of
     RERANK_BATCH. Each posting is shown as its title, any sentence stating
     years of experience, and its best stage-1 sentences — so fifteen
     postings fit Groq's free tier (8k tokens a minute) where fifteen full
     descriptions would not. The model returns a 1-10 score and a
     one-line reason per posting, judged against a short candidate card
     built from resume.json (roles with dates, so level can be judged).

The Extractor is deliberately not called: briefs are written for the
shortlist only, in Phase 5 (Sourcer -> Ranker -> Extractor -> Retriever ...).

Both expensive dependencies are injected — the embedder and the LLM client —
so tests run on fakes with no model download and no API call.

Stage-1 options (Stage1Options), added after the first labelled run showed
stage 1 keeping only 5 of 12 wanted postings in its top 15:
  * titles  — the posting's title counts as one of its sentences. Some
              descriptions are pure company boilerplate and only the title
              says what the job is (Infosys: "Data Scientist | Python |
              GenAI | LLM | RAG" over "A day in the life of an Infoscion").
  * exclude — atom ids left out of stage 1. A soft-skill atom
              ("cross-functional communication") matches the collaboration
              line every posting has, so it takes a top-atom slot
              everywhere and says nothing about fit. Same lesson as
              retrieval's v5: soft skills are phrasing, not evidence.
  * center  — the automatic alternative: score each atom's match *above
              its own average across the pool*, so an atom that matches
              every posting equally contributes nothing. Scores become
              relative to the batch being ranked, which is all ranking needs.
  * level_slack — the level gate (below). None turns it off.
The embedder is pluggable too (EMBEDDERS, make_embedder). The 2026-09-29
comparison (data/evals/README.md) picked the defaults: Qwen3-Embedding-0.6B,
titles on, no centering, the atoms in RANKER_EXCLUDE_ATOMS left out
(default_options()). Centering lost on every embedder.

The level gate. The same comparison showed that similarity cannot fix
stage 1: all 28 postings come from one "data scientist" search, so all are
on-topic, and the best embedder kept 7 of 12 wanted postings — a random
cut keeps about 6.4. What separated the labels was level: no wanted posting asked
for 7+ years, and 7 of the 8 unwanted ones did. So after scoring, each
posting's stated minimum years (years_asked) is compared with the
candidate's total experience from resume.json; postings asking more than
that plus `level_slack` years move below all the others, keeping their
order, with a one-line explanation ("asks 8+ yrs (you: 5.2)"). Demoted, not
dropped: a rule that reads text by pattern will sometimes misread it, and a
demoted posting is still in shortlist.json for you to see. The demotion
also holds through stage 2 — the reranker orders postings within each tier.

    python -m src.agents.ranker "data scientist" --location Bangalore --pages 3
    python -m src.agents.ranker "data scientist" --location Bangalore --pages 3 --stage1-only
    python -m src.agents.ranker ... --embed minilm --no-gate --exclude-atoms ""   # the run-1 setup
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Annotated, Any, Protocol, Sequence

import httpx
import numpy as np
from pydantic import Field, StringConstraints, ValidationError

# Borrowed from the Extractor: the constrained-decoding plumbing is the same
# for every open-model node. When a third node needs it (the Judge, Phase 6)
# it moves to a shared module — two copies are cheaper than a premature
# abstraction, three are not.
from src.agents.extractor import (
    _default_client,
    _is_schema_rejection,
    _strip_constraints,
    _strip_fences,
)
from src.atoms import Atom, Profile
from src.config import ModelRoute, model_settings, ranker_exclude_atoms, ranker_route, route_for
from src.models import NonEmptyStr, Resume, StrictModel
from src.tools.jobs import JobPost

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_EMBEDDER = "qwen3-0.6b"   # steadiest P@5 in the 2026-09-29 comparison; MiniLM was run 1's
TOP_ATOMS = 3                 # atoms averaged into a posting's stage-1 score
LEVEL_SLACK = 1.5             # years a posting may ask beyond yours before the gate demotes it
MIN_SENTENCE_CHARS = 20       # shorter is a heading ("Key Responsibilities"), not content
MAX_SENTENCE_CHARS = 400      # well inside MiniLM's 256-token window

RERANK_TOP = 15               # postings that reach stage 2
RERANK_BATCH = 5              # postings per rerank call: ~3-5k tokens with reasoning
SENTENCES_PER_POSTING = 5     # best stage-1 sentences shown to the reranker
MAX_ATTEMPTS = 2              # content retries per batch, as in the other agents


# ---------------------------------------------------------------------------
# Stage 1 — sentences against atoms
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        """One unit-length row per text, so a dot product is the cosine.
        `query=True` marks the side that searches (the atoms): instruction-
        tuned models embed a query differently from the text it searches."""
        ...


# The search here is asymmetric: an atom (what the candidate did) looks for
# the posting sentence that asks for it. Instruction-tuned embedders are told so.
QWEN_INSTRUCTION = ("Instruct: Given a description of a candidate's work experience, retrieve "
                    "sentences from job postings that ask for this experience\nQuery: ")
BGE_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def _unit_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


class SentenceTransformerEmbedder:
    """Any Hugging Face embedding model, run locally by sentence-transformers.
    The library (and torch, ~2 GB) is imported on first use, so importing
    this module stays cheap; the model downloads once and is cached. It runs
    on the GPU only if torch was installed with CUDA support."""

    def __init__(self, model_id: str, query_prompt: str | None = None, name: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.name = name or model_id
        self._prompt = query_prompt
        self._model = SentenceTransformer(model_id)

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        kwargs = {"prompt": self._prompt} if query and self._prompt else {}
        return _unit_rows(np.asarray(self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, **kwargs), dtype=float))


class OllamaEmbedder:
    """An embedding model served by a local Ollama (`ollama pull <model>`).
    The route for models too big for plain PyTorch on a 6 GB GPU: Ollama
    runs a 4-bit copy (Qwen3-Embedding-4B: ~2.5 GB instead of ~8 GB) and
    uses the GPU without a CUDA build of torch. Ollama applies no prompts,
    so the query instruction is prefixed here."""

    BATCH = 64

    def __init__(self, model: str, query_prompt: str | None = None, name: str | None = None,
                 base_url: str | None = None, client: httpx.Client | None = None):
        self.name = name or f"ollama:{model}"
        self._model = model
        self._prompt = query_prompt
        self._url = (base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self._client = client or httpx.Client(timeout=httpx.Timeout(300.0, connect=5.0))

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if query and self._prompt:
            texts = [self._prompt + t for t in texts]
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            try:
                response = self._client.post(f"{self._url}/api/embed",
                                             json={"model": self._model, "input": texts[i:i + self.BATCH]})
            except httpx.ConnectError as e:
                raise RuntimeError(f"Ollama is not reachable at {self._url}; start it (the Ollama app "
                                   "or `ollama serve`)") from e
            if response.status_code != 200:
                raise RuntimeError(f"Ollama returned HTTP {response.status_code} for {self._model!r}: "
                                   f"{response.text[:200]} (pulled it? `ollama pull {self._model}`)")
            rows += response.json()["embeddings"]
        return _unit_rows(np.asarray(rows, dtype=float))


# Short names for the models worth comparing. Anything else can be given as
# "st:<hugging face id>" or "ollama:<model tag>".
EMBEDDERS: dict[str, tuple[str, str, str | None]] = {
    "minilm":     ("st", "sentence-transformers/all-MiniLM-L6-v2", None),       # 22M, the baseline
    "bge-base":   ("st", "BAAI/bge-base-en-v1.5", BGE_INSTRUCTION),              # 110M
    "qwen3-0.6b": ("st", "Qwen/Qwen3-Embedding-0.6B", QWEN_INSTRUCTION),         # 600M
    "qwen3-4b":   ("ollama", "qwen3-embedding:4b", QWEN_INSTRUCTION),            # 4B, 4-bit via Ollama
}


def make_embedder(spec: str = DEFAULT_EMBEDDER) -> Embedder:
    """'minilm' | 'bge-base' | 'qwen3-0.6b' | 'qwen3-4b' | 'st:<hf id>' | 'ollama:<tag>'."""
    if spec in EMBEDDERS:
        backend, model, prompt = EMBEDDERS[spec]
    elif spec.startswith(("st:", "ollama:")):
        backend, model = spec.split(":", 1)
        prompt = QWEN_INSTRUCTION if "qwen3" in model.lower() else None
    else:
        raise ValueError(f"unknown embedder {spec!r}; use one of {sorted(EMBEDDERS)}, "
                         "st:<hugging face id> or ollama:<model tag>")
    if backend == "st":
        return SentenceTransformerEmbedder(model, prompt, name=spec)
    return OllamaEmbedder(model, prompt, name=spec)


class CachedEmbedder:
    """Remembers every text it has embedded, so comparing stage-1 variants on
    one model embeds each posting sentence once, not once per variant."""

    def __init__(self, inner: Embedder):
        self.inner = inner
        self.name = inner.name
        self._cache: dict[tuple[bool, str], np.ndarray] = {}

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        todo = list(dict.fromkeys(t for t in texts if (query, t) not in self._cache))
        if todo:
            for text, row in zip(todo, self.inner.encode(todo, query=query)):
                self._cache[(query, text)] = row
        return np.vstack([self._cache[(query, t)] for t in texts]) if texts else np.zeros((0, 0))


_BULLET = re.compile(r"^\s*(?:[-*•·●▪◦‣–—]+|\(?\d{1,2}[.)])\s*")
_SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-Z0-9(\"'“])")


def split_sentences(text: str) -> list[str]:
    """Posting text -> the short units stage 1 compares against atoms.

    Lines first (most postings are bullet lists), then sentence ends within
    a line, so a description with no line breaks at all still splits.
    Bullet markers are stripped, fragments under MIN_SENTENCE_CHARS dropped,
    runs over MAX_SENTENCE_CHARS cut at a space, exact repeats kept once.
    """
    out: list[str] = []
    seen: set[str] = set()

    def keep(piece: str) -> None:
        piece = " ".join(piece.split())
        if len(piece) >= MIN_SENTENCE_CHARS and piece not in seen:
            seen.add(piece)
            out.append(piece)

    for line in text.splitlines():
        line = _BULLET.sub("", line)
        for piece in _SENTENCE_END.split(line):
            piece = " ".join(piece.split())
            while len(piece) > MAX_SENTENCE_CHARS:
                cut = piece.rfind(" ", MIN_SENTENCE_CHARS, MAX_SENTENCE_CHARS)
                cut = cut if cut > 0 else MAX_SENTENCE_CHARS
                keep(piece[:cut])
                piece = piece[cut:].strip()
            keep(piece)
    return out


def atom_text(atom: Atom) -> str:
    """What an atom looks like to the embedder: its claim plus its tags."""
    parts = [f"{atom.title}.", atom.summary]
    if atom.skills:
        parts.append("Skills: " + ", ".join(atom.skills) + ".")
    if atom.keywords:
        parts.append("Topics: " + ", ".join(atom.keywords) + ".")
    return " ".join(parts)


@dataclass(frozen=True)
class Stage1Options:
    """How stage 1 runs. The field defaults are the plain baseline (no
    exclusions, no gate) so a bare Stage1Options() never reads the
    environment; default_options() is the configured setup — what the
    CLIs run with no flags, and what the Phase 5 graph node will use."""

    titles: bool = True                       # the title counts as one of the posting's sentences
    center: bool = False                      # score atoms above their own pool average
    exclude: frozenset[str] = field(default_factory=frozenset)   # atom ids left out
    top_atoms: int = TOP_ATOMS
    level_slack: float | None = None          # level gate: demote postings asking > yours + this; None = off

    @property
    def label(self) -> str:
        parts = ["titles" if self.titles else "no titles"]
        if self.exclude:
            parts.append(f"-{len(self.exclude)} atom(s)")
        if self.center:
            parts.append("centered")
        if self.level_slack is not None:
            parts.append(f"level gate +{self.level_slack:g}y")
        return ", ".join(parts)


def default_options() -> Stage1Options:
    """The chosen setup: titles on, no centering, RANKER_EXCLUDE_ATOMS left
    out, level gate at LEVEL_SLACK. Atom ids come from .env because they
    name entries of the private profile, which the public repo never shows."""
    return Stage1Options(titles=True, center=False, exclude=ranker_exclude_atoms(), level_slack=LEVEL_SLACK)


def ranking_atoms(profile: Profile, exclude: frozenset[str] = frozenset()) -> list[Atom]:
    """Education atoms describe a degree, not work a posting could ask for;
    `exclude` removes more by id. An unknown id is an error, not a silent no-op."""
    unknown = set(exclude) - {a.id for a in profile.atoms}
    if unknown:
        raise ValueError(f"unknown atom id(s) to exclude: {sorted(unknown)}")
    return [a for a in profile.atoms if a.section != "education" and a.id not in exclude]


@dataclass(frozen=True)
class Match:
    atom_id: str
    sentence: str
    similarity: float             # raw cosine between the atom and its best sentence
    lift: float | None = None     # similarity minus the atom's pool average (centered runs only)


@dataclass(frozen=True)
class Scored:
    post: JobPost
    score: float                  # mean over the top atoms of similarity (or lift, if centered)
    matches: tuple[Match, ...]    # those atoms, best first: the explanation
    sentences: tuple[str, ...]    # the posting's sentences, most relevant to any atom first
    years_asked: int | None = None    # set by the level gate: the posting's stated minimum years
    demotion: str | None = None       # why the level gate moved it down; None if it didn't

    @property
    def ref(self) -> str:
        """The 8-character id shown by the jobs CLI and used in the labels file."""
        return self.post.slug[-8:]


def _units(post: JobPost, titles: bool) -> list[str]:
    sentences = split_sentences(post.description)
    if not titles:
        return sentences
    title = " ".join(post.title.split())
    return [title] + [s for s in sentences if s != title]


def score_postings(posts: Sequence[JobPost], atoms: Sequence[Atom], embedder: Embedder,
                   options: Stage1Options = Stage1Options()) -> list[Scored]:
    """Stage 1. Highest score first; ties keep the fetch order.

    Two passes: first each atom's best sentence in every posting (an
    atoms x postings matrix), then the per-posting score — which is where
    centering, if on, subtracts each atom's average across the pool.
    """
    if not atoms:
        raise ValueError("no atoms to rank against")
    atom_vecs = embedder.encode([atom_text(a) for a in atoms], query=True)
    units = [_units(p, options.titles) for p in posts]
    best = np.zeros((len(atoms), len(posts)))
    best_at = np.zeros((len(atoms), len(posts)), dtype=int)
    relevance: list[tuple[str, ...]] = []
    for j, post_units in enumerate(units):
        if not post_units:
            relevance.append(())
            continue
        sims = atom_vecs @ embedder.encode(post_units).T      # atoms x sentences, cosines
        best[:, j], best_at[:, j] = sims.max(axis=1), sims.argmax(axis=1)
        relevance.append(tuple(post_units[i] for i in np.argsort(-sims.max(axis=0), kind="stable")))

    value = best - best.mean(axis=1, keepdims=True) if options.center else best
    k = min(options.top_atoms, len(atoms))
    scored: list[Scored] = []
    for j, post in enumerate(posts):
        if not units[j]:
            scored.append(Scored(post, float(value[:, j].min()) if options.center else 0.0, (), ()))
            continue
        top = np.argsort(-value[:, j], kind="stable")[:k]
        matches = tuple(Match(atoms[i].id, units[j][best_at[i, j]], float(best[i, j]),
                              float(value[i, j]) if options.center else None) for i in top)
        scored.append(Scored(post, float(np.mean(value[top, j])), matches, relevance[j]))
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Stage 1, second step — the level gate
# ---------------------------------------------------------------------------

# "5+ years", "4-8 years", "6 to 8 yrs", "Exp : 6-8yrs", "12 + years", and
# "7 11 Years" (a range whose dash the job board dropped). Numbers 1-20 only:
# "we've done that for 90 years" is company history, not a requirement.
# Two-digit numbers come first in the alternation so a capture of "12+" is
# 12 — never 1 — whatever else the pattern could settle for.
_NUM = r"(?:1\d|20|[1-9])"
_YEARS = re.compile(rf"\b({_NUM})\s*(?:\+|(?:\s*(?:-|–|—|to)\s*|\s+)({_NUM}))?\s*\+?\s*(?:years?|yrs?)\b",
                    re.I)
# A bare "N years" is a requirement only near a word that says so; "over the
# last 5 years, Gartner ranked us #1" is not one. A "+" or a range always is.
_ASKS = re.compile(r"\b(?:exp(?:erience[ds]?|\.)?|min(?:imum|\.)?|at\s*least|relevant|professional)\b", re.I)
_ASKS_WINDOW = 60             # characters either side of the number, within its line


def years_asked(post: JobPost) -> int | None:
    """The minimum years of experience a posting asks for, or None if it
    states none: the largest lower bound among its years-of-experience
    phrases, read from the title and every line of the description.

    Largest, because a posting's secondary asks ("8+ years in data science
    with 2-3 in GenAI") are parts of its main one. Lower bound, because "6-8
    years" means six is enough. The title counts: "Data Scientist (Exp :
    6-8yrs)" says it nowhere else. Alternatives ("6+ years with a Masters or
    4+ with a PhD") read as the larger — a known over-read, and one reason
    the gate demotes instead of dropping.
    """
    best: int | None = None
    for line in [post.title, *post.description.splitlines()]:
        for m in _YEARS.finditer(line):
            low, high = int(m.group(1)), m.group(2)
            if high is not None:
                low = min(low, int(high))
            window = line[max(m.start() - _ASKS_WINDOW, 0):m.end() + _ASKS_WINDOW]
            if high is None and "+" not in m.group(0) and not _ASKS.search(window):
                continue
            best = low if best is None else max(best, low)
    return best


def _months(month_year: str) -> tuple[int, int]:
    month, year = month_year.split("/")
    return int(year), int(month)


def role_months(resume: Resume, today: date | None = None) -> list[int]:
    """Months in each role of resume.json, in resume order ("Present" = today)."""
    today = today or date.today()
    out = []
    for e in resume.experience:
        start = _months(e.start)
        end = (today.year, today.month) if e.end == "Present" else _months(e.end)
        out.append(max((end[0] - start[0]) * 12 + end[1] - start[1], 0))
    return out


def experience_years(resume: Resume, today: date | None = None) -> float:
    """Total years of experience — the number the candidate card states."""
    return sum(role_months(resume, today)) / 12


def level_gate(scored: Sequence[Scored], candidate_years: float, slack: float = LEVEL_SLACK) -> list[Scored]:
    """Stage 1's ranking with every posting that asks for more than
    `candidate_years + slack` years moved below all the others. Both groups
    keep their stage-1 order; every posting gets `years_asked` set, and each
    demoted one says why. A posting that states no years is never demoted:
    silence is not a requirement."""
    passed, demoted = [], []
    for s in scored:
        asked = years_asked(s.post)
        if asked is not None and asked > candidate_years + slack:
            demoted.append(replace(s, years_asked=asked,
                                   demotion=f"asks {asked}+ yrs (you: {candidate_years:.1f})"))
        else:
            passed.append(replace(s, years_asked=asked))
    return passed + demoted


# ---------------------------------------------------------------------------
# Stage 2 — open-model rerank
# ---------------------------------------------------------------------------

class Rating(StrictModel):
    ref: NonEmptyStr
    score: int = Field(ge=1, le=10)
    reason: Annotated[str, StringConstraints(min_length=1, max_length=240)]


class RatingBatch(StrictModel):
    ratings: list[Rating] = Field(min_length=1)


def output_schema() -> dict[str, Any]:
    """RatingBatch for a constrained decoder: every key required, the
    constraints a grammar can't express stripped (Pydantic re-applies them)."""
    schema = RatingBatch.model_json_schema()
    for node in [schema, *schema.get("$defs", {}).values()]:
        if "properties" in node:
            node["required"] = list(node["properties"])
    return _strip_constraints(schema)


SYSTEM_PROMPT = """\
You are the Ranker in an automated job-search pipeline. You receive one \
candidate's profile and a batch of job postings. A cheap similarity filter has \
already shortlisted them; each posting is shown as its title, any sentence that \
states the experience required, and the sentences most relevant to the \
candidate. Rate how strongly this candidate should consider applying to each.

Scores, 1-10:
  9-10  strong fit: the work matches the candidate's experience and the level fits
  6-8   good fit with one gap: a missing core skill, a domain shift, or a level one step off
  3-5   weak fit: several gaps, or a level clearly above or below the candidate
  1-2   not a fit

Rules:
- Judge level from the years asked and the title against the candidate's roles \
and dates. A role that asks for far more experience (director, principal, \
10+ years) or far less (junior, fresher) scores lower even when the domain matches.
- Judge only from what is shown; do not assume requirements that are not stated.
- Score each posting on its own merits; do not force a spread within a batch. \
Scores are compared across batches.
- reason: one line, under 25 words, naming the main match and the main gap.

Return exactly one rating per posting, using its [ref] exactly as given.\
"""


def candidate_card(resume: Resume, today: date | None = None) -> str:
    """The candidate as the reranker sees them: the resume summary, each role
    with its dates and length (level is judged from these), and the skills."""
    months = role_months(resume, today)
    roles = [f"- {e.title}, {e.company}, {e.start} to {e.end} ({m / 12:.1f} years)"
             for e, m in zip(resume.experience, months)]
    skills = [item for group in resume.skills for item in group.items]
    return "\n".join([f"Summary: {resume.summary}",
                      f"Total experience: {sum(months) / 12:.1f} years",
                      "Roles:", *roles,
                      "Skills: " + ", ".join(skills)])


def posting_card(scored: Scored, n_sentences: int = SENTENCES_PER_POSTING) -> str:
    """One posting, compressed for the reranker: ~150-300 tokens instead of
    ~800. The years line is found by pattern, not by relevance, because a
    requirement the candidate misses is exactly what similarity won't surface."""
    post = scored.post
    where = f" ({post.location})" if post.location else ""
    lines = [f"[{scored.ref}] {post.title} — {post.company}{where}"]
    title = " ".join(post.title.split())
    body = [s for s in scored.sentences if s != title]       # the header already shows it
    years = [s for s in body if _YEARS.search(s)][:2]
    if years:
        lines.append("Experience asked: " + " | ".join(years))
    lines += [f"- {s}" for s in body if s not in years][:n_sentences]
    return "\n".join(lines)


def build_prompt(card: str, batch: Sequence[Scored]) -> str:
    return "\n\n".join(["# Candidate", card, "# Postings", *(posting_card(s) for s in batch),
                        "Rate every posting above."])


def _feedback(problems: list[str]) -> str:
    return ("Your previous output was rejected. Fix every problem below and return all "
            "the ratings again as JSON.\n\n" + "\n".join(f"- {p}" for p in problems))


class RankerError(RuntimeError):
    def __init__(self, problems: list[str]) -> None:
        super().__init__("Ranker output rejected after retry:\n" + "\n".join(f"- {p}" for p in problems))
        self.problems = problems


def _response_format(mode: str, schema: dict[str, Any]) -> dict[str, Any]:
    if mode == "json_schema":
        return {"type": "json_schema", "json_schema": {"name": "job_ratings", "schema": schema}}
    return {"type": "json_object"}


def _complete(client: Any, model: str, messages: list[dict[str, str]], mode: str,
              schema: dict[str, Any]) -> tuple[Any, str]:
    """One chat completion; one downgrade to plain JSON mode if the provider
    rejects the schema (same policy as the Extractor). Temperature and
    reasoning settings come from config.MODEL_SETTINGS, as for the Extractor."""
    extra = model_settings(model).request_kwargs()      # temperature, max_tokens, reasoning
    try:
        return client.chat.completions.create(
            model=model, messages=messages,
            response_format=_response_format(mode, schema), **extra), mode
    except Exception as e:
        if mode == "json_schema" and _is_schema_rejection(e):
            return client.chat.completions.create(
                model=model, messages=messages,
                response_format=_response_format("json_object", schema), **extra), "json_object"
        raise


def _check(batch: RatingBatch, refs: list[str]) -> list[str]:
    """Every posting rated exactly once, nothing else rated."""
    got = [r.ref for r in batch.ratings]
    problems = [f"posting [{ref}] has no rating" for ref in refs if ref not in got]
    problems += [f"[{ref}] is not one of the postings in this batch" for ref in got if ref not in refs]
    problems += [f"[{ref}] is rated more than once" for ref in sorted(set(got)) if got.count(ref) > 1]
    return problems


def rate_batch(batch: Sequence[Scored], card: str, *, client: Any, model: str,
               usage: dict[str, int]) -> dict[str, Rating]:
    """One rerank call (plus at most one content retry). `usage` is summed in place."""
    refs = [s.ref for s in batch]
    schema = output_schema()
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(card, batch)}]
    mode = "json_schema"
    problems: list[str] = []
    for _attempt in range(MAX_ATTEMPTS):
        response, mode = _complete(client, model, messages, mode, schema)
        if response.usage is not None:
            usage["prompt_tokens"] += response.usage.prompt_tokens or 0
            usage["completion_tokens"] += response.usage.completion_tokens or 0
        choice = response.choices[0]
        text = choice.message.content or ""
        if not text.strip():
            raise RankerError([f"empty response (finish_reason={choice.finish_reason!r}); a reasoning "
                               "model may have spent the whole token budget before answering"])
        if choice.finish_reason == "length":
            problems = ["output was cut off at the token limit; keep each reason to one short line"]
        else:
            try:
                parsed = RatingBatch.model_validate_json(_strip_fences(text))
            except ValidationError as e:
                problems = [f"output was not valid JSON: {err['msg']}" if err["type"] == "json_invalid"
                            else f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
                            for err in e.errors()]
            else:
                problems = _check(parsed, refs)
                if not problems:
                    return {r.ref: r for r in parsed.ratings}
        messages = messages + [{"role": "assistant", "content": text},
                               {"role": "user", "content": _feedback(problems)}]
    raise RankerError(problems)


# ---------------------------------------------------------------------------
# Both stages
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Ranked:
    stage1: Scored
    score: int | None = None      # stage-2 rating; None when not reranked
    reason: str | None = None

    @property
    def post(self) -> JobPost:
        return self.stage1.post

    @property
    def ref(self) -> str:
        return self.stage1.ref


@dataclass(frozen=True)
class RankResult:
    ranked: list[Ranked]
    usage: dict[str, int]
    model: str | None             # None when stage 2 did not run


def rank(posts: Sequence[JobPost], profile: Profile, resume: Resume, *,
         embedder: Embedder | None = None, options: Stage1Options = Stage1Options(),
         client: Any = None, route: ModelRoute | None = None,
         model: str | None = None, rerank_top: int = RERANK_TOP, batch_size: int = RERANK_BATCH,
         today: date | None = None) -> RankResult:
    """postings -> ranked postings, best first.

    Stage 1 scores every posting, then (if `options.level_slack` is set)
    the level gate moves over-level postings to the bottom. The top
    `rerank_top` are ordered by the stage-2 score (stage-1 score breaks
    ties) — within their tier, so a demoted posting stays below the others
    however stage 2 rates it. The rest follow in stage-1 order, unrated.
    `rerank_top=0` runs stage 1 only and makes no API call.
    """
    scored = score_postings(posts, ranking_atoms(profile, options.exclude),
                            embedder or make_embedder(), options)
    if options.level_slack is not None:
        scored = level_gate(scored, experience_years(resume, today), options.level_slack)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    head, tail = scored[:max(rerank_top, 0)], scored[max(rerank_top, 0):]
    if not head:
        return RankResult([Ranked(s) for s in scored], usage, None)

    default = ranker_route()
    model = model or (route or default).model
    route = route or route_for(model, default)       # the model's own provider, if it has one
    client = client or _default_client(route)
    card = candidate_card(resume, today)
    ratings: dict[str, Rating] = {}
    for i in range(0, len(head), batch_size):
        ratings.update(rate_batch(head[i:i + batch_size], card, client=client, model=model, usage=usage))

    reranked = sorted(head, key=lambda s: (s.demotion is None, ratings[s.ref].score, s.score), reverse=True)
    ranked = [Ranked(s, ratings[s.ref].score, ratings[s.ref].reason) for s in reranked]
    return RankResult(ranked + [Ranked(s) for s in tail], usage, model)


def shortlist_json(result: RankResult) -> list[dict[str, Any]]:
    """The ranked list as plain data: what Phase 5's graph state will carry."""
    return [{"rank": i, "ref": r.ref, "id": r.post.id, "title": r.post.title,
             "company": r.post.company, "url": r.post.url, "score": r.score, "reason": r.reason,
             "years_asked": r.stage1.years_asked, "demotion": r.stage1.demotion,
             "similarity": round(r.stage1.score, 4),
             "matches": [{"atom": m.atom_id, "sentence": m.sentence, "similarity": round(m.similarity, 4),
                          **({"lift": round(m.lift, 4)} if m.lift is not None else {})}
                         for m in r.stage1.matches]}
            for i, r in enumerate(result.ranked, 1)]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import warnings

    from dotenv import load_dotenv

    from src.tools.jobs import fetch_jobs

    load_dotenv()

    ap = argparse.ArgumentParser(description="Rank job postings against the profile.")
    ap.add_argument("keyword")
    ap.add_argument("--location", default=None)
    ap.add_argument("--pages", type=int, default=1,
                    help="pages of postings to rank; served from data/jobs_cache when cached")
    ap.add_argument("--top", type=int, default=5, help="how many to print in detail")
    ap.add_argument("--rerank", type=int, default=RERANK_TOP, help="postings sent to stage 2")
    ap.add_argument("--stage1-only", action="store_true", help="embeddings only; no API call")
    ap.add_argument("--model", default=None, help="override RANKER_MODEL")
    ap.add_argument("--embed", default=DEFAULT_EMBEDDER,
                    help=f"stage-1 embedder: {', '.join(EMBEDDERS)}, st:<hf id> or ollama:<tag>")
    ap.add_argument("--no-titles", action="store_true", help="don't count the title as a sentence")
    ap.add_argument("--center", action="store_true", help="score atoms above their pool average")
    ap.add_argument("--exclude-atoms", default=None,
                    help="comma-separated atom ids to leave out of stage 1 (default: RANKER_EXCLUDE_ATOMS; "
                         '"" for none)')
    ap.add_argument("--no-gate", action="store_true", help="turn the level gate off")
    ap.add_argument("--gate-slack", type=float, default=LEVEL_SLACK,
                    help=f"years a posting may ask beyond yours before it is demoted (default {LEVEL_SLACK})")
    ap.add_argument("--profile", default=ROOT / "data" / "profile.yaml")
    ap.add_argument("--resume", default=ROOT / "assets" / "resume.json")
    ap.add_argument("--out", default=ROOT / "output" / "shortlist.json")
    args = ap.parse_args()
    # With no flags this is default_options(): the flags' defaults are its values.
    options = Stage1Options(titles=not args.no_titles, center=args.center,
                            exclude=ranker_exclude_atoms(args.exclude_atoms),
                            level_slack=None if args.no_gate else args.gate_slack)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        posts = fetch_jobs(args.keyword, args.location, pages=args.pages)
    for w in caught:
        print(f"warning: {w.message}")
    if not posts:
        ap.error("no postings to rank")

    print(f"ranking {len(posts)} postings (stage 1: {args.embed}, {options.label}"
          f"{'' if args.stage1_only else f'; stage 2: top {args.rerank}'})")
    resume = Resume.load(args.resume)
    if options.level_slack is not None:
        you = experience_years(resume)
        print(f"level gate: you have {you:.1f} years; postings asking more than "
              f"{you + options.level_slack:.1f} are demoted")
    result = rank(posts, Profile.load(args.profile), resume,
                  embedder=make_embedder(args.embed), options=options,
                  model=args.model, rerank_top=0 if args.stage1_only else args.rerank)
    if result.model:
        print(f"stage 2: {result.model}, tokens in/out "
              f"{result.usage['prompt_tokens']}/{result.usage['completion_tokens']}")

    print(f"\n{'#':>2}  {'ref':<8}  {'llm':>3}  {'sim':>5}  {'yrs':>3}  {'title':<46} company")
    for i, r in enumerate(result.ranked, 1):
        llm = "-" if r.score is None else str(r.score)
        asked = years_asked(r.post)                  # shown even with the gate off
        yrs = "-" if asked is None else str(asked)
        demoted = f"  [demoted: {r.stage1.demotion}]" if r.stage1.demotion else ""
        print(f"{i:>2}  {r.ref:<8}  {llm:>3}  {r.stage1.score:5.3f}  {yrs:>3}  {r.post.title[:46]:<46} "
              f"{r.post.company[:28]}{demoted}")
    for i, r in enumerate(result.ranked[:args.top], 1):
        print(f"\n{i}. {r.post.title} — {r.post.company}")
        if r.reason:
            print(f"   {r.score}/10: {r.reason}")
        for m in r.stage1.matches:
            lift = f" (+{m.lift:.3f})" if m.lift is not None and m.lift >= 0 else \
                   (f" ({m.lift:.3f})" if m.lift is not None else "")
            print(f"   {m.similarity:.3f}{lift}  {m.atom_id:<40} \"{m.sentence[:90]}\"")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(shortlist_json(result), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")
