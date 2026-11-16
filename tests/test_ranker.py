"""Phase 4 Ranker tests. No model download, no API call: a bag-of-words
embedder stands in for MiniLM (similarity = shared words, deterministic),
and a stub client with the slice of the OpenAI SDK the Ranker touches
returns canned ratings.

Run from the repo root:  pytest tests/test_ranker.py -v
"""

from __future__ import annotations

import json
import zlib
from datetime import date
from types import SimpleNamespace

import numpy as np
import pytest

from src.agents.ranker import (
    LEVEL_SLACK,
    CachedEmbedder,
    OllamaEmbedder,
    RankerError,
    Scored,
    Stage1Options,
    atom_text,
    build_prompt,
    candidate_card,
    default_options,
    experience_years,
    level_gate,
    output_schema,
    posting_card,
    rank,
    ranking_atoms,
    score_postings,
    shortlist_json,
    split_sentences,
    years_asked,
)
from src.atoms import Profile
from src.config import ModelRoute, ranker_exclude_atoms, ranker_route
from src.models import Resume
from src.tools.evidence import tokenize
from src.tools.jobs import JobPost

ROUTE = ModelRoute(base_url="http://stub", key_env="STUB_KEY", model="stub-model")
BOILERPLATE = ("We are a global leader in materials engineering with offices in forty countries. "
               "We offer flexible hours, a learning budget and a vibrant culture. ")


class BagOfWords:
    """Deterministic stand-in for MiniLM: hashed word counts, unit length.
    Records how many texts it embedded, and whether as queries."""

    name = "bag-of-words"

    def __init__(self) -> None:
        self.calls = 0
        self.embedded: list[tuple[bool, str]] = []

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        self.calls += 1
        self.embedded += [(query, t) for t in texts]
        m = np.zeros((len(texts), 256))
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                m[i, zlib.crc32(tok.encode()) % 256] += 1
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return m / norms


def _atom(id: str, section: str = "experience", **over) -> dict:
    return {"id": id, "section": section, "source": "Acme", "title": id, "summary": "nothing here",
            "year": "2024", **over}


PROFILE = Profile.model_validate({"atoms": [
    _atom("fraud", title="Fraud detection rules", summary="Built fraud detection rules and risk models.",
          skills=["SQL", "Python"], keywords=["fraud detection", "risk analytics"]),
    _atom("logistics", title="Shipping cost savings", summary="Cut last mile delivery shipping cost.",
          keywords=["logistics", "supply chain"]),
    _atom("nlp", section="project", title="Tweet classification", summary="Fine-tuned BERT for text classification.",
          skills=["NLP", "transformers"]),
    _atom("degree", section="education", title="B.Tech electronics", summary="Engineering degree in electronics."),
]})

RESUME = Resume.model_validate({
    "contact": {"name": "Test Person", "phone": "+1 555 0100", "email": "t@example.com", "links": []},
    "summary": "Data scientist working on fraud prevention.",
    "experience": [
        {"company": "PayPal", "title": "Data Scientist", "location": "Bangalore", "start": "09/2025",
         "end": "Present", "bullets": ["Built rules."]},
        {"company": "Target", "title": "Analyst", "location": "Bangalore", "start": "09/2022",
         "end": "05/2025", "bullets": ["Cut costs."]},
    ],
    "skills": [{"category": "Technical", "items": ["SQL", "Python"]}],
    "education": [{"degree": "B.Tech", "institution": "U", "location": "City",
                   "start": "07/2017", "end": "04/2021"}],
})


def _post(id: str, title: str, body: str) -> JobPost:
    return JobPost(id=f"jsearch:{id}", source="jsearch", title=title, company=f"Co {id}",
                   description=(BOILERPLATE * 2) + body)


POSTS = [
    _post("ship", "Supply Chain Data Scientist", "You will reduce last mile delivery shipping cost across logistics."),
    # Matches two atoms (fraud, nlp): under the top-atoms mean it should beat a
    # posting that matches one atom (logistics), however well.
    _post("fraud", "Fraud Data Scientist", "Build fraud detection rules and risk models in SQL and Python.\n"
                                           "Fine-tune BERT transformers for text classification.\n"
                                           "5+ years of experience in risk analytics."),
    _post("none", "Chip Designer", "Design analog circuits for wafer inspection tools."),
]


# ------------------------------------------------------------ sentences

def test_split_sentences_handles_bullets_headings_and_run_on_text():
    text = ("What you'll do\n"                             # heading: too short, dropped
            "• Build fraud detection models for payments.\n"
            "- Build fraud detection models for payments.\n"   # repeat, kept once
            "2) Partner with product teams on metrics. Own the roadmap for risk analytics.")
    assert split_sentences(text) == ["Build fraud detection models for payments.",
                                     "Partner with product teams on metrics.",
                                     "Own the roadmap for risk analytics."]


def test_split_sentences_cuts_overlong_runs_inside_the_model_window():
    """One posting has no line breaks at all; nothing may exceed MiniLM's window."""
    run_on = "word " * 300
    pieces = split_sentences(run_on)
    assert len(pieces) > 1 and all(len(p) <= 400 for p in pieces)


# -------------------------------------------------------------- stage 1

def test_best_matching_posting_ranks_first_and_explains_itself():
    scored = score_postings(POSTS, ranking_atoms(PROFILE), BagOfWords())
    assert [s.post.id for s in scored][0] == "jsearch:fraud"
    top = scored[0].matches[0]
    assert top.atom_id == "fraud" and "fraud detection rules" in top.sentence


def test_boilerplate_never_counts_and_education_is_excluded():
    scored = score_postings(POSTS, ranking_atoms(PROFILE), BagOfWords())
    for s in scored:
        assert all("materials engineering" not in m.sentence for m in s.matches if m.similarity > 0.3)
        assert "degree" not in {m.atom_id for m in s.matches}
    assert [a.id for a in ranking_atoms(PROFILE)] == ["fraud", "logistics", "nlp"]


def test_score_is_the_mean_of_the_top_atoms():
    scored = score_postings(POSTS, ranking_atoms(PROFILE), BagOfWords(), Stage1Options(top_atoms=2))
    for s in scored:
        assert len(s.matches) == 2
        assert s.score == pytest.approx(np.mean([m.similarity for m in s.matches]))
        assert s.matches[0].similarity >= s.matches[1].similarity


def test_atom_text_carries_claim_and_tags():
    text = atom_text(ranking_atoms(PROFILE)[0])
    assert text.startswith("Fraud detection rules.") and "Skills: SQL, Python." in text
    assert "Topics: fraud detection, risk analytics." in text


# ---------------------------------------------------------------- cards

def test_candidate_card_states_roles_dates_and_total_years():
    card = candidate_card(RESUME, today=date(2026, 9, 1))
    assert "Total experience: 3.7 years" in card                      # 12 + 32 months
    assert "- Data Scientist, PayPal, 09/2025 to Present (1.0 years)" in card
    assert card.splitlines()[0] == "Summary: Data scientist working on fraud prevention."


def test_posting_card_is_compact_and_surfaces_the_years_asked():
    scored = score_postings(POSTS, ranking_atoms(PROFILE), BagOfWords())
    card = posting_card(next(s for s in scored if s.post.id == "jsearch:fraud"), n_sentences=2)
    lines = card.splitlines()
    assert lines[0].startswith("[") and "Fraud Data Scientist — Co fraud" in lines[0]
    assert lines[1] == "Experience asked: 5+ years of experience in risk analytics."
    assert len(lines) == 4                                             # header, years, 2 sentences
    assert len(card) < len(POSTS[1].description)


def test_prompt_lists_every_posting_ref():
    scored = score_postings(POSTS, ranking_atoms(PROFILE), BagOfWords())
    prompt = build_prompt("CARD", scored)
    assert all(f"[{s.ref}]" in prompt for s in scored) and prompt.startswith("# Candidate\n\nCARD")


def test_output_schema_is_decoder_ready():
    schema = output_schema()
    assert schema["additionalProperties"] is False
    rating = schema["$defs"]["Rating"]
    assert set(rating["required"]) == {"ref", "score", "reason"}
    assert "maxLength" not in json.dumps(schema) and "minimum" not in json.dumps(schema)


# -------------------------------------------------------------- stage 2

def _response(text: str, finish_reason: str = "stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish_reason)],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40))


def rating_client(scores: dict[str, int] | None = None, *, script: list[str] | None = None):
    """Rates whatever refs the prompt contains (score from `scores` by post id,
    default 5), or plays `script` verbatim. Records every call."""
    calls: list[dict] = []

    def create(**kw):
        calls.append(kw)
        if script:
            return _response(script[len(calls) - 1])
        prompt = kw["messages"][1]["content"]
        ratings = [{"ref": s.ref, "score": (scores or {}).get(s.post.id, 5), "reason": "fits"}
                   for s in _SCORED_BY_REF.values() if f"[{s.ref}]" in prompt]
        return _response(json.dumps({"ratings": ratings}))

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    client.calls = calls
    return client


_SCORED_BY_REF = {s.ref: s for s in score_postings(POSTS, ranking_atoms(PROFILE), BagOfWords())}
REF = {s.post.id: ref for ref, s in _SCORED_BY_REF.items()}


def test_stage1_only_makes_no_api_call():
    result = rank(POSTS, PROFILE, RESUME, embedder=BagOfWords(), rerank_top=0,
                  client=SimpleNamespace())       # would fail if touched
    assert result.model is None and all(r.score is None for r in result.ranked)
    assert result.ranked[0].post.id == "jsearch:fraud"


def test_rerank_reorders_the_head_and_keeps_the_tail_in_stage1_order():
    """The LLM prefers logistics; the unreranked tail follows, unrated."""
    client = rating_client({"jsearch:ship": 9, "jsearch:fraud": 6})
    result = rank(POSTS, PROFILE, RESUME, embedder=BagOfWords(), client=client, route=ROUTE,
                  rerank_top=2, batch_size=5)
    assert [r.post.id for r in result.ranked] == ["jsearch:ship", "jsearch:fraud", "jsearch:none"]
    assert [r.score for r in result.ranked] == [9, 6, None]
    assert result.model == "stub-model" and result.usage == {"prompt_tokens": 100, "completion_tokens": 40}
    call = client.calls[0]
    assert call["temperature"] == 0.0 and call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["name"] == "job_ratings"


def test_ties_in_llm_score_fall_back_to_stage1():
    result = rank(POSTS, PROFILE, RESUME, embedder=BagOfWords(), client=rating_client(), route=ROUTE)
    assert [r.post.id for r in result.ranked][0] == "jsearch:fraud"   # all 5s: stage 1 decides


def test_postings_are_rated_in_batches():
    client = rating_client()
    rank(POSTS, PROFILE, RESUME, embedder=BagOfWords(), client=client, route=ROUTE, batch_size=2)
    assert len(client.calls) == 2                                      # 3 postings, batches of 2


def test_a_missing_rating_is_retried_with_the_problem():
    only_one = json.dumps({"ratings": [{"ref": REF["jsearch:fraud"], "score": 7, "reason": "x"}]})
    full = json.dumps({"ratings": [{"ref": REF[p.id], "score": 5, "reason": "x"} for p in POSTS]})
    client = rating_client(script=[only_one, full])
    result = rank(POSTS, PROFILE, RESUME, embedder=BagOfWords(), client=client, route=ROUTE)
    assert len(client.calls) == 2 and all(r.score == 5 for r in result.ranked)
    feedback = client.calls[1]["messages"][-1]["content"]
    assert f"posting [{REF['jsearch:ship']}] has no rating" in feedback


def test_unknown_refs_and_bad_scores_fail_after_one_retry():
    bad = json.dumps({"ratings": [{"ref": "zzzzzzzz", "score": 11, "reason": "x"}]})
    client = rating_client(script=[bad, bad])
    with pytest.raises(RankerError) as exc:
        rank(POSTS, PROFILE, RESUME, embedder=BagOfWords(), client=client, route=ROUTE)
    assert len(client.calls) == 2 and any("score" in p for p in exc.value.problems)


def test_empty_response_is_not_retried():
    client = rating_client(script=[""])
    with pytest.raises(RankerError, match="empty response"):
        rank(POSTS, PROFILE, RESUME, embedder=BagOfWords(), client=client, route=ROUTE)
    assert len(client.calls) == 1


def test_shortlist_json_carries_rank_reason_and_evidence():
    result = rank(POSTS, PROFILE, RESUME, embedder=BagOfWords(), client=rating_client(), route=ROUTE)
    first = shortlist_json(result)[0]
    assert first["rank"] == 1 and first["score"] == 5 and first["reason"] == "fits"
    assert first["matches"][0]["atom"] == "fraud" and first["ref"] == REF["jsearch:fraud"]


def test_ranker_route_defaults_and_overrides(monkeypatch):
    for var in ("RANKER_BASE_URL", "RANKER_KEY_ENV", "RANKER_MODEL"):
        monkeypatch.delenv(var, raising=False)
    route = ranker_route()
    assert route.base_url.startswith("https://api.groq.com") and route.key_env == "GROQ_API_KEY"
    monkeypatch.setenv("RANKER_MODEL", "some/small-model")
    assert ranker_route().model == "some/small-model"


# ------------------------------------------------------------------ eval

def test_precision_at_k_strict_lenient_and_best_possible():
    from src.tools.rank_eval import best_possible, precision_at_k
    labels = {"a": "yes", "b": "maybe", "c": "no", "d": "yes", "e": "no", "f": "no"}
    ranking = ["a", "b", "c", "d", "e", "f"]
    assert precision_at_k(ranking, labels, k=5) == pytest.approx(2 / 5)
    assert precision_at_k(ranking, labels, k=5, accept=("yes", "maybe")) == pytest.approx(3 / 5)
    assert best_possible(labels, k=5) == pytest.approx(2 / 5)


def test_load_labels_rejects_a_half_labelled_file(tmp_path):
    from src.tools.rank_eval import load_labels
    path = tmp_path / "x.labels.yaml"
    path.write_text("query: {keyword: q}\npostings:\n"
                    "- {ref: aaaaaaaa, title: A, label: yes}\n"
                    "- {ref: bbbbbbbb, title: B, label: null}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="1 posting.*bbbbbbbb"):
        load_labels(path)
    path.write_text(path.read_text(encoding="utf-8").replace("label: null", "label: Maybe"), encoding="utf-8")
    query, labels = load_labels(path)
    assert query == {"keyword": "q"} and labels == {"aaaaaaaa": "yes", "bbbbbbbb": "maybe"}


def test_load_labels_reads_bare_yes_and_no(tmp_path):
    """YAML 1.1 turns a bare yes/no into True/False — exactly what the labels
    file asks you to type. They must come back as labels, not errors."""
    from src.tools.rank_eval import load_labels
    path = tmp_path / "x.labels.yaml"
    path.write_text("postings:\n- {ref: a, label: yes}\n- {ref: b, label: no}\n- {ref: c, label: maybe}\n",
                    encoding="utf-8")
    assert load_labels(path)[1] == {"a": "yes", "b": "no", "c": "maybe"}


def test_rank_eval_reads_the_cache_and_never_the_network(tmp_path):
    from src.tools.jobs import JobQuery, ProviderError
    from src.tools.rank_eval import CacheOnlyProvider
    with pytest.raises(ProviderError, match="not in data/jobs_cache"):
        CacheOnlyProvider().fetch_raw(JobQuery(keyword="q"))


# ------------------------------------------------- stage-1 options (v2)

def test_title_counts_when_the_body_is_boilerplate():
    """Infosys, run 1: the description was all company boilerplate and only
    the title named the job, so stage 1 ranked it last. Titles now count."""
    blank = _post("gen", "Fraud Detection Rules Analyst", "Join our digital transformation journey today.")
    atoms = ranking_atoms(PROFILE)
    with_titles = score_postings([blank], atoms, BagOfWords(), Stage1Options(titles=True))[0]
    without = score_postings([blank], atoms, BagOfWords(), Stage1Options(titles=False))[0]
    assert with_titles.score > without.score
    assert with_titles.matches[0].sentence == "Fraud Detection Rules Analyst"
    assert posting_card(with_titles).count("Fraud Detection Rules Analyst") == 1   # header only, not a bullet too


def test_excluded_atoms_never_match_and_typos_are_errors():
    atoms = ranking_atoms(PROFILE, exclude=frozenset({"logistics"}))
    assert [a.id for a in atoms] == ["fraud", "nlp"]
    for s in score_postings(POSTS, atoms, BagOfWords()):
        assert "logistics" not in {m.atom_id for m in s.matches}
    with pytest.raises(ValueError, match="unknown atom id"):
        ranking_atoms(PROFILE, exclude=frozenset({"logistcs"}))


def test_centering_neutralises_an_atom_that_matches_everything():
    """The soft-skill problem: an atom matching the collaboration line every
    posting has took a top slot everywhere. Centered, its lift is ~0 in every
    posting, so a posting's score comes from what is distinctive about it."""
    collab = "Collaborate with product and engineering stakeholders across teams."
    profile = Profile.model_validate({"atoms": [
        _atom("soft", title="Stakeholder collaboration", summary=collab),
        _atom("fraud", title="Fraud detection rules", summary="Built fraud detection rules and risk models."),
        _atom("ship", title="Shipping cost", summary="Cut last mile delivery shipping cost."),
    ]})
    posts = [_post("f", "Analyst", f"{collab}\nBuild fraud detection rules and risk models."),
             _post("s", "Analyst", f"{collab}\nReduce last mile delivery shipping cost."),
             _post("x", "Analyst", f"{collab}\nDesign analog circuits for wafer tools.")]
    options = Stage1Options(center=True, top_atoms=1)
    scored = score_postings(posts, ranking_atoms(profile), BagOfWords(), options)
    assert {s.post.id for s in scored[:2]} == {"jsearch:f", "jsearch:s"}
    assert all(m.atom_id != "soft" for s in scored[:2] for m in s.matches)
    soft_lifts = [m.lift for s in score_postings(posts, ranking_atoms(profile), BagOfWords(),
                                                 Stage1Options(center=True, top_atoms=3))
                  for m in s.matches if m.atom_id == "soft"]
    assert soft_lifts and all(abs(l) < 1e-9 for l in soft_lifts)
    uncentered = score_postings(posts, ranking_atoms(profile), BagOfWords(), Stage1Options(top_atoms=1))
    assert all(s.matches[0].atom_id == "soft" for s in uncentered)          # the problem, reproduced


def test_atoms_are_embedded_as_queries_and_sentences_as_documents():
    emb = BagOfWords()
    score_postings(POSTS[:1], ranking_atoms(PROFILE), emb)
    queries = [t for q, t in emb.embedded if q]
    assert queries == [atom_text(a) for a in ranking_atoms(PROFILE)]
    assert all(not q for q, t in emb.embedded if t not in queries)


def test_options_label_names_each_change():
    assert Stage1Options(titles=False).label == "no titles"
    assert Stage1Options(exclude=frozenset({"a"}), center=True).label == "titles, -1 atom(s), centered"


# ------------------------------------------------------------ embedders

def test_cached_embedder_embeds_each_text_once():
    inner = BagOfWords()
    cached = CachedEmbedder(inner)
    first = cached.encode(["a b", "c d"])
    second = cached.encode(["c d", "a b", "e f"])
    assert np.allclose(first[0], second[1]) and len(inner.embedded) == 3
    cached.encode(["a b"], query=True)                  # same text as a query is a different vector
    assert len(inner.embedded) == 4 and cached.name == "bag-of-words"


def test_make_embedder_resolves_names_and_prefixes(monkeypatch):
    """A fake sentence_transformers module: the real one imports torch
    (seconds, ~2 GB) and would download models."""
    import sys
    import types
    made: list[str] = []

    class FakeST:
        def __init__(self, model_id):
            made.append(model_id)

        def encode(self, texts, **kw):
            FakeST.last_kw = kw
            return np.ones((len(texts), 4))

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=FakeST))
    from src.agents.ranker import QWEN_INSTRUCTION, make_embedder
    qwen = make_embedder("qwen3-0.6b")
    assert made == ["Qwen/Qwen3-Embedding-0.6B"] and qwen.name == "qwen3-0.6b"
    qwen.encode(["x"], query=True)
    assert FakeST.last_kw["prompt"] == QWEN_INSTRUCTION         # the instruction goes on queries...
    qwen.encode(["x"])
    assert "prompt" not in FakeST.last_kw                       # ...and not on posting sentences
    assert np.allclose(np.linalg.norm(qwen.encode(["x", "y"]), axis=1), 1.0)
    make_embedder("st:some/model")
    assert made[-1] == "some/model"
    assert isinstance(make_embedder("qwen3-4b"), OllamaEmbedder)
    assert make_embedder("ollama:nomic-embed-text").name == "ollama:nomic-embed-text"
    with pytest.raises(ValueError, match="unknown embedder"):
        make_embedder("gpt-embed")


def test_ollama_embedder_batches_prefixes_queries_and_normalises():
    import httpx
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        return httpx.Response(200, json={"embeddings": [[3.0, 4.0]] * len(body["input"])})

    emb = OllamaEmbedder("qwen3-embedding:4b", query_prompt="Q: ", base_url="http://ollama",
                         client=httpx.Client(transport=httpx.MockTransport(handler)))
    emb.BATCH = 2
    out = emb.encode(["a", "b", "c"], query=True)
    assert [b["input"] for b in seen] == [["Q: a", "Q: b"], ["Q: c"]]
    assert seen[0]["model"] == "qwen3-embedding:4b"
    assert out.shape == (3, 2) and np.allclose(out[0], [0.6, 0.8])
    emb.encode(["d"])
    assert seen[-1]["input"] == ["d"]                                     # documents: no prefix


def test_ollama_errors_say_what_to_do():
    import httpx
    missing = OllamaEmbedder("qwen3-embedding:4b", base_url="http://ollama", client=httpx.Client(
        transport=httpx.MockTransport(lambda r: httpx.Response(404, text="model not found"))))
    with pytest.raises(RuntimeError, match="ollama pull qwen3-embedding:4b"):
        missing.encode(["a"])

    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    down = OllamaEmbedder("m", base_url="http://ollama", client=httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(RuntimeError, match="not reachable"):
        down.encode(["a"])


# ---------------------------------------------------------- eval (v2)

LABELS = {"a": "yes", "b": "skip", "c": "no", "d": "yes", "e": "maybe", "f": "no", "g": "yes"}


def test_skipped_postings_neither_help_nor_hurt_precision():
    from src.tools.rank_eval import judged, precision_at_k
    ranking = ["b", "a", "c", "d", "e", "f", "g"]
    assert judged(ranking, LABELS) == ["a", "c", "d", "e", "f", "g"]
    assert precision_at_k(ranking, LABELS, k=5) == pytest.approx(2 / 5)    # a, c, d, e, f judged
    assert precision_at_k(ranking, LABELS, k=5, accept=("yes", "maybe")) == pytest.approx(3 / 5)


def test_stage1_recall_counts_raw_seats():
    """Stage 2 sees the raw top 15, skipped postings included."""
    from src.tools.rank_eval import recall_at_k
    assert recall_at_k(["b", "a", "c", "d", "e", "f", "g"], LABELS, k=3) == pytest.approx(1 / 3)   # only a
    assert recall_at_k(["a", "d", "g", "b"], LABELS, k=3) == pytest.approx(1.0)


def test_baselines_ignore_skips():
    from src.tools.rank_eval import best_possible, random_expected
    assert random_expected(LABELS) == pytest.approx(3 / 6)
    assert best_possible(LABELS, k=5) == pytest.approx(3 / 5)
    assert best_possible(LABELS, k=5, accept=("yes", "maybe")) == pytest.approx(4 / 5)


def test_load_labels_accepts_skip():
    from src.tools.rank_eval import load_labels
    import tempfile, pathlib
    path = pathlib.Path(tempfile.mkdtemp()) / "x.labels.yaml"
    path.write_text("postings:\n- {ref: a, label: skip}\n- {ref: b, label: yes}\n", encoding="utf-8")
    assert load_labels(path)[1] == {"a": "skip", "b": "yes"}


def test_compare_table_has_one_row_per_setup():
    from src.tools.rank_eval import compare_table
    rows = [("minilm", "no titles", ["c", "f", "a", "d", "g"]), ("qwen3-0.6b", "titles", ["a", "d", "g", "e", "c"])]
    table = compare_table(rows, LABELS, k=5)
    assert table[0].startswith("| embedder | stage-1 variant | recall@15")
    assert table[2] == "| minilm | no titles | 1.00 | 0.60 | 0.60 |"
    assert table[3] == "| qwen3-0.6b | titles | 1.00 | 0.60 | 0.80 |"


# ------------------------------------------------------ level gate (v3)

@pytest.mark.parametrize("text, expected", [
    ("7+ years of professional experience in data science", 7),
    ("Experience: 12 – 16 Years", 12),                        # two digits, en dash
    ("12 + years of industry experience", 12),                # "12+" is 12, never 1
    ("Experience: 7 11 Years", 7),                            # a range whose dash was dropped
    ("Experience: 11–15+ Years", 11),
    ("5-8 years of professional experience", 5),              # a range's lower bound
    ("8+ years in data science / ML with at least 2-3 years in GenAI", 8),   # the largest bound
    ("6+ years with a Masters Degree or 4+ years with a PhD", 6),
    ("Minimum 5 years in analytics", 5),                      # bare, but says it is asked
    ("Over the last 5 years, Gartner has ranked our supply chain #1.", None),   # company history
    ("We've done that for 90 years.", None),                  # out of range
    ("Relevant years of experience with regression methods.", None),   # no number
])
def test_years_asked_reads_the_largest_lower_bound(text, expected):
    assert years_asked(_post("y", "Data Scientist", text)) == expected


def test_years_asked_reads_the_title_too():
    """ZecData, 2026-09-27 pool: the years were in the title and nowhere else."""
    assert years_asked(_post("z", "Data Scientist(Exp : 6-8yrs)", "Build models.")) == 6
    assert years_asked(_post("z", "Senior Data Scientist | 6+ Years", "4+ years with a PhD")) == 6


def test_experience_years_is_the_card_total():
    assert experience_years(RESUME, today=date(2026, 9, 1)) == pytest.approx(44 / 12)   # 12 + 32 months
    assert "Total experience: 3.7 years" in candidate_card(RESUME, today=date(2026, 9, 1))


GATE_POSTS = [
    _post("lead", "Lead Fraud Data Scientist", "Build fraud detection rules and risk models in SQL and Python.\n"
                                               "Fine-tune BERT transformers for text classification.\n"
                                               "10+ years of experience in risk analytics."),
    _post("ship", "Supply Chain Data Scientist", "Reduce last mile delivery shipping cost across logistics.\n"
                                                 "4-6 years of experience."),
    _post("mid", "Data Scientist", "Fine-tune BERT transformers for text classification.\n"
                                   "Experience: 5+ years"),
    _post("open", "Chip Designer", "Design analog circuits for wafer inspection tools."),
]
GATE_REF = {p.id: p.slug[-8:] for p in GATE_POSTS}


def test_level_gate_demotes_below_the_rest_and_keeps_each_groups_order():
    scored = score_postings(GATE_POSTS, ranking_atoms(PROFILE), BagOfWords())
    assert scored[0].post.id == "jsearch:lead"                     # best on topic...
    gated = level_gate(scored, candidate_years=3.7, slack=1.5)     # ...but asks 10 against 5.2
    assert [s.post.id for s in gated] == [s.post.id for s in scored if s.post.id != "jsearch:lead"] + ["jsearch:lead"]
    assert gated[-1].years_asked == 10 and gated[-1].demotion == "asks 10+ yrs (you: 3.7)"
    kept = {s.post.id: s for s in gated[:-1]}
    assert kept["jsearch:ship"].years_asked == 4 and kept["jsearch:ship"].demotion is None
    assert kept["jsearch:open"].years_asked is None                # states nothing: never demoted
    assert kept["jsearch:mid"].demotion is None                    # 5 <= 3.7 + 1.5


def test_level_gate_boundary_is_strictly_more_than_yours_plus_slack():
    scored = score_postings(GATE_POSTS, ranking_atoms(PROFILE), BagOfWords())
    assert [s.post.id for s in level_gate(scored, 3.5, 1.5) if s.demotion] == ["jsearch:lead"]   # 5 == 5.0 stays
    assert {s.post.id for s in level_gate(scored, 3.4, 1.5) if s.demotion} == {"jsearch:lead", "jsearch:mid"}


def _ref_client(scores: dict[str, int]):
    """Rates each ref found in the prompt with scores[ref]."""
    def create(**kw):
        prompt = kw["messages"][1]["content"]
        return _response(json.dumps({"ratings": [{"ref": ref, "score": score, "reason": "r"}
                                                 for ref, score in scores.items() if f"[{ref}]" in prompt]}))
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))


def test_rank_gates_stage1_and_stage2_orders_within_tiers():
    """Stage 2 loves the over-level posting (10/10); it still ranks last
    among the reranked, because the demotion holds through stage 2."""
    scores = {GATE_REF["jsearch:lead"]: 10, GATE_REF["jsearch:ship"]: 4,
              GATE_REF["jsearch:mid"]: 6, GATE_REF["jsearch:open"]: 2}
    options = Stage1Options(level_slack=1.5)
    result = rank(GATE_POSTS, PROFILE, RESUME, embedder=BagOfWords(), options=options,
                  client=_ref_client(scores), route=ROUTE, today=date(2026, 9, 1))
    assert [r.post.id for r in result.ranked] == ["jsearch:mid", "jsearch:ship", "jsearch:open", "jsearch:lead"]
    assert result.ranked[-1].score == 10
    ungated = rank(GATE_POSTS, PROFILE, RESUME, embedder=BagOfWords(), client=_ref_client(scores),
                   route=ROUTE, today=date(2026, 9, 1))
    assert ungated.ranked[0].post.id == "jsearch:lead"                     # gate off: the old behaviour
    assert all(r.stage1.demotion is None and r.stage1.years_asked is None for r in ungated.ranked)


def test_stage1_only_with_the_gate_puts_demoted_postings_last():
    result = rank(GATE_POSTS, PROFILE, RESUME, embedder=BagOfWords(), options=Stage1Options(level_slack=1.5),
                  rerank_top=0, today=date(2026, 9, 1))
    assert result.ranked[-1].post.id == "jsearch:lead" and result.model is None
    row = shortlist_json(result)[-1]
    assert row["years_asked"] == 10 and row["demotion"] == "asks 10+ yrs (you: 3.7)"
    assert shortlist_json(result)[0]["demotion"] is None


def test_options_label_names_the_gate():
    assert Stage1Options(level_slack=1.5).label == "titles, level gate +1.5y"
    assert Stage1Options(exclude=frozenset({"a"}), level_slack=2).label == "titles, -1 atom(s), level gate +2y"


def test_default_options_are_the_chosen_setup(monkeypatch):
    monkeypatch.setenv("RANKER_EXCLUDE_ATOMS", " soft-skill-atom , other-atom ,")
    opts = default_options()
    assert opts.titles and not opts.center and opts.level_slack == LEVEL_SLACK == 1.5
    assert opts.exclude == frozenset({"soft-skill-atom", "other-atom"})
    monkeypatch.delenv("RANKER_EXCLUDE_ATOMS")
    assert default_options().exclude == frozenset()
    assert Stage1Options().level_slack is None and Stage1Options().exclude == frozenset()   # never reads env


def test_ranker_exclude_atoms_flag_beats_env(monkeypatch):
    monkeypatch.setenv("RANKER_EXCLUDE_ATOMS", "from-env")
    assert ranker_exclude_atoms() == frozenset({"from-env"})
    assert ranker_exclude_atoms("a,b") == frozenset({"a", "b"})
    assert ranker_exclude_atoms("") == frozenset()                          # --exclude-atoms "" = none


def test_default_embedder_is_qwen3_small():
    from src.agents.ranker import DEFAULT_EMBEDDER, EMBEDDERS
    assert DEFAULT_EMBEDDER == "qwen3-0.6b" and DEFAULT_EMBEDDER in EMBEDDERS


def test_gate_report_counts_demotions_by_label():
    from src.tools.rank_eval import gate_report
    titles = {r: f"T{r}" for r in "acf"}
    lines = gate_report([("c", "asks 8+ yrs (you: 5.2)"), ("f", "asks 12+ yrs (you: 5.2)")], LABELS, titles)
    assert lines[0] == "## level gate — demoted 2: 2 no"
    assert "asks 12+ yrs (you: 5.2)" in lines[2] and lines[2].split()[:2] == ["no", "f"]
    assert gate_report([], LABELS, titles) == ["## level gate — demoted nothing"]


def test_rerank_requests_carry_the_models_settings():
    """Stage 2 uses config.MODEL_SETTINGS like the Extractor: explicit
    reasoning effort for gpt-oss, the same values the acceptance run got."""
    client = rating_client()
    rank(POSTS, PROFILE, RESUME, embedder=BagOfWords(), client=client, route=ROUTE, model="openai/gpt-oss-120b")
    assert client.calls[0]["reasoning_effort"] == "medium" and client.calls[0]["temperature"] == 0.0
