# resume-agent

A multi-agent pipeline that finds relevant job postings, ranks them, and
generates a tailored one-page CV as PDF for each one.

## Architecture

The deterministic backbone (Phases 1–3):

```
job brief ──▶ retrieve ──▶ Tailor (Claude) ──▶ schema gate ──▶ verify() ──▶ fit loop ──▶ PDF
(JobBrief)   (atoms +     structured output   (src/models.py)  grounding:   render → pages
              covers)     + change log                          ids, sections, + fill → scale
                                                                figures, skills  or cut
```

Agents edit *structured data*, never markup — a bad edit fails schema
validation instead of breaking the render. Hard constraints are enforced
by deterministic tools, not model judgment: JSON shape by constrained
decoding, field constraints by Pydantic, grounding by string checks
against the evidence store, page count by pypdf. A single `--scale` CSS
variable is the fit loop's only layout knob.

Every claim on a tailored CV traces to an *atom* in `data/profile.yaml` —
one verified fact per entry. The Tailor may rewrite and reorder, and may
insert only atoms it was offered, citing each by id in a change log.

The front of the pipeline (Phase 4) finds and ranks the jobs, cheap
before expensive:

```
keyword ──▶ Sourcer ──▶ Ranker stage 1 ──▶ level gate ──▶ Ranker stage 2 ──▶ shortlist ──▶ Extractor ──▶ JobBrief
            (JSearch,   (local embeddings:  (years asked   (open-model        .json         (open model:
             cached)     posting sentences   vs. yours:     rerank of the                    must / nice /
                         vs. profile atoms)  demote over-   top 15, 1-10                     soft, grounded)
                                             level)         + a reason)
```

Phase 5 wires the two halves into one LangGraph pipeline (`src/graph/`),
run by `run.py`:

```
RUN GRAPH       START ─▶ sourcer ─▶ ranker ─▶ posting (one task per shortlisted posting) ─▶ present ─▶ END

POSTING GRAPH   START ─▶ extract ─▶ retrieve ─▶ tailor ─▶ fit ─▶ judge ─┬─▶ finish ─▶ END
(per posting)                                     ▲                     │
                                                  └── Judge has fixes, fewer than 2 retries
                any step that fails records error.json and goes straight to finish
```

The state is typed (`src/graph/state.py`). The ranker fans out with one
`Send` per posting; results merge back through a reducer. The Judge's loop
is bounded by an `attempts` counter in the state. A node never raises: a
posting that fails is recorded and the others carry on. `python run.py
--graph` prints both graphs as Mermaid.

| Node | Role | Status |
|---|---|---|
| Sourcer | Fetch + normalize job postings by keyword, cached | ✅ `src/tools/jobs.py` |
| Ranker | Sentence-level embedding match + level gate, then LLM rerank of the top 15 | ✅ `src/agents/ranker.py` |
| Extractor | Raw JD → `JobBrief` (must / nice / soft, one concept each); grounded; one retry | ✅ `src/agents/extractor.py` |
| Retriever | Rank evidence atoms against the brief, with coverage | ✅ `retrieve_for_brief` |
| Tailor | Rewrite resume JSON against the brief; grounded; one retry | ✅ `src/agents/tailor.py` |
| Fit | Render, measure, scale, cut or expand until it fits `max_pages` | ✅ `src/tools/fit.py` |
| Judge | Tool-read checks: page budget, ATS-extractable text, coverage gaps | ✅ `src/tools/checks.py` |
| Judge (LLM) | Rubric: claims traceable, keywords embedded, every bullet an achievement | Phase 6 |

Claude for tailoring and judging (different models — the Judge never
grades its own homework); `openai/gpt-oss-120b` on Groq for extraction and
reranking; a local embedding model (Qwen3-Embedding-0.6B) for the free
first ranking stage. Which model, which provider and which settings
(temperature, reasoning effort, token budget) each request uses lives in
`src/config.py`.

Every LLM node has an eval against hand-made references, run on repeats
because hosted models are not deterministic even at temperature 0:
`src/tools/extract_eval.py` (briefs vs. hand-written briefs) and
`src/tools/rank_eval.py` (shortlists vs. hand labels). Results and every
decision they drove are logged in `data/evals/README.md`.

## Setup (Windows + conda)

```bash
conda create -n resume-agent python=3.12
conda activate resume-agent
pip install -r requirements.txt
playwright install chromium
copy .env.example .env    # then fill in the keys; each line is explained there
```

`sentence-transformers` pulls in PyTorch (~2 GB). The CPU build is enough
at this scale (28 postings rank in seconds). A CUDA build is optional and
not in `requirements.txt`: install it from pytorch.org if you want the GPU.
Embedding models download from Hugging Face on first use and are cached.

Verify (no API calls):

```bash
pytest                                                       # full suite; no credits used
python -m src.tools.render                                   # base CV -> PDF, page count + fill
python -m src.tools.evidence --brief data/sample_jd.brief.yaml -k 8   # what evidence a brief pulls
python -m src.tools.extract_eval --rescore                   # re-score the saved Extractor runs
```

Run the whole pipeline (uses API credits):

```bash
python run.py --keyword "data scientist" --location Bangalore --top 3
```

It writes `output/<keyword>-<location>/`: `shortlist.json` (the ranking),
`summary.json` (one row per posting: status, pages, fill, gaps, tokens),
and a folder per shortlisted posting, `<rank>-<ref>/`, holding
`brief.yaml`, `resume.pdf`, `resume.json`, `resume.html` (debug),
`changes.json` (what changed, citing atoms) and `judge.json`. A posting
that fails holds `error.json` instead (the stage, the reason, the
traceback), and the run's exit code is 1.

Cost of a run: one JSearch request per page not already cached; the rerank
and one extraction per shortlisted posting on Groq; per posting, one Claude
call to tailor plus up to two to fill leftover room (`--no-expand` skips
those). Postings are read from the cache however old it is, with a warning
after 7 days; `--refresh` re-fetches and overwrites the cached pool.

Each node still runs on its own:

```bash
python -m src.tools.jobs "data scientist" --location Bangalore --pages 3      # JSearch, cached
python -m src.agents.ranker "data scientist" --location Bangalore --pages 3   # -> output/shortlist.json
python -m src.agents.extractor data/sample_jd.txt            # raw JD -> JobBrief (Groq)
python -m src.tools.fit --brief data/sample_jd.brief.yaml    # JD brief -> tailored, fitted PDF (Claude)
```

## Layout

- `assets/` — `resume.json` (the CV as data), canonical `.tex` + reference PDF (gitignored)
- `data/profile.yaml` — evidence store: everything true about the candidate, superset of the CV (gitignored; `profile.example.yaml` committed)
- `data/<jd>.txt` + `data/<jd>.brief.yaml` — raw JDs and their hand-written briefs; the briefs are the Extractor's eval set
- `data/evals/` — the eval log (`README.md`) and the Extractor's saved runs (`extracted/`)
- `data/jobs_cache/` — cached JSearch responses, cached sentence embeddings and your Ranker labels (gitignored)
- `templates/` — Jinja2 template + print CSS; all presentation lives here
- `src/models.py`, `src/atoms.py`, `src/brief.py` — the three data contracts: what may be on the page, what is true, what the job wants
- `src/config.py` — model routing: providers, model ids, per-model request settings
- `src/tools/` — deterministic tools: `render` (PDF, pages, fill), `evidence` (retrieval), `fit` (page loop), `jobs` (sourcing), `checks` (the Judge's tool-read checks), and the evals `extract_eval`, `rank_eval`
- `src/agents/` — LLM nodes: `tailor`, `extractor`, `ranker`; the LLM Judge comes in Phase 6
- `src/graph/` — the pipeline: `state` (typed shared state), `nodes` (one thin wrapper per step), `build` (edges, the fan-out, the Judge's loop)
- `run.py` — the one command: keyword → tailored PDFs
- `tests/` — pytest suite; everything that touches gitignored data skips cleanly without it
- `output/` — generated artifacts (gitignored)

## Status

Phases 0–4 complete. Phase 5 is built; its live acceptance run is pending.

- **Phase 5:** `run.py` takes a keyword to one tailored PDF per shortlisted
  posting through a LangGraph pipeline. Verified on the cached pool with
  real ranking, extraction and rendering and the Tailor's Claude call
  stubbed: three postings, three one-page PDFs. The first fully live run
  (2026-11-16) stopped at the Tailor because the Anthropic account had no
  credit; the run recorded the failure and ended cleanly. Acceptance is the
  same command with credit: three tailored PDFs.
- **Phase 3:** one command turns a job brief into a validated, grounded,
  one-page PDF with a change log. Two real JDs end to end, both first
  attempt.
- **Phase 4:** real postings are fetched and cached. The Extractor writes
  briefs as good as the hand-written ones on one JD (must-have P/R about
  0.95 / 0.95) and decent on the other (about 0.8 / 0.72). The Ranker's
  top 5 had no unwanted posting in any of three runs on a labelled pool of
  28 postings. That result is in-sample; the check on a fresh pool is
  planned for Phase 5.

Next: the live acceptance run, then the two evals Phase 5 still owes (the
Ranker on a fresh, pre-labelled pool; keyword vs. embedding retrieval), then
Phase 6: the LLM Judge, tracing and cost.
