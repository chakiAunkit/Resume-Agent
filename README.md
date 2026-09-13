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

The agent layer (Phases 4–6) adds these nodes, orchestrated with LangGraph:

| Node | Role | Status |
|---|---|---|
| Sourcer | Fetch + normalize job postings by keyword | Phase 4 |
| Extractor | Raw JD → `JobBrief` (must / nice / soft, one concept each) | Phase 4 — hand-written in Phase 3 |
| Ranker | Embedding similarity, then LLM rerank of the top matches | Phase 4 |
| Retriever | Rank evidence atoms against the brief, with coverage | ✅ `retrieve_for_brief` |
| Tailor | Rewrite resume JSON against the brief; grounded; one retry | ✅ `src/agents/tailor.py` |
| Fit | Render, measure, scale or cut until it fits `max_pages` | ✅ `src/tools/fit.py` |
| Judge | Rubric check: claims traceable, keywords covered, ATS-safe | Phase 6 |

Claude for tailoring and judging (different models — the Judge never
grades its own homework); open models (Groq/OpenRouter) for extraction
and reranking; local embeddings for cheap pre-filtering.

## Setup (Windows + conda)

```bash
conda create -n resume-agent python=3.12
conda activate resume-agent
pip install -r requirements.txt
playwright install chromium
copy .env.example .env    # ANTHROPIC_API_KEY, TAILOR_MODEL, TAILOR_EFFORT
```

Verify:

```bash
pytest                                                       # full suite
python -m src.tools.render                                   # base CV -> PDF, page count + fill
python -m src.tools.evidence --brief data/sample_jd.brief.yaml -k 8   # what evidence a brief pulls
python -m src.tools.fit --brief data/sample_jd.brief.yaml    # JD brief -> tailored, fitted PDF
```

The last command writes `output/<brief>/resume.pdf`, `resume.json`,
`resume.html` (debug) and `changes.json` (what changed, citing atoms).

## Layout

- `assets/` — `resume.json` (the CV as data), canonical `.tex` + reference PDF (gitignored)
- `data/profile.yaml` — evidence store: everything true about the candidate, superset of the CV (gitignored; `profile.example.yaml` committed)
- `data/<jd>.txt` + `data/<jd>.brief.yaml` — raw JDs and their briefs; the briefs are the Extractor's eval set
- `data/eval/` — retrieval eval log: hand-labelled expectations and per-iteration rankings
- `templates/` — Jinja2 template + print CSS; all presentation lives here
- `src/models.py`, `src/atoms.py`, `src/brief.py` — the three data contracts: what may be on the page, what is true, what the job wants
- `src/tools/` — deterministic tools: `render` (PDF, pages, fill), `evidence` (retrieval), `fit` (page loop)
- `src/agents/` — LLM nodes: `tailor` (Phase 3); extractor, ranker, judge to come
- `tests/` — pytest suite; everything that touches gitignored data skips cleanly without it
- `output/` — generated artifacts (gitignored)

## Status

Phases 0–3 complete. One command turns a job brief into a validated,
grounded, one-page PDF with a change log. Two real JDs end to end, both
first attempt, no corrective fit steps. Next: Phase 4 — sourcing,
the Extractor (which replaces the hand-written briefs), ranking, and the
keyword-vs-embedding retrieval experiment against the v5 baseline.
