# resume-agent

A multi-agent pipeline that finds relevant job postings, ranks them, and
generates a tailored 2-page CV as PDF for each one.

## Architecture

The deterministic backbone (built, Phase 1):

```
resume.json ──▶ schema gate ──▶ Jinja2 + print.css ──▶ Chromium ──▶ PDF ──▶ page count
   (data)     (src/models.py)     (templates/)       (Playwright)        (pypdf: measured,
                                                                          never guessed)
```

Agents edit *structured data*, never markup — a bad edit fails schema
validation instead of breaking the render. Hard constraints (page count,
date formats, field shapes) are enforced by deterministic tools, not
model judgment. A single `--scale` CSS variable is the fit loop's only
layout knob.

The agent layer (Phases 3–5) adds these nodes, orchestrated with LangGraph:

| Node | Role |
|---|---|
| Sourcer | Fetch + normalize job postings by keyword |
| Ranker | Embedding similarity, then LLM rerank of the top matches |
| Retriever | Pull relevant evidence atoms from `data/profile.yaml` |
| Tailor | Rewrite resume JSON against the JD; loop until ≤ 2 pages |
| Judge | Rubric check: claims traceable, keywords covered, ATS-safe |

Claude for tailoring and judging (different models — the Judge never
grades its own homework); open models (Groq/OpenRouter) for extraction
and reranking; local embeddings for cheap pre-filtering.

## Setup (Windows + conda)

```bash
conda create -n resume-agent python=3.12
conda activate resume-agent
pip install -r requirements.txt
playwright install chromium
copy .env.example .env    # API keys — not needed until Phase 3
```

Verify:

```bash
python -m src.tools.render    # assets/resume.json -> output/resume.pdf + page count
pytest                        # full acceptance suite
```

## Layout

- `assets/` — `resume.json` (the CV as data), canonical `.tex` + reference PDF
- `data/profile.yaml` — evidence store: everything true about me, superset of the CV (Phase 2)
- `templates/` — Jinja2 template + print CSS; all presentation lives here
- `src/models.py` — Pydantic schema: the contract every agent edit must satisfy
- `src/tools/` — deterministic tools (render → PDF + page count; job fetch later)
- `src/agents/` — LLM nodes (Phase 3+)
- `tests/` — pytest acceptance suite
- `output/` — generated artifacts (gitignored, like `assets/` and `data/profile.yaml`)

## Status

Phases 0–1 complete: the deterministic render pipeline reproduces the
current CV from structured data, under test — schema-validated JSON →
HTML → Chromium print → measured page count. Next: Phase 2, the evidence
store and retrieval.