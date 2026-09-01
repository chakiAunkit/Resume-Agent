# resume-agent

A multi-agent pipeline that finds relevant job postings, ranks them, and
generates a tailored 2-page CV as PDF for each one.

## Architecture

JSON resume data + Jinja2 HTML template → Chromium → PDF.
Agents edit *structured data*, never markup, so a bad edit fails
validation instead of breaking the render.

| Node | Role |
|---|---|
| Sourcer | Fetch + normalize job postings by keyword |
| Ranker | Embedding similarity, then LLM rerank of the top matches |
| Retriever | Pull relevant evidence atoms from `data/profile.yaml` |
| Tailor | Rewrite resume JSON against the JD; loop until ≤ 2 pages |
| Judge | Rubric check: claims traceable, keywords covered, ATS-safe |

Orchestrated with LangGraph. Claude for tailoring and judging;
open models (Groq/OpenRouter) for extraction and reranking.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
cp .env.example .env      # fill in your API keys
python tests/smoke_render.py
```

## Layout

- `assets/` — canonical Overleaf `.tex` and current PDF (reference)
- `data/profile.yaml` — evidence store: projects, metrics, links
- `templates/` — Jinja2 HTML + print CSS
- `src/tools/` — deterministic tools (render, page count, job fetch)
- `src/agents/` — LangGraph nodes

## Status

Phase 0: scaffold + render toolchain. See project plan for roadmap.