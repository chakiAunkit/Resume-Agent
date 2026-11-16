"""Phase 5 graph tests. The whole pipeline runs on fakes (tests/fakes.py):
no network, no model, no browser, no credits.

Run from the repo root:  pytest tests/test_graph.py -v
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from src.agents.ranker import Stage1Options
from src.atoms import Profile
from src.graph.build import build_graph, build_posting_graph
from src.graph.nodes import MAX_JUDGE_RETRIES, Deps
from src.models import Resume
from src.tools.checks import JudgeReport
from tests.fakes import (BASE_RESUME, BagOfWords, FakeAnthropic, FakeBoard, FakeRenderer, ScriptedJudge,
                         echo_resume, extractor_client, make_post, rating_client)

ROOT = Path(__file__).resolve().parents[1]

POSTS = [
    make_post("shop", "Data Scientist, Catalog", "Shopwise",
              "Analyze e-commerce orders in SQL. Fix shipping service selection across logistics. Own the taxonomy."),
    make_post("agent", "Data Scientist, Agents", "Querycraft",
              "Build a Text2SQL agent with long-term memory in Python. E-commerce data, SQL, taxonomy."),
    make_post("chip", "Chip Designer", "Siliconia",
              "Design analog circuits for wafer inspection. Some SQL for e-commerce dashboards and a taxonomy."),
]


def make_deps(tmp_path, **over) -> Deps:
    base = dict(
        profile=Profile.load(ROOT / "data" / "profile.example.yaml"),
        resume=Resume.model_validate(BASE_RESUME),
        out_dir=tmp_path / "out", cache_dir=tmp_path / "cache", today=date(2026, 9, 1),
        provider=FakeBoard(POSTS), embedder=BagOfWords(), options=Stage1Options(),
        ranker_client=rating_client(), extractor_client=extractor_client(),
        tailor_client=FakeAnthropic(), renderer_factory=FakeRenderer(),
        judge=ScriptedJudge(), log=lambda _line: None)
    return Deps(**{**base, **over})


def run(deps: Deps, top: int = 3, parallel: int = 1) -> dict:
    return build_graph(deps).invoke({"keyword": "data scientist", "location": "Bangalore", "top": top},
                                    config={"max_concurrency": parallel})


def by_company(out: dict) -> dict:
    return {r.company: r for r in out["results"]}


# --------------------------------------------------------------- happy path

def test_keyword_to_one_resume_per_shortlisted_posting(tmp_path):
    deps = make_deps(tmp_path)
    out = run(deps)

    assert len(out["jobs"]) == 3 and len(out["shortlist"]) == 3
    results = sorted(out["results"], key=lambda r: r.rank)
    assert [r.rank for r in results] == [1, 2, 3] and {r.status for r in results} == {"ok"}
    assert [r.ref for r in results] == [e.ref for e in out["shortlist"]]          # the ranking's order
    assert deps.provider.fetches == 1

    for r in results:
        folder = Path(r.folder)
        assert folder.name == f"{r.rank}-{r.ref}"
        assert {p.name for p in folder.iterdir()} == {"brief.yaml", "resume.pdf", "resume.json",
                                                      "changes.json", "judge.json"}
        assert Resume.load(folder / "resume.json") == deps.resume                 # the fake Tailor echoes
        assert r.page_count == 1 and r.must_haves == 3 and r.uncovered == 1       # "taxonomy" has no atom
        assert r.usage["extract"] == {"prompt_tokens": 100, "completion_tokens": 40}
        assert r.usage["tailor"] == {"input_tokens": 1000, "output_tokens": 400}

    summary = json.loads((deps.out_dir / "summary.json").read_text(encoding="utf-8"))
    assert [p["rank"] for p in summary["postings"]] == [1, 2, 3]
    assert summary["tokens"]["tailor"] == {"input_tokens": 3000, "output_tokens": 1200}
    assert summary["tokens"]["ranker"] == {"prompt_tokens": 100, "completion_tokens": 40}
    assert len(json.loads((deps.out_dir / "shortlist.json").read_text(encoding="utf-8"))) == 3


def test_only_the_top_n_are_extracted_and_tailored(tmp_path):
    """Cheap before expensive: the Extractor and the Tailor run for the
    shortlist, never for the whole pool."""
    deps = make_deps(tmp_path, ranker_client=rating_client())
    out = run(deps, top=2)
    assert len(out["results"]) == 2 and {r.rank for r in out["results"]} == {1, 2}
    assert len(deps.extractor_client.calls) == 2 and len(deps.tailor_client.calls) == 2
    assert len(out["shortlist"]) == 3                                              # all ranked, two tailored


def test_extractor_reads_the_jd_with_its_title_and_company_header(tmp_path):
    deps = make_deps(tmp_path)
    run(deps, top=1)
    jd = deps.extractor_client.calls[0]["messages"][1]["content"]
    assert "\n# " in jd and " — " in jd.split("\n# ", 1)[1].splitlines()[0]


def test_results_are_the_same_run_in_parallel(tmp_path):
    serial = run(make_deps(tmp_path / "a"))
    parallel = run(make_deps(tmp_path / "b"), parallel=3)
    key = lambda out: sorted((r.rank, r.ref, r.status) for r in out["results"])
    assert key(serial) == key(parallel)


# ------------------------------------------------------ failure isolation

def test_one_posting_failing_does_not_stop_the_others(tmp_path):
    deps = make_deps(tmp_path, extractor_client=extractor_client(broken=("Querycraft",)))
    results = by_company(run(deps))

    bad = results["Querycraft"]
    assert bad.status == "failed" and bad.failed_stage == "extract" and "Nowhere Inc" in bad.error
    assert bad.pdf_path is None
    error = json.loads((Path(bad.folder) / "error.json").read_text(encoding="utf-8"))
    assert error["stage"] == "extract" and error["error"] == "ExtractorError"
    assert not (Path(bad.folder) / "resume.pdf").exists()                          # it never reached the Tailor

    assert results["Shopwise"].status == "ok" and results["Siliconia"].status == "ok"
    assert len(deps.tailor_client.calls) == 2


def test_a_resume_that_cannot_fit_fails_with_its_tailored_version_kept(tmp_path):
    """Always two pages and the cutter never shortens it: FitError. The
    Tailor's accepted work is saved beside the error."""
    deps = make_deps(tmp_path, renderer_factory=FakeRenderer(pages=lambda resume: 2))
    results = by_company(run(deps, top=1))
    (result,) = results.values()
    assert result.status == "failed" and result.failed_stage == "fit" and "FitError" in result.error
    folder = Path(result.folder)
    assert (folder / "error.json").exists() and Resume.load(folder / "resume.unfitted.json") == deps.resume
    assert result.usage["fit"]["input_tokens"] > 0                                 # the cuts it tried still cost


def test_a_failed_rerank_falls_back_to_stage_one_order(tmp_path):
    deps = make_deps(tmp_path, ranker_client=rating_client(fail=RuntimeError("429 rate limit")))
    out = run(deps)
    assert len(out["results"]) == 3 and {r.status for r in out["results"]} == {"ok"}
    assert all(e.score is None for e in out["shortlist"])                          # stage 1 only
    assert any("stage-2 rerank failed" in w for w in out["warnings"])


def test_no_postings_ends_the_run_with_a_reason(tmp_path):
    with pytest.raises(RuntimeError, match="no postings for 'data scientist' in Bangalore"):
        run(make_deps(tmp_path, provider=FakeBoard([])))


# ------------------------------------------------------------ the judge loop

def _drop_last_bullet(resume: dict) -> None:
    resume["experience"][0]["bullets"].pop()


def test_judge_sends_the_resume_back_once_and_the_tailor_revises_it(tmp_path):
    fix = "the rendered resume is 2 pages and the budget is 1: remove the bullet or project that serves this brief least"
    judge = ScriptedJudge(JudgeReport(passed=False, fixes=[fix]), JudgeReport(passed=True))
    # The Tailor echoes on the first call and drops a bullet when revising.
    tailor = FakeAnthropic(lambda kw: echo_resume(
        kw, _drop_last_bullet if "# Revision requested" in kw["messages"][0]["content"] else None))
    deps = make_deps(tmp_path, judge=judge, tailor_client=tailor)

    (result,) = run(deps, top=1)["results"]
    assert result.status == "ok" and result.judge_retries == 1 and result.findings == []
    assert len(judge.seen) == 2 and len(tailor.calls) == 2
    assert fix in tailor.prompts()[1] and fix not in tailor.prompts()[0]           # the fix reached the Tailor
    assert len(judge.seen[0].experience[0].bullets) == 3
    assert len(judge.seen[1].experience[0].bullets) == 2                           # the Judge saw the revision
    changes = json.loads((Path(result.folder) / "changes.json").read_text(encoding="utf-8"))
    assert len(changes) == 2                                                       # both passes are in the log
    assert result.usage["tailor"] == {"input_tokens": 2000, "output_tokens": 800}


def test_judge_loop_stops_at_the_retry_cap(tmp_path):
    """A Judge that never passes costs exactly MAX_JUDGE_RETRIES revisions,
    then the resume is delivered with the finding attached."""
    judge = ScriptedJudge(JudgeReport(passed=False, fixes=["still too long"]))
    deps = make_deps(tmp_path, judge=judge)
    (result,) = run(deps, top=1)["results"]
    assert len(judge.seen) == MAX_JUDGE_RETRIES + 1 == 3
    assert len(deps.tailor_client.calls) == MAX_JUDGE_RETRIES + 1
    assert result.status == "flagged" and result.judge_retries == 2 and result.findings == ["still too long"]
    assert result.pdf_path is not None                                             # delivered, not discarded


def test_a_defect_is_flagged_without_asking_the_tailor_again(tmp_path):
    """Text an ATS cannot read is not something a rewrite can fix."""
    judge = ScriptedJudge(JudgeReport(passed=False, defects=["not extractable from the PDF text: name"]))
    deps = make_deps(tmp_path, judge=judge)
    (result,) = run(deps, top=1)["results"]
    assert len(judge.seen) == 1 and len(deps.tailor_client.calls) == 1
    assert result.status == "flagged" and result.judge_retries == 0


def test_gaps_do_not_fail_a_posting(tmp_path):
    judge = ScriptedJudge(JudgeReport(passed=True, gaps=["must-have 'taxonomy': no evidence in the profile"]))
    (result,) = run(make_deps(tmp_path, judge=judge), top=1)["results"]
    assert result.status == "ok" and result.gaps == ["must-have 'taxonomy': no evidence in the profile"]


def test_expansion_runs_on_the_first_fit_only(tmp_path):
    """A Judge retry is a repair: the second fit must not ask for more content."""
    judge = ScriptedJudge(JudgeReport(passed=False, fixes=["too long"]), JudgeReport(passed=True))
    tailor = FakeAnthropic()
    deps = make_deps(tmp_path, judge=judge, tailor_client=tailor, renderer_factory=FakeRenderer(fill=0.80))
    run(deps, top=1)
    expands = [p for p in tailor.prompts() if "fits the page budget with about" in p]
    assert len(expands) == 1                # one expansion attempt (it echoed, so "nothing to add"), first fit only


# ------------------------------------------------------------------ run.py

def test_run_py_prints_both_graphs_without_running_anything(capsys):
    import run
    assert run.main(["--graph"]) == 0
    text = capsys.readouterr().out
    assert "%% run graph" in text and "ranker -.-> posting" in text
    assert "%% posting graph" in text and "judge -.-> tailor" in text and "judge -.-> finish" in text


def test_run_py_arguments():
    import run
    args = run.parse_args(["--keyword", "ml engineer", "--top", "2", "--no-expand"])
    assert (args.keyword, args.top, args.pages, args.max_pages, args.parallel) == ("ml engineer", 2, 3, 1, 1)
    assert args.no_expand and not args.refresh
    with pytest.raises(SystemExit):
        run.parse_args([])                                   # a keyword, or --graph
    with pytest.raises(SystemExit):
        run.parse_args(["--keyword", "x", "--top", "0"])
    assert run._slug("Data Scientist  Bangalore ") == "data-scientist-bangalore"


# ------------------------------------------------------------------ wiring

def test_graphs_have_the_roadmap_shape(tmp_path):
    deps = make_deps(tmp_path)
    run_graph = build_graph(deps).get_graph()
    assert {"sourcer", "ranker", "posting", "present"} <= set(run_graph.nodes)
    run_edges = {(e.source, e.target) for e in run_graph.edges}
    assert {("__start__", "sourcer"), ("sourcer", "ranker"), ("ranker", "posting"),
            ("posting", "present"), ("present", "__end__")} <= run_edges

    posting = build_posting_graph(deps).get_graph()
    edges = {(e.source, e.target) for e in posting.edges}
    assert {("__start__", "extract"), ("extract", "retrieve"), ("retrieve", "tailor"), ("tailor", "fit"),
            ("fit", "judge"), ("judge", "tailor"), ("judge", "finish"), ("finish", "__end__")} <= edges
    for node in ("extract", "retrieve", "tailor", "fit"):                          # every step can bail out
        assert (node, "finish") in edges
