"""Wiring: which node runs after which, and where control can branch or loop.

Read this file to see the pipeline's control flow; read nodes.py to see
what each step does.

    RUN GRAPH (RunState)

        START -> sourcer -> ranker -+-> posting (one task per shortlisted posting) -> present -> END
                                    +-> present            (nothing shortlisted)

    POSTING GRAPH (PostingState), run once inside each `posting` task

        START -> extract -> retrieve -> tailor -> fit -> judge -+-> finish -> END
                                          ^                     |
                                          +-- Judge has fixes and attempts < MAX_JUDGE_RETRIES

        extract, retrieve, tailor and fit each go to `finish` instead of
        their next node when they set `error`.

Three LangGraph ideas, each used once:

  * Conditional edge: a function of the state picks the next node
    (`_unless_failed`, `nodes.route_after_judge`).
  * Cycle: judge -> tailor. The `attempts` counter in the state is what
    bounds it; nothing else stops a graph from looping.
  * Fan-out with Send: `nodes.fan_out` returns one Send per posting, each
    with its own input, and LangGraph runs `posting` once per Send. Their
    results meet again in `RunState.results` through its reducer.
    `max_concurrency` in the invoke config decides how many run at once.

No `from __future__ import annotations` here: LangGraph reads each node's
parameter annotation to know which state type to hand it.
"""

from typing import Callable

from langgraph.graph import END, START, StateGraph

from src.graph import nodes
from src.graph.nodes import Deps
from src.graph.state import PostingState, RunState


def _unless_failed(next_node: str) -> Callable[[PostingState], str]:
    """An edge that goes to `next_node`, or to `finish` if the node just run failed."""
    def route(state: PostingState) -> str:
        return "finish" if state.error else next_node
    return route


def build_posting_graph(deps: Deps):
    """extract -> retrieve -> tailor -> fit -> judge, with the Judge's loop."""

    def extract(state: PostingState) -> dict:
        return nodes.extract(state, deps)

    def retrieve(state: PostingState) -> dict:
        return nodes.retrieve(state, deps)

    def tailor(state: PostingState) -> dict:
        return nodes.tailor(state, deps)

    def fit(state: PostingState) -> dict:
        return nodes.fit(state, deps)

    def judge(state: PostingState) -> dict:
        return nodes.judge(state, deps)

    def finish(state: PostingState) -> dict:
        return nodes.finish(state, deps)

    g = StateGraph(PostingState)
    for node in (extract, retrieve, tailor, fit, judge, finish):
        g.add_node(node.__name__, node)

    g.add_edge(START, "extract")
    g.add_conditional_edges("extract", _unless_failed("retrieve"), ["retrieve", "finish"])
    g.add_conditional_edges("retrieve", _unless_failed("tailor"), ["tailor", "finish"])
    g.add_conditional_edges("tailor", _unless_failed("fit"), ["fit", "finish"])
    g.add_conditional_edges("fit", _unless_failed("judge"), ["judge", "finish"])
    g.add_conditional_edges("judge", nodes.route_after_judge, ["tailor", "finish"])    # the loop
    g.add_edge("finish", END)
    return g.compile()


def build_graph(deps: Deps):
    """The whole run: keyword -> postings -> shortlist -> one posting graph each -> summary."""
    posting_graph = build_posting_graph(deps)

    def sourcer(state: RunState) -> dict:
        return nodes.sourcer(state, deps)

    def ranker(state: RunState) -> dict:
        return nodes.ranker(state, deps)

    def posting(payload: dict) -> dict:
        """One shortlisted posting through its own graph. Whatever happens
        inside, this returns exactly one result row.

        `payload` is what `nodes.fan_out` put in the Send: {"rank", "post"}.
        A Send's payload reaches the node as given (a plain dict), and the
        posting graph returns a dict of the fields that were written, so
        both ends are validated into a PostingState here."""
        start = PostingState.model_validate(payload)
        try:
            final = PostingState.model_validate(posting_graph.invoke({"rank": start.rank, "post": start.post}))
            result = nodes.to_result(final, deps)
        except Exception as exc:            # noqa: BLE001 — a bug in one posting must not end the run
            deps.log(f"  [#{start.rank}] posting graph crashed: {type(exc).__name__}: {exc}")
            result = nodes.crashed_result(start.rank, start.post, deps, exc)
        return {"results": [result]}

    def present(state: RunState) -> dict:
        return nodes.present(state, deps)

    g = StateGraph(RunState)
    for node in (sourcer, ranker, posting, present):
        g.add_node(node.__name__, node)

    g.add_edge(START, "sourcer")
    g.add_edge("sourcer", "ranker")
    g.add_conditional_edges("ranker", nodes.fan_out, ["posting", "present"])           # the fan-out
    g.add_edge("posting", "present")
    g.add_edge("present", END)
    return g.compile()
