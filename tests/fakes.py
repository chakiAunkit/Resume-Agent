"""Stand-ins for everything the Phase 5 graph would otherwise pay for or
wait on: the job board, the embedding model, the three LLM clients and the
browser. Each records its calls so a test can assert on what the graph did.

They mirror the slice of each real interface the nodes touch, the same way
the per-module tests do (FakeProvider in test_jobs, BagOfWords in
test_ranker, fake_client in test_tailor, FakeRenderer in test_fit).
"""

from __future__ import annotations

import json
import re
import zlib
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np

from src.models import Resume
from src.tools.checks import JudgeReport
from src.tools.evidence import tokenize
from src.tools.jobs import JobPost
from src.tools.render import RenderResult

BASE_RESUME = {
    "contact": {"name": "Test Person", "phone": "+1 555 0100", "email": "t@example.com", "links": []},
    "summary": "Data scientist with 4 years of experience.",
    "experience": [{
        "company": "Target Corporation", "title": "Analyst", "location": "Bangalore",
        "start": "09/2022", "end": "05/2025",
        "bullets": ["Ran analyses on 12 datasets.", "Built SQL reports.", "Automated a weekly job."],
    }],
    "skills": [{"category": "Technical", "items": ["SQL"]}],
    "education": [{"degree": "B.Tech", "institution": "U", "location": "City",
                   "start": "07/2017", "end": "04/2021"}],
}

_FILLER = "We are a growing team that values ownership and clear thinking in everything we ship. " * 3


def make_post(key: str, title: str, company: str, body: str) -> JobPost:
    return JobPost(id=f"jsearch:{key}", source="jsearch", title=title, company=company,
                   url=f"https://example.com/{key}", description=_FILLER + body)


class FakeBoard:
    """A job board that returns the given postings on one page."""

    name = "fake"

    def __init__(self, posts: list[JobPost]):
        self.posts = posts
        self.fetches = 0

    def fetch_raw(self, query) -> dict:
        self.fetches += 1
        return {"count": len(self.posts)}

    def parse(self, raw: dict) -> list[JobPost]:
        return list(self.posts)

    def next_cursor(self, raw: dict) -> str | None:
        return None


class BagOfWords:
    """Deterministic embedder: hashed word counts, unit length."""

    name = "bag-of-words"

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        m = np.zeros((len(texts), 256))
        for i, text in enumerate(texts):
            for tok in tokenize(text):
                m[i, zlib.crc32(tok.encode()) % 256] += 1
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return m / norms


class FakeOpenAI:
    """`client.chat.completions.create(**kw)` answered by `reply(kw) -> str`.
    A reply that is an Exception is raised instead."""

    def __init__(self, reply: Callable[[dict], str | Exception]):
        self.calls: list[dict] = []

        def create(**kw):
            self.calls.append(kw)
            answer = reply(kw)
            if isinstance(answer, Exception):
                raise answer
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=answer), finish_reason="stop")],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=40))

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


def rating_client(scores: dict[str, int] | None = None, fail: Exception | None = None) -> FakeOpenAI:
    """The Ranker's stage 2: rates every [ref] in the prompt (default 5)."""
    def reply(kw: dict):
        if fail is not None:
            return fail
        refs = re.findall(r"\[([0-9a-f]{8})\]", kw["messages"][1]["content"])
        return json.dumps({"ratings": [{"ref": r, "score": (scores or {}).get(r, 5), "reason": "fits"}
                                       for r in refs]})
    return FakeOpenAI(reply)


def extractor_client(broken: tuple[str, ...] = ()) -> FakeOpenAI:
    """The Extractor: reads the title and company from the JD header and
    returns a small grounded brief. For a company in `broken` it names a
    company the JD never mentions, so verify() rejects it on both attempts."""
    def reply(kw: dict):
        jd = kw["messages"][1]["content"]
        title, company = re.search(r"^# (.+?) — (.+)$", jd, re.M).groups()
        return json.dumps({
            "title": title, "company": "Nowhere Inc" if company in broken else company,
            "seniority": "unspecified", "years_experience": None,
            "must_haves": ["SQL", "e-commerce", "taxonomy"], "nice_to_haves": [], "soft_skills": []})
    return FakeOpenAI(reply)


class FakeAnthropic:
    """`client.messages.stream(**kw)` as a context manager whose
    get_final_message() returns `reply(kw)` as a text block. The default
    reply is the resume it was shown, unchanged, with one change entry:
    valid and grounded whatever the prompt asked."""

    def __init__(self, reply: Callable[[dict], str] | None = None):
        self.calls: list[dict] = []
        self._reply = reply or (lambda kw: echo_resume(kw))

        @contextmanager
        def stream(**kw):
            self.calls.append(kw)
            message = SimpleNamespace(
                content=[SimpleNamespace(type="text", text=self._reply(kw))],
                stop_reason="end_turn", usage=SimpleNamespace(input_tokens=1000, output_tokens=400))
            yield SimpleNamespace(get_final_message=lambda: message)

        self.messages = SimpleNamespace(stream=stream)

    def prompts(self) -> list[str]:
        return [kw["messages"][0]["content"] for kw in self.calls]


def echo_resume(kw: dict, edit: Callable[[dict], None] | None = None) -> str:
    """The "# Current resume" block of the prompt, optionally edited, as a
    TailoredResume answer."""
    prompt = kw["messages"][0]["content"]
    block = prompt.split("# Current resume\n", 1)[1]
    resume, _end = json.JSONDecoder().raw_decode(block)
    if edit:
        edit(resume)
    return json.dumps({"resume": resume, "changes": [{"where": "summary", "atom_id": None, "why": "rephrased"}]})


class FakeRenderer:
    """A renderer that writes a placeholder file and reports `pages(resume)`
    pages at a fill that needs no expansion. Used as `renderer_factory`:
    calling the instance returns a context manager that yields it."""

    def __init__(self, pages: Callable[[Resume], int] = lambda resume: 1, fill: float = 0.97):
        self.pages, self.fill = pages, fill
        self.renders: list[Path] = []
        self.opened = 0

    @contextmanager
    def __call__(self):
        self.opened += 1
        yield self

    def render(self, data: Resume, scale: float = 1.0, out_path=None) -> RenderResult:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("placeholder pdf", encoding="utf-8")
        self.renders.append(out)
        pages = self.pages(data)
        return RenderResult(pdf_path=out, html_path=out.with_suffix(".html"), page_count=pages,
                            fill=self.fill if pages == 1 else pages - 0.5, scale=scale)


class ScriptedJudge:
    """A Judge that returns the next report from `script` on each call (the
    last one repeats). Records what it was shown."""

    def __init__(self, *script: JudgeReport):
        self.script = list(script) or [JudgeReport(passed=True)]
        self.seen: list[Resume] = []

    def __call__(self, resume, pdf_path, page_count, brief, hits, *, max_pages=1) -> JudgeReport:
        self.seen.append(resume)
        return self.script[min(len(self.seen), len(self.script)) - 1]
