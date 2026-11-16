"""Phase 4 sourcing tests. No network, no key: a fake provider pins the
cache/merge mechanics, and `httpx.MockTransport` pins the JSearch
request contract (URL, params, headers) and its parser.

Run from the repo root:  pytest tests/test_jobs.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from src.tools.jobs import (
    MIN_DESCRIPTION_CHARS,
    JobPost,
    JobQuery,
    JSearchProvider,
    MissingApiKey,
    ProviderError,
    cache_path,
    dump_jd,
    fetch_jobs,
)

LONG = "Responsibilities. " * 20  # comfortably over MIN_DESCRIPTION_CHARS


def _item(job_id: str, title: str = "Data Scientist", company: str = "Acme", **over) -> dict:
    base = {
        "job_id": job_id,
        "job_title": title,
        "employer_name": company,
        "job_description": LONG,
        "job_apply_link": f"https://example.com/{job_id}",
        "job_city": "Bangalore",
        "job_state": "Karnataka",
        "job_country": "IN",
        "job_is_remote": False,
        "job_employment_type": "FULLTIME",
        "job_posted_at_datetime_utc": "2026-09-10T00:00:00.000Z",
        "job_publisher": "LinkedIn",
    }
    return {**base, **over}


def _raw(*items: dict, cursor: str | None = "next-token") -> dict:
    """JSearch v5 shape: jobs and the pagination cursor live under `data`."""
    return {"status": "OK", "request_id": "r", "parameters": {},
            "data": {"jobs": list(items), "cursor": cursor}}


class FakeProvider:
    """JSearch-shaped canned responses, one per fetch in order (the last one
    repeats); an exception in the list is raised instead. Counts fetches."""

    name = "fake"

    def __init__(self, *responses: dict | Exception):
        self.responses = list(responses)
        self.fetches: list[JobQuery] = []

    def fetch_raw(self, query: JobQuery) -> dict:
        self.fetches.append(query)
        response = self.responses[min(len(self.fetches), len(self.responses)) - 1]
        if isinstance(response, Exception):
            raise response
        return response

    def parse(self, raw: dict) -> list[JobPost]:
        return JSearchProvider(api_key="x").parse(raw)

    def next_cursor(self, raw: dict) -> str | None:
        return JSearchProvider.next_cursor(raw)


# ----------------------------------------------------------------- cache

def test_first_call_fetches_and_caches_raw(tmp_path):
    prov = FakeProvider(_raw(_item("a")))
    posts = fetch_jobs("data scientist", "Bangalore", provider=prov, cache_dir=tmp_path)
    assert [p.id for p in posts] == ["jsearch:a"]
    assert len(prov.fetches) == 1

    files = list((tmp_path / "fake").glob("*.json"))
    assert len(files) == 1 and files[0].name.startswith("data-scientist-bangalore-n1-")
    entry = json.loads(files[0].read_text(encoding="utf-8"))
    assert entry["raw"] == _raw(_item("a"))          # raw, not parsed
    assert entry["query"]["keyword"] == "data scientist"
    assert "fetched_at" in entry


def _age_cache(cache_dir, days: float) -> None:
    """Rewrite every cache entry's fetched_at to `days` ago."""
    from datetime import datetime, timedelta, timezone
    for path in cache_dir.rglob("*.json"):
        entry = json.loads(path.read_text(encoding="utf-8"))
        entry["fetched_at"] = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds")
        path.write_text(json.dumps(entry), encoding="utf-8")


def test_old_cache_warns_with_its_age_and_never_refetches(tmp_path):
    """The cache has no expiry; a caller that wants live postings is told
    how old the pool is. It is not re-fetched: that costs requests and
    overwrites the pool."""
    import warnings

    from src.tools.jobs import StaleCacheWarning
    prov = FakeProvider(_raw(_item("a"), cursor=None))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                       # a fresh fetch never warns
        fetch_jobs("q", provider=prov, cache_dir=tmp_path, stale_after_days=7)
    _age_cache(tmp_path, days=50)
    with pytest.warns(StaleCacheWarning, match=r"\(50 days ago\)"):
        posts = fetch_jobs("q", provider=prov, cache_dir=tmp_path, stale_after_days=7)
    assert [p.id for p in posts] == ["jsearch:a"] and len(prov.fetches) == 1     # served, not re-fetched
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fetch_jobs("q", provider=prov, cache_dir=tmp_path)                        # the evals: no limit, no warning
        fetch_jobs("q", provider=prov, cache_dir=tmp_path, stale_after_days=60)   # within the limit
        fetch_jobs("q", provider=prov, cache_dir=tmp_path, stale_after_days=7, refresh=True)   # fetched just now
    assert len(prov.fetches) == 2


def test_second_call_is_served_from_cache(tmp_path):
    prov = FakeProvider(_raw(_item("a")))
    fetch_jobs("data scientist", provider=prov, cache_dir=tmp_path)
    fetch_jobs("data scientist", provider=prov, cache_dir=tmp_path)
    assert len(prov.fetches) == 1


def test_refresh_bypasses_cache(tmp_path):
    prov = FakeProvider(_raw(_item("a")))
    fetch_jobs("data scientist", provider=prov, cache_dir=tmp_path)
    fetch_jobs("data scientist", provider=prov, cache_dir=tmp_path, refresh=True)
    assert len(prov.fetches) == 2


def test_cache_key_is_stable_and_query_sensitive(tmp_path):
    q = JobQuery(keyword="ml engineer", location="Bangalore")
    assert cache_path(tmp_path, "x", q) == cache_path(tmp_path, "x", q)
    for other in (JobQuery(keyword="ml engineer"),
                  JobQuery(keyword="ml engineer", location="Bangalore", num_pages=2),
                  JobQuery(keyword="ml engineer", location="Bangalore", cursor="abc"),
                  JobQuery(keyword="ml engineer", location="Bangalore", country="us"),
                  JobQuery(keyword="ml engineer", location="Bangalore", date_posted="week")):
        assert cache_path(tmp_path, "x", other) != cache_path(tmp_path, "x", q)
    assert cache_path(tmp_path, "x", q) != cache_path(tmp_path, "y", q)  # provider in the key


def test_parse_is_rerun_from_cache_not_frozen(tmp_path):
    """Improving the parser must not require a re-fetch: the cache holds
    the provider's raw JSON, and parse() runs on every call."""
    prov = FakeProvider(_raw(_item("a"), _item("b", title="Analyst", job_description="short")))
    fetch_jobs("q", provider=prov, cache_dir=tmp_path)
    path = next((tmp_path / "fake").glob("*.json"))
    entry = json.loads(path.read_text(encoding="utf-8"))
    jobs = entry["raw"]["data"]["jobs"]
    assert len(jobs) == 2                                   # the dropped item is still in the raw
    jobs[1]["job_description"] = LONG                       # simulate "better data" on disk
    path.write_text(json.dumps(entry), encoding="utf-8")
    assert len(fetch_jobs("q", provider=prov, cache_dir=tmp_path)) == 2
    assert len(prov.fetches) == 1


# ----------------------------------------------------------------- merge

def test_results_are_deduplicated(tmp_path):
    prov = FakeProvider(_raw(
        _item("a"), _item("b", title="ML Engineer"),
        _item("a"),                                                # same id again
        _item("c", title="ML Engineer", job_publisher="Indeed"),   # same job, other board
        _item("d", title="Analyst"),
    ))
    posts = fetch_jobs("q", provider=prov, cache_dir=tmp_path)
    assert [p.id for p in posts] == ["jsearch:a", "jsearch:b", "jsearch:d"]


# ---------------------------------------------------------------- paging

def test_pages_are_one_request_each_following_the_cursor(tmp_path):
    """Three pages in one request made RapidAPI's gateway time out (504).
    Each page is its own request now, carrying the previous page's cursor,
    and duplicates are removed across pages too."""
    prov = FakeProvider(_raw(_item("a"), cursor="c1"),
                        _raw(_item("b", title="ML Engineer"), _item("a"), cursor="c2"),
                        _raw(_item("c", title="Analyst"), cursor="c3"))
    posts = fetch_jobs("q", provider=prov, pages=3, cache_dir=tmp_path)
    assert [p.id for p in posts] == ["jsearch:a", "jsearch:b", "jsearch:c"]
    assert [(q.num_pages, q.cursor) for q in prov.fetches] == [(1, None), (1, "c1"), (1, "c2")]
    assert len(list((tmp_path / "fake").glob("*.json"))) == 3          # one cache file per page


def test_paging_stops_when_results_run_out(tmp_path):
    no_cursor = FakeProvider(_raw(_item("a"), cursor=None))
    assert len(fetch_jobs("q", provider=no_cursor, pages=3, cache_dir=tmp_path / "1")) == 1
    assert len(no_cursor.fetches) == 1                                  # no credit spent on page 2
    empty = FakeProvider(_raw(_item("a"), cursor="c1"), _raw(cursor="c2"))
    fetch_jobs("q", provider=empty, pages=5, cache_dir=tmp_path / "2")
    assert len(empty.fetches) == 2                                      # stopped at the empty page


def test_later_page_failure_keeps_earlier_pages_and_resumes_from_cache(tmp_path):
    failing = FakeProvider(_raw(_item("a"), cursor="c1"), ProviderError("HTTP 504"))
    with pytest.warns(UserWarning, match="stopped after page 1 of 3"):
        posts = fetch_jobs("q", provider=failing, pages=3, cache_dir=tmp_path)
    assert [p.id for p in posts] == ["jsearch:a"]
    # Re-run without refresh: page 1 comes from cache, only page 2 onward is fetched.
    healthy = FakeProvider(_raw(_item("b", title="Analyst"), cursor=None))
    posts = fetch_jobs("q", provider=healthy, pages=3, cache_dir=tmp_path)
    assert [p.id for p in posts] == ["jsearch:a", "jsearch:b"]
    assert [q.cursor for q in healthy.fetches] == ["c1"]


def test_first_page_failure_raises(tmp_path):
    with pytest.raises(ProviderError, match="504"):
        fetch_jobs("q", provider=FakeProvider(ProviderError("HTTP 504")), cache_dir=tmp_path)


# ----------------------------------------------------------------- parse

def test_parse_drops_unusable_items():
    raw = _raw(
        _item("ok"),
        _item("teaser", job_description="Apply now!"),
        _item("no-desc", job_description=None),
        _item("no-title", job_title=""),
        _item("no-company", employer_name=None),
        {**_item("x"), "job_id": None},
    )
    posts = JSearchProvider(api_key="x").parse(raw)
    assert [p.id for p in posts] == ["jsearch:ok"]


def test_parse_reads_v5_and_legacy_shapes():
    v5 = _raw(_item("a"))
    legacy = {"status": "OK", "data": [_item("a")]}          # pre-v5: data was the list
    prov = JSearchProvider(api_key="x")
    assert [p.id for p in prov.parse(v5)] == ["jsearch:a"]
    assert [p.id for p in prov.parse(legacy)] == ["jsearch:a"]
    assert prov.next_cursor(v5) == "next-token" and prov.next_cursor(legacy) is None
    assert prov.parse({"status": "OK", "data": {"jobs": [], "cursor": "c"}}) == []


def test_parse_normalises_fields():
    post = JSearchProvider(api_key="x").parse(_raw(_item("a", job_is_remote=True)))[0]
    assert post.source == "jsearch" and post.publisher == "LinkedIn"
    assert post.location == "Bangalore, Karnataka, IN"
    assert post.remote is True
    assert post.employment_type == "FULLTIME"
    assert post.url == "https://example.com/a"
    assert post.slug.startswith("acme-data-scientist-")
    assert len(post.description) >= MIN_DESCRIPTION_CHARS


# --------------------------------------------------------------- request

def test_jsearch_request_contract():
    """Pins what we send RapidAPI (JSearch v5): endpoint, query text,
    paging, headers. If JSearch changes its interface, this is the test
    that should break — as it did going from /search to /search-v2."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        seen["headers"] = request.headers
        return httpx.Response(200, json=_raw(_item("a")))

    prov = JSearchProvider(api_key="secret", client=httpx.Client(transport=httpx.MockTransport(handler)))
    raw = prov.fetch_raw(JobQuery(keyword="data scientist fraud", location="Bangalore", num_pages=2))

    assert raw["status"] == "OK"
    url = seen["url"]
    assert url.host == "jsearch.p.rapidapi.com" and url.path == "/search-v2"
    assert url.params["query"] == "data scientist fraud in Bangalore"
    assert url.params["location"] == "Bangalore"                       # UULE anchor, not just query text
    assert url.params["num_pages"] == "2"
    assert "page" not in url.params and "cursor" not in url.params   # v5: no page; cursor only when given
    assert url.params["country"] == "in" and url.params["date_posted"] == "month"
    assert seen["headers"]["x-rapidapi-key"] == "secret"
    assert seen["headers"]["x-rapidapi-host"] == "jsearch.p.rapidapi.com"


def test_jsearch_cursor_is_sent_when_given():
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        return httpx.Response(200, json=_raw())

    prov = JSearchProvider(api_key="x", client=httpx.Client(transport=httpx.MockTransport(handler)))
    prov.fetch_raw(JobQuery(keyword="q", cursor="next-token"))
    assert seen["url"].params["cursor"] == "next-token"
    assert "location" not in seen["url"].params                        # omitted when not given


def _scripted(*outcomes):
    """A JSearchProvider whose transport plays `outcomes` in order: an
    httpx.Response is returned, an exception class is raised. Records the
    calls and the waits instead of sleeping."""
    calls, waits = [], []

    def handler(request: httpx.Request) -> httpx.Response:
        outcome = outcomes[len(calls)]
        calls.append(request)
        if isinstance(outcome, type) and issubclass(outcome, Exception):
            raise outcome("slow", request=request)
        return outcome

    prov = JSearchProvider(api_key="x", client=httpx.Client(transport=httpx.MockTransport(handler)),
                           sleep=waits.append)
    return prov, calls, waits


@pytest.mark.parametrize("response", [
    httpx.Response(429, text="quota exceeded"),
    httpx.Response(401, text="bad key"),
    httpx.Response(200, json={"status": "ERROR", "error": {"message": "bad"}}),
])
def test_jsearch_errors_are_loud_and_not_retried(response):
    """Failures that would repeat identically cost one request, not three."""
    prov, calls, waits = _scripted(response)
    with pytest.raises(ProviderError):
        prov.fetch_raw(JobQuery(keyword="q"))
    assert len(calls) == 1 and waits == []


def test_gateway_timeout_is_retried_then_succeeds():
    """The 2026-09-27 failure: RapidAPI's gateway returned 504 because the
    upstream took too long. That may not repeat, so it is retried."""
    prov, calls, waits = _scripted(httpx.Response(504, text="timed out"), httpx.Response(200, json=_raw(_item("a"))))
    assert prov.fetch_raw(JobQuery(keyword="q"))["status"] == "OK"
    assert len(calls) == 2 and waits == [JSearchProvider.RETRY_WAITS[0]]


def test_transient_failures_give_up_after_three_attempts():
    prov, calls, waits = _scripted(*[httpx.Response(504, text="timed out")] * 3)
    with pytest.raises(ProviderError, match=r"HTTP 504.*gave up after 3 attempts"):
        prov.fetch_raw(JobQuery(keyword="q"))
    assert len(calls) == 3 and waits == list(JSearchProvider.RETRY_WAITS)


def test_jsearch_timeout_is_retried_then_a_provider_error():
    prov, calls, waits = _scripted(httpx.ReadTimeout, httpx.ReadTimeout, httpx.ReadTimeout)
    with pytest.raises(ProviderError, match="did not answer.*gave up after 3 attempts"):
        prov.fetch_raw(JobQuery(keyword="q"))
    assert len(calls) == 3


def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv(JSearchProvider.ENV_KEY, raising=False)
    with pytest.raises(MissingApiKey, match=JSearchProvider.ENV_KEY):
        JSearchProvider()


# ------------------------------------------------------------------ dump

def test_dump_jd_writes_header_and_description(tmp_path):
    post = JSearchProvider(api_key="x").parse(_raw(_item("a")))[0]
    path = dump_jd(post, tmp_path)
    assert path == tmp_path / f"{post.slug}.txt"
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Data Scientist — Acme\n")
    assert text.rstrip().endswith(LONG.strip())
