"""Job sourcing: keyword -> cached raw responses -> normalised `JobPost`s.

The Sourcer node's tool (roadmap Phase 4 / 5). Given a keyword and an
optional location it returns job postings with their *full* description
text — the Extractor's input — and it never hits the network twice for
the same query during development.

Design:
  * `fetch_raw()` and `parse()` are separate steps on the provider, and
    only the raw response is cached. Parsing is re-run from cache on
    every call, so a parser fix or a new `JobPost` field never costs a
    request — the same split as `build_html()` / `Renderer` in render.py:
    the cheap, deterministic half is repeatable; the expensive half is
    done once.
  * `Provider` is a Protocol, and `fetch_jobs()` takes the provider as an
    argument. Tests inject a fake with canned responses (credit-free,
    like the Tailor's injected `client`); a second job board is a second
    class, not a rewrite.
  * JSearch (RapidAPI) first: it returns full JD text, which the
    Extractor needs. Adzuna's free API truncates descriptions to a
    snippet. The free tier is ~200 requests/month — hence the cache.
  * A cache entry's filename is the slugged query plus a hash of the
    canonical query dict, so `data/jobs_cache/` stays greppable and two
    queries differing only in page count or country never collide.
  * Posts with no usable description are dropped at parse time and
    duplicates (same id, or same title+company from two publishers) are
    dropped at merge time. No evidence is better than a JD the Extractor
    would hallucinate requirements from.
  * One page per request, following the cursor. Asking JSearch for three
    geo-anchored pages in one request made RapidAPI's gateway give up
    (HTTP 504, 2026-09-27). Short requests, each page cached on its own,
    and a failure on page 3 keeps pages 1-2.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol

import httpx
from pydantic import Field, StringConstraints

from src.models import NonEmptyStr, StrictModel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_DIR = ROOT / "data" / "jobs_cache"

# A JD shorter than this is a teaser, not a description. The Extractor
# would be guessing; drop it rather than brief the Tailor on a guess.
MIN_DESCRIPTION_CHARS = 200

DatePosted = Literal["all", "today", "3days", "week", "month"]


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

class JobPost(StrictModel):
    """One normalised posting. Provider-agnostic: the Extractor, Ranker
    and the graph state see this shape, never a provider's JSON."""

    id: NonEmptyStr                     # "<provider>:<provider's id>"
    source: NonEmptyStr                 # provider name, e.g. "jsearch"
    publisher: str | None = None        # board it was scraped from (LinkedIn, Indeed, ...)
    title: NonEmptyStr
    company: NonEmptyStr
    location: str | None = None
    remote: bool = False
    employment_type: str | None = None
    posted_at: str | None = None        # ISO 8601 as the provider gives it
    url: str | None = None
    description: Annotated[str, StringConstraints(min_length=MIN_DESCRIPTION_CHARS)]

    @property
    def slug(self) -> str:
        """Filesystem-safe handle: 'ebay-data-scientist-3f2a9c1b'."""
        base = _slugify(f"{self.company} {self.title}")[:60].rstrip("-")
        return f"{base}-{_short_hash(self.id)}"


class JobQuery(StrictModel):
    """What one provider request asks for. The cache key is derived from
    this, so every field that changes the results must live here."""

    keyword: NonEmptyStr
    location: str | None = None
    country: str = "in"                 # ISO 3166-1 alpha-2, lower-case
    num_pages: int = Field(default=1, ge=1, le=20)   # 10 results per page, one credit per page
    cursor: str | None = None           # v5 pagination: `cursor` from a previous response
    date_posted: DatePosted = "month"

    @property
    def text(self) -> str:
        """The free-text query providers want: 'data scientist in Bangalore'."""
        return f"{self.keyword} in {self.location}" if self.location else self.keyword

    def cache_key(self, provider: str) -> str:
        canonical = json.dumps({"provider": provider, **self.model_dump()}, sort_keys=True)
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()


class Provider(Protocol):
    name: str

    def fetch_raw(self, query: JobQuery) -> dict[str, Any]: ...
    def parse(self, raw: dict[str, Any]) -> list[JobPost]: ...
    def next_cursor(self, raw: dict[str, Any]) -> str | None: ...


class ProviderError(RuntimeError):
    """The provider answered, but not with jobs (bad key, quota, API error)."""


class MissingApiKey(ProviderError):
    pass


class StaleCacheWarning(UserWarning):
    """Postings were served from a cache older than the caller allows."""


# What the CLIs pass as `stale_after_days`. Postings are searched with
# date_posted="month", so a pool older than a week already holds closed jobs.
STALE_AFTER_DAYS = 7


# ---------------------------------------------------------------------------
# JSearch (RapidAPI)
# ---------------------------------------------------------------------------

class JSearchProvider:
    """https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch

    API v5. GET https://jsearch.p.rapidapi.com/search-v2
        ?query=...&country=in&num_pages=1&date_posted=month
        [&location=...][&cursor=...]
    headers: x-rapidapi-key, x-rapidapi-host
    Location is sent twice on purpose: "in <location>" inside `query` is
    only a hint to Google's ranking; the separate `location` parameter
    (Google's UULE — where the search is made *from*) is what anchors the
    results. Query-only gave Mumbai jobs for a Bangalore search.
    Pagination is cursor-based: `num_pages` pages (10 results each, one
    credit per page) come back in one response, and a `cursor` in the
    response fetches the next page on a later request. There is no `page`
    parameter in v5 — that was the pre-v5 interface.
    response: {"status": "OK", "request_id", "parameters": {...},
               "data": {"jobs": [ {job_id, job_title, employer_name,
                                   job_description, job_apply_link,
                                   job_city, job_state, job_country,
                                   job_is_remote, job_employment_type,
                                   job_posted_at_datetime_utc,
                                   job_publisher, ...} ],
                        "cursor": "..."}}
    (pre-v5, `data` was the jobs list itself; `jobs()` accepts both.)
    """

    name = "jsearch"
    HOST = "jsearch.p.rapidapi.com"
    ENV_KEY = "RAPIDAPI_KEY"
    # JSearch does a live search behind this call. ~8s is typical, but
    # geo-anchored queries (the `location` parameter) can take far longer;
    # 30s produced ReadTimeouts. Connect fast, read patiently.
    TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)
    # Transport retries, the same idea as the Extractor's MAX_PROVIDER_RETRIES:
    # resend the identical request when the failure is on the provider's side
    # and may not repeat — the gateway timing out on a slow upstream (504),
    # the upstream briefly down (502/503), or our own read timeout. A bad key
    # (401/403), an exhausted quota (429) or a bad request (400) would fail
    # the same way again, so those raise at once. Kept small: the free tier
    # is ~200 requests a month.
    RETRY_STATUSES = frozenset({502, 503, 504})
    RETRY_WAITS = (5.0, 15.0)   # seconds before retry 1 and retry 2

    def __init__(self, api_key: str | None = None, client: httpx.Client | None = None,
                 sleep: Any = time.sleep):
        key = api_key or os.environ.get(self.ENV_KEY)
        if not key:
            raise MissingApiKey(
                f"{self.ENV_KEY} is not set. Subscribe to JSearch on RapidAPI "
                f"and put the key in .env as {self.ENV_KEY}=..."
            )
        self._key = key
        self._client = client or httpx.Client(timeout=self.TIMEOUT)
        self._sleep = sleep     # injectable so tests don't wait

    def fetch_raw(self, query: JobQuery) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query": query.text,
            "country": query.country,
            "num_pages": query.num_pages,
            "date_posted": query.date_posted,
        }
        if query.location:
            params["location"] = query.location
        if query.cursor:
            params["cursor"] = query.cursor

        attempts = len(self.RETRY_WAITS) + 1
        for attempt in range(1, attempts + 1):
            try:
                response = self._client.get(
                    f"https://{self.HOST}/search-v2",
                    params=params,
                    headers={"x-rapidapi-key": self._key, "x-rapidapi-host": self.HOST},
                )
            except httpx.TimeoutException as e:
                transient = ProviderError(
                    f"JSearch did not answer within {self.TIMEOUT.read:.0f}s for {query.text!r}")
                transient.__cause__ = e
            except httpx.HTTPError as e:
                raise ProviderError(f"network error calling JSearch: {e!r}") from e
            else:
                if response.status_code not in self.RETRY_STATUSES:
                    break
                transient = ProviderError(
                    f"JSearch returned HTTP {response.status_code}: {response.text[:300]}")
            if attempt == attempts:
                raise ProviderError(f"{transient} (gave up after {attempts} attempts)") from transient
            self._sleep(self.RETRY_WAITS[attempt - 1])

        if response.status_code != 200:
            raise ProviderError(
                f"JSearch returned HTTP {response.status_code}: {response.text[:300]}"
            )
        raw = response.json()
        if raw.get("status") != "OK":
            raise ProviderError(f"JSearch status {raw.get('status')!r}: {str(raw)[:300]}")
        return raw

    @staticmethod
    def jobs(raw: dict[str, Any]) -> list[dict[str, Any]]:
        data = raw.get("data") or {}
        return data.get("jobs") or [] if isinstance(data, dict) else data

    @staticmethod
    def next_cursor(raw: dict[str, Any]) -> str | None:
        data = raw.get("data") or {}
        return data.get("cursor") or None if isinstance(data, dict) else None

    def parse(self, raw: dict[str, Any]) -> list[JobPost]:
        posts: list[JobPost] = []
        for item in self.jobs(raw):
            description = (item.get("job_description") or "").strip()
            job_id = item.get("job_id")
            title = (item.get("job_title") or "").strip()
            company = (item.get("employer_name") or "").strip()
            if not (job_id and title and company) or len(description) < MIN_DESCRIPTION_CHARS:
                continue
            posts.append(JobPost(
                id=f"{self.name}:{job_id}",
                source=self.name,
                publisher=item.get("job_publisher") or None,
                title=title,
                company=company,
                location=_join_location(item.get("job_city"), item.get("job_state"),
                                        item.get("job_country")),
                remote=bool(item.get("job_is_remote")),
                employment_type=item.get("job_employment_type") or None,
                posted_at=item.get("job_posted_at_datetime_utc") or None,
                url=item.get("job_apply_link") or None,
                description=description,
            ))
        return posts


def _join_location(*parts: str | None) -> str | None:
    kept = [p.strip() for p in parts if p and p.strip()]
    return ", ".join(kept) or None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def cache_path(cache_dir: Path, provider: str, query: JobQuery) -> Path:
    slug = _slugify(f"{query.keyword} {query.location or ''}")
    return cache_dir / provider / f"{slug}-n{query.num_pages}-{query.cache_key(provider)[:8]}.json"


def _cached_raw(provider: Provider, query: JobQuery, cache_dir: Path,
                refresh: bool) -> tuple[dict[str, Any], datetime | None]:
    """-> (raw response, when it was fetched). The time is None for a
    response fetched just now, and for a cache entry that records none."""
    path = cache_path(cache_dir, provider.name, query)
    if path.exists() and not refresh:
        entry = json.loads(path.read_text(encoding="utf-8"))
        try:
            fetched_at = datetime.fromisoformat(entry["fetched_at"])
        except (KeyError, TypeError, ValueError):
            fetched_at = None
        return entry["raw"], fetched_at
    raw = provider.fetch_raw(query)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "provider": provider.name,
        "query": query.model_dump(),
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw": raw,
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    return raw, None


# ---------------------------------------------------------------------------
# The roadmap signature
# ---------------------------------------------------------------------------

def fetch_jobs(
    keyword: str,
    location: str | None = None,
    *,
    provider: Provider | None = None,
    pages: int = 1,
    country: str = "in",
    date_posted: DatePosted = "month",
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    stale_after_days: float | None = None,
) -> list[JobPost]:
    """keyword (+ location) -> de-duplicated JobPosts from up to `pages` pages.

    One request per page (`num_pages=1`, one credit each), following the
    `cursor` each response carries to the next page. Paging stops early
    when a response has no cursor or a page yields no usable posts, so a
    thin query does not spend credits on empty pages.

    Every page is cached on its own under `cache_dir/<provider>/`; the
    cursor is part of the cache key, so page 2's file is tied to the page 1
    it came from. `refresh=True` re-fetches every page.

    Page 1 failing raises. A later page failing (after the provider's own
    retries) stops paging with a warning and returns what was fetched: those
    pages are cached, so running again *without* `refresh` resumes at the
    failed page and costs only the pages still missing.

    The cache has no expiry: a cached page is served however old it is,
    which is what the evals need (they score a fixed, labelled pool). A
    caller that wants live postings passes `stale_after_days`; if any page
    served from the cache is older than that, a StaleCacheWarning says how
    old the pool is. It warns and never re-fetches: a re-fetch spends
    requests and overwrites the cached pool, so that stays the caller's
    decision (`refresh=True`).

    `provider` defaults to JSearch, which needs RAPIDAPI_KEY; inject a fake
    in tests.
    """
    if pages < 1:
        raise ValueError("pages must be at least 1")
    provider = provider or JSearchProvider()

    posts: list[JobPost] = []
    seen: set[str] = set()
    cursor: str | None = None
    oldest: datetime | None = None      # the oldest page served from the cache
    for page in range(1, pages + 1):
        query = JobQuery(keyword=keyword, location=location, country=country,
                         num_pages=1, cursor=cursor, date_posted=date_posted)
        try:
            raw, fetched_at = _cached_raw(provider, query, Path(cache_dir), refresh)
            if fetched_at is not None and (oldest is None or fetched_at < oldest):
                oldest = fetched_at
        except ProviderError as e:
            if page == 1:
                raise
            warnings.warn(f"stopped after page {page - 1} of {pages}: {e}. The fetched pages are "
                          "cached; run again without --refresh to resume.", stacklevel=2)
            break

        page_posts = provider.parse(raw)
        for post in page_posts:
            # Same posting from two boards gets two ids; title+company catches it.
            keys = {post.id, _slugify(f"{post.company} {post.title}")}
            if keys & seen:
                continue
            seen |= keys
            posts.append(post)

        cursor = provider.next_cursor(raw)
        if not cursor or not page_posts:
            break

    if stale_after_days is not None and oldest is not None:
        if oldest.tzinfo is None:
            oldest = oldest.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - oldest).total_seconds() / 86400
        if age_days > stale_after_days:
            warnings.warn(StaleCacheWarning(
                f"these postings come from the cache, fetched {oldest:%Y-%m-%d} ({age_days:.0f} days "
                "ago); some will have closed. --refresh fetches current ones: one JSearch request per "
                "page, and it overwrites this cached pool, including one a labels file was made for."),
                stacklevel=2)
    return posts


def dump_jd(post: JobPost, out_dir: Path | str) -> Path:
    """Write one JD as `<slug>.txt` with a short header, the same shape as
    the hand-collected `data/*.txt` JDs — for hand-labelling eval cases."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{post.slug}.txt"
    header = [f"# {post.title} — {post.company}",
              f"# {post.location or 'location n/a'} | {post.source}:{post.publisher or '?'} | {post.posted_at or 'date n/a'}",
              f"# {post.url or ''}".rstrip(), ""]
    path.write_text("\n".join(header) + post.description.strip() + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    # python -m src.tools.jobs "data scientist fraud" --location Bangalore
    # python -m src.tools.jobs "data scientist fraud" --pages 2 --dump
    # python -m src.tools.jobs "ml engineer" --show 3          (row number from the table)
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    ap = argparse.ArgumentParser(description="Fetch (and cache) job postings.")
    ap.add_argument("keyword")
    ap.add_argument("--location", default=None)
    ap.add_argument("--country", default="in")
    ap.add_argument("--pages", type=int, default=1,
                    help="up to N pages of 10, one request and one credit each; stops early when results run out")
    ap.add_argument("--date-posted", default="month", choices=["all", "today", "3days", "week", "month"])
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-fetch")
    ap.add_argument("--stale-after", type=float, default=STALE_AFTER_DAYS, metavar="DAYS",
                    help=f"warn when cached postings are older than this (default {STALE_AFTER_DAYS})")
    ap.add_argument("--dump", action="store_true", help="write each JD to data/jobs_cache/jd/<slug>.txt")
    ap.add_argument("--show", default=None, help="print one JD in full: a row number from the table, or a job id")
    args = ap.parse_args()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        posts = fetch_jobs(args.keyword, args.location, pages=args.pages, country=args.country,
                           date_posted=args.date_posted, refresh=args.refresh,
                           stale_after_days=args.stale_after)
    for w in caught:
        print(f"warning: {w.message}")
    print(f"{len(posts)} posting(s) for {args.keyword!r}"
          f"{' in ' + args.location if args.location else ''} (up to {args.pages} page(s))")
    # Ids can run to hundreds of characters (Google-sourced ones do), so the
    # table shows the slug's hash instead; --show takes the row number.
    for i, p in enumerate(posts, 1):
        print(f"{i:>2}. {p.slug[-8:]}  {p.title[:44]:<44} {p.company[:24]:<24} "
              f"{(p.location or '-')[:20]:<20} {len(p.description):>5} chars")

    if args.dump:
        out = DEFAULT_CACHE_DIR / "jd"
        for p in posts:
            dump_jd(p, out)
        print(f"wrote {len(posts)} JD(s) to {out}")

    if args.show:
        if args.show.isdigit() and 1 <= int(args.show) <= len(posts):
            match = posts[int(args.show) - 1]
        else:
            match = next((p for p in posts if p.id == args.show or p.slug.endswith(args.show)), None)
        if match is None:
            ap.error(f"{args.show!r} is not a row number, id, or slug hash in this result set")
        print(f"\n# {match.title} — {match.company}\n# {match.url}\n\n{match.description}")
