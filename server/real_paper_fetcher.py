"""
server/real_paper_fetcher.py — Real paper fetching for ScholarEnv tasks T6/T7/T8.

Provides:
  ArxivFetcher          — fetch abstract+metadata from arXiv (free, no key needed)
  SemanticScholarFetcher — fetch citations, metadata, tldr (free, no key needed)
  CrossRefFetcher       — check DOI metadata, retraction status (free, no key needed)
  RetractionWatchCache  — load local retraction CSV (no API needed)
  FirecrawlFetcher      — full paper text via Firecrawl (key needed, optional)
  RealPaperFetcher      — unified facade used by tasks T6, T7, T8

ALL fetchers degrade gracefully:
  - Network unavailable  → returns empty dict, task uses synthetic fallback
  - API key missing      → skips that source, others still run
  - Rate limited         → exponential backoff, max 2 retries

Authors: Nensi Pansuriya, Krushna Parmar, Ishita Bhojani
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ── Timeouts + retry config ───────────────────────────────────────────────────
_TIMEOUT    = 8    # seconds per request
_MAX_RETRY  = 2
_RETRY_WAIT = 1.5  # seconds

_HEADERS = {
    "User-Agent": "ScholarEnv/2.0 (hackathon research tool; contact: scholarenv@example.com)",
    "Accept":     "application/json",
}


def _get(url: str, headers: dict | None = None, timeout: int = _TIMEOUT) -> dict | list | None:
    """GET with retry + graceful failure. Returns parsed JSON or None."""
    h = {**_HEADERS, **(headers or {})}
    for attempt in range(_MAX_RETRY + 1):
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception:
            if attempt < _MAX_RETRY:
                time.sleep(_RETRY_WAIT * (attempt + 1))
    return None


def _post(url: str, payload: dict, headers: dict | None = None) -> dict | None:
    """POST with retry. Returns parsed JSON or None."""
    h = {**_HEADERS, "Content-Type": "application/json", **(headers or {})}
    data = json.dumps(payload).encode()
    for attempt in range(_MAX_RETRY + 1):
        try:
            req = urllib.request.Request(url, data=data, headers=h, method="POST")
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                return json.loads(r.read().decode("utf-8", errors="replace"))
        except Exception:
            if attempt < _MAX_RETRY:
                time.sleep(_RETRY_WAIT * (attempt + 1))
    return None


# ── Dataclasses ───────────────────────────────────────────────────────────────
@dataclass
class RealPaperMeta:
    arxiv_id:    str   = ""
    doi:         str   = ""
    title:       str   = ""
    authors:     list  = field(default_factory=list)
    year:        int   = 0
    abstract:    str   = ""
    full_text:   str   = ""     # populated by Firecrawl if available
    tldr:        str   = ""     # from Semantic Scholar
    citations:   list  = field(default_factory=list)   # list of {title, year, authors}
    retracted:   bool  = False
    retraction_reason: str = ""
    versions:    list  = field(default_factory=list)   # arxiv version list
    semantic_id: str   = ""
    source:      str   = ""     # which API provided this
    fetched_at:  float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "arxiv_id": self.arxiv_id, "doi": self.doi,
            "title": self.title[:200], "authors": self.authors[:5],
            "year": self.year, "abstract": self.abstract[:1000],
            "tldr": self.tldr[:300], "retracted": self.retracted,
            "retraction_reason": self.retraction_reason[:300],
            "n_versions": len(self.versions), "n_citations": len(self.citations),
            "source": self.source,
        }


# ── 1. arXiv Fetcher (no key needed) ─────────────────────────────────────────
class ArxivFetcher:
    """
    Uses arXiv REST API v2 (atom+json export).
    Rate limit: ~3 req/sec. We only ever do 1 per task so this is fine.
    """
    BASE = "https://export.arxiv.org/abs/"
    API  = "https://export.arxiv.org/search/?searchtype=all&query={q}&start=0&max_results=3"
    ATOM = "https://export.arxiv.org/api/query?id_list={id}&max_results=1"

    _ARXIV_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)", re.I)

    def fetch_by_id(self, arxiv_id: str) -> RealPaperMeta | None:
        """Fetch paper by arXiv ID like '2312.11805' or '2312.11805v2'."""
        clean_id = self._ARXIV_RE.search(arxiv_id)
        if not clean_id:
            return None
        aid = clean_id.group(1).split("v")[0]  # strip version for base fetch

        # Use arXiv Atom API (most reliable)
        url  = self.ATOM.format(id=aid)
        data = _get(url, headers={"Accept": "application/atom+xml"})
        # Atom returns XML, not JSON — parse manually
        try:
            import urllib.request as _ur
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                xml = r.read().decode("utf-8", errors="replace")
        except Exception:
            return None

        meta        = RealPaperMeta(arxiv_id=aid, source="arxiv")
        meta.title  = self._extract_xml(xml, "title", strip_tags=True)
        meta.abstract = self._extract_xml(xml, "summary", strip_tags=True).strip()
        year_m      = re.search(r"<published>(\d{4})", xml)
        if year_m:
            meta.year = int(year_m.group(1))
        authors     = re.findall(r"<name>([^<]+)</name>", xml)
        meta.authors = authors[:8]

        # Fetch version list
        meta.versions = self._fetch_versions(aid)
        return meta

    def fetch_by_query(self, query: str, max_results: int = 3) -> list[RealPaperMeta]:
        """Search arXiv for papers matching a query."""
        q   = urllib.parse.quote(query)
        url = self.API.format(q=q)
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                xml = r.read().decode("utf-8", errors="replace")
        except Exception:
            return []
        ids = re.findall(r"<id>http[^<]+abs/([^<v]+)", xml)
        results = []
        for aid in ids[:max_results]:
            m = self.fetch_by_id(aid.strip())
            if m:
                results.append(m)
        return results

    def _fetch_versions(self, arxiv_id: str) -> list[dict]:
        """Fetch version history from arXiv abstract page."""
        try:
            url = self.BASE + arxiv_id
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as r:
                html = r.read().decode("utf-8", errors="replace")
            # Versions listed as vN links in submission history
            versions = re.findall(
                r'<a href="/abs/' + re.escape(arxiv_id) + r'v(\d+)"[^>]*>[^<]+</a>\s*\(([^)]+)\)',
                html,
            )
            return [{"version": f"v{v}", "date": d.strip()} for v, d in versions]
        except Exception:
            return []

    @staticmethod
    def _extract_xml(xml: str, tag: str, strip_tags: bool = False) -> str:
        m = re.search(fr"<{tag}[^>]*>(.*?)</{tag}>", xml, re.DOTALL)
        if not m:
            return ""
        text = m.group(1)
        if strip_tags:
            text = re.sub(r"<[^>]+>", " ", text)
        return re.sub(r"\s+", " ", text).strip()


# ── 2. Semantic Scholar (no key needed, 100 req/5min) ────────────────────────
class SemanticScholarFetcher:
    """
    Uses Semantic Scholar public API.
    Returns: title, year, authors, tldr, citations (titles only).
    """
    BASE = "https://api.semanticscholar.org/graph/v1"

    def fetch_by_arxiv(self, arxiv_id: str) -> RealPaperMeta | None:
        clean = arxiv_id.split("v")[0].strip()
        url   = (f"{self.BASE}/paper/arXiv:{clean}"
                 f"?fields=title,year,authors,abstract,tldr,citations.title,citations.year")
        data  = _get(url)
        if not data or "title" not in data:
            return None
        return self._parse(data)

    def fetch_by_doi(self, doi: str) -> RealPaperMeta | None:
        doi_enc = urllib.parse.quote(doi, safe="")
        url     = (f"{self.BASE}/paper/{doi_enc}"
                   f"?fields=title,year,authors,abstract,tldr,citations.title,citations.year")
        data    = _get(url)
        if not data or "title" not in data:
            return None
        return self._parse(data)

    def search(self, query: str, limit: int = 3) -> list[RealPaperMeta]:
        q   = urllib.parse.quote(query)
        url = f"{self.BASE}/paper/search?query={q}&limit={limit}&fields=title,year,authors,abstract,tldr"
        data = _get(url)
        if not data or "data" not in data:
            return []
        results = []
        for item in data["data"][:limit]:
            m = self._parse(item)
            if m:
                results.append(m)
        return results

    @staticmethod
    def _parse(data: dict) -> RealPaperMeta:
        meta = RealPaperMeta(source="semantic_scholar")
        meta.semantic_id = data.get("paperId", "")
        meta.title       = data.get("title", "")
        meta.year        = data.get("year", 0) or 0
        meta.abstract    = (data.get("abstract") or "")[:1000]
        meta.tldr        = (data.get("tldr") or {}).get("text", "")
        meta.authors     = [a.get("name", "") for a in data.get("authors", [])[:5]]
        cites            = data.get("citations", []) or []
        meta.citations   = [
            {"title": c.get("title", ""), "year": c.get("year", 0)}
            for c in cites[:20]
        ]
        return meta


# ── 3. CrossRef (DOI resolution + retraction check, no key needed) ────────────
class CrossRefFetcher:
    """
    Uses CrossRef public API.
    Checks: title, year, authors, update-type (retraction marker).
    """
    BASE = "https://api.crossref.org/works"

    def fetch_by_doi(self, doi: str) -> RealPaperMeta | None:
        enc  = urllib.parse.quote(doi, safe="")
        url  = f"{self.BASE}/{enc}"
        data = _get(url, headers={"mailto": "scholarenv@example.com"})
        if not data:
            return None
        msg = data.get("message", {})
        return self._parse(msg)

    def is_retracted(self, doi: str) -> tuple[bool, str]:
        """Returns (retracted: bool, reason: str)."""
        meta = self.fetch_by_doi(doi)
        if not meta:
            return False, "could not resolve DOI"
        return meta.retracted, meta.retraction_reason

    @staticmethod
    def _parse(msg: dict) -> RealPaperMeta:
        meta       = RealPaperMeta(source="crossref")
        meta.doi   = msg.get("DOI", "")
        meta.title = " ".join(msg.get("title", [""])) or ""
        meta.year  = (msg.get("published", {}).get("date-parts", [[0]])[0] or [0])[0]
        meta.authors = [
            f"{a.get('given','')} {a.get('family','')}".strip()
            for a in msg.get("author", [])[:5]
        ]
        # Retraction detection: update-to + type
        update_type = msg.get("update-type", "")
        if "retract" in update_type.lower():
            meta.retracted          = True
            meta.retraction_reason  = update_type
        # Also check relation field for retraction-of
        relation = msg.get("relation", {})
        if "is-retraction-of" in relation or "is-correction-of" in relation:
            meta.retracted         = True
            meta.retraction_reason = "CrossRef relation: is-retraction-of"
        return meta


# ── 4. Retraction Watch CSV (offline, ships with ScholarEnv) ─────────────────
class RetractionWatchCache:
    """
    Uses local copy of the Retraction Watch database.
    Download from: https://retractionwatch.com/retraction-watch-database-user-guide/
    Place at: data/retraction_watch.csv
    Falls back gracefully if file not present.
    """
    _DATA_PATH = Path(__file__).parent.parent / "data" / "retraction_watch.csv"
    _cache: dict[str, dict] | None = None

    def _load(self) -> dict[str, dict]:
        if self.__class__._cache is not None:
            return self.__class__._cache
        if not self._DATA_PATH.exists():
            self.__class__._cache = {}
            return {}
        import csv
        cache: dict[str, dict] = {}
        try:
            with open(self._DATA_PATH, encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    doi = (row.get("OriginalPaperDOI") or "").strip().lower()
                    if doi:
                        cache[doi] = {
                            "title":   row.get("Title", ""),
                            "reason":  row.get("Reason", ""),
                            "date":    row.get("RetractionDate", ""),
                            "journal": row.get("Journal", ""),
                        }
        except Exception:
            pass
        self.__class__._cache = cache
        return cache

    def check_doi(self, doi: str) -> tuple[bool, str]:
        """Returns (retracted: bool, reason: str)."""
        db     = self._load()
        record = db.get(doi.lower().strip())
        if record:
            return True, record.get("reason", "listed in Retraction Watch")
        return False, ""

    def check_title(self, title: str) -> tuple[bool, str]:
        """Fuzzy title match against retraction database."""
        if not title:
            return False, ""
        db = self._load()
        title_l = title.lower().strip()
        for doi, record in db.items():
            if _title_sim(title_l, record.get("title", "").lower()) > 0.85:
                return True, record.get("reason", "title match in Retraction Watch")
        return False, ""


def _title_sim(a: str, b: str) -> float:
    try:
        from thefuzz import fuzz
        return fuzz.token_set_ratio(a, b) / 100.0
    except ImportError:
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()


# ── 5. Firecrawl (full text, optional, key from env) ─────────────────────────
class FirecrawlFetcher:
    """
    Fetches full paper text via Firecrawl API.
    Requires FIRECRAWL_KEY env var. Degrades gracefully if absent.
    """
    API = "https://api.firecrawl.dev/v1/scrape"

    def __init__(self, api_key: str = ""):
        import os
        self.key = api_key or os.environ.get("FIRECRAWL_KEY", "")

    def fetch_url(self, url: str) -> str:
        """Returns markdown text of the page, or empty string."""
        if not self.key:
            return ""
        payload = {"url": url, "formats": ["markdown"], "onlyMainContent": True}
        resp = _post(self.API, payload,
                     headers={"Authorization": f"Bearer {self.key}"})
        if not resp:
            return ""
        return resp.get("data", {}).get("markdown", "") or ""

    def fetch_arxiv_full(self, arxiv_id: str) -> str:
        return self.fetch_url(f"https://arxiv.org/abs/{arxiv_id}")

    def fetch_doi_full(self, doi: str) -> str:
        return self.fetch_url(f"https://doi.org/{doi}")


# ── 6. Unified facade ────────────────────────────────────────────────────────
class RealPaperFetcher:
    """
    Unified entry point for all real-paper fetching.
    Used by T6 (cross-paper consistency), T7 (version drift), T8 (retraction).

    Resolution order:
      1. arXiv (always tried, no key)
      2. Semantic Scholar (always tried, no key)
      3. CrossRef (for DOIs, no key)
      4. RetractionWatch (local CSV)
      5. Firecrawl (full text, only if key present)
    """

    def __init__(self, firecrawl_key: str = ""):
        self.arxiv    = ArxivFetcher()
        self.ss       = SemanticScholarFetcher()
        self.crossref = CrossRefFetcher()
        self.rw       = RetractionWatchCache()
        self.fc       = FirecrawlFetcher(api_key=firecrawl_key)

    def fetch(self, arxiv_id: str = "", doi: str = "",
              query: str = "") -> RealPaperMeta | None:
        """
        Fetch paper metadata from all available sources and merge.
        At least one of arxiv_id / doi / query must be provided.
        """
        meta = RealPaperMeta()

        # Try arXiv first (most reliable for ML papers)
        if arxiv_id:
            m = self.arxiv.fetch_by_id(arxiv_id)
            if m:
                meta = m

        # Enrich with Semantic Scholar
        if arxiv_id:
            m2 = self.ss.fetch_by_arxiv(arxiv_id)
            if m2:
                if not meta.title: meta.title = m2.title
                if not meta.abstract: meta.abstract = m2.abstract
                meta.tldr      = m2.tldr
                meta.citations = m2.citations
                meta.semantic_id = m2.semantic_id
                if not meta.year and m2.year: meta.year = m2.year
                if not meta.authors and m2.authors: meta.authors = m2.authors

        # CrossRef for retraction status
        if doi:
            m3 = self.crossref.fetch_by_doi(doi)
            if m3:
                meta.doi       = doi
                meta.retracted = m3.retracted
                meta.retraction_reason = m3.retraction_reason

        # Retraction Watch check (both DOI and title)
        if doi:
            rw_r, rw_reason = self.rw.check_doi(doi)
            if rw_r:
                meta.retracted = True
                meta.retraction_reason = rw_reason
        if meta.title and not meta.retracted:
            rw_r, rw_reason = self.rw.check_title(meta.title)
            if rw_r:
                meta.retracted = True
                meta.retraction_reason = rw_reason

        # Full text via Firecrawl
        if arxiv_id and self.fc.key:
            meta.full_text = self.fc.fetch_arxiv_full(arxiv_id)

        # Query fallback
        if not meta.title and query:
            results = self.ss.search(query, limit=1)
            if results:
                meta = results[0]

        return meta if (meta.title or meta.abstract) else None

    def fetch_cited_papers(self, refs: list[dict],
                           max_fetch: int = 5) -> dict[str, RealPaperMeta]:
        """
        Given a list of reference dicts {citation_id, raw, doi, arxiv_id, title},
        fetch real metadata for each. Returns {citation_id: RealPaperMeta}.
        """
        results: dict[str, RealPaperMeta] = {}
        fetched = 0
        for ref in refs:
            if fetched >= max_fetch:
                break
            cid     = ref.get("citation_id") or ref.get("id", "")
            arxiv_m = re.search(r"(\d{4}\.\d{4,5})", ref.get("raw", "") + ref.get("arxiv_id", ""))
            doi_m   = re.search(r"10\.\d{4}/\S+", ref.get("raw", "") + ref.get("doi", ""))
            arxiv_id = arxiv_m.group(1) if arxiv_m else ""
            doi      = doi_m.group(0)  if doi_m   else ""
            query    = ref.get("title", "") or ref.get("raw", "")[:80]
            meta = self.fetch(arxiv_id=arxiv_id, doi=doi, query=query)
            if meta:
                results[cid] = meta
                fetched += 1
            time.sleep(0.3)  # be polite to APIs
        return results
