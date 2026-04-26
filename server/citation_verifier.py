"""
server/citation_verifier.py — Task 4 citation verification engine.
Upgraded with HackNU CitationEngine regex patterns (PARENTHETICAL_RE,
NARRATIVE_RE, APA_JOURNAL_RE, NUMBERED_JOURNAL_RE) and thefuzz matching.
Authors: Nensi Pansuriya, Krushna Parmar, Ishita Bhojani
"""
from __future__ import annotations

import json, re, sqlite3, time, urllib.parse, urllib.request
from dataclasses import dataclass, field
from pathlib import Path

DB_PATH = Path("data/citation_cache.db")

# ── HackNU CitationEngine regex patterns (battle-tested on real papers) ───────
# Parenthetical: (Smith, 2023) or (Smith & Jones, 2023) or (Smith et al., 2023)
PARENTHETICAL_RE = re.compile(
    r'\(([A-Z][a-zà-ÿ]+(?:\s(?:&|and)\s[A-Z][a-zà-ÿ]+)*(?:\set\sal\.)?'
    r',\s*\d{4}[a-z]?'
    r'(?:;\s*[A-Z][a-zà-ÿ]+(?:\s(?:&|and)\s[A-Z][a-zà-ÿ]+)*(?:\set\sal\.)?'
    r',\s*\d{4}[a-z]?)*)\)'
)
# Narrative: Smith (2023) or Smith and Jones (2023)
NARRATIVE_RE = re.compile(
    r'([A-Z][a-zà-ÿ]+(?:\s(?:and|&)\s[A-Z][a-zà-ÿ]+)?(?:\set\sal\.)?)'
    r'\s*\((\d{4}[a-z]?)\)'
)
# APA journal: Author, A. A. (Year). Title. Journal, Vol(Issue), Pages.
APA_JOURNAL_RE = re.compile(
    r'^(.+?)\s*\((\d{4})\)\.\s*(.+?)\.\s*(.+?),\s*(\d+)'
    r'(?:\((\d+)\))?,\s*(.+?)\.(?:\s*(?:https?://)?doi\.org/(.+))?$'
)
# Numbered journal: "1. Author (Year) Title. Journal Vol(Issue):Pages."
NUMBERED_JOURNAL_RE = re.compile(
    r'^(\d{1,3})[\.]\s*(.+?)\s*\((\d{4})\)\s*(.+?)\.\s*'
    r'([A-Z][\w\s&:,\-]+?)\s+(\d+)'
    r'(?:\((\d+)\))?\s*[:\-,]\s*([\d\-–]+)'
)

# thefuzz fallback
try:
    from thefuzz import fuzz as _fuzz
    def _sim(a: str, b: str) -> float:
        return _fuzz.token_sort_ratio(str(a).lower(), str(b).lower()) / 100.0
except ImportError:
    import difflib
    def _sim(a: str, b: str) -> float:
        return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


@dataclass
class ParsedReference:
    citation_id: str; raw_string: str
    authors: list[str] = field(default_factory=list)
    title: str = ""; year: int | None = None
    doi: str = ""; arxiv_id: str = ""; journal: str = ""

    def to_dict(self) -> dict:
        return {"citation_id": self.citation_id, "raw_string": self.raw_string[:200],
                "authors": self.authors[:3], "title": self.title[:120],
                "year": self.year, "doi": self.doi, "arxiv_id": self.arxiv_id}


@dataclass
class VerificationResult:
    citation_id: str; status: str; confidence: float; source: str
    issue: str = ""; matched_title: str = ""


class ReferenceParser:
    """Parses IEEE + APA + numbered references using HackNU regex patterns."""

    _ENTRY = re.compile(r'^\s*\[(\d+)\]\s+(.+?)(?=^\s*\[\d+\]|\Z)', re.M | re.S)
    _YEAR  = re.compile(r'\b(20[0-2]\d|19[89]\d)\b')
    _DOI   = re.compile(r'(?:doi:|DOI:|https?://doi\.org/)([^\s,]+)', re.I)
    _ARXIV = re.compile(r'(?:arXiv:|arxiv\.org/abs/)(\d{4}\.\d{4,5}(?:v\d+)?)', re.I)

    def parse_references_section(self, text: str) -> list[ParsedReference]:
        refs = []
        for m in self._ENTRY.finditer(text):
            raw = m.group(2).strip()
            ref = ParsedReference(citation_id=f"[{m.group(1)}]", raw_string=raw)
            # Year
            ys = self._YEAR.findall(raw)
            if ys: ref.year = int(ys[0])
            # DOI
            dm = self._DOI.search(raw)
            if dm: ref.doi = dm.group(1).rstrip(".")
            # arXiv
            am = self._ARXIV.search(raw)
            if am: ref.arxiv_id = am.group(1)
            # Try APA pattern
            am2 = APA_JOURNAL_RE.match(raw)
            if am2:
                ref.title   = am2.group(3).strip()[:120]
                ref.journal = am2.group(4).strip()[:80]
                if am2.group(8): ref.doi = am2.group(8).rstrip(".")
            else:
                segs = raw.split(".")
                if len(segs) >= 2:
                    ref.title = segs[1].strip()[:120]
            # Authors from text before year
            if ys and ys[0] in raw:
                before = raw[:raw.index(ys[0])].strip()
                ref.authors = [p.strip() for p in re.split(r',|;|\band\b', before)
                               if len(p.strip()) > 2 and p.strip()[0].isupper()][:6]
            refs.append(ref)
        return refs


class CitationCache:
    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path))
        self._init()

    def _init(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                citation_id TEXT, paper_id TEXT, raw_string TEXT,
                authors TEXT, title TEXT, year INTEGER, doi TEXT, arxiv_id TEXT,
                verification_status TEXT, verified_at REAL, confidence REAL,
                source TEXT, issue TEXT, matched_title TEXT,
                PRIMARY KEY (citation_id, paper_id)
            )""")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS verification_stats (
                paper_id TEXT PRIMARY KEY, total_citations INTEGER,
                verified_valid INTEGER, verified_ghost INTEGER,
                verified_misattr INTEGER, unverifiable INTEGER, last_updated REAL
            )""")
        self.conn.commit()

    def get(self, cid: str, pid: str) -> dict | None:
        cur = self.conn.execute("SELECT * FROM citations WHERE citation_id=? AND paper_id=?", (cid, pid))
        row = cur.fetchone()
        return dict(zip([d[0] for d in cur.description], row)) if row else None

    def put(self, pid: str, ref: ParsedReference, res: VerificationResult) -> None:
        self.conn.execute("""
            INSERT OR REPLACE INTO citations
            (citation_id, paper_id, raw_string, authors, title, year, doi, arxiv_id,
             verification_status, verified_at, confidence, source, issue, matched_title)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
            ref.citation_id, pid, ref.raw_string[:500],
            json.dumps(ref.authors), ref.title, ref.year,
            ref.doi, ref.arxiv_id, res.status, time.time(),
            res.confidence, res.source, res.issue, res.matched_title,
        ))
        self.conn.commit()

    def update_stats(self, pid: str, results: list[VerificationResult]) -> None:
        stats = {"valid": 0, "ghost": 0, "misattributed": 0, "unverifiable": 0}
        for r in results:
            stats[r.status if r.status in stats else "unverifiable"] += 1
        self.conn.execute("""
            INSERT OR REPLACE INTO verification_stats
            (paper_id, total_citations, verified_valid, verified_ghost,
             verified_misattr, unverifiable, last_updated) VALUES (?,?,?,?,?,?,?)""", (
            pid, len(results), stats["valid"], stats["ghost"],
            stats["misattributed"], stats["unverifiable"], time.time(),
        ))
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


class CitationVerifier:
    CROSSREF_API = "https://api.crossref.org/works"
    ARXIV_API    = "https://export.arxiv.org/abs/"
    TIMEOUT      = 5

    def __init__(self, cache: CitationCache | None = None) -> None:
        self.cache  = cache or CitationCache()
        self.parser = ReferenceParser()

    def verify_citation(self, ref: ParsedReference, paper_id: str) -> VerificationResult:
        cached = self.cache.get(ref.citation_id, paper_id)
        if cached:
            return VerificationResult(citation_id=ref.citation_id,
                                       status=cached["verification_status"],
                                       confidence=cached["confidence"], source="cache",
                                       issue=cached.get("issue", ""),
                                       matched_title=cached.get("matched_title", ""))
        if ref.arxiv_id:
            r = self._via_arxiv(ref)
            if r.status != "unverifiable":
                self.cache.put(paper_id, ref, r); return r
        if ref.doi or ref.title:
            r = self._via_crossref(ref)
            if r.status != "unverifiable":
                self.cache.put(paper_id, ref, r); return r
        r = self._heuristic(ref)
        self.cache.put(paper_id, ref, r)
        return r

    def _via_arxiv(self, ref: ParsedReference) -> VerificationResult:
        try:
            req = urllib.request.Request(f"{self.ARXIV_API}{ref.arxiv_id}",
                                          headers={"User-Agent": "ScholarEnv/1.0"})
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                html = resp.read().decode("utf-8", errors="replace")
            if ref.arxiv_id in html:
                title_m = re.search(r'<h1[^>]*class="title[^"]*"[^>]*>(.*?)</h1>', html, re.S)
                matched = re.sub(r'<[^>]+>', '', title_m.group(1)).replace("Title:", "").strip()[:100] if title_m else ""
                return VerificationResult(citation_id=ref.citation_id, status="valid",
                                           confidence=0.85, source="arxiv", matched_title=matched)
            return VerificationResult(citation_id=ref.citation_id, status="ghost",
                                       confidence=0.80, source="arxiv",
                                       issue=f"arXiv {ref.arxiv_id} not found")
        except Exception:
            return VerificationResult(citation_id=ref.citation_id, status="unverifiable",
                                       confidence=0.0, source="arxiv", issue="Network error")

    def _via_crossref(self, ref: ParsedReference) -> VerificationResult:
        try:
            url = (f"{self.CROSSREF_API}/{urllib.parse.quote(ref.doi)}" if ref.doi
                   else f"{self.CROSSREF_API}?query={urllib.parse.quote(ref.title[:80])}&rows=1")
            req = urllib.request.Request(url, headers={"User-Agent": "ScholarEnv/1.0"})
            with urllib.request.urlopen(req, timeout=self.TIMEOUT) as resp:
                data = json.loads(resp.read())
            if ref.doi:
                if data.get("status") == "ok":
                    matched = data["message"].get("title", [""])[0]
                    return VerificationResult(citation_id=ref.citation_id, status="valid",
                                               confidence=0.92, source="crossref", matched_title=matched[:100])
                return VerificationResult(citation_id=ref.citation_id, status="ghost",
                                           confidence=0.88, source="crossref",
                                           issue=f"DOI {ref.doi} not found")
            items = data.get("message", {}).get("items", [])
            if not items:
                return VerificationResult(citation_id=ref.citation_id, status="unverifiable",
                                           confidence=0.2, source="crossref")
            # Use thefuzz for title similarity
            ctitle  = items[0].get("title", [""])[0]
            overlap = _sim(ref.title, ctitle)
            if overlap > 0.55:
                return VerificationResult(citation_id=ref.citation_id, status="valid",
                                           confidence=min(0.5 + overlap * 0.4, 0.90),
                                           source="crossref", matched_title=ctitle[:100])
            return VerificationResult(citation_id=ref.citation_id, status="misattributed",
                                       confidence=0.55, source="crossref",
                                       issue=f"Title overlap={overlap:.2f}", matched_title=ctitle[:100])
        except Exception:
            return VerificationResult(citation_id=ref.citation_id, status="unverifiable",
                                       confidence=0.0, source="crossref", issue="Network error")

    def _heuristic(self, ref: ParsedReference) -> VerificationResult:
        if not ref.year: pass
        elif ref.year > 2026 or ref.year < 1900:
            return VerificationResult(citation_id=ref.citation_id, status="ghost",
                                       confidence=0.70, source="heuristic",
                                       issue=f"Implausible year: {ref.year}")
        if len(ref.raw_string) < 20:
            return VerificationResult(citation_id=ref.citation_id, status="ghost",
                                       confidence=0.65, source="heuristic",
                                       issue="Reference string too short")
        return VerificationResult(citation_id=ref.citation_id, status="unverifiable",
                                   confidence=0.3, source="heuristic")


class CitationGrader:
    """Grades Task 4 (citation_verification) submissions."""

    def grade(self, verdicts: list[dict], ground_truth: list[dict],
              refs_checked: int = 0) -> dict:
        if not ground_truth:
            return {"score": 1.0, "precision_valid": 1.0, "recall_invalid": 1.0,
                    "evidence_score": 0.0, "rule_results": {}}
        if not verdicts:
            return {"score": 0.0, "precision_valid": 0.0, "recall_invalid": 0.0,
                    "evidence_score": 0.0,
                    "rule_results": {gt["id"]: False for gt in ground_truth}}

        gt_map     = {gt["id"]: gt for gt in ground_truth}
        valid_gt   = [g for g in ground_truth if g.get("status") == "valid"]
        invalid_gt = [g for g in ground_truth if g.get("status") != "valid"]
        tp_v = tp_i = 0

        for v in verdicts:
            gt = gt_map.get(v.get("citation_id", ""))
            if not gt: continue
            if v.get("status") == gt.get("status"):
                if gt.get("status") == "valid": tp_v += 1
                else: tp_i += 1

        p_v  = tp_v / len(valid_gt)   if valid_gt   else 1.0
        r_i  = tp_i / len(invalid_gt) if invalid_gt else 1.0
        evid = min(1.0, refs_checked / max(len(ground_truth), 1))
        score = min(1.0, 0.5 * p_v + 0.4 * r_i + 0.1 * evid)

        return {
            "score":           round(score, 4),
            "precision_valid": round(p_v, 4),
            "recall_invalid":  round(r_i, 4),
            "evidence_score":  round(evid, 4),
            "rule_results":    {gt["id"]: any(
                v.get("citation_id") == gt["id"] and v.get("status") == gt.get("status")
                for v in verdicts) for gt in ground_truth},
        }
