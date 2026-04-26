"""
server/graders/cross_paper_grader.py — Tasks T6, T7, T8 graders.

T6: CrossPaperConsistencyGrader
    Checks whether numerical claims in the main paper MATCH values in cited papers.
    Uses RealPaperFetcher (arXiv + Semantic Scholar, no key needed).

T7: VersionDriftGrader
    Checks whether claims differ between arXiv versions of the same paper.
    Uses ArxivFetcher version history.

T8: RetractionCheckGrader
    Checks whether any cited paper has been retracted.
    Uses CrossRef + RetractionWatch (local CSV).

All three work WITHOUT training — they use deterministic rule-based grading
on top of real API data. They add genuine research-integrity value and
demonstrate that the environment has real-world utility beyond synthetic data.

Authors: Nensi Pansuriya, Krushna Parmar, Ishita Bhojani
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corpus import Paper

try:
    from thefuzz import fuzz as _fuzz
    def _sim(a, b): return _fuzz.token_set_ratio(str(a).lower(), str(b).lower()) / 100.0
except ImportError:
    import difflib
    def _sim(a, b): return difflib.SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()

_NUM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUM_RE.findall(text)]


def _numbers_match(a: float, b: float, tol: float = 0.05) -> bool:
    """Numbers match if within 5% relative tolerance."""
    if a == 0 and b == 0:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= tol


# ── T6: Cross-Paper Consistency ───────────────────────────────────────────────
@dataclass
class CrossPaperGradeResult:
    score:                  float
    f_beta:                 float
    n_claims_checked:       int
    n_matches:              int
    n_mismatches:           int
    n_unverifiable:         int
    mismatches:             list[dict] = field(default_factory=list)
    sources_fetched:        list[str]  = field(default_factory=list)
    evidence_specificity:   float = 0.0

    def hint(self) -> str:
        if self.n_claims_checked == 0:
            return "No numerical claims were checked against cited papers."
        if self.n_unverifiable > self.n_claims_checked * 0.6:
            return "Most cited papers could not be fetched. Check DOI/arXiv IDs in references."
        if self.n_mismatches > 0:
            return (f"Found {self.n_mismatches} mismatch(es). "
                    "Include citation_id, claimed_value, actual_value in each finding.")
        return f"Checked {self.n_claims_checked} claims. Score: {self.score:.3f}"


class CrossPaperConsistencyGrader:
    """
    T6: Verifies that numerical claims attributed to cited papers
    actually appear (within tolerance) in those papers.

    Works without training — rule-based comparison of fetched paper values.
    Network calls are cached for the session.
    """

    _FETCH_CACHE: dict[str, object] = {}

    def __init__(self, fetcher=None, max_fetch: int = 5):
        self.max_fetch = max_fetch
        if fetcher is None:
            try:
                from server.real_paper_fetcher import RealPaperFetcher
                self._fetcher = RealPaperFetcher()
            except ImportError:
                self._fetcher = None
        else:
            self._fetcher = fetcher

    def grade(self, submissions: list[dict], paper: "Paper") -> CrossPaperGradeResult:
        """
        submissions: list of agent findings like:
          {"type":"cross_paper_mismatch","citation_id":"ref_2",
           "claimed_value":"91.5","actual_value":"89.2","metric":"GLUE F1"}
        paper: the main paper being audited
        """
        gt_mismatches = paper.ground_truth.get("task6_cross_mismatches", [])
        refs          = paper.ground_truth.get("task4_citations", [])

        # Fetch cited papers (cached)
        fetched = self._fetch_refs(refs)
        sources = list(fetched.keys())

        # Auto-detect mismatches from fetched papers (ground truth enrichment)
        auto_gt = self._auto_detect_mismatches(paper, fetched) if not gt_mismatches else gt_mismatches

        if not auto_gt:
            # No mismatches found — correct answer is empty list
            ok    = len(submissions) == 0
            score = 0.90 if ok else max(0.05, 0.90 - 0.15 * len(submissions))
            return CrossPaperGradeResult(
                score=score, f_beta=score, n_claims_checked=len(refs),
                n_matches=len(refs), n_mismatches=0, n_unverifiable=len(refs)-len(fetched),
                sources_fetched=sources, evidence_specificity=score,
            )

        if not submissions:
            return CrossPaperGradeResult(
                score=0.0001, f_beta=0.0, n_claims_checked=len(refs),
                n_matches=0, n_mismatches=len(auto_gt), n_unverifiable=len(refs)-len(fetched),
                mismatches=auto_gt, sources_fetched=sources,
            )

        # Match submissions to auto_gt
        matched = 0
        for sub in submissions:
            cid    = sub.get("citation_id", "")
            cval   = sub.get("claimed_value", "") or sub.get("claim", "")
            aval   = sub.get("actual_value", "") or sub.get("table_value", "")
            for gt in auto_gt:
                if (_sim(cid, gt.get("citation_id","")) > 0.7 and
                        _sim(str(cval), str(gt.get("claimed_value",""))) > 0.6):
                    matched += 1
                    break

        beta      = 1.5
        precision = matched / len(submissions) if submissions else 0
        recall    = matched / len(auto_gt)
        f_beta    = 0.0
        if precision + recall > 0:
            f_beta = (1+beta**2)*precision*recall / (beta**2*precision+recall)

        spec = 1.0 if all(
            s.get("citation_id") and s.get("claimed_value") and s.get("actual_value")
            for s in submissions
        ) else 0.4

        return CrossPaperGradeResult(
            score=round(min(0.9999, max(0.0001, f_beta)), 4),
            f_beta=round(f_beta, 4),
            n_claims_checked=len(refs),
            n_matches=len(fetched) - len(auto_gt),
            n_mismatches=len(auto_gt),
            n_unverifiable=len(refs) - len(fetched),
            mismatches=auto_gt,
            sources_fetched=sources,
            evidence_specificity=spec,
        )

    def _fetch_refs(self, refs: list[dict]) -> dict[str, object]:
        if not self._fetcher:
            return {}
        results = {}
        fetched = 0
        for ref in refs:
            if fetched >= self.max_fetch:
                break
            cid = ref.get("id", ref.get("citation_id", ""))
            if cid in self._FETCH_CACHE:
                results[cid] = self._FETCH_CACHE[cid]
                fetched += 1
                continue
            arxiv_m = re.search(r"(\d{4}\.\d{4,5})", ref.get("raw","") + ref.get("arxiv_id",""))
            aid     = arxiv_m.group(1) if arxiv_m else ""
            meta    = self._fetcher.fetch(arxiv_id=aid, query=ref.get("title","")[:60])
            if meta:
                self._FETCH_CACHE[cid] = meta
                results[cid] = meta
                fetched += 1
            time.sleep(0.25)
        return results

    def _auto_detect_mismatches(self, paper, fetched: dict) -> list[dict]:
        """
        Compare numerical claims in the main paper's abstract against
        values found in the cited papers' abstracts.
        """
        mismatches = []
        main_abstract = getattr(paper, "sections", {}).get("abstract", "")
        main_nums = _extract_numbers(main_abstract)

        for cid, meta in fetched.items():
            cited_abstract = getattr(meta, "abstract", "")
            if not cited_abstract:
                continue
            cited_nums = _extract_numbers(cited_abstract)
            # Find numbers claimed in main paper that don't appear in cited paper
            for mn in main_nums:
                if mn < 1.0:  # skip percentages expressed as decimals
                    continue
                # Check if this number appears in the cited paper within tolerance
                if cited_nums and not any(_numbers_match(mn, cn) for cn in cited_nums):
                    # High-value number (>50) that doesn't match anything is suspicious
                    if mn > 50:
                        mismatches.append({
                            "citation_id":    cid,
                            "claimed_value":  str(mn),
                            "actual_value":   f"not found in {cid}",
                            "metric":         "unspecified numeric claim",
                            "confidence":     0.6,
                        })
                        break  # one mismatch per citation is enough
        return mismatches[:3]  # cap at 3 for grading stability


# ── T7: Version Drift Grader ─────────────────────────────────────────────────
@dataclass
class VersionDriftResult:
    score:             float
    f_beta:            float
    n_versions:        int
    drifts_found:      list[dict] = field(default_factory=list)
    versions_fetched:  list[dict] = field(default_factory=list)
    evidence_specificity: float = 0.0

    def hint(self) -> str:
        if self.n_versions < 2:
            return "Paper has only one version — no drift possible."
        if not self.drifts_found:
            return "No numerical drift detected across versions."
        return (f"Found {len(self.drifts_found)} drift(s). "
                "Include arxiv_id, version_a, version_b, metric, value_a, value_b.")


class VersionDriftGrader:
    """
    T7: Checks whether key numerical claims changed between arXiv versions.
    A paper that quietly changes 91.5→93.2 between v1 and v3 without disclosure
    is a form of undisclosed revision / result inflation.
    """

    def __init__(self, fetcher=None):
        if fetcher is None:
            try:
                from server.real_paper_fetcher import ArxivFetcher
                self._arxiv = ArxivFetcher()
            except ImportError:
                self._arxiv = None
        else:
            self._arxiv = fetcher

    def grade(self, submissions: list[dict], paper: "Paper") -> VersionDriftResult:
        gt_drifts = paper.ground_truth.get("task7_version_drifts", [])
        arxiv_id  = getattr(paper, "arxiv_id", "") or paper.ground_truth.get("arxiv_id", "")

        # Try to detect drifts from real arXiv versions
        auto_drifts = []
        versions    = []
        if self._arxiv and arxiv_id:
            try:
                meta     = self._arxiv.fetch_by_id(arxiv_id)
                versions = meta.versions if meta else []
                if len(versions) >= 2:
                    auto_drifts = self._detect_drift(arxiv_id, versions)
            except Exception:
                pass

        all_drifts = gt_drifts or auto_drifts

        if not all_drifts:
            ok    = len(submissions) == 0
            score = 0.90 if ok else max(0.05, 0.90 - 0.15 * len(submissions))
            return VersionDriftResult(score=score, f_beta=score,
                n_versions=len(versions), versions_fetched=versions,
                evidence_specificity=score)

        if not submissions:
            return VersionDriftResult(score=0.0001, f_beta=0.0,
                n_versions=len(versions), drifts_found=all_drifts,
                versions_fetched=versions)

        matched = 0
        for sub in submissions:
            for gt in all_drifts:
                if (_sim(sub.get("metric",""), gt.get("metric","")) > 0.6 and
                        _sim(str(sub.get("value_a","")), str(gt.get("value_a",""))) > 0.7):
                    matched += 1
                    break

        beta      = 1.5
        precision = matched / len(submissions)
        recall    = matched / len(all_drifts)
        f_beta    = 0.0
        if precision + recall > 0:
            f_beta = (1+beta**2)*precision*recall / (beta**2*precision+recall)

        spec = 1.0 if all(
            s.get("version_a") and s.get("version_b") and s.get("metric")
            for s in submissions
        ) else 0.5

        return VersionDriftResult(
            score=round(min(0.9999, max(0.0001, f_beta)), 4),
            f_beta=round(f_beta, 4), n_versions=len(versions),
            drifts_found=all_drifts, versions_fetched=versions,
            evidence_specificity=spec,
        )

    def _detect_drift(self, arxiv_id: str, versions: list[dict]) -> list[dict]:
        """Fetch v1 and latest, compare numerical claims in abstracts."""
        if not self._arxiv or len(versions) < 2:
            return []
        try:
            meta_v1 = self._arxiv.fetch_by_id(f"{arxiv_id}v1")
            meta_vl = self._arxiv.fetch_by_id(f"{arxiv_id}v{len(versions)}")
            if not (meta_v1 and meta_vl):
                return []
            nums_v1 = _extract_numbers(meta_v1.abstract)
            nums_vl = _extract_numbers(meta_vl.abstract)
            drifts  = []
            for n1 in nums_v1:
                if n1 < 50:  # focus on high-value metrics
                    continue
                for nl in nums_vl:
                    if abs(n1 - nl) > 1.0 and not _numbers_match(n1, nl):
                        drifts.append({
                            "arxiv_id": arxiv_id,
                            "version_a": "v1",
                            "version_b": f"v{len(versions)}",
                            "metric":    "numeric claim in abstract",
                            "value_a":   str(n1),
                            "value_b":   str(nl),
                            "delta":     round(nl - n1, 2),
                        })
            return drifts[:2]
        except Exception:
            return []


# ── T8: Retraction Check Grader ───────────────────────────────────────────────
@dataclass
class RetractionCheckResult:
    score:                float
    f_beta:               float
    n_refs_checked:       int
    n_retracted:          int
    retracted_refs:       list[dict] = field(default_factory=list)
    evidence_specificity: float = 0.0

    def hint(self) -> str:
        if self.n_retracted == 0:
            return "No retracted references found. Correct answer is empty FINDINGS."
        return (f"Found {self.n_retracted} retracted reference(s). "
                "Include citation_id, doi, retraction_reason in each finding.")


class RetractionCheckGrader:
    """
    T8: Checks whether any cited papers have been retracted.
    Uses CrossRef + local RetractionWatch CSV.
    No API key needed.
    """

    def __init__(self, fetcher=None):
        if fetcher is None:
            try:
                from server.real_paper_fetcher import CrossRefFetcher, RetractionWatchCache
                self._crossref = CrossRefFetcher()
                self._rw       = RetractionWatchCache()
            except ImportError:
                self._crossref = None
                self._rw       = None
        else:
            self._crossref, self._rw = fetcher

    def grade(self, submissions: list[dict], paper: "Paper") -> RetractionCheckResult:
        refs = paper.ground_truth.get("task4_citations", [])

        # Find retracted refs
        retracted = []
        for ref in refs:
            if ref.get("status") == "retracted" or ref.get("injected", False):
                retracted.append({
                    "citation_id":       ref.get("id", ref.get("citation_id", "")),
                    "doi":               ref.get("doi", ""),
                    "retraction_reason": ref.get("retraction_reason", "flagged in ground truth"),
                    "title":             ref.get("title", ref.get("raw", "")[:80]),
                })

        # Also check via real APIs if DOIs present
        if self._crossref or self._rw:
            for ref in refs:
                doi   = ref.get("doi", "")
                title = ref.get("title", ref.get("raw", "")[:80])
                if not doi and not title:
                    continue
                cid = ref.get("id", ref.get("citation_id", ""))
                if any(r["citation_id"] == cid for r in retracted):
                    continue  # already found
                if doi and self._crossref:
                    try:
                        r_flag, reason = self._crossref.is_retracted(doi)
                        if r_flag:
                            retracted.append({"citation_id":cid,"doi":doi,
                                              "retraction_reason":reason,"title":title[:80]})
                            continue
                    except Exception:
                        pass
                if self._rw:
                    try:
                        r_flag, reason = self._rw.check_doi(doi) if doi else (False,"")
                        if not r_flag and title:
                            r_flag, reason = self._rw.check_title(title)
                        if r_flag:
                            retracted.append({"citation_id":cid,"doi":doi,
                                              "retraction_reason":reason,"title":title[:80]})
                    except Exception:
                        pass

        if not retracted:
            ok    = len(submissions) == 0
            score = 0.90 if ok else max(0.05, 0.90 - 0.15 * len(submissions))
            return RetractionCheckResult(score=score, f_beta=score,
                n_refs_checked=len(refs), n_retracted=0, evidence_specificity=score)

        if not submissions:
            return RetractionCheckResult(score=0.0001, f_beta=0.0,
                n_refs_checked=len(refs), n_retracted=len(retracted),
                retracted_refs=retracted)

        matched = 0
        for sub in submissions:
            for gt in retracted:
                if _sim(sub.get("citation_id",""), gt["citation_id"]) > 0.7:
                    matched += 1
                    break

        beta      = 1.5
        precision = matched / len(submissions)
        recall    = matched / len(retracted)
        f_beta    = 0.0
        if precision + recall > 0:
            f_beta = (1+beta**2)*precision*recall / (beta**2*precision+recall)

        spec = 1.0 if all(
            s.get("citation_id") and s.get("retraction_reason")
            for s in submissions
        ) else 0.4

        return RetractionCheckResult(
            score=round(min(0.9999, max(0.0001, f_beta)), 4),
            f_beta=round(f_beta, 4),
            n_refs_checked=len(refs), n_retracted=len(retracted),
            retracted_refs=retracted, evidence_specificity=spec,
        )
