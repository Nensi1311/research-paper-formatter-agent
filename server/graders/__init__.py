from .formatting_grader import FormattingGrader, FormattingGradeResult
from .consistency_grader import ConsistencyGrader, ConsistencyGradeResult
from .audit_grader import AuditGrader, AuditGradeResult
from .prompt_injection_grader import (
    PromptInjectionGrader, PromptInjectionGradeResult, InjectionScanner,
)

__all__ = [
    "FormattingGrader", "FormattingGradeResult",
    "ConsistencyGrader", "ConsistencyGradeResult",
    "AuditGrader", "AuditGradeResult",
    "PromptInjectionGrader", "PromptInjectionGradeResult", "InjectionScanner",
]

# Optional: cross-paper graders (T6/T7/T8) — degrade gracefully
try:
    from .cross_paper_grader import (
        CrossPaperConsistencyGrader, CrossPaperGradeResult,
        VersionDriftGrader, VersionDriftResult,
        RetractionCheckGrader, RetractionCheckResult,
    )
    __all__ += [
        "CrossPaperConsistencyGrader", "CrossPaperGradeResult",
        "VersionDriftGrader", "VersionDriftResult",
        "RetractionCheckGrader", "RetractionCheckResult",
    ]
except ImportError:
    pass
