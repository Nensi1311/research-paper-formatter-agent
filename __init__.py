"""
ScholarEnv — OpenEnv environment for scholarly integrity verification.
"""
from .models import FormattingAction, ScholarAction, ScholarObservation, EpisodeStatus
from .corpus import PaperCorpus, Paper

__version__ = "0.4.0"
__all__ = [
    "FormattingAction",
    "ScholarAction",
    "ScholarObservation",
    "EpisodeStatus",
    "PaperCorpus",
    "Paper",
]
