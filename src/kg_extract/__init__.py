"""NSF Awards CSV to knowledge graph triple extraction."""

from .extractor import ExtractionStats, Triple, extract_awards

from .abstracts import (
    AbstractRelation,
    KGGenBackend,
    UIEBackend,
    extract_abstract_triples,
)
from .keywords import (
    KeywordAssignment,
    KeywordCandidate,
    KeywordExtractionStats,
    LLMKeywordBackend,
    extract_keyword_triples,
)

__all__ = [
    "AbstractRelation",
    "ExtractionStats",
    "KGGenBackend",
    "KeywordAssignment",
    "KeywordCandidate",
    "KeywordExtractionStats",
    "LLMKeywordBackend",
    "Triple",
    "UIEBackend",
    "extract_abstract_triples",
    "extract_awards",
    "extract_keyword_triples",
]
__version__ = "0.1.0"
