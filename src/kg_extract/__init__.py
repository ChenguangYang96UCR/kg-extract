"""NSF Awards CSV to knowledge graph triple extraction."""

from .extractor import ExtractionStats, Triple, extract_awards

from .abstracts import (
    AbstractRelation,
    KGGenBackend,
    UIEBackend,
    extract_abstract_triples,
)

__all__ = [
    "AbstractRelation",
    "ExtractionStats",
    "KGGenBackend",
    "Triple",
    "UIEBackend",
    "extract_abstract_triples",
    "extract_awards",
]
__version__ = "0.1.0"
