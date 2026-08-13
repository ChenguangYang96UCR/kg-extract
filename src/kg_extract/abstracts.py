from __future__ import annotations

import csv
import inspect
import json
import os
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Protocol, Sequence

from .extractor import (
    DEFAULT_BASE_URI,
    RDF_TYPE,
    SCHEMA,
    Triple,
    clean_excel_literal,
    entity_uri,
    slug,
)

DEFAULT_UIE_SCHEMA: list[dict[str, list[str]]] = [
    {
        "Project": [
            "develops",
            "addresses problem",
            "uses method",
            "targets domain",
            "evaluated in",
            "has expected outcome",
            "mitigates risk",
        ]
    }
]

KGGEN_BOILERPLATE_PATTERNS = [
    re.compile(
        r"This\s+award\s+reflects\s+NSF's\s+statutory\s+mission\s+and\s+has\s+been\s+"
        r"deemed\s+worthy\s+of\s+support\s+through\s+evaluation\s+using\s+the\s+"
        r"Foundation's\s+intellectual\s+merit\s+and\s+broader\s+impacts\s+review\s+"
        r"criteria\.?",
        flags=re.IGNORECASE,
    ),
]


class MissingBackendDependency(RuntimeError):
    """Raised when an optional abstract extraction backend is not installed."""


@dataclass(frozen=True, slots=True)
class AbstractRelation:
    subject: str
    predicate: str
    object: str
    confidence: float | None = None
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class AbstractExtractionStats:
    awards: int
    relations: int
    triples: int


class AbstractBackend(Protocol):
    name: str

    def extract(self, text: str, *, context: str = "") -> Sequence[AbstractRelation]: ...


class KGGenBackend:
    """Open-schema relation extraction through the official kg-gen package."""

    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o",
        temperature: float = 0.0,
        api_key: str | None = None,
        base_url: str | None = None,
        chunk_size: int = 5000,
        cluster: bool = True,
        client: Any | None = None,
    ) -> None:
        self.name = f"kggen:{model}"
        self.chunk_size = chunk_size
        self.cluster = cluster
        if client is not None:
            self.client = client
            return
        try:
            from kg_gen import KGGen
        except ImportError as exc:
            raise MissingBackendDependency(
                "KGGen is not installed. Run: python3 -m pip install -e '.[kggen]'"
            ) from exc

        options: dict[str, Any] = {"model": model, "temperature": temperature}
        if api_key:
            options["api_key"] = api_key
        if base_url:
            options["base_url"] = base_url
        self.client = KGGen(**options)

    def extract(self, text: str, *, context: str = "") -> list[AbstractRelation]:
        processed_text = preprocess_abstract_for_kggen(text)
        if not processed_text:
            return []
        generate_options: dict[str, Any] = {
            "input_data": processed_text,
            "context": context or "",
            "chunk_size": self.chunk_size,
        }
        generate_parameters = inspect.signature(self.client.generate).parameters
        if "cluster" in generate_parameters:
            generate_options["cluster"] = self.cluster
        elif not self.cluster and "deduplication_method" in generate_parameters:
            generate_options["deduplication_method"] = None

        graph = self.client.generate(**generate_options)
        relations: list[AbstractRelation] = []
        for relation in sorted(graph.relations):
            if len(relation) != 3:
                continue
            subject, predicate, obj = (str(value).strip() for value in relation)
            if subject and predicate and obj:
                relations.append(
                    AbstractRelation(subject, predicate, obj, evidence=processed_text)
                )
        return relations


def preprocess_abstract_for_kggen(text: str) -> str:
    """Clean Abstract text before KGGen while preserving relation-bearing syntax."""
    cleaned = text.replace("\ufeff", " ").replace("\u00a0", " ")
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", cleaned)
    for pattern in KGGEN_BOILERPLATE_PATTERNS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


class UIEBackend:
    """Controlled-schema extraction through PaddleNLP's English UIE model."""

    def __init__(
        self,
        *,
        model: str = "uie-base-en",
        schema: Any = None,
        chunk_size: int = 256,
        pipeline: Any | None = None,
    ) -> None:
        self.name = f"uie:{model}"
        self.schema = schema or DEFAULT_UIE_SCHEMA
        if chunk_size <= 0:
            raise ValueError("UIE chunk size must be greater than zero")
        self.chunk_size = chunk_size
        if pipeline is not None:
            self.pipeline = pipeline
            return
        try:
            from paddlenlp import Taskflow
        except ImportError as exc:
            raise MissingBackendDependency(
                "UIE is not installed. Run: python3 -m pip install -e '.[uie]'"
            ) from exc
        self.pipeline = Taskflow(
            "information_extraction",
            schema=self.schema,
            model=model,
        )

    def extract(self, text: str, *, context: str = "") -> list[AbstractRelation]:
        del context
        chunks = _split_uie_text(text, self.chunk_size)
        if not chunks:
            return []
        pipeline_input: str | list[str] = chunks[0] if len(chunks) == 1 else chunks
        predictions = self.pipeline(pipeline_input)
        if not predictions:
            return []
        relations: list[AbstractRelation] = []
        for result, chunk in zip(predictions, chunks):
            for items in result.values():
                for item in items:
                    relations.extend(_uie_item_relations(item, chunk))
        return relations


def _split_uie_text(text: str, max_chars: int) -> list[str]:
    """Split text into bounded chunks so nested UIE prompts remain valid."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            while len(sentence) > max_chars:
                split_at = sentence.rfind(" ", 0, max_chars + 1)
                if split_at <= 0:
                    split_at = max_chars
                chunks.append(sentence[:split_at].strip())
                sentence = sentence[split_at:].strip()
            current = sentence
            continue

        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        chunks.append(current)
    return chunks


def _uie_item_relations(item: dict[str, Any], text: str) -> list[AbstractRelation]:
    subject = str(item.get("text", "")).strip()
    if not subject:
        return []
    results: list[AbstractRelation] = []
    for predicate, objects in item.get("relations", {}).items():
        for obj in objects:
            object_text = str(obj.get("text", "")).strip()
            if not object_text:
                continue
            probabilities = [
                value
                for value in (item.get("probability"), obj.get("probability"))
                if isinstance(value, Real)
            ]
            confidence = float(min(probabilities)) if probabilities else None
            evidence = _evidence_sentence(text, item, obj)
            results.append(
                AbstractRelation(
                    subject=subject,
                    predicate=str(predicate).strip(),
                    object=object_text,
                    confidence=confidence,
                    evidence=evidence,
                )
            )
            results.extend(_uie_item_relations(obj, text))
    return results


def _evidence_sentence(
    text: str, subject: dict[str, Any], obj: dict[str, Any]
) -> str:
    starts = [value for value in (subject.get("start"), obj.get("start")) if isinstance(value, int)]
    ends = [value for value in (subject.get("end"), obj.get("end")) if isinstance(value, int)]
    if not starts or not ends:
        return text
    start = min(starts)
    end = max(ends)
    left = max(text.rfind(marker, 0, start) for marker in (".", "?", "!", "\n")) + 1
    right_candidates = [
        position
        for marker in (".", "?", "!", "\n")
        if (position := text.find(marker, end)) != -1
    ]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    return text[left:right].strip()


def load_uie_schema(path: str | Path | None) -> Any:
    if path is None:
        return DEFAULT_UIE_SCHEMA
    with Path(path).open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, (list, dict, str)):
        raise ValueError("A UIE schema must be a JSON list, object, or string")
    return schema


def api_key_from_environment(variable: str | None) -> str | None:
    if not variable:
        return None
    value = os.environ.get(variable)
    if not value:
        raise ValueError(f"Environment variable {variable!r} is not set")
    return value


def extract_abstract_triples(
    input_csv: str | Path,
    backend: AbstractBackend,
    *,
    base_uri: str = DEFAULT_BASE_URI,
    limit: int | None = None,
    min_confidence: float = 0.0,
) -> tuple[list[Triple], AbstractExtractionStats]:
    triples: list[Triple] = []
    processed = 0
    relation_count = 0
    labelled: set[tuple[str, str]] = set()
    mentioned: set[tuple[str, str]] = set()
    with Path(input_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            number = clean_excel_literal(row.get("AwardNumber"))
            text = (row.get("Abstract") or "").strip()
            if not number or not text:
                continue
            if limit is not None and processed >= limit:
                break
            processed += 1
            award_uri = f"{base_uri}award/{number}"
            context = clean_excel_literal(row.get("Title"))
            for relation in backend.extract(text, context=context):
                if relation.confidence is not None and relation.confidence < min_confidence:
                    continue
                relation_count += 1
                subject_uri = _relation_entity_uri(
                    relation.subject, award_uri=award_uri, base_uri=base_uri
                )
                object_uri = entity_uri(base_uri, "concept", relation.object)
                predicate_uri = f"{base_uri}vocab/{slug(relation.predicate)}"
                confidence = "" if relation.confidence is None else f"{relation.confidence:.6f}"
                evidence = relation.evidence or text
                triples.append(
                    Triple(
                        subject=subject_uri,
                        predicate=predicate_uri,
                        object=object_uri,
                        object_type="iri",
                        award_number=number,
                        source_column="Abstract",
                        evidence=evidence,
                        confidence=confidence,
                        extractor=backend.name,
                    )
                )
                for uri, label in (
                    (subject_uri, relation.subject),
                    (object_uri, relation.object),
                ):
                    if uri == award_uri:
                        continue
                    include_metadata = (uri, label) not in labelled
                    include_mention = (award_uri, uri) not in mentioned
                    if not include_metadata and not include_mention:
                        continue
                    if include_metadata:
                        labelled.add((uri, label))
                    if include_mention:
                        mentioned.add((award_uri, uri))
                    triples.extend(
                        _concept_metadata(
                            uri,
                            label,
                            award_uri=award_uri,
                            number=number,
                            evidence=evidence,
                            confidence=confidence,
                            extractor=backend.name,
                            base_uri=base_uri,
                            include_metadata=include_metadata,
                            include_mention=include_mention,
                        )
                    )
    return triples, AbstractExtractionStats(processed, relation_count, len(triples))


def _relation_entity_uri(label: str, *, award_uri: str, base_uri: str) -> str:
    normalized = re.sub(r"[^a-z]+", " ", label.casefold()).strip()
    if normalized in {"project", "this project", "the project", "proposed project"}:
        return award_uri
    return entity_uri(base_uri, "concept", label)


def _concept_metadata(
    uri: str,
    label: str,
    *,
    award_uri: str,
    number: str,
    evidence: str,
    confidence: str,
    extractor: str,
    base_uri: str,
    include_metadata: bool,
    include_mention: bool,
) -> list[Triple]:
    common = {
        "award_number": number,
        "source_column": "Abstract",
        "evidence": evidence,
        "confidence": confidence,
        "extractor": extractor,
    }
    triples: list[Triple] = []
    if include_metadata:
        triples.extend(
            [
                Triple(uri, RDF_TYPE, f"{base_uri}vocab/Concept", "iri", **common),
                Triple(uri, SCHEMA + "name", label, "literal", **common),
            ]
        )
    if include_mention:
        triples.append(
            Triple(award_uri, f"{base_uri}vocab/mentions", uri, "iri", **common)
        )
    return triples
