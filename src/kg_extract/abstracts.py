from __future__ import annotations

import csv
import inspect
import json
import os
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

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

DEFAULT_ABSTRACT_NODE_CLEANER_MODEL = "ollama_chat/deepseek-r1:14b"
DEFAULT_ABSTRACT_NODE_CLEANER_API_BASE = "http://localhost:11434"

KGGEN_BOILERPLATE_PATTERNS = [
    re.compile(
        r"This\s+award\s+reflects\s+NSF's\s+statutory\s+mission\s+and\s+has\s+been\s+"
        r"deemed\s+worthy\s+of\s+support\s+through\s+evaluation\s+using\s+the\s+"
        r"Foundation's\s+intellectual\s+merit\s+and\s+broader\s+impacts\s+review\s+"
        r"criteria\.?",
        flags=re.IGNORECASE,
    ),
]

ABSTRACT_NODE_MAX_EXPANSIONS = 6
ABSTRACT_NODE_MAX_WORDS = 10
ABSTRACT_NODE_TRAILING_PREPOSITIONS = {
    "about",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "through",
    "to",
    "with",
}


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


@dataclass(frozen=True, slots=True)
class AbstractNodeCleaningDecision:
    raw_label: str
    action: str
    clean_labels: tuple[str, ...]
    reason: str = ""


@dataclass(frozen=True, slots=True)
class AbstractNodeCleaningRecord:
    award_number: str
    raw_label: str
    action: str
    clean_labels: str
    reason: str
    cleaner: str


class AbstractBackend(Protocol):
    name: str

    def extract(self, text: str, *, context: str = "") -> Sequence[AbstractRelation]: ...


class AbstractNodeCleaner(Protocol):
    name: str
    records: list[AbstractNodeCleaningRecord]

    def clean_labels(
        self,
        labels: Sequence[str],
        *,
        award_number: str,
        title: str,
        abstract: str,
    ) -> dict[str, AbstractNodeCleaningDecision]: ...


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
        no_dspy: bool = True,
        client: Any | None = None,
    ) -> None:
        self.name = f"kggen:{model}"
        self.chunk_size = chunk_size
        self.cluster = cluster
        self.no_dspy = no_dspy
        if client is not None:
            self.client = client
            return
        try:
            from kg_gen import KGGen
        except ImportError as exc:
            raise MissingBackendDependency(
                "KGGen could not be imported. "
                f"Original import error: {exc}. "
                "Install the KGGen dependencies with: "
                "python -m pip install -e '.[kggen,litellm]' && "
                "python -m pip install -e src/kg-gen"
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
        if "no_dspy" in generate_parameters:
            generate_options["no_dspy"] = self.no_dspy

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


class LLMAbstractNodeCleaner:
    """Semantic node cleaning through a LiteLLM-compatible chat model."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_ABSTRACT_NODE_CLEANER_MODEL,
        api_base: str | None = DEFAULT_ABSTRACT_NODE_CLEANER_API_BASE,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.name = f"llm-node-cleaner:{model}"
        self.model = model
        self.api_base = api_base or None
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.records: list[AbstractNodeCleaningRecord] = []
        self._cache: dict[tuple[str, str, str], AbstractNodeCleaningDecision] = {}
        try:
            from litellm import completion
        except ImportError as exc:
            raise MissingBackendDependency(
                "LiteLLM is not installed. Run: python3 -m pip install -e '.[litellm]'"
            ) from exc
        self.completion = completion

    def clean_labels(
        self,
        labels: Sequence[str],
        *,
        award_number: str,
        title: str,
        abstract: str,
    ) -> dict[str, AbstractNodeCleaningDecision]:
        unique_labels = _unique_clean_labels(labels)
        if not unique_labels:
            return {}

        cache_key_prefix = (title, _abstract_context_fingerprint(abstract))
        decisions: dict[str, AbstractNodeCleaningDecision] = {}
        missing: list[str] = []
        for label in unique_labels:
            key = (*cache_key_prefix, label)
            if key in self._cache:
                decisions[label] = self._cache[key]
            else:
                missing.append(label)

        if missing:
            response = self.completion(**self._request_options(missing, title=title, abstract=abstract))
            content = response.choices[0].message.content or ""
            content = _strip_thinking_blocks(content)
            parsed = _parse_node_cleaning_response(content, missing)
            for label in missing:
                decision = parsed.get(label) or AbstractNodeCleaningDecision(
                    raw_label=label,
                    action="keep",
                    clean_labels=(label,),
                    reason="LLM response did not include this label; kept original label.",
                )
                self._cache[(*cache_key_prefix, label)] = decision
                decisions[label] = decision

        for label in unique_labels:
            decision = decisions[label]
            self.records.append(
                AbstractNodeCleaningRecord(
                    award_number=award_number,
                    raw_label=label,
                    action=decision.action,
                    clean_labels="|".join(decision.clean_labels),
                    reason=decision.reason,
                    cleaner=self.name,
                )
            )
        return decisions

    def _request_options(self, labels: Sequence[str], *, title: str, abstract: str) -> dict[str, Any]:
        prompt = _node_cleaning_prompt(labels, title=title, abstract=abstract)
        options: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You clean noisy knowledge-graph node labels extracted from NSF "
                        "award abstracts. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "api_base": self.api_base,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_key:
            options["api_key"] = self.api_key
        return options


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
    node_cleaner: AbstractNodeCleaner | None = None,
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
            raw_relations = list(backend.extract(text, context=context))
            node_decisions = _clean_relation_node_labels(
                raw_relations,
                node_cleaner=node_cleaner,
                award_number=number,
                title=context,
                abstract=text,
            )
            for raw_relation in raw_relations:
                for relation in _postprocess_abstract_relation(
                    raw_relation,
                    node_decisions=node_decisions,
                ):
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


def _clean_relation_node_labels(
    relations: Sequence[AbstractRelation],
    *,
    node_cleaner: AbstractNodeCleaner | None,
    award_number: str,
    title: str,
    abstract: str,
) -> dict[str, AbstractNodeCleaningDecision]:
    if node_cleaner is None:
        return {}
    labels = [label for relation in relations for label in (relation.subject, relation.object)]
    return node_cleaner.clean_labels(
        labels,
        award_number=award_number,
        title=title,
        abstract=abstract,
    )


def _postprocess_abstract_relation(
    relation: AbstractRelation,
    *,
    node_decisions: dict[str, AbstractNodeCleaningDecision] | None = None,
) -> list[AbstractRelation]:
    subjects = _postprocess_abstract_node_label(
        relation.subject,
        node_decisions=node_decisions,
    )
    objects = _postprocess_abstract_node_label(
        relation.object,
        node_decisions=node_decisions,
    )
    if not subjects or not objects:
        return []

    processed: list[AbstractRelation] = []
    seen: set[tuple[str, str]] = set()
    for subject in subjects:
        for obj in objects:
            key = (subject.casefold(), obj.casefold())
            if key in seen:
                continue
            seen.add(key)
            processed.append(
                AbstractRelation(
                    subject=subject,
                    predicate=relation.predicate,
                    object=obj,
                    confidence=relation.confidence,
                    evidence=relation.evidence,
                )
            )
            if len(processed) >= ABSTRACT_NODE_MAX_EXPANSIONS:
                return processed
    return processed


def _postprocess_abstract_node_label(
    label: str,
    *,
    node_decisions: dict[str, AbstractNodeCleaningDecision] | None = None,
) -> list[str]:
    decision = (node_decisions or {}).get(label)
    if decision and decision.action == "drop":
        return []
    source_labels = list(decision.clean_labels) if decision and decision.clean_labels else [label]

    labels: list[str] = []
    seen: set[str] = set()
    for source_label in source_labels:
        for processed_label in _postprocess_single_abstract_node_label(source_label):
            key = processed_label.casefold()
            if key in seen:
                continue
            seen.add(key)
            labels.append(processed_label)
            if len(labels) >= ABSTRACT_NODE_MAX_EXPANSIONS:
                return labels
    return labels


def _postprocess_single_abstract_node_label(label: str) -> list[str]:
    cleaned = _clean_abstract_node_label(label)
    if not cleaned:
        return []
    if _is_project_action_label(cleaned):
        return ["project"]

    candidates = _expand_supported_concept_label(cleaned)
    if not candidates:
        candidates = [cleaned]

    labels: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = _condense_abstract_node_label(candidate)
        for part in _split_abstract_node_label(candidate):
            part = _condense_abstract_node_label(part)
            part = _clean_abstract_node_label(part)
            if not _is_usable_abstract_node_label(part):
                continue
            key = part.casefold()
            if key in seen:
                continue
            seen.add(key)
            labels.append(part)
            if len(labels) >= ABSTRACT_NODE_MAX_EXPANSIONS:
                return labels
    return labels


def _node_cleaning_prompt(labels: Sequence[str], *, title: str, abstract: str) -> str:
    labels_json = json.dumps(list(labels), ensure_ascii=False, indent=2)
    return (
        "Clean the node labels below for a knowledge graph about one NSF award.\n"
        "For each raw label, decide whether it should be kept, dropped, rewritten, or split.\n"
        "\n"
        "Goal: keep labels that describe award-specific technologies, methods, resources, "
        "domains, organizations, datasets, systems, educational interventions, outcomes, "
        "or concrete scientific concepts.\n"
        "\n"
        "Drop labels that are too generic, rhetorical, administrative, incomplete, "
        "or do not help characterize this award. Rewrite verbose labels into concise "
        "noun phrases. Split labels only when the raw label contains multiple meaningful "
        "concepts.\n"
        "\n"
        "Return only a JSON array. Each item must have exactly these fields:\n"
        "- raw_label: one original label from the input list\n"
        "- action: one of keep, drop, rewrite, split\n"
        "- labels: [] for drop, otherwise one or more concise cleaned labels\n"
        "- reason: short explanation\n"
        "\n"
        f"Award title:\n{title or '(missing)'}\n\n"
        f"Abstract:\n{abstract[:3500]}\n\n"
        f"Raw labels:\n{labels_json}"
    )


def _parse_node_cleaning_response(
    response: str,
    expected_labels: Sequence[str],
) -> dict[str, AbstractNodeCleaningDecision]:
    expected = {label: label for label in expected_labels}
    expected_casefold = {label.casefold(): label for label in expected_labels}
    data = _extract_json_value(response)
    if isinstance(data, dict):
        items = data.get("labels", data.get("decisions", []))
    else:
        items = data
    if not isinstance(items, list):
        return {}

    decisions: dict[str, AbstractNodeCleaningDecision] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_label = str(item.get("raw_label", "")).strip()
        if raw_label not in expected:
            raw_label = expected_casefold.get(raw_label.casefold(), "")
        if not raw_label:
            continue
        action = str(item.get("action", "keep")).strip().casefold()
        if action not in {"keep", "drop", "rewrite", "split"}:
            action = "keep"
        clean_labels = _coerce_clean_labels(item.get("labels"))
        if action == "drop":
            clean_labels = ()
        elif not clean_labels:
            clean_labels = (raw_label,)
            action = "keep"
        decisions[raw_label] = AbstractNodeCleaningDecision(
            raw_label=raw_label,
            action=action,
            clean_labels=clean_labels,
            reason=str(item.get("reason", "")).strip(),
        )
    return decisions


def _coerce_clean_labels(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = [str(item) for item in value]
    else:
        values = []
    labels: list[str] = []
    seen: set[str] = set()
    for label in values:
        cleaned = _clean_abstract_node_label(label)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        labels.append(cleaned)
    return tuple(labels[:ABSTRACT_NODE_MAX_EXPANSIONS])


def _extract_json_value(response: str) -> Any:
    response = response.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    start_positions = [pos for pos in (response.find("["), response.find("{")) if pos != -1]
    if not start_positions:
        return None
    start = min(start_positions)
    for end in range(len(response), start, -1):
        try:
            return json.loads(response[start:end])
        except json.JSONDecodeError:
            continue
    return None


def _strip_thinking_blocks(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()


def _unique_clean_labels(labels: Sequence[str]) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for label in labels:
        cleaned = _clean_abstract_node_label(label)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(cleaned)
    return unique


def _abstract_context_fingerprint(abstract: str) -> str:
    return re.sub(r"\s+", " ", abstract[:500]).strip()


def _clean_abstract_node_label(label: str) -> str:
    label = str(label or "").replace("\u00a0", " ")
    label = re.sub(r"\s+", " ", label).strip(" \t\r\n\"'`.,;:")
    label = re.sub(r"\s+\)", ")", label)
    label = re.sub(r"\(\s+", "(", label)
    label = re.sub(r"\s{2,}", " ", label)
    label = re.sub(r"^(?:and|or)\s+", "", label, flags=re.IGNORECASE)
    label = re.sub(r"^(?:a|an|the)\s+", "", label, flags=re.IGNORECASE)
    return label.strip()


def _is_project_action_label(label: str) -> bool:
    return bool(
        re.match(
            r"^(?:this\s+|the\s+|proposed\s+)?project(?:'s)?\s+"
            r"(?:will\s+)?(?:aims?|seeks?|plans?|intends?|is\s+designed)\s+to\b",
            label,
            flags=re.IGNORECASE,
        )
    )


def _expand_supported_concept_label(label: str) -> list[str]:
    match = re.match(
        r"^(?P<head>.+?)\s+that\s+supports?\s+(?P<tail>.+)$",
        label,
        flags=re.IGNORECASE,
    )
    if not match:
        return []
    head = _clean_abstract_node_label(match.group("head"))
    tail_parts = _split_abstract_node_label(match.group("tail"))
    if not head or len(tail_parts) < 2:
        return []
    if _word_count(head) <= 3:
        return [_clean_abstract_node_label(f"{part} {head}") for part in tail_parts]
    return tail_parts


def _condense_abstract_node_label(label: str) -> str:
    label = _clean_abstract_node_label(label)
    if not label:
        return ""

    support_match = re.search(r"\bsupport\s+for\s+(.+)$", label, flags=re.IGNORECASE)
    if support_match and _word_count(label) > ABSTRACT_NODE_MAX_WORDS:
        label = support_match.group(1)

    through_match = re.match(
        r"^(?P<head>.+?)\s+through\s+.+?\s+in\s+(?P<domain>.+)$",
        label,
        flags=re.IGNORECASE,
    )
    if through_match:
        head = _clean_abstract_node_label(through_match.group("head"))
        domain = _clean_abstract_node_label(through_match.group("domain"))
        if head and domain:
            label = f"{domain} {head}"

    for marker in (" that ", " which ", " where "):
        left, separator, _ = label.partition(marker)
        if separator and _word_count(left) >= 2:
            label = left
            break

    enhanced_match = re.match(r"^(.+?)\s+enhanced\s+by\s+.+$", label, flags=re.IGNORECASE)
    if enhanced_match and _word_count(enhanced_match.group(1)) >= 2:
        label = enhanced_match.group(1)

    label = re.sub(
        r"\s+for\s+(?:instructional|educational|training|learning|research)\s+purposes?$",
        "",
        label,
        flags=re.IGNORECASE,
    )
    return _clean_abstract_node_label(label)


def _split_abstract_node_label(label: str) -> list[str]:
    label = _clean_abstract_node_label(label)
    if not re.search(r",|;|\s+(?:and|or)\s+", label, flags=re.IGNORECASE):
        return [label] if label else []

    has_list_punctuation = bool(re.search(r",|;", label))
    parts = [
        _clean_abstract_node_label(part)
        for part in re.split(r"\s*(?:,|;)\s*|\s+(?:and|or)\s+", label, flags=re.IGNORECASE)
    ]
    parts = [part for part in parts if part]
    if len(parts) < 2 or len(parts) > ABSTRACT_NODE_MAX_EXPANSIONS:
        return [label]

    first_words = parts[0].split()
    if not has_list_punctuation and len(first_words) >= 2:
        prefix = " ".join(first_words[:-1])
        for index in range(1, len(parts)):
            if _word_count(parts[index]) == 1:
                parts[index] = f"{prefix} {parts[index]}"
    return parts


def _is_usable_abstract_node_label(label: str) -> bool:
    if not label:
        return False
    words = label.split()
    if len(words) > ABSTRACT_NODE_MAX_WORDS:
        return False
    if words[-1].casefold() in ABSTRACT_NODE_TRAILING_PREPOSITIONS:
        return False
    if label.casefold() in {"and", "or", "that", "which", "where"}:
        return False
    return True


def _word_count(label: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+", label))


def write_abstract_node_cleaning_csv(
    records: Iterable[AbstractNodeCleaningRecord],
    output_path: str | Path,
) -> None:
    fields = list(AbstractNodeCleaningRecord.__dataclass_fields__)
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: getattr(record, field) for field in fields})


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
