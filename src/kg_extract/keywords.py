from __future__ import annotations

import csv
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

from .abstracts import MissingBackendDependency
from .extractor import (
    DEFAULT_BASE_URI,
    RDF_TYPE,
    SCHEMA,
    Triple,
    clean_excel_literal,
    entity_uri,
)

NSF_BOILERPLATE_PATTERN = re.compile(
    r"This\s+award\s+reflects\s+NSF's\s+statutory\s+mission\s+and\s+has\s+been\s+"
    r"deemed\s+worthy\s+of\s+support\s+through\s+evaluation\s+using\s+the\s+"
    r"Foundation's\s+intellectual\s+merit\s+and\s+broader\s+impacts\s+review\s+"
    r"criteria\.?",
    flags=re.IGNORECASE,
)

DEFAULT_LLM_KEYWORD_MODEL_PATH = "/data/cyang314/kg/Qwen3-8B"
DEFAULT_LLM_KEYWORD_REPO_ID = "Qwen/Qwen3-8B"
DEFAULT_LITELLM_KEYWORD_MODEL = "ollama_chat/deepseek-r1:14b"
DEFAULT_LITELLM_API_BASE = "http://localhost:11434"

STOPWORDS = {
    "a",
    "about",
    "above",
    "across",
    "after",
    "also",
    "an",
    "and",
    "are",
    "as",
    "at",
    "award",
    "be",
    "been",
    "being",
    "between",
    "both",
    "by",
    "can",
    "could",
    "criteria",
    "deemed",
    "during",
    "each",
    "for",
    "foundation",
    "from",
    "has",
    "have",
    "having",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "mission",
    "more",
    "nsf",
    "of",
    "on",
    "or",
    "other",
    "project",
    "proposal",
    "proposed",
    "provide",
    "reflects",
    "review",
    "statutory",
    "support",
    "supports",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "to",
    "team",
    "teams",
    "universities",
    "university",
    "using",
    "was",
    "were",
    "which",
    "will",
    "with",
    "worthy",
}

NOUN_FILTER_TRIM_PREFIXES = {
    "advance",
    "advances",
    "advanced",
    "advancing",
    "build",
    "building",
    "builds",
    "built",
    "combine",
    "combines",
    "combining",
    "create",
    "created",
    "creates",
    "creating",
    "design",
    "designed",
    "designing",
    "designs",
    "develop",
    "developed",
    "developing",
    "develops",
    "enable",
    "enabled",
    "enables",
    "enabling",
    "evaluate",
    "evaluated",
    "evaluates",
    "evaluating",
    "improve",
    "improved",
    "improves",
    "improving",
    "integrate",
    "integrated",
    "integrates",
    "integrating",
    "install",
    "installed",
    "installing",
    "installs",
    "make",
    "made",
    "makes",
    "making",
    "measure",
    "measured",
    "measures",
    "measuring",
    "reach",
    "reached",
    "reaches",
    "reaching",
    "transform",
    "transformed",
    "transforming",
    "transforms",
    "use",
    "used",
    "uses",
}

NOUN_FILTER_REJECT_PREFIXES = {
    "address",
    "addresses",
    "addressing",
    "drive",
    "drives",
    "driving",
    "characterize",
    "characterized",
    "characterizes",
    "characterizing",
    "facilitate",
    "facilitates",
    "facilitating",
    "led",
    "promote",
    "promotes",
    "promoting",
    "revolutionize",
    "revolutionizes",
    "revolutionizing",
    "strengthen",
    "strengthened",
    "strengthening",
    "strengthens",
    "streamline",
    "streamlines",
    "streamlining",
    "understand",
    "understanding",
    "understands",
}

NOUN_FILTER_BAD_FINAL_WORDS = {
    "accessible",
    "deployable",
    "existing",
    "practical",
    "societal",
}


class MissingKeywordDependency(MissingBackendDependency):
    """Raised when an optional keyword extraction backend is not installed."""


@dataclass(frozen=True, slots=True)
class KeywordCandidate:
    label: str
    score: float
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class KeywordAssignment:
    award_number: str
    keyword: str
    canonical_keyword: str
    score: str
    extractor: str
    evidence: str


@dataclass(frozen=True, slots=True)
class KeywordExtractionStats:
    awards: int
    keywords: int
    triples: int


@dataclass(frozen=True, slots=True)
class _KeywordRecord:
    award_number: str
    award_uri: str
    raw_keyword: str
    normalized_keyword: str
    score: float
    confidence: str
    evidence: str


class KeywordBackend(Protocol):
    name: str

    def extract(
        self,
        text: str,
        *,
        top_k: int,
        ngram_range: tuple[int, int],
    ) -> Sequence[KeywordCandidate]: ...


class KeywordClusterer(Protocol):
    name: str

    def cluster(
        self,
        labels: Sequence[str],
        *,
        scores: dict[str, float],
    ) -> dict[str, str]: ...


class SimpleKeywordBackend:
    """Dependency-free keyword baseline based on filtered n-gram frequency."""

    name = "simple:frequency"

    def extract(
        self,
        text: str,
        *,
        top_k: int,
        ngram_range: tuple[int, int],
    ) -> list[KeywordCandidate]:
        tokens = _tokenize(text)
        counts: Counter[str] = Counter()
        evidence: dict[str, str] = {}
        min_n, max_n = ngram_range
        for n in range(min_n, max_n + 1):
            for index in range(0, max(len(tokens) - n + 1, 0)):
                phrase_tokens = tokens[index : index + n]
                if not _valid_phrase_tokens(phrase_tokens):
                    continue
                phrase = " ".join(phrase_tokens)
                counts[phrase] += 1
                evidence.setdefault(phrase, _find_evidence_sentence(text, phrase))

        scored = [
            (phrase, count * math.sqrt(len(phrase.split())))
            for phrase, count in counts.items()
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        max_score = scored[0][1] if scored else 1.0
        return [
            KeywordCandidate(
                label=phrase,
                score=score / max_score,
                evidence=evidence.get(phrase, ""),
            )
            for phrase, score in scored[:top_k]
        ]


class KeyBERTKeywordBackend:
    """Semantic keyphrase extraction with KeyBERT."""

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.name = f"keybert:{model}"
        try:
            from keybert import KeyBERT
        except ImportError as exc:
            raise MissingKeywordDependency(
                "KeyBERT is not installed. Run: python3 -m pip install -e '.[keywords]'"
            ) from exc
        self.client = KeyBERT(model=model)

    def extract(
        self,
        text: str,
        *,
        top_k: int,
        ngram_range: tuple[int, int],
    ) -> list[KeywordCandidate]:
        raw_keywords = self.client.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            top_n=top_k,
            use_mmr=True,
            diversity=0.5,
        )
        return [
            KeywordCandidate(
                label=str(label),
                score=float(score),
                evidence=_find_evidence_sentence(text, str(label)),
            )
            for label, score in raw_keywords
        ]


class YakeKeywordBackend:
    """Unsupervised keyphrase extraction with YAKE."""

    name = "yake:en"

    def __init__(self) -> None:
        try:
            import yake
        except ImportError as exc:
            raise MissingKeywordDependency(
                "YAKE is not installed. Run: python3 -m pip install -e '.[yake]'"
            ) from exc
        self.yake = yake

    def extract(
        self,
        text: str,
        *,
        top_k: int,
        ngram_range: tuple[int, int],
    ) -> list[KeywordCandidate]:
        _, max_n = ngram_range
        extractor = self.yake.KeywordExtractor(lan="en", n=max_n, top=top_k)
        raw_keywords = extractor.extract_keywords(text)
        return [
            KeywordCandidate(
                label=str(label),
                score=1.0 / (1.0 + float(raw_score)),
                evidence=_find_evidence_sentence(text, str(label)),
            )
            for label, raw_score in raw_keywords
        ]


class LLMKeywordBackend:
    """Local causal-LLM keyword extraction through Hugging Face Transformers."""

    def __init__(
        self,
        *,
        model_path: str | Path = DEFAULT_LLM_KEYWORD_MODEL_PATH,
        repo_id: str = DEFAULT_LLM_KEYWORD_REPO_ID,
        revision: str | None = None,
        allow_download: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        device_map: str = "auto",
        trust_remote_code: bool = True,
    ) -> None:
        self.model_path = str(model_path)
        self.name = f"llm:{Path(self.model_path).name or self.model_path}"
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self._ensure_model_available(
            repo_id=repo_id,
            revision=revision,
            allow_download=allow_download,
        )
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise MissingKeywordDependency(
                "Transformers is not installed. Run: python3 -m pip install -e '.[llm]'"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype="auto",
            local_files_only=True,
            trust_remote_code=trust_remote_code,
        )

    def _ensure_model_available(
        self,
        *,
        repo_id: str,
        revision: str | None,
        allow_download: bool,
    ) -> None:
        model_path = Path(self.model_path)
        if _is_huggingface_model_dir(model_path):
            return
        if not allow_download:
            raise MissingKeywordDependency(
                f"Local LLM model files were not found in {self.model_path}. "
                "Remove --keyword-llm-no-download to download them, or pass "
                "--keyword-llm-model-path to an existing Hugging Face model directory."
            )
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise MissingKeywordDependency(
                "huggingface-hub is not installed. Run: python3 -m pip install -e '.[llm]'"
            ) from exc

        model_path.mkdir(parents=True, exist_ok=True)
        snapshot_options = {
            "repo_id": repo_id,
            "local_dir": str(model_path),
        }
        if revision:
            snapshot_options["revision"] = revision
        snapshot_download(**snapshot_options)

        if not _is_huggingface_model_dir(model_path):
            if _looks_like_dspark_draft_repo(repo_id):
                raise MissingKeywordDependency(
                    f"{repo_id} was downloaded to {self.model_path}, but it is a "
                    "DeepSpec/DSpark draft checkpoint rather than a standalone "
                    "text-generation model. Use the target model instead, for "
                    "example --keyword-llm-repo-id Qwen/Qwen3-8B "
                    "--keyword-llm-model-path /data/cyang314/kg/Qwen3-8B."
                )
            raise MissingKeywordDependency(
                f"Downloaded files from {repo_id} to {self.model_path}, but the "
                "directory still does not look like a complete Hugging Face model."
            )

    def extract(
        self,
        text: str,
        *,
        top_k: int,
        ngram_range: tuple[int, int],
    ) -> list[KeywordCandidate]:
        prompt = _llm_keyword_prompt(text, top_k=top_k, ngram_range=ngram_range)
        model_input = self._format_prompt(prompt)
        inputs = self.tokenizer(model_input, return_tensors="pt")
        if hasattr(self.model, "device"):
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        generation_options = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            generation_options["temperature"] = self.temperature

        output_ids = self.model.generate(**inputs, **generation_options)
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return _parse_llm_keywords(response, top_k=top_k, text=text)

    def _format_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You extract concise scientific topic keywords from NSF award "
                    "abstracts. Return only valid JSON."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return (
            "System: You extract concise scientific topic keywords from NSF award "
            "abstracts. Return only valid JSON.\n\n"
            f"User: {prompt}\n\nAssistant:"
        )


class LiteLLMKeywordBackend:
    """Keyword extraction through LiteLLM-compatible chat providers."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_LITELLM_KEYWORD_MODEL,
        api_base: str | None = DEFAULT_LITELLM_API_BASE,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> None:
        self.name = f"litellm:{model}"
        self.model = model
        self.api_base = api_base or None
        self.temperature = temperature
        self.max_tokens = max_tokens
        try:
            from litellm import completion
        except ImportError as exc:
            raise MissingKeywordDependency(
                "LiteLLM is not installed. Run: python3 -m pip install -e '.[litellm]'"
            ) from exc
        self.completion = completion

    def extract(
        self,
        text: str,
        *,
        top_k: int,
        ngram_range: tuple[int, int],
    ) -> list[KeywordCandidate]:
        prompt = _llm_keyword_prompt(text, top_k=top_k, ngram_range=ngram_range)
        response = self.completion(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract concise scientific topic keywords from NSF "
                        "award abstracts. Return only valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            api_base=self.api_base,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = response.choices[0].message.content or ""
        content = _strip_thinking_blocks(content)
        return _parse_llm_keywords(content, top_k=top_k, text=text)


class EmbeddingKeywordClusterer:
    """Merge semantically similar keyword labels using sentence embeddings."""

    def __init__(
        self,
        *,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.82,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Keyword cluster threshold must be between 0.0 and 1.0")
        self.name = f"embedding:{model}"
        self.threshold = threshold
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise MissingKeywordDependency(
                "sentence-transformers is not installed. Run: python3 -m pip install -e '.[keywords]'"
            ) from exc
        self.model = SentenceTransformer(model)

    def cluster(
        self,
        labels: Sequence[str],
        *,
        scores: dict[str, float],
    ) -> dict[str, str]:
        unique_labels = sorted(set(labels), key=lambda label: (-scores.get(label, 0.0), len(label), label))
        if not unique_labels:
            return {}
        embeddings = self.model.encode(unique_labels, normalize_embeddings=True)
        vectors = {
            label: [float(value) for value in vector]
            for label, vector in zip(unique_labels, embeddings)
        }
        mapping: dict[str, str] = {}
        for representative in unique_labels:
            if representative in mapping:
                continue
            mapping[representative] = representative
            representative_vector = vectors[representative]
            for label in unique_labels:
                if label in mapping:
                    continue
                if _cosine_similarity(representative_vector, vectors[label]) >= self.threshold:
                    mapping[label] = representative
        return mapping


def clean_abstract_for_keywords(text: str) -> str:
    """Remove corpus-level boilerplate that would otherwise become a shared keyword."""
    text = NSF_BOILERPLATE_PATTERN.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def _llm_keyword_prompt(
    text: str,
    *,
    top_k: int,
    ngram_range: tuple[int, int],
) -> str:
    min_n, max_n = ngram_range
    return (
        f"Extract up to {top_k} scientific topic keywords from the abstract below.\n"
        f"Each keyword must be a short noun topic of {min_n} to {max_n} words.\n"
        "Prefer compact domain concepts over full phrases. Use singular nouns when possible.\n"
        "Do not return verbs, verb phrases, actions, complete clauses, institution names, "
        "person names, grant boilerplate, generic words, or duplicates.\n"
        "Bad examples: \"develop\", \"improving systems\", \"students learn\", "
        "\"using sensors\".\n"
        "Good examples: \"quantum sensing\", \"digital twin\", \"robotics\", "
        "\"privacy\".\n"
        "Return only a JSON array of strings, for example: "
        '["quantum sensing", "digital twin"].\n\n'
        f"Abstract:\n{text}"
    )


def _parse_llm_keywords(
    response: str,
    *,
    top_k: int,
    text: str,
) -> list[KeywordCandidate]:
    values = _extract_json_keywords(response)
    if not values:
        values = _extract_line_keywords(response)

    keywords: list[KeywordCandidate] = []
    seen: set[str] = set()
    for value in values:
        label = str(value).strip()
        if not label:
            continue
        canonical = canonical_keyword(label)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        rank = len(keywords)
        score = max(0.0, 1.0 - rank * 0.01)
        keywords.append(
            KeywordCandidate(
                label=label,
                score=score,
                evidence=_find_evidence_sentence(text, label),
            )
        )
        if len(keywords) >= top_k:
            break
    return keywords


def _extract_json_keywords(response: str) -> list[str]:
    start = response.find("[")
    end = response.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(response[start : end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list):
        return []

    values: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            values.append(item)
        elif isinstance(item, dict):
            value = item.get("keyword") or item.get("label") or item.get("term")
            if value:
                values.append(str(value))
    return values


def _extract_line_keywords(response: str) -> list[str]:
    values: list[str] = []
    for line in response.splitlines():
        line = re.sub(r"^\s*[-*•]?\s*\d*[\.)]?\s*", "", line).strip()
        line = line.strip("\"'`,; ")
        if line:
            values.append(line)
    return values


def _strip_thinking_blocks(response: str) -> str:
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()


def _is_huggingface_model_dir(path: str | Path) -> bool:
    path = Path(path)
    if not path.is_dir():
        return False
    has_config = (path / "config.json").is_file()
    has_tokenizer = any(
        (path / name).is_file()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "vocab.txt",
        )
    )
    has_weights = any(
        any(path.glob(pattern)) for pattern in _MODEL_WEIGHT_PATTERNS
    )
    return has_config and has_tokenizer and has_weights


def _looks_like_dspark_draft_repo(repo_id: str) -> bool:
    normalized = repo_id.casefold()
    return "dspark_" in normalized or "/dspark" in normalized


_MODEL_WEIGHT_PATTERNS = (
    "*.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)


def canonical_keyword(label: str, *, noun_filter: bool = False) -> str:
    """Normalize a candidate keyphrase into a stable, readable graph label."""
    label = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode()
    label = _expand_keyword_aliases(label)
    label = re.sub(r"[^a-zA-Z0-9]+", " ", label).casefold()
    tokens = [token for token in label.split() if token and token not in STOPWORDS]
    tokens = [_singularize(token) for token in tokens]
    if noun_filter:
        tokens = _noun_filter_tokens(tokens)
    return " ".join(tokens)


def extract_keyword_triples(
    input_csv: str | Path,
    backend: KeywordBackend,
    *,
    base_uri: str = DEFAULT_BASE_URI,
    limit: int | None = None,
    top_k: int = 8,
    min_score: float = 0.0,
    ngram_range: tuple[int, int] = (1, 3),
    clusterer: KeywordClusterer | None = None,
    noun_filter: bool = False,
) -> tuple[list[Triple], list[KeywordAssignment], KeywordExtractionStats]:
    if top_k <= 0:
        raise ValueError("Keyword top-k must be greater than zero")
    min_n, max_n = ngram_range
    if min_n <= 0 or max_n < min_n:
        raise ValueError("Keyword n-gram range must be positive and ordered")

    triples: list[Triple] = []
    records: list[_KeywordRecord] = []
    processed = 0

    with Path(input_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            number = clean_excel_literal(row.get("AwardNumber"))
            text = clean_abstract_for_keywords((row.get("Abstract") or "").strip())
            if not number or not text:
                continue
            if limit is not None and processed >= limit:
                break
            processed += 1
            award_uri = f"{base_uri}award/{number}"
            seen_for_award: set[str] = set()
            for candidate in backend.extract(text, top_k=top_k, ngram_range=ngram_range):
                if candidate.score < min_score:
                    continue
                canonical = canonical_keyword(candidate.label, noun_filter=noun_filter)
                if not canonical or canonical in seen_for_award:
                    continue
                seen_for_award.add(canonical)
                confidence = f"{candidate.score:.6f}"
                evidence = candidate.evidence or _find_evidence_sentence(text, canonical) or text
                records.append(
                    _KeywordRecord(
                        award_number=number,
                        award_uri=award_uri,
                        raw_keyword=candidate.label,
                        normalized_keyword=canonical,
                        score=candidate.score,
                        confidence=confidence,
                        evidence=evidence,
                    )
                )

    cluster_map = _keyword_cluster_map(
        records,
        clusterer=clusterer,
        noun_filter=noun_filter,
    )
    extractor_name = backend.name if clusterer is None else f"{backend.name}+{clusterer.name}"
    assignments: list[KeywordAssignment] = []
    labelled: set[str] = set()
    linked: set[tuple[str, str]] = set()
    assigned: set[tuple[str, str]] = set()
    for record in records:
        canonical = cluster_map.get(record.normalized_keyword, record.normalized_keyword)
        if not canonical:
            continue
        assignment_key = (record.award_number, canonical)
        if assignment_key in assigned:
            continue
        assigned.add(assignment_key)
        keyword_uri = entity_uri(base_uri, "keyword", canonical)
        assignments.append(
            KeywordAssignment(
                award_number=record.award_number,
                keyword=record.raw_keyword,
                canonical_keyword=canonical,
                score=record.confidence,
                extractor=extractor_name,
                evidence=record.evidence,
            )
        )

        link_key = (record.award_uri, keyword_uri)
        if link_key not in linked:
            linked.add(link_key)
            triples.append(
                Triple(
                    record.award_uri,
                    f"{base_uri}vocab/hasKeyword",
                    keyword_uri,
                    "iri",
                    award_number=record.award_number,
                    source_column="Abstract",
                    evidence=record.evidence,
                    confidence=record.confidence,
                    extractor=extractor_name,
                )
            )
        if keyword_uri not in labelled:
            labelled.add(keyword_uri)
            triples.extend(
                [
                    Triple(
                        keyword_uri,
                        RDF_TYPE,
                        f"{base_uri}vocab/Keyword",
                        "iri",
                        award_number=record.award_number,
                        source_column="Abstract",
                        evidence=record.evidence,
                        confidence=record.confidence,
                        extractor=extractor_name,
                    ),
                    Triple(
                        keyword_uri,
                        SCHEMA + "name",
                        canonical,
                        "literal",
                        award_number=record.award_number,
                        source_column="Abstract",
                        evidence=record.evidence,
                        confidence=record.confidence,
                        extractor=extractor_name,
                    ),
                ]
            )

    return triples, assignments, KeywordExtractionStats(processed, len(assignments), len(triples))


def _keyword_cluster_map(
    records: Sequence[_KeywordRecord],
    *,
    clusterer: KeywordClusterer | None,
    noun_filter: bool,
) -> dict[str, str]:
    labels = [record.normalized_keyword for record in records]
    if clusterer is None:
        return {label: label for label in labels}
    scores: dict[str, float] = {}
    for record in records:
        scores[record.normalized_keyword] = max(
            scores.get(record.normalized_keyword, 0.0), record.score
        )
    cluster_map = clusterer.cluster(labels, scores=scores)
    return {
        label: canonical_keyword(cluster_map.get(label, label), noun_filter=noun_filter)
        for label in labels
    }


def write_keyword_assignments_csv(
    assignments: Iterable[KeywordAssignment], output_path: str | Path
) -> None:
    fields = list(KeywordAssignment.__dataclass_fields__)
    with Path(output_path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for assignment in assignments:
            writer.writerow(asdict(assignment))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z][a-zA-Z0-9-]*", text.casefold())


def _valid_phrase_tokens(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    if tokens[0] in STOPWORDS or tokens[-1] in STOPWORDS:
        return False
    informative = [token for token in tokens if token not in STOPWORDS]
    return bool(informative) and any(len(token) > 2 for token in informative)


def _singularize(token: str) -> str:
    irregular = {
        "analyses": "analysis",
        "children": "child",
        "data": "data",
        "indices": "index",
        "matrices": "matrix",
        "methods": "method",
        "models": "model",
        "people": "person",
        "systems": "system",
        "technologies": "technology",
    }
    if token in irregular:
        return irregular[token]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _noun_filter_tokens(tokens: Sequence[str]) -> list[str]:
    tokens = [token for token in tokens if token and token not in STOPWORDS]
    while tokens and tokens[0] in NOUN_FILTER_TRIM_PREFIXES:
        tokens = tokens[1:]
    if not tokens:
        return []
    if tokens[0] in NOUN_FILTER_REJECT_PREFIXES:
        return []
    if len(tokens) < 2:
        return []
    if tokens[-1] in NOUN_FILTER_BAD_FINAL_WORDS:
        return []
    if tokens[-1].endswith("ing") or tokens[-1].endswith("ly"):
        return []
    return list(tokens)


def _expand_keyword_aliases(label: str) -> str:
    replacements = {
        r"\bAI\b": "artificial intelligence",
        r"\bLLMs\b": "large language models",
        r"\bLLM\b": "large language model",
        r"\bML\b": "machine learning",
    }
    for pattern, replacement in replacements.items():
        label = re.sub(pattern, replacement, label, flags=re.IGNORECASE)
    return label


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _find_evidence_sentence(text: str, phrase: str) -> str:
    normalized_phrase = phrase.casefold()
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        if normalized_phrase in sentence.casefold():
            return sentence
    return ""
