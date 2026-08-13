from __future__ import annotations

import argparse
import os
from pathlib import Path

from .abstracts import (
    DEFAULT_ABSTRACT_NODE_CLEANER_API_BASE,
    DEFAULT_ABSTRACT_NODE_CLEANER_MODEL,
    KGGenBackend,
    LLMAbstractNodeCleaner,
    MissingBackendDependency,
    UIEBackend,
    api_key_from_environment,
    extract_abstract_triples,
    load_uie_schema,
    write_abstract_node_cleaning_csv,
)
from .extractor import DEFAULT_BASE_URI, extract_awards, write_csv, write_ntriples
from .keywords import (
    DEFAULT_LLM_KEYWORD_MODEL_PATH,
    DEFAULT_LLM_KEYWORD_REPO_ID,
    DEFAULT_LITELLM_API_BASE,
    DEFAULT_LITELLM_KEYWORD_MODEL,
    DEFAULT_TINKER_API_BASE,
    DEFAULT_TINKER_API_KEY_ENV,
    DEFAULT_TINKER_MODEL_ENV,
    KeyBERTKeywordBackend,
    LLMKeywordBackend,
    LiteLLMKeywordBackend,
    MissingKeywordDependency,
    SimpleKeywordBackend,
    YakeKeywordBackend,
    EmbeddingKeywordClusterer,
    extract_keyword_triples,
    litellm_openai_compatible_model,
    write_keyword_assignments_csv,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kg-extract",
        description="Convert an NSF Awards CSV file into traceable knowledge graph triples.",
    )
    parser.add_argument("input_csv", type=Path, help="Path to the NSF Awards CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output directory (default: output)",
    )
    parser.add_argument(
        "--base-uri",
        default=DEFAULT_BASE_URI,
        help=f"Base URI for generated entities (default: {DEFAULT_BASE_URI})",
    )
    parser.add_argument(
        "--include-contact",
        action="store_true",
        help="Include email, phone, and postal-address triples (off by default)",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "nt", "both"),
        default="both",
        help="Output format (default: both)",
    )
    parser.add_argument(
        "--abstract-backend",
        choices=("none", "kggen", "uie"),
        default="none",
        help="Optional Abstract relation extractor (default: none)",
    )
    parser.add_argument(
        "--abstract-model",
        help="Backend model name (defaults: openai/gpt-4o for KGGen, uie-base-en for UIE)",
    )
    parser.add_argument(
        "--abstract-limit",
        type=int,
        help="Process at most this many non-empty abstracts",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Discard model relations below this confidence (UIE only)",
    )
    parser.add_argument(
        "--api-key-env",
        help="Environment variable containing the KGGen provider API key",
    )
    parser.add_argument("--base-url", help="Optional KGGen model provider base URL")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="KGGen chunk size in characters (default: 5000)",
    )
    parser.add_argument(
        "--no-cluster",
        action="store_true",
        help="Disable KGGen entity and relation clustering (kept for compatibility)",
    )
    parser.add_argument(
        "--kggen-deduplicate",
        action="store_true",
        help=(
            "Enable KGGen's built-in deduplication/clustering. Off by default because "
            "some kg-gen and semhash version combinations are incompatible."
        ),
    )
    parser.add_argument(
        "--uie-schema",
        type=Path,
        help="Path to a custom UIE schema JSON file",
    )
    parser.add_argument(
        "--uie-chunk-size",
        type=int,
        default=256,
        help="Maximum UIE input chunk size in characters (default: 256)",
    )
    parser.add_argument(
        "--abstract-node-cleaner",
        choices=("none", "llm"),
        default="none",
        help="Optional semantic cleaner for Abstract subject/object node labels (default: none)",
    )
    parser.add_argument(
        "--abstract-node-cleaner-model",
        default=DEFAULT_ABSTRACT_NODE_CLEANER_MODEL,
        help=(
            "LiteLLM model for --abstract-node-cleaner llm "
            f"(default: {DEFAULT_ABSTRACT_NODE_CLEANER_MODEL})"
        ),
    )
    parser.add_argument(
        "--abstract-node-cleaner-api-base",
        default=DEFAULT_ABSTRACT_NODE_CLEANER_API_BASE,
        help=(
            "LiteLLM API base URL for --abstract-node-cleaner llm "
            f"(default: {DEFAULT_ABSTRACT_NODE_CLEANER_API_BASE})"
        ),
    )
    parser.add_argument(
        "--abstract-node-cleaner-api-key-env",
        help="Environment variable containing the node cleaner provider API key",
    )
    parser.add_argument(
        "--abstract-node-cleaner-max-tokens",
        type=int,
        default=1024,
        help="Maximum generated tokens for Abstract node cleaning (default: 1024)",
    )
    parser.add_argument(
        "--abstract-node-cleaner-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Abstract node cleaning (default: 0.0)",
    )
    parser.add_argument(
        "--keyword-backend",
        choices=("none", "simple", "keybert", "yake", "llm", "litellm", "tinker"),
        default="none",
        help="Optional Abstract keyword extractor (default: none)",
    )
    parser.add_argument(
        "--keyword-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="KeyBERT sentence-transformers model (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--keyword-limit",
        type=int,
        help="Process at most this many non-empty abstracts for keyword extraction",
    )
    parser.add_argument(
        "--keyword-top-k",
        type=int,
        default=8,
        help="Maximum keywords to keep per award (default: 8)",
    )
    parser.add_argument(
        "--keyword-min-score",
        type=float,
        default=0.0,
        help="Discard keyword candidates below this backend score (default: 0.0)",
    )
    parser.add_argument(
        "--keyword-ngram-min",
        type=int,
        default=1,
        help="Minimum keyword n-gram length (default: 1)",
    )
    parser.add_argument(
        "--keyword-ngram-max",
        type=int,
        default=3,
        help="Maximum keyword n-gram length (default: 3)",
    )
    parser.add_argument(
        "--keyword-llm-model-path",
        default=DEFAULT_LLM_KEYWORD_MODEL_PATH,
        help="Local Hugging Face causal-LM path for --keyword-backend llm",
    )
    parser.add_argument(
        "--keyword-llm-repo-id",
        default=DEFAULT_LLM_KEYWORD_REPO_ID,
        help=(
            "Hugging Face repo id to download when the local LLM model path is "
            f"incomplete (default: {DEFAULT_LLM_KEYWORD_REPO_ID})"
        ),
    )
    parser.add_argument(
        "--keyword-llm-revision",
        help="Optional Hugging Face model revision, branch, or commit for LLM download",
    )
    parser.add_argument(
        "--keyword-llm-no-download",
        action="store_true",
        help="Fail instead of downloading the local LLM model when files are missing",
    )
    parser.add_argument(
        "--keyword-llm-max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens for local LLM keyword generation (default: 256)",
    )
    parser.add_argument(
        "--keyword-llm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for local LLM keyword generation (default: 0.0)",
    )
    parser.add_argument(
        "--keyword-llm-device-map",
        default="auto",
        help="Transformers device_map for local LLM loading (default: auto)",
    )
    parser.add_argument(
        "--keyword-litellm-model",
        default=DEFAULT_LITELLM_KEYWORD_MODEL,
        help=f"LiteLLM model for --keyword-backend litellm (default: {DEFAULT_LITELLM_KEYWORD_MODEL})",
    )
    parser.add_argument(
        "--keyword-litellm-api-base",
        default=DEFAULT_LITELLM_API_BASE,
        help=f"LiteLLM API base URL (default: {DEFAULT_LITELLM_API_BASE})",
    )
    parser.add_argument(
        "--keyword-litellm-max-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens for LiteLLM keyword extraction (default: 256)",
    )
    parser.add_argument(
        "--keyword-litellm-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for LiteLLM keyword extraction (default: 0.0)",
    )
    parser.add_argument(
        "--keyword-tinker-model",
        help=(
            "Tinker sampler checkpoint path for --keyword-backend tinker, for example "
            "'tinker://.../sampler_weights/...'. Can also be set with "
            f"{DEFAULT_TINKER_MODEL_ENV}."
        ),
    )
    parser.add_argument(
        "--keyword-tinker-api-base",
        default=DEFAULT_TINKER_API_BASE,
        help=f"Tinker OpenAI-compatible API base URL (default: {DEFAULT_TINKER_API_BASE})",
    )
    parser.add_argument(
        "--keyword-tinker-api-key-env",
        default=DEFAULT_TINKER_API_KEY_ENV,
        help=f"Environment variable containing the Tinker API key (default: {DEFAULT_TINKER_API_KEY_ENV})",
    )
    parser.add_argument(
        "--keyword-tinker-max-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens for Tinker keyword extraction (default: 256)",
    )
    parser.add_argument(
        "--keyword-tinker-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for Tinker keyword extraction (default: 0.0)",
    )
    parser.add_argument(
        "--keyword-tinker-reasoning-effort",
        choices=("none", "minimal", "low", "medium", "high", "xhigh"),
        help="Optional Tinker reasoning effort for chat completions.",
    )
    parser.add_argument(
        "--keyword-noun-filter",
        action="store_true",
        help="Prefer noun-like keyword phrases by trimming or dropping verb-led candidates",
    )
    parser.add_argument(
        "--keyword-cluster",
        action="store_true",
        help="Merge semantically similar keywords with sentence-transformers embeddings",
    )
    parser.add_argument(
        "--keyword-cluster-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model for keyword clustering (default: all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--keyword-cluster-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for keyword clustering (default: 0.82)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input_csv.is_file():
        raise SystemExit(f"Input CSV does not exist: {args.input_csv}")

    base_uri = args.base_uri.rstrip("/") + "/"
    triples, stats = extract_awards(
        args.input_csv,
        base_uri=base_uri,
        include_contact=args.include_contact,
    )
    abstract_stats = None
    abstract_triples = []
    abstract_node_cleaner = None
    keyword_stats = None
    keyword_triples = []
    keyword_assignments = []
    if args.abstract_backend != "none":
        try:
            backend = _build_abstract_backend(args)
            abstract_node_cleaner = _build_abstract_node_cleaner(args)
            abstract_triples, abstract_stats = extract_abstract_triples(
                args.input_csv,
                backend,
                base_uri=base_uri,
                limit=args.abstract_limit,
                min_confidence=args.min_confidence,
                node_cleaner=abstract_node_cleaner,
            )
        except (MissingBackendDependency, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        triples.extend(abstract_triples)
    if args.keyword_backend != "none":
        try:
            keyword_backend = _build_keyword_backend(args)
            keyword_clusterer = _build_keyword_clusterer(args)
            keyword_triples, keyword_assignments, keyword_stats = extract_keyword_triples(
                args.input_csv,
                keyword_backend,
                base_uri=base_uri,
                limit=args.keyword_limit,
                top_k=args.keyword_top_k,
                min_score=args.keyword_min_score,
                ngram_range=(args.keyword_ngram_min, args.keyword_ngram_max),
                clusterer=keyword_clusterer,
                noun_filter=args.keyword_noun_filter,
            )
        except (MissingKeywordDependency, ValueError) as exc:
            raise SystemExit(str(exc)) from exc
        triples.extend(keyword_triples)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    if args.format in {"csv", "both"}:
        path = args.output_dir / "triples.csv"
        write_csv(triples, path)
        written.append(path)
    if args.format in {"nt", "both"}:
        path = args.output_dir / "triples.nt"
        write_ntriples(triples, path)
        written.append(path)
    if abstract_stats:
        if args.format in {"csv", "both"}:
            path = args.output_dir / "abstract_triples.csv"
            write_csv(abstract_triples, path)
            written.append(path)
        if args.format in {"nt", "both"}:
            path = args.output_dir / "abstract_triples.nt"
            write_ntriples(abstract_triples, path)
            written.append(path)
        if abstract_node_cleaner is not None and abstract_node_cleaner.records:
            path = args.output_dir / "abstract_node_cleaning.csv"
            write_abstract_node_cleaning_csv(abstract_node_cleaner.records, path)
            written.append(path)
    if keyword_stats:
        path = args.output_dir / "keywords.csv"
        write_keyword_assignments_csv(keyword_assignments, path)
        written.append(path)
        if args.format in {"csv", "both"}:
            path = args.output_dir / "keyword_triples.csv"
            write_csv(keyword_triples, path)
            written.append(path)
        if args.format in {"nt", "both"}:
            path = args.output_dir / "keyword_triples.nt"
            write_ntriples(keyword_triples, path)
            written.append(path)

    print(f"Processed {stats.awards} awards and generated {len(triples)} triples.")
    if abstract_stats:
        print(
            f"Abstract backend processed {abstract_stats.awards} awards, extracted "
            f"{abstract_stats.relations} relations, and added {abstract_stats.triples} triples."
        )
    if keyword_stats:
        print(
            f"Keyword backend processed {keyword_stats.awards} awards, extracted "
            f"{keyword_stats.keywords} keyword assignments, and added "
            f"{keyword_stats.triples} triples."
        )
    for path in written:
        print(f"Wrote {path}")
    if stats.skipped_rows:
        print(f"Skipped {stats.skipped_rows} rows without an AwardNumber.")
    return 0


def _build_abstract_backend(args: argparse.Namespace):
    if args.abstract_backend == "kggen":
        return KGGenBackend(
            model=args.abstract_model or "openai/gpt-4o",
            api_key=api_key_from_environment(args.api_key_env),
            base_url=args.base_url,
            chunk_size=args.chunk_size,
            cluster=args.kggen_deduplicate and not args.no_cluster,
        )
    return UIEBackend(
        model=args.abstract_model or "uie-base-en",
        schema=load_uie_schema(args.uie_schema),
        chunk_size=args.uie_chunk_size,
    )


def _build_abstract_node_cleaner(args: argparse.Namespace):
    if args.abstract_node_cleaner == "none":
        return None
    return LLMAbstractNodeCleaner(
        model=args.abstract_node_cleaner_model,
        api_base=args.abstract_node_cleaner_api_base,
        api_key=api_key_from_environment(args.abstract_node_cleaner_api_key_env),
        temperature=args.abstract_node_cleaner_temperature,
        max_tokens=args.abstract_node_cleaner_max_tokens,
    )


def _build_keyword_backend(args: argparse.Namespace):
    if args.keyword_backend == "simple":
        return SimpleKeywordBackend()
    if args.keyword_backend == "keybert":
        return KeyBERTKeywordBackend(model=args.keyword_model)
    if args.keyword_backend == "llm":
        return LLMKeywordBackend(
            model_path=args.keyword_llm_model_path,
            repo_id=args.keyword_llm_repo_id,
            revision=args.keyword_llm_revision,
            allow_download=not args.keyword_llm_no_download,
            max_new_tokens=args.keyword_llm_max_new_tokens,
            temperature=args.keyword_llm_temperature,
            device_map=args.keyword_llm_device_map,
        )
    if args.keyword_backend == "litellm":
        return LiteLLMKeywordBackend(
            model=args.keyword_litellm_model,
            api_base=args.keyword_litellm_api_base,
            temperature=args.keyword_litellm_temperature,
            max_tokens=args.keyword_litellm_max_tokens,
        )
    if args.keyword_backend == "tinker":
        model = args.keyword_tinker_model or os.environ.get(DEFAULT_TINKER_MODEL_ENV)
        if not model:
            raise ValueError(
                "Tinker keyword extraction requires --keyword-tinker-model or "
                f"{DEFAULT_TINKER_MODEL_ENV}."
            )
        extra_body: dict[str, object] = {"separate_reasoning": True}
        if args.keyword_tinker_reasoning_effort:
            extra_body["reasoning_effort"] = args.keyword_tinker_reasoning_effort
        return LiteLLMKeywordBackend(
            model=litellm_openai_compatible_model(model),
            api_base=args.keyword_tinker_api_base,
            api_key=api_key_from_environment(args.keyword_tinker_api_key_env),
            temperature=args.keyword_tinker_temperature,
            max_tokens=args.keyword_tinker_max_tokens,
            extra_body=extra_body,
        )
    return YakeKeywordBackend()


def _build_keyword_clusterer(args: argparse.Namespace):
    if not args.keyword_cluster:
        return None
    return EmbeddingKeywordClusterer(
        model=args.keyword_cluster_model,
        threshold=args.keyword_cluster_threshold,
    )
