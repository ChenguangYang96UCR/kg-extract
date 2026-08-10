from __future__ import annotations

import argparse
from pathlib import Path

from .abstracts import (
    KGGenBackend,
    MissingBackendDependency,
    UIEBackend,
    api_key_from_environment,
    extract_abstract_triples,
    load_uie_schema,
)
from .extractor import DEFAULT_BASE_URI, extract_awards, write_csv, write_ntriples
from .keywords import (
    KeyBERTKeywordBackend,
    MissingKeywordDependency,
    SimpleKeywordBackend,
    YakeKeywordBackend,
    EmbeddingKeywordClusterer,
    extract_keyword_triples,
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
        "--keyword-backend",
        choices=("none", "simple", "keybert", "yake"),
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
    keyword_stats = None
    keyword_triples = []
    keyword_assignments = []
    if args.abstract_backend != "none":
        try:
            backend = _build_abstract_backend(args)
            abstract_triples, abstract_stats = extract_abstract_triples(
                args.input_csv,
                backend,
                base_uri=base_uri,
                limit=args.abstract_limit,
                min_confidence=args.min_confidence,
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


def _build_keyword_backend(args: argparse.Namespace):
    if args.keyword_backend == "simple":
        return SimpleKeywordBackend()
    if args.keyword_backend == "keybert":
        return KeyBERTKeywordBackend(model=args.keyword_model)
    return YakeKeywordBackend()


def _build_keyword_clusterer(args: argparse.Namespace):
    if not args.keyword_cluster:
        return None
    return EmbeddingKeywordClusterer(
        model=args.keyword_cluster_model,
        threshold=args.keyword_cluster_threshold,
    )
