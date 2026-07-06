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


    print(f"Processed {stats.awards} awards and generated {len(triples)} triples.")
    if abstract_stats:
        print(
            f"Abstract backend processed {abstract_stats.awards} awards, extracted "
            f"{abstract_stats.relations} relations, and added {abstract_stats.triples} triples."
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
