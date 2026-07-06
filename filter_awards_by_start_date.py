#!/usr/bin/env python3
"""Split an NSF Awards CSV into 2024 and 2025-2026 start-date groups."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable


DATE_FORMATS = ("%m/%d/%Y", "%Y-%m-%d")


def parse_start_year(value: str) -> int:
    """Return the year from a supported StartDate value."""
    cleaned = value.strip()
    for date_format in DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, date_format).year
        except ValueError:
            continue
    raise ValueError(f"Unsupported StartDate value: {value!r}")


def write_rows(
    path: Path, fieldnames: list[str], rows: Iterable[dict[str, str]]
) -> int:
    """Write rows while preserving the source CSV column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def filter_awards(input_path: Path, output_dir: Path) -> tuple[Path, int, Path, int]:
    """Filter awards and return both output paths and their row counts."""
    with input_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if not reader.fieldnames:
            raise ValueError("The input CSV has no header row.")
        if "StartDate" not in reader.fieldnames:
            raise ValueError("The input CSV must contain a StartDate column.")
        fieldnames = reader.fieldnames

        rows_2024: list[dict[str, str]] = []
        rows_2025_2026: list[dict[str, str]] = []
        invalid_rows: list[tuple[int, str]] = []

        for line_number, row in enumerate(reader, start=2):
            start_date = row.get("StartDate", "")
            try:
                year = parse_start_year(start_date)
            except ValueError:
                invalid_rows.append((line_number, start_date))
                continue

            if year == 2024:
                rows_2024.append(row)
            elif 2025 <= year <= 2026:
                rows_2025_2026.append(row)

    if invalid_rows:
        examples = ", ".join(
            f"line {line_number}: {value!r}"
            for line_number, value in invalid_rows[:5]
        )
        raise ValueError(
            f"Found {len(invalid_rows)} invalid StartDate value(s): {examples}"
        )

    output_2024 = output_dir / "awards_start_2024.csv"
    output_2025_2026 = output_dir / "awards_start_2025_2026.csv"
    count_2024 = write_rows(output_2024, fieldnames, rows_2024)
    count_2025_2026 = write_rows(
        output_2025_2026, fieldnames, rows_2025_2026
    )
    return output_2024, count_2024, output_2025_2026, count_2025_2026


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Filter an NSF Awards CSV into a 2024 file and a 2025-2026 file "
            "using the StartDate column."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to the source CSV file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("filtered-awards"),
        help="Output directory (default: filtered-awards).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_2024, count_2024, output_2025_2026, count_2025_2026 = filter_awards(
        args.input_csv, args.output_dir
    )
    print(f"Saved {count_2024} rows to {output_2024}")
    print(f"Saved {count_2025_2026} rows to {output_2025_2026}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
