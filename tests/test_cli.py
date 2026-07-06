import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kg_extract.abstracts import AbstractRelation
from kg_extract.cli import main


class FakeBackend:
    name = "fake:model"

    def extract(self, text, *, context=""):
        return [
            AbstractRelation(
                "This project",
                "develops",
                "a debug graph",
                confidence=0.9,
                evidence="This project develops a debug graph.",
            )
        ]


class CliTests(unittest.TestCase):
    def test_writes_abstract_only_debug_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_csv = root / "awards.csv"
            output_dir = root / "output"
            with input_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["AwardNumber", "Title", "Abstract"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "AwardNumber": "123",
                        "Title": "Debug Award",
                        "Abstract": "This project develops a debug graph.",
                    }
                )

            with patch("kg_extract.cli._build_abstract_backend", return_value=FakeBackend()):
                status = main(
                    [
                        str(input_csv),
                        "--output-dir",
                        str(output_dir),
                        "--abstract-backend",
                        "kggen",
                    ]
                )

            self.assertEqual(status, 0)
            self.assertTrue((output_dir / "triples.csv").is_file())
            self.assertTrue((output_dir / "triples.nt").is_file())
            self.assertTrue((output_dir / "abstract_triples.csv").is_file())
            self.assertTrue((output_dir / "abstract_triples.nt").is_file())

            with (output_dir / "abstract_triples.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertTrue(any(row["source_column"] == "Abstract" for row in rows))
            self.assertTrue(all(row["extractor"] == "fake:model" for row in rows))


if __name__ == "__main__":
    unittest.main()
