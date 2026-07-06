import csv
import tempfile
import unittest
from pathlib import Path

from kg_extract.extractor import (
    SCHEMA,
    clean_excel_literal,
    extract_awards,
    normalize_date,
    normalize_money,
    parse_co_investigators,
    write_csv,
    write_ntriples,
)


class NormalizationTests(unittest.TestCase):
    def test_excel_literal(self):
        self.assertEqual(clean_excel_literal('="2541748"'), "2541748")
        self.assertEqual(clean_excel_literal("ordinary"), "ordinary")

    def test_date(self):
        self.assertEqual(normalize_date("10/01/2026"), "2026-10-01")

    def test_money(self):
        self.assertEqual(normalize_money("$349,875.00"), "349875.00")

    def test_co_investigators(self):
        value = "Lizy K John ljohn@example.edu,Aman Arora aman@example.edu"
        self.assertEqual(
            parse_co_investigators(value),
            [("Lizy K John", "ljohn@example.edu"), ("Aman Arora", "aman@example.edu")],
        )


class ExtractionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.input = self.root / "awards.csv"
        fields = [
            "AwardNumber",
            "Title",
            "NSFOrganization",
            "Program(s)",
            "StartDate",
            "LastAmendmentDate",
            "PrincipalInvestigator",
            "State",
            "Organization",
            "AwardInstrument",
            "ProgramManager",
            "EndDate",
            "AwardedAmountToDate",
            "Co-PIName(s)",
            "PIEmailAddress",
            "OrganizationStreet",
            "OrganizationCity",
            "OrganizationState",
            "OrganizationZip",
            "OrganizationPhone",
            "NSFDirectorate",
            "ProgramElementCode(s)",
            "ProgramReferenceCode(s)",
            "ARRAAmount",
            "Abstract",
        ]
        row = {field: "" for field in fields}
        row.update(
            {
                "AwardNumber": '="2541748"',
                "Title": "A Test Award",
                "Program(s)": "Program A, Program B",
                "StartDate": "10/01/2026",
                "PrincipalInvestigator": "Chenguang Wang",
                "Organization": "University of California-Santa Cruz",
                "EndDate": "09/30/2031",
                "AwardedAmountToDate": "$349,875.00",
                "PIEmailAddress": "private@example.edu",
                "Abstract": "This project develops a safety-aware learning framework.",
            }
        )
        with self.input.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow(row)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extracts_entities_and_typed_values_without_contact_by_default(self):
        triples, stats = extract_awards(self.input)
        self.assertEqual(stats.awards, 1)
        self.assertGreater(stats.triples, 10)
        self.assertTrue(any(t.predicate == SCHEMA + "amount" and t.object == "349875.00" for t in triples))
        self.assertTrue(any(t.predicate == SCHEMA + "description" for t in triples))
        self.assertFalse(any(t.predicate == SCHEMA + "email" for t in triples))
        self.assertTrue(all(t.extractor == "rules" for t in triples))
        self.assertTrue(all(t.confidence == "1.0" for t in triples))

    def test_contact_is_opt_in(self):
        triples, _ = extract_awards(self.input, include_contact=True)
        self.assertTrue(any(t.predicate == SCHEMA + "email" for t in triples))

    def test_writers(self):
        triples, _ = extract_awards(self.input)
        csv_path = self.root / "triples.csv"
        nt_path = self.root / "triples.nt"
        write_csv(triples, csv_path)
        write_ntriples(triples, nt_path)
        self.assertIn("source_column", csv_path.read_text(encoding="utf-8"))
        self.assertIn(
            "<https://schema.org/MonetaryGrant>",
            nt_path.read_text(encoding="utf-8"),
        )



if __name__ == "__main__":
    unittest.main()
