import csv
import tempfile
import unittest
from pathlib import Path

from kg_extract.abstracts import (
    AbstractRelation,
    KGGenBackend,
    UIEBackend,
    extract_abstract_triples,
    preprocess_abstract_for_kggen,
)


class FakeGraph:
    relations = {
        ("This project", "develops", "a safety-aware framework"),
        ("the framework", "targets domain", "healthcare"),
    }


class FakeKGGenClient:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return FakeGraph()


class FakeModernKGGenClient:
    def __init__(self):
        self.calls = []

    def generate(
        self,
        input_data,
        context="",
        chunk_size=None,
        deduplication_method="semhash",
    ):
        self.calls.append(
            {
                "input_data": input_data,
                "context": context,
                "chunk_size": chunk_size,
                "deduplication_method": deduplication_method,
            }
        )
        return FakeGraph()


class FakeUIEPipeline:
    def __call__(self, text):
        start = text.index("This project")
        object_start = text.index("framework")
        return [
            {
                "Project": [
                    {
                        "text": "This project",
                        "start": start,
                        "end": start + len("This project"),
                        "probability": 0.98,
                        "relations": {
                            "develops": [
                                {
                                    "text": "framework",
                                    "start": object_start,
                                    "end": object_start + len("framework"),
                                    "probability": 0.91,
                                }
                            ]
                        },
                    }
                ]
            }
        ]


class RecordingUIEPipeline:
    def __init__(self):
        self.inputs = []

    def __call__(self, text):
        self.inputs = text if isinstance(text, list) else [text]
        return [{} for _ in self.inputs]


class FakeBackend:
    name = "fake:model"

    def extract(self, text, *, context=""):
        return [
            AbstractRelation(
                "This project",
                "develops",
                "a framework",
                confidence=0.9,
                evidence="This project develops a framework.",
            ),
            AbstractRelation(
                "This project",
                "uses method",
                "low confidence method",
                confidence=0.2,
            ),
        ]


class BackendTests(unittest.TestCase):
    def test_kggen_normalizes_graph_relations(self):
        client = FakeKGGenClient()
        backend = KGGenBackend(client=client, model="test/model")
        relations = backend.extract("Source text", context="Award title")
        self.assertEqual(len(relations), 2)
        self.assertEqual(relations[0].evidence, "Source text")
        self.assertEqual(client.calls[0]["context"], "Award title")

    def test_kggen_preprocesses_abstract_before_generation(self):
        client = FakeKGGenClient()
        backend = KGGenBackend(client=client, model="test/model")
        source = (
            "This project develops a safety-aware framework.\n\n"
            "This award reflects NSF's statutory mission and has been deemed worthy "
            "of support through evaluation using the Foundation's intellectual merit "
            "and broader impacts review criteria.\n\n"
            "It targets healthcare."
        )
        backend.extract(source)

        generated_text = client.calls[0]["input_data"]
        self.assertIn("safety-aware framework", generated_text)
        self.assertIn("It targets healthcare.", generated_text)
        self.assertNotIn("statutory mission", generated_text)

    def test_kggen_preprocessing_preserves_case_and_punctuation(self):
        text = "AI-enabled robots improve STEM learning.\n\nThey use sensors."
        self.assertEqual(
            preprocess_abstract_for_kggen(text),
            "AI-enabled robots improve STEM learning. They use sensors.",
        )

    def test_kggen_disables_modern_deduplication_when_cluster_is_false(self):
        client = FakeModernKGGenClient()
        backend = KGGenBackend(client=client, model="test/model", cluster=False)
        backend.extract("Source text", context="Award title")
        self.assertIsNone(client.calls[0]["deduplication_method"])

    def test_uie_flattens_nested_predictions_and_preserves_confidence(self):
        backend = UIEBackend(pipeline=FakeUIEPipeline(), model="uie-test")
        text = "This project develops a framework. It is evaluated carefully."
        relations = backend.extract(text)
        self.assertEqual(relations[0].subject, "This project")
        self.assertEqual(relations[0].predicate, "develops")
        self.assertEqual(relations[0].object, "framework")
        self.assertEqual(relations[0].confidence, 0.91)
        self.assertEqual(relations[0].evidence, "This project develops a framework.")

    def test_uie_chunks_long_abstracts_before_running_nested_schema(self):
        pipeline = RecordingUIEPipeline()
        backend = UIEBackend(pipeline=pipeline, model="uie-test", chunk_size=80)
        backend.extract("This project develops a framework. " * 12)

        self.assertGreater(len(pipeline.inputs), 1)
        self.assertTrue(all(len(chunk) <= 80 for chunk in pipeline.inputs))


class AbstractTripleTests(unittest.TestCase):
    def test_model_relations_are_grounded_to_the_award(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "awards.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["AwardNumber", "Title", "Abstract"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "AwardNumber": '="123"',
                        "Title": "Test Award",
                        "Abstract": "This project develops a framework.",
                    }
                )

            triples, stats = extract_abstract_triples(
                path, FakeBackend(), min_confidence=0.5
            )

        relation = next(t for t in triples if t.predicate.endswith("/develops"))
        self.assertEqual(relation.subject, "https://example.org/nsf/award/123")
        self.assertEqual(relation.confidence, "0.900000")
        self.assertEqual(relation.extractor, "fake:model")
        self.assertEqual(stats.relations, 1)
        self.assertEqual(stats.triples, 4)

    def test_shared_concepts_are_linked_to_each_award(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "awards.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["AwardNumber", "Title", "Abstract"]
                )
                writer.writeheader()
                for number in ("1", "2"):
                    writer.writerow(
                        {
                            "AwardNumber": number,
                            "Title": f"Award {number}",
                            "Abstract": "This project develops a framework.",
                        }
                    )

            triples, _ = extract_abstract_triples(
                path, FakeBackend(), min_confidence=0.5
            )

        mentions = [t for t in triples if t.predicate.endswith("/mentions")]
        self.assertEqual(len(mentions), 2)
        self.assertEqual({t.award_number for t in mentions}, {"1", "2"})


if __name__ == "__main__":
    unittest.main()
