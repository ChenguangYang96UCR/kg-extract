import csv
import tempfile
import unittest
from pathlib import Path

from kg_extract.abstracts import (
    AbstractRelation,
    AbstractNodeCleaningDecision,
    AbstractNodeCleaningRecord,
    KGGenBackend,
    UIEBackend,
    extract_abstract_triples,
    preprocess_abstract_for_kggen,
    write_abstract_node_cleaning_csv,
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


class FakeLongNodeBackend:
    name = "fake:long-node"

    def extract(self, text, *, context=""):
        del text, context
        return [
            AbstractRelation(
                "This project",
                "updates",
                "infrastructure that supports data science, machine learning, and AI",
                confidence=0.8,
            ),
            AbstractRelation(
                "AI-Driven Assessments integration with human evaluations",
                "enhance",
                "evaluation accuracy and reliability",
                confidence=0.7,
            ),
            AbstractRelation(
                "project aim to understand how learners interact with",
                "provide valuable insights into",
                "human-AI dynamics in education",
                confidence=0.6,
            ),
        ]


class FakeSemanticNodeCleaner:
    name = "fake:semantic-node-cleaner"

    def __init__(self):
        self.records = []

    def clean_labels(self, labels, *, award_number, title, abstract):
        del title, abstract
        decisions = {}
        for label in labels:
            if label in {"novelty", "goals"}:
                decision = AbstractNodeCleaningDecision(
                    raw_label=label,
                    action="drop",
                    clean_labels=(),
                    reason="Too generic to characterize the award.",
                )
            elif label == "controlled space for practical training":
                decision = AbstractNodeCleaningDecision(
                    raw_label=label,
                    action="rewrite",
                    clean_labels=("practical training environment",),
                    reason="More concise award-specific concept.",
                )
            else:
                decision = AbstractNodeCleaningDecision(
                    raw_label=label,
                    action="keep",
                    clean_labels=(label,),
                    reason="Meaningful label.",
                )
            decisions[label] = decision
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


class FakeNoisyNodeBackend:
    name = "fake:noisy-node"

    def extract(self, text, *, context=""):
        del text, context
        return [
            AbstractRelation("This project", "has", "novelty"),
            AbstractRelation("This project", "has", "goals"),
            AbstractRelation(
                "This project",
                "creates",
                "controlled space for practical training",
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

    def test_long_abstract_nodes_are_condensed_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "awards.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["AwardNumber", "Title", "Abstract"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "AwardNumber": "42",
                        "Title": "Long Node Award",
                        "Abstract": "This project updates AI infrastructure.",
                    }
                )

            triples, stats = extract_abstract_triples(path, FakeLongNodeBackend())

        relation_objects = {
            triple.object
            for triple in triples
            if triple.source_column == "Abstract" and triple.object_type == "iri"
        }
        self.assertIn(
            "https://example.org/nsf/concept/data-science-infrastructure",
            relation_objects,
        )
        self.assertIn(
            "https://example.org/nsf/concept/machine-learning-infrastructure",
            relation_objects,
        )
        self.assertIn("https://example.org/nsf/concept/ai-infrastructure", relation_objects)
        self.assertIn("https://example.org/nsf/concept/evaluation-accuracy", relation_objects)
        self.assertIn("https://example.org/nsf/concept/evaluation-reliability", relation_objects)
        self.assertGreater(stats.relations, 3)

    def test_project_action_nodes_are_grounded_to_the_award(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "awards.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["AwardNumber", "Title", "Abstract"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "AwardNumber": "42",
                        "Title": "Project Action Award",
                        "Abstract": "This project studies human-AI dynamics.",
                    }
                )

            triples, _ = extract_abstract_triples(path, FakeLongNodeBackend())

        insight_relation = next(
            triple
            for triple in triples
            if triple.predicate.endswith("/provide-valuable-insights-into")
        )
        self.assertEqual(insight_relation.subject, "https://example.org/nsf/award/42")

    def test_semantic_node_cleaner_drops_and_rewrites_noisy_nodes(self):
        cleaner = FakeSemanticNodeCleaner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "awards.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["AwardNumber", "Title", "Abstract"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "AwardNumber": "42",
                        "Title": "Semantic Cleaner Award",
                        "Abstract": "This project creates a practical training environment.",
                    }
                )

            triples, stats = extract_abstract_triples(
                path,
                FakeNoisyNodeBackend(),
                node_cleaner=cleaner,
            )

        concept_names = {
            triple.object
            for triple in triples
            if triple.predicate == "https://schema.org/name"
        }
        self.assertNotIn("novelty", concept_names)
        self.assertNotIn("goals", concept_names)
        self.assertIn("practical training environment", concept_names)
        self.assertEqual(stats.relations, 1)
        self.assertTrue(cleaner.records)

    def test_writes_abstract_node_cleaning_debug_csv(self):
        records = [
            AbstractNodeCleaningRecord(
                award_number="42",
                raw_label="novelty",
                action="drop",
                clean_labels="",
                reason="Too generic.",
                cleaner="fake",
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "abstract_node_cleaning.csv"
            write_abstract_node_cleaning_csv(records, output_path)

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(rows[0]["raw_label"], "novelty")
        self.assertEqual(rows[0]["action"], "drop")


if __name__ == "__main__":
    unittest.main()
