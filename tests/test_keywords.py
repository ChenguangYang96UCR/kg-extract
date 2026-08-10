import csv
import tempfile
import unittest
from pathlib import Path

from kg_extract.keywords import (
    KeywordCandidate,
    SimpleKeywordBackend,
    canonical_keyword,
    clean_abstract_for_keywords,
    extract_keyword_triples,
)


class FakeKeywordBackend:
    name = "fake:keywords"

    def extract(self, text, *, top_k, ngram_range):
        del text, top_k, ngram_range
        return [
            KeywordCandidate("Digital Tutors", 0.91, "Digital tutors support students."),
            KeywordCandidate("digital tutor", 0.87, "A digital tutor adapts feedback."),
            KeywordCandidate("machine learning", 0.83, "Machine learning is used."),
        ]


class FakeClusterer:
    name = "fake:cluster"

    def cluster(self, labels, *, scores):
        del scores
        return {
            label: "large language model"
            if label in {"language model", "large language model"}
            else label
            for label in labels
        }


class FakeClusterBackend:
    name = "fake:cluster-keywords"

    def extract(self, text, *, top_k, ngram_range):
        del text, top_k, ngram_range
        return [
            KeywordCandidate("language models", 0.91, "Language models are evaluated."),
            KeywordCandidate("large language model", 0.88, "A large language model is used."),
            KeywordCandidate("AI", 0.77, "AI supports analysis."),
        ]


class KeywordNormalizationTests(unittest.TestCase):
    def test_nsf_boilerplate_is_removed(self):
        text = (
            "This project studies digital tutors. This award reflects NSF's statutory "
            "mission and has been deemed worthy of support through evaluation using "
            "the Foundation's intellectual merit and broader impacts review criteria."
        )
        cleaned = clean_abstract_for_keywords(text)
        self.assertIn("digital tutors", cleaned)
        self.assertNotIn("statutory mission", cleaned)

    def test_keyword_labels_are_canonicalized(self):
        self.assertEqual(canonical_keyword("Digital Tutors"), "digital tutor")
        self.assertEqual(canonical_keyword("LLMs"), "large language model")

    def test_simple_backend_scores_are_normalized(self):
        backend = SimpleKeywordBackend()
        keywords = backend.extract(
            "Quantum sensing improves quantum measurements. The university team studies quantum sensing.",
            top_k=3,
            ngram_range=(1, 2),
        )
        self.assertTrue(keywords)
        self.assertTrue(all(0.0 <= keyword.score <= 1.0 for keyword in keywords))
        self.assertNotIn("university", {keyword.label for keyword in keywords})


class KeywordTripleTests(unittest.TestCase):
    def test_keywords_are_linked_to_awards_and_deduplicated(self):
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
                        "Abstract": "This project studies digital tutors.",
                    }
                )

            triples, assignments, stats = extract_keyword_triples(
                path, FakeKeywordBackend(), top_k=3
            )

        has_keyword = [triple for triple in triples if triple.predicate.endswith("/hasKeyword")]
        self.assertEqual(len(has_keyword), 2)
        self.assertEqual(has_keyword[0].subject, "https://example.org/nsf/award/123")
        self.assertEqual(
            {assignment.canonical_keyword for assignment in assignments},
            {"digital tutor", "machine learning"},
        )
        self.assertEqual(stats.keywords, 2)
        self.assertTrue(
            any(
                triple.object == "https://example.org/nsf/vocab/Keyword"
                for triple in triples
            )
        )

    def test_embedding_clusterer_can_merge_similar_keyword_nodes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "awards.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["AwardNumber", "Title", "Abstract"]
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "AwardNumber": "1",
                        "Title": "Award 1",
                        "Abstract": "This project evaluates language models.",
                    }
                )

            triples, assignments, stats = extract_keyword_triples(
                path,
                FakeClusterBackend(),
                top_k=3,
                clusterer=FakeClusterer(),
            )

        self.assertEqual(stats.keywords, 2)
        self.assertIn(
            "large language model",
            {assignment.canonical_keyword for assignment in assignments},
        )
        has_keyword_objects = {
            triple.object for triple in triples if triple.predicate.endswith("/hasKeyword")
        }
        self.assertIn(
            "https://example.org/nsf/keyword/large-language-model",
            has_keyword_objects,
        )
        self.assertTrue(all("+fake:cluster" in assignment.extractor for assignment in assignments))


if __name__ == "__main__":
    unittest.main()
