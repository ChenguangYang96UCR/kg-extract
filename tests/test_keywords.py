import csv
import tempfile
import unittest
from pathlib import Path

from kg_extract.keywords import (
    KeywordCandidate,
    SimpleKeywordBackend,
    _is_huggingface_model_dir,
    _looks_like_dspark_draft_repo,
    _llm_keyword_prompt,
    _parse_llm_keywords,
    _strip_thinking_blocks,
    canonical_keyword,
    clean_abstract_for_keywords,
    extract_keyword_triples,
    litellm_openai_compatible_model,
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

    def test_noun_filter_removes_or_trims_verb_led_candidates(self):
        self.assertEqual(
            canonical_keyword("developing groundbreaking chips", noun_filter=True),
            "groundbreaking chip",
        )
        self.assertEqual(canonical_keyword("led university michigan", noun_filter=True), "")
        self.assertEqual(
            canonical_keyword("driving substantial societal", noun_filter=True),
            "",
        )

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

    def test_llm_keyword_parser_reads_json_array(self):
        keywords = _parse_llm_keywords(
            '["quantum sensing", "photonic integrated circuit", "quantum sensing"]',
            top_k=5,
            text="This project develops quantum sensing with photonic integrated circuits.",
        )
        self.assertEqual(
            [keyword.label for keyword in keywords],
            ["quantum sensing", "photonic integrated circuit"],
        )
        self.assertEqual(keywords[0].score, 1.0)

    def test_strips_reasoning_blocks_before_parsing_litellm_output(self):
        response = '<think>I should choose terms.</think>\n["quantum sensing"]'
        self.assertEqual(_strip_thinking_blocks(response), '["quantum sensing"]')

    def test_llm_prompt_restricts_short_noun_topics(self):
        prompt = _llm_keyword_prompt(
            "This project develops privacy-preserving digital twins.",
            top_k=3,
            ngram_range=(1, 2),
        )
        self.assertIn("1 to 2 words", prompt)
        self.assertIn("short noun topic", prompt)
        self.assertIn("Do not return verbs", prompt)

    def test_openai_compatible_model_prefix_is_added_once(self):
        model = "tinker://example:train:0/sampler_weights/000080"
        self.assertEqual(
            litellm_openai_compatible_model(model),
            f"openai/{model}",
        )
        self.assertEqual(
            litellm_openai_compatible_model(f"openai/{model}"),
            f"openai/{model}",
        )

    def test_huggingface_model_dir_detection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertFalse(_is_huggingface_model_dir(path))
            (path / "config.json").write_text("{}", encoding="utf-8")
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            self.assertFalse(_is_huggingface_model_dir(path))
            (path / "model.safetensors").write_text("", encoding="utf-8")
            self.assertTrue(_is_huggingface_model_dir(path))

    def test_dspark_draft_repo_detection(self):
        self.assertTrue(_looks_like_dspark_draft_repo("deepseek-ai/dspark_qwen3_8b_block7"))
        self.assertFalse(_looks_like_dspark_draft_repo("Qwen/Qwen3-8B"))


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

    def test_noun_filter_keeps_clean_topic_nodes(self):
        class FakeActionBackend:
            name = "fake:actions"

            def extract(self, text, *, top_k, ngram_range):
                del text, top_k, ngram_range
                return [
                    KeywordCandidate(
                        "developing groundbreaking chips",
                        0.91,
                        "The team is developing groundbreaking chips.",
                    ),
                    KeywordCandidate(
                        "led university michigan",
                        0.82,
                        "The team is led by the University of Michigan.",
                    ),
                    KeywordCandidate(
                        "quantum photonic circuits",
                        0.80,
                        "Quantum photonic circuits are developed.",
                    ),
                ]

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
                        "Abstract": "This project develops quantum photonic circuits.",
                    }
                )

            _, assignments, stats = extract_keyword_triples(
                path,
                FakeActionBackend(),
                top_k=3,
                noun_filter=True,
            )

        self.assertEqual(stats.keywords, 2)
        self.assertEqual(
            {assignment.canonical_keyword for assignment in assignments},
            {"groundbreaking chip", "quantum photonic circuit"},
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
