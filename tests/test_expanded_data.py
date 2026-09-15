"""Anchor leakage, provenance and natural-token boundaries for corpus expansion."""
from collections import Counter
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from fly_wordbrain import data
from fly_wordbrain.expanded_data import (
    build_expanded_dataset, expanded_dataset_from_records, fetch_json_with_retries,
    retry_delay,
)


class ExpandedDataTests(unittest.TestCase):
    def setUp(self):
        self.source = {"dataset_id": data.DATASET_ID, "revision": "fixed-test-revision", "receipts": []}
        self.records = [{"id": f"story/{i}", "text": f"{word} sees the sun and goes home"}
                        for i, word in enumerate(("lily", "tim", "sam", "tom", "amy", "bob", "mia", "sue",
                                                  "ali", "ben", "eva", "fox", "gus", "hal", "ivy", "jay"))]
        self.records[1]["text"] = "tim rests"
        self.pilot = data.dataset_from_records(self.records[:6], train_stories=2, val_stories=2,
            test_stories=2, max_words=5, vocab_size=100, seed=9, source_metadata=self.source)

    def build(self, records=None, **kwargs):
        return expanded_dataset_from_records(records or self.records, self.pilot, self.source,
            train_stories=6, val_stories=4, test_stories=4, max_words=5, vocab_size=100, seed=1729, **kwargs)

    def test_pilot_membership_survives_shuffling_and_content_alias_cannot_leak(self):
        evaluation = self.pilot["splits"]["val"][0]
        original = next(row for row in self.records if row["id"] == evaluation["id"])
        alias = {"id": "aaa-alias-of-eval", "text": original["text"].upper() + "!!!"}
        result = self.build([alias] + self.records)
        for split in ("train", "val", "test"):
            old = {row["id"] for row in self.pilot["splits"][split]}
            new = {row["id"] for row in result["splits"][split]}
            self.assertTrue(old <= new)
        self.assertNotIn(alias["id"], {row["id"] for rows in result["splits"].values() for row in rows})
        for train in result["splits"]["train"]:
            self.assertNotEqual(train["normalized_story_sha256"], evaluation["normalized_story_sha256"])

    def test_conflicting_id_and_changed_anchor_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting source"):
            self.build(self.records + [{"id": self.records[0]["id"], "text": "changed content"}])
        changed = copy.deepcopy(self.records)
        changed[0]["text"] += " different ending"
        with self.assertRaisesRegex(ValueError, "anchor source/content mismatch"):
            self.build(changed)

    def test_source_revision_mismatch_and_missing_anchor_rejected(self):
        wrong = {**self.source, "revision": "different"}
        with self.assertRaisesRegex(ValueError, "pinned"):
            expanded_dataset_from_records(self.records, self.pilot, wrong)
        with self.assertRaisesRegex(ValueError, "Missing raw source"):
            self.build(self.records[1:])

    def test_deterministic_under_input_reordering_and_final_splits_shuffled(self):
        first, second = self.build(), self.build(list(reversed(self.records)))
        self.assertEqual(first, second)
        self.assertEqual(first["metadata"]["split_shuffle_seeds"], {"train": 1730, "val": 1731, "test": 1732})
        for split in ("train", "val", "test"):
            self.assertNotEqual([row["id"] for row in first["splits"][split]][:2],
                                [row["id"] for row in self.pilot["splits"][split]])

    def test_final_vocabulary_is_fitted_only_on_expanded_train(self):
        result = self.build()
        counts = Counter(word for row in result["splits"]["train"] for word in row["words"])
        expected = list(data.SPECIAL_TOKENS) + sorted(counts, key=lambda word: (-counts[word], word))
        self.assertEqual(result["vocabulary"], expected)
        lookup = {word: i for i, word in enumerate(expected)}
        for rows in result["splits"].values():
            for row in rows:
                self.assertEqual(row["word_ids"][1:1 + len(row["words"])],
                                 [lookup.get(word, data.UNK_ID) for word in row["words"]])

    def test_complete_and_truncated_story_eos_and_masks(self):
        result = self.build()
        for rows in result["splits"].values():
            for row in rows:
                self.assertEqual(row["word_ids"][0], data.BOS_ID)
                self.assertFalse(row["target_mask"][0])
                self.assertTrue(all(row["target_mask"][1:]))
                self.assertLessEqual(len(row["words"]), 5)
                self.assertEqual(data.EOS_ID in row["word_ids"], row["ended_naturally"])
        short = next(row for rows in result["splits"].values() for row in rows if row["id"] == "story/1")
        self.assertEqual(short["word_ids"][-1], data.EOS_ID)

    def test_immutable_outputs_fail_before_fetch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "dataset.json").write_text("already published")
            with patch("fly_wordbrain.expanded_data.fetch_pinned_stories") as fetch:
                with self.assertRaises(FileExistsError):
                    build_expanded_dataset(root, root / "pilot.json")
                fetch.assert_not_called()
            self.assertEqual((root / "dataset.json").read_text(), "already published")

    def test_existing_source_cache_allowed_and_manifest_hashes_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            (output / "source").mkdir(parents=True)
            (output / "source" / "keep.txt").write_text("verified source placeholder")
            pilot_path = root / "pilot.json"
            pilot_path.write_text(json.dumps(self.pilot))
            with patch("fly_wordbrain.expanded_data.fetch_pinned_stories", return_value=(self.records, self.source)):
                build_expanded_dataset(output, pilot_path, train_stories=6, val_stories=4,
                                       test_stories=4, max_words=5, vocab_size=100)
            self.assertTrue((output / "manifest.json").exists())
            self.assertTrue((output / "source" / "keep.txt").exists())
            self.assertFalse((output / "build.lock").exists())

    def test_network_retries_are_bounded_and_honor_retry_after(self):
        rate_limit = HTTPError("https://example.org", 429, "slow down", {"Retry-After": "17"}, None)
        self.assertEqual(retry_delay(rate_limit, 0), 17)
        capped = HTTPError("https://example.org", 503, "wait", {"Retry-After": "999"}, None)
        self.assertEqual(retry_delay(capped, 0), 60)
        sleeps = []
        calls = [rate_limit, URLError("temporary connection"), ({"ok": True}, {"sha256": "receipt"})]
        def fetch(url, path):
            value = calls.pop(0)
            if isinstance(value, Exception):
                raise value
            return value
        result = fetch_json_with_retries("https://example.org", Path("unused"), fetch=fetch, sleep=sleeps.append)
        self.assertEqual(result[0], {"ok": True})
        self.assertEqual(sleeps, [17, 4])
        forbidden = HTTPError("https://example.org", 403, "forbidden", {}, None)
        with self.assertRaises(HTTPError):
            fetch_json_with_retries("https://example.org", Path("unused"), fetch=lambda *args: (_ for _ in ()).throw(forbidden), sleep=sleeps.append)


if __name__ == "__main__":
    unittest.main()
