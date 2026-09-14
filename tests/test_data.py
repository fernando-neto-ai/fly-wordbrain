"""Behavioral checks for data leakage, target alignment and memory probes."""

from collections import Counter, defaultdict
import unittest

from fly_wordbrain.data import (
    BOS_ID, EOS_ID, UNK_ID, SPECIAL_TOKENS, build_recall_probes,
    dataset_from_records, words_from_text,
)


class DataTests(unittest.TestCase):
    def records(self):
        return [
            {"id": "one", "text": "Lily sees the red apple."},
            {"id": "two", "text": "Tim has a little ball."},
            {"id": "three", "text": "Tom plays near the house."},
            {"id": "four", "text": "Sam likes the big tree."},
            {"id": "five", "text": "A bird flies above clouds."},
            {"id": "six", "text": "The fish swims very quickly."},
        ]

    def test_words_are_not_punctuation_or_subwords(self):
        self.assertEqual(words_from_text("Lily's cat—can't run! Tim’s well-loved 128 toys."),
                         ["lily's", "cat", "can't", "run", "tim's", "well", "loved", "toys"])

    def test_split_is_disjoint_vocab_is_training_only(self):
        result = dataset_from_records(self.records(), train_stories=2, val_stories=2,
                                      test_stories=2, vocab_size=100)
        train_words = {w for row in result["splits"]["train"] for w in row["words"]}
        self.assertEqual(set(result["vocabulary"]), train_words | set(SPECIAL_TOKENS))
        hashes = [row["normalized_story_sha256"] for rows in result["splits"].values() for row in rows]
        self.assertEqual(len(hashes), len(set(hashes)))
        for split in ("val", "test"):
            for row in result["splits"][split]:
                for word, wid in zip(row["words"], row["word_ids"][1:-1]):
                    self.assertEqual(wid == UNK_ID, word not in train_words)

    def test_dedup_happens_before_split_including_retained_prefix(self):
        records = self.records() + [
            {"id": "dupe", "text": "LILY SEES THE RED APPLE!!!"},
            {"id": "prefix-dupe", "text": "Lily sees the red apple and takes it home."},
        ]
        result = dataset_from_records(records, train_stories=2, val_stories=2,
                                      test_stories=2, max_words=5)
        self.assertEqual(result["metadata"]["duplicate_stories_dropped"], 2)
        self.assertEqual(result["metadata"]["unique_candidates"], 6)

    def test_targets_have_no_bos_pad_or_fake_eos(self):
        records = [{"id": "long", "text": "one two three four five six"},
                   {"id": "short", "text": "seven eight"}]
        result = dataset_from_records(records, train_stories=2, val_stories=0,
                                      test_stories=0, max_words=4, vocab_size=30)
        rows = {row["id"]: row for row in result["splits"]["train"]}
        self.assertNotIn(EOS_ID, rows["long"]["word_ids"])
        self.assertEqual(rows["short"]["word_ids"][-1], EOS_ID)
        for row in rows.values():
            self.assertEqual(row["word_ids"][0], BOS_ID)
            self.assertFalse(row["target_mask"][0])
            self.assertTrue(all(row["target_mask"][1:]))
            self.assertLessEqual(len(row["words"]), 4)

    def test_repeatability_and_seed(self):
        args = dict(train_stories=2, val_stories=2, test_stories=2)
        self.assertEqual(dataset_from_records(self.records(), **args),
                         dataset_from_records(self.records(), **args))
        self.assertNotEqual(dataset_from_records(self.records(), seed=1, **args)["splits"],
                            dataset_from_records(self.records(), seed=2, **args)["splits"])

    def test_recall_pairs_positions_balance_and_no_leak(self):
        vocabulary = list(SPECIAL_TOKENS) + ["lily", "tim", "tom", "sam"]
        result = build_recall_probes(vocabulary)
        self.assertTrue(result["metadata"]["usable_with_text_word_encoding"])
        template_sets = []
        prompts = set()
        for split, items in result["splits"].items():
            template_sets.append({x["template_id"] for x in items})
            groups = defaultdict(list)
            by_context = defaultdict(Counter)
            for item in items:
                groups[item["pair_group"]].append(item)
                by_context[item["context_words"]][item["target_word"]] += 1
                words = item["words"]
                self.assertEqual(len(words), item["context_words"])
                self.assertEqual(words[1], item["target_word"])
                self.assertEqual(sum(w in result["labels"] for w in words), 1)
                self.assertFalse(set(words[-5:]) & set(result["labels"]))
                self.assertNotIn(tuple(words), prompts)
                prompts.add(tuple(words))
            for counts in by_context.values():
                self.assertEqual(len(counts), 4)
                self.assertEqual(len(set(counts.values())), 1)
            for pair in groups.values():
                self.assertEqual(len(pair), 4)
                masked = {tuple([*row["words"][:1], "NAME", *row["words"][2:]]) for row in pair}
                self.assertEqual(len(masked), 1)
        self.assertTrue(template_sets[0].isdisjoint(template_sets[1]))
        self.assertTrue(template_sets[0].isdisjoint(template_sets[2]))
        self.assertTrue(template_sets[1].isdisjoint(template_sets[2]))

    def test_recall_reports_invalid_unknown_label_encoding(self):
        result = build_recall_probes(list(SPECIAL_TOKENS) + ["lily"])
        self.assertFalse(result["metadata"]["usable_with_text_word_encoding"])
        self.assertEqual(result["metadata"]["missing_label_words_from_text_vocab"], ["tim", "tom", "sam"])


if __name__ == "__main__":
    unittest.main()
