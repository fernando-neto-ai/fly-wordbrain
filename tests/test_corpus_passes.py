"""Passes over the corpus are derived from the measured epoch, never restated.

The repository claimed 16,800 updates was "18.3 passes" of the 1,000-story corpus and
"1.83" of the 10,000-story one, in the top-level README, two HuggingFace model cards, the
Stage 6 and 7 write-ups, a plotting script and two machine-readable records. Both numbers
came from an assumed 918-update epoch. Every arm's manifest records the real figures --
1,196 updates for 1,000 stories and 12,042 for 10,000 -- so the true counts are 14.05 and
1.40, and the claim overstated each arm's data exposure by 31%.

No measured score depended on it: all arms ran the same 16,800 updates and every published
comparison is matched. What it distorted was the convergence caveat, which is *stronger*
than it was written, not weaker.
"""
import json
import pathlib
import re
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

EPOCH_UPDATES = {1000: 1196, 10000: 12042}
BUDGET = 16800


class MeasuredEpochTests(unittest.TestCase):
    def test_the_ten_thousand_story_epoch_is_not_ten_times_the_thousand_story_one(self):
        # The old constant assumed exact 10x scaling from a 918-update epoch. Neither holds.
        self.assertNotEqual(EPOCH_UPDATES[10000], EPOCH_UPDATES[1000] * 10)
        self.assertAlmostEqual(EPOCH_UPDATES[10000] / EPOCH_UPDATES[1000], 10.07, places=2)

    def test_the_published_pass_counts_follow_from_the_measured_epochs(self):
        self.assertAlmostEqual(BUDGET / EPOCH_UPDATES[1000], 14.05, places=2)
        self.assertAlmostEqual(BUDGET / EPOCH_UPDATES[10000], 1.40, places=2)

    def test_the_superseded_figures_are_what_a_918_update_epoch_would_give(self):
        # Documents where 18.3 and 1.83 came from, so the error is legible rather than odd.
        self.assertAlmostEqual(BUDGET / 918, 18.3, places=1)
        self.assertAlmostEqual(BUDGET / 9180, 1.83, places=2)


class NoStaleFiguresTests(unittest.TestCase):
    """Nothing user-facing may still quote the superseded counts."""

    def files(self):
        """Every file git tracks or would track -- not build output.

        Scanning the working tree instead pulls in `space/dist/`, which is gitignored
        generated output. That copy IS what HuggingFace serves, so it needs rebuilding and
        republishing, but it is not a source file and cannot be fixed by editing it.
        This mirrors how `check_markdown_links.py` scopes itself.
        """
        listing = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\n")
        for name in listing:
            if not name or name.endswith(pathlib.Path(__file__).name):
                continue
            p = ROOT / name
            if p.suffix.lower() in (".md", ".py", ".json", ".html", ".svg", ".txt") and p.is_file():
                yield p

    def test_no_file_still_claims_the_old_pass_counts(self):
        stale = []
        for p in self.files():
            try:
                text = p.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for line in text.splitlines():
                if re.search(r"\b18\.3 passes|\b1\.83 passes|only \*\*1\.83\*\*", line):
                    stale.append(f"{p.relative_to(ROOT)}: {line.strip()[:90]}")
        self.assertEqual(stale, [], "superseded pass counts still present:\n" + "\n".join(stale))

    def test_the_packager_derives_passes_rather_than_hardcoding_them(self):
        source = (ROOT / "scripts/package_control_release.py").read_text()
        self.assertIn("EPOCH_UPDATES_10K = 12042", source)
        self.assertNotIn('"passes_over_corpus": 1.83', source)
        self.assertIn('status["updates"] / EPOCH_UPDATES_10K', source)


if __name__ == "__main__":
    unittest.main()
