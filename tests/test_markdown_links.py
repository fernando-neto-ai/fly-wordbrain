"""The link checker has to see files that are not committed yet.

It used to list only `git ls-files`, so a newly written page was skipped in silence -- and
a new page is exactly where a broken cross-reference is most likely. That regression is
invisible from the outside: the tool prints "All relative links resolve" either way. So the
behaviour is pinned here, against a scratch repository rather than this one.
"""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts/check_markdown_links.py"


def run_copy(script):
    """Run the *copied* checker.

    It resolves the repository from its own __file__, so invoking the original with a
    different working directory would silently scan this repository instead of the
    scratch one -- which is how the first version of this test passed against 222 real
    links while asserting nothing.
    """
    return subprocess.run([sys.executable, str(script)], capture_output=True, text=True)


class MarkdownLinkTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        (self.root / "scripts").mkdir(parents=True)
        (self.root / "scripts/check_markdown_links.py").write_text(CHECKER.read_text())
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)
        (self.root / "target.md").write_text("# target\n")

    def tearDown(self):
        self.temporary.cleanup()

    def check(self):
        return run_copy(self.root / "scripts/check_markdown_links.py")

    def test_an_untracked_page_with_a_broken_link_is_caught(self):
        (self.root / "new.md").write_text("[gone](missing.md)\n")
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("missing.md", result.stdout)

    def test_an_untracked_page_with_a_good_link_passes(self):
        (self.root / "new.md").write_text("[here](target.md)\n")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("All relative links resolve", result.stdout)

    def test_a_tracked_page_is_still_checked(self):
        (self.root / "tracked.md").write_text("[gone](missing.md)\n")
        subprocess.run(["git", "add", "tracked.md"], cwd=self.root, check=True)
        result = self.check()
        self.assertEqual(result.returncode, 1, result.stdout)

    def test_ignored_files_are_not_checked(self):
        (self.root / ".gitignore").write_text("build/\n")
        (self.root / "build").mkdir()
        (self.root / "build/generated.md").write_text("[gone](missing.md)\n")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_external_links_and_fragments_are_out_of_scope(self):
        (self.root / "new.md").write_text(
            "[a](https://example.com/x) [b](mailto:a@b.c) [c](#section) [d](target.md#part)\n")
        result = self.check()
        self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
