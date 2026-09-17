#!/usr/bin/env python3
"""Fail if any relative Markdown link in the repository points at a missing file.

Public documentation is only useful if its cross-references resolve. This walks
every tracked Markdown file, resolves each relative link target against that
file's own directory, and reports the ones that do not exist. External links,
fragments and mail addresses are out of scope.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
EXTERNAL = ("http://", "https://", "mailto:", "#", "data:")
# Vendored upstream notices cite the upstream repository's own layout, which this
# repository deliberately does not copy wholesale. Editing a third-party licence
# notice to satisfy a link checker would misrepresent it, so those files are read
# but not enforced.
SKIP_DIRS = ("vendor/",)


def tracked_markdown():
    """Every Markdown file git would consider part of the repository, committed or not.

    `git ls-files` alone lists only what is already in the index, so a newly written page
    is skipped in silence -- and a new page is exactly where a broken cross-reference is
    most likely. `--others --exclude-standard` adds the untracked files that are not
    ignored, which is what a reader of the repository will actually see.
    """
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "--cached", "--others",
                          "--exclude-standard", "*.md"],
                         capture_output=True, text=True, check=True).stdout.split()
    return sorted({ROOT / name for name in out if not name.startswith(SKIP_DIRS)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="Only print broken links")
    args = parser.parse_args()
    broken, checked = [], 0
    for path in tracked_markdown():
        for target in LINK.findall(path.read_text(encoding="utf-8", errors="replace")):
            if target.startswith(EXTERNAL):
                continue
            checked += 1
            resolved = (path.parent / target.split("#", 1)[0]).resolve()
            if not resolved.exists():
                broken.append((path.relative_to(ROOT), target))
    if not args.quiet:
        print(f"Checked {checked} relative links across {len(tracked_markdown())} Markdown files.")
    for source, target in broken:
        print(f"BROKEN  {source} -> {target}")
    if broken:
        print(f"\n{len(broken)} broken link(s).")
        return 1
    print("All relative links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
