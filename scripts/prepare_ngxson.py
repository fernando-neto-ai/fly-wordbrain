#!/usr/bin/env python3
"""Download the reviewed ngxson checkpoint, verifying pinned upstream hashes."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "ngxson/fly-llm-hf"
REVISION = "65c677b3d566a2e9793d5f72999cdb441c6c0a9f"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "data" / "ngxson-fly-llm-hf" / REVISION
FILES = {
    "README.md": (8205, "df1ee681a0804bbdf59214b8f1dfdc866e3782d4", "git"),
    "config.json": (723, "bad00de7c077eb17762c8eb47d0184b44e5b0733", "git"),
    "configuration_fly.py": (1569, "ea0fd7381f6cd6a9103e8d23f7bb65be05723479", "git"),
    "generation_config.json": (216, "92633eeb0790412fffaf8e8c70fd75e7b3a40d43", "git"),
    "model.safetensors": (284134372, "355f06c44d14e38af50e9c801f51839c37a0a56ac4ca016da2be9b3ae6f215ad", "sha256"),
    "modeling_fly.py": (10322, "c11110d0aaa6db2cc5c500b2a87e64a83c4d3a5d", "git"),
    "tokenizer.json": (54528, "bbd4e01e8bca8ace788e0f3498186a2953da7f20", "git"),
    "tokenizer_config.json": (245, "2be9bbf6111a5252897b47d578ab9c71075f6a90", "git"),
}


def verify(path, spec):
    size, expected, kind = spec
    if path.stat().st_size != size:
        raise ValueError(f"Wrong size: {path}")
    sha = hashlib.sha256()
    git = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
            git.update(chunk)
    actual = git.hexdigest() if kind == "git" else sha.hexdigest()
    if actual != expected:
        raise ValueError(f"Upstream checksum mismatch: {path}")
    return {"bytes": size, "sha256": sha.hexdigest(), "upstream_hash": expected, "hash_kind": kind}


def fetch(directory, name):
    path = directory / name
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{name}"
    if not path.exists():
        partial = path.with_name(path.name + ".partial")
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        verify(partial, FILES[name])
        partial.replace(path)
    receipt = verify(path, FILES[name])
    print(f"verified {name}", flush=True)
    return name, {"url": url, **receipt}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        receipts = dict(pool.map(lambda name: fetch(args.output, name), FILES))
    manifest = {"repo": REPO, "revision": REVISION, "files": receipts}
    (args.output / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
