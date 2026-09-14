"""Cache causal neural observations from independent, bounded story windows."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
from .brain import FrozenBrain

WORKER = None


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def init_worker(config):
    global WORKER
    WORKER = FrozenBrain(**config)


def story_features(story):
    b = WORKER
    b.reset()
    ids = story["word_ids"]
    mask = story.get("target_mask", [True] * len(ids))
    features, targets, positions, currents = [], [], [], []
    kernel_seconds, spikes, output_spikes = 0., 0, 0
    started = time.perf_counter()
    # No recurrent state crosses a story/window boundary. No future input is
    # presented before an observation is paired with its next-word target.
    for p, current in enumerate(ids[:-1]):
        if p > 128:
            raise ValueError("Context exceeds 128 lexical words")
        x = b.step(current)
        d = b.last_diagnostics
        kernel_seconds += d["kernel_seconds"]
        spikes += d["all_spikes"]
        output_spikes += d["readout_spikes"]
        if mask[p + 1]:
            features.append(x)
            targets.append(ids[p + 1])
            positions.append(p)
            currents.append(current)
    return {"X": np.asarray(features, np.float32), "y": np.asarray(targets, np.int64),
        "positions": np.asarray(positions, np.int64), "current_ids": np.asarray(currents, np.int64),
        "story_ids": np.asarray([story["id"]] * len(targets)),
        "diagnostics": {"story_id": story["id"], "examples": len(targets),
            "wall_seconds": time.perf_counter() - started, "kernel_seconds": kernel_seconds,
            "all_spikes": spikes, "readout_spikes": output_spikes,
            "frozen_graph_sha256": b.verify_frozen()}}


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".partial")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def extract(dataset_path, output, workers=4, word_ms=20., warmup_ms=100., high=0.02):
    dataset_path, output = Path(dataset_path), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(dataset_path.read_text())
    config = {"vocab_size": len(dataset["vocabulary"]), "word_ms": word_ms,
              "warmup_ms": warmup_ms, "high": high}
    brain = FrozenBrain(**config)
    manifest = {"schema": 1, "dataset_sha256": sha256_file(dataset_path),
        "brain": brain.metadata(), "workers": workers,
        "code_sha256": {f: sha256_file(Path(__file__).with_name(f)) for f in ["brain.py", "extract.py"]},
        "scope": "full graph frozen, story-reset state, at most 128 lexical context words"}
    identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    lock_path = output / "manifest.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text())
        if previous.get("identity") != identity:
            raise RuntimeError("Cache configuration/source differs; use a new output directory")
    manifest["identity"] = identity
    atomic_json(lock_path, manifest)
    del brain
    started = time.perf_counter()
    progress = {"state": "extracting", "identity": identity, "completed": 0,
                "total": sum(len(s) for s in dataset["splits"].values())}
    atomic_json(output / "progress.json", progress)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             initializer=init_worker, initargs=(config,)) as pool:
        for split in ["train", "val", "test"]:
            stories = dataset["splits"][split]
            chunks = output / "chunks" / split
            chunks.mkdir(parents=True, exist_ok=True)
            pending = {}
            for i, story in enumerate(stories):
                path = chunks / ("%05d.npz" % i)
                if path.exists() and path.with_suffix(".json").exists():
                    with np.load(path, allow_pickle=False) as z:
                        if len(z["story_ids"]) and str(z["story_ids"][0]) != story["id"]:
                            raise RuntimeError("Cached story identity mismatch")
                    progress["completed"] += 1
                else:
                    pending[pool.submit(story_features, story)] = path
            for future in as_completed(pending):
                path = pending[future]
                result = future.result()
                diagnostics = result.pop("diagnostics")
                if diagnostics["frozen_graph_sha256"] != manifest["brain"]["graph_sha256"]:
                    raise RuntimeError("Worker changed the frozen graph")
                save_npz(path, **result)
                atomic_json(path.with_suffix(".json"), diagnostics)
                progress.update(completed=progress["completed"] + 1, split=split,
                                elapsed_seconds=time.perf_counter() - started,
                                latest=diagnostics)
                atomic_json(output / "progress.json", progress)
                if progress["completed"] % 8 == 0:
                    print(json.dumps(progress), flush=True)
            keys = ["X", "y", "positions", "current_ids", "story_ids"]
            arrays = {key: [] for key in keys}
            for i in range(len(stories)):
                with np.load(chunks / ("%05d.npz" % i), allow_pickle=False) as z:
                    for key in keys:
                        arrays[key].append(z[key])
            save_npz(output / (split + ".npz"), **{k: np.concatenate(v) for k, v in arrays.items()})
    # Each worker uses const weights; verify the disk graph and newly loaded
    # provenance here. Probe checks the in-memory hash after repeated steps.
    verify = FrozenBrain(**config)
    manifest["final_graph_sha256"] = verify.verify_frozen()
    if manifest["final_graph_sha256"] != manifest["brain"]["graph_sha256"]:
        raise RuntimeError("Graph changed during extraction")
    manifest["split_hashes"] = {s: sha256_file(output / (s + ".npz")) for s in dataset["splits"]}
    manifest["completed"] = True
    manifest["elapsed_seconds"] = time.perf_counter() - started
    atomic_json(lock_path, manifest)
    progress.update(state="complete", elapsed_seconds=manifest["elapsed_seconds"])
    atomic_json(output / "progress.json", progress)
    print(json.dumps(progress), flush=True)
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--word-ms", type=float, default=20.)
    p.add_argument("--warmup-ms", type=float, default=100.)
    p.add_argument("--high", type=float, default=0.02)
    a = p.parse_args()
    extract(a.dataset, a.output, a.workers, a.word_ms, a.warmup_ms, a.high)


if __name__ == "__main__":
    main()
