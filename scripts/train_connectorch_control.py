#!/usr/bin/env python3
"""Train an arm on a randomised graph, to test whether the fly's wiring matters.

Every result in this repository so far runs on the same measured connectome, so
none of them can separate *"this wiring is a good prior for language"* from
*"this recurrent shape suits short stories"*. This is the control that can.

The graph is rewired by a **global permutation of the presynaptic index array**,
which is a strong degree-preserving shuffle:

    preserved exactly   every neuron's in-degree (row offsets untouched)
                        every neuron's out-degree (each index value still occurs
                          the same number of times, just in other positions)
                        every edge weight, and each neuron's incoming weight multiset
                        the input-injection and readout interfaces
    destroyed           which particular neuron connects to which

So a model trained on the shuffled graph has the same connectivity *statistics*
and the same dynamic range, and differs only in the specific wiring the fly
actually has. If it scores the same, the anatomy is contributing nothing beyond
its degree structure.

This never edits the pinned trainer. It patches the two seams where the reference
graph enters — model loading, and the grouping's graph-identity check — records a
receipt of exactly what changed, and then delegates to the real trainer so the
recipe, the optimizer and every audit stay byte-identical to the arm it controls.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_connectorch as trainer
from train_connectorch import file_hash

CONTROL = {}


def digest(array):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(tuple(array.shape)).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def conflicts(source, destination, neurons):
    """Edge slots that are self-loops or repeat a (destination, source) pair."""
    key = destination * neurons + source.astype(np.int64)
    order = np.argsort(key, kind="stable")
    seen, bad = set(), []
    for index in order:
        k = int(key[index])
        if k in seen or source[index] == destination[index]:
            bad.append(int(index))
        else:
            seen.add(k)
    return bad, seen


def repair(source, destination, neurons, generator, attempts=64, rounds=8):
    """Remove self-loops and duplicate edges by swapping sources between edge slots.

    Swapping sources between two slots keeps every neuron's in-degree (slots stay in
    their rows) and the out-degree multiset (the same source values are still used
    the same number of times), so the degree guarantees survive the repair.

    A single pass can leave a handful of slots whose random partners all collided, so
    this repeats until the graph is clean. The sparse backend rejects duplicates rather
    than coalescing them, which makes a leftover a hard failure rather than a silent
    change of topology.
    """
    fixed = 0
    for _ in range(rounds):
        bad, present = conflicts(source, destination, neurons)
        if not bad:
            return source, fixed
        for index in bad:
            for _ in range(attempts):
                other = int(generator.integers(0, source.size))
                if other == index:
                    continue
                a, b = int(destination[index]), int(source[other])
                c, d = int(destination[other]), int(source[index])
                if a == b or c == d:
                    continue
                new_a, new_c = a * neurons + b, c * neurons + d
                if new_a in present or new_c in present or new_a == new_c:
                    continue
                present.discard(c * neurons + int(source[other]))
                source[index], source[other] = source[other], source[index]
                present.add(new_a)
                present.add(new_c)
                fixed += 1
                break
    remaining, _ = conflicts(source, destination, neurons)
    if remaining:
        raise SystemExit(f"Could not produce a simple graph: {len(remaining)} conflicting slots")
    return source, fixed


def control_indices(offsets, source, mode, seed):
    """Derive the control graph's presynaptic index array. Pure, and deterministic in `seed`.

    This is the single definition of what the control graph *is*. The trainer uses it to
    build the graph, and the evaluator uses it to rebuild the same graph from the seed
    alone, so a scoring run cannot silently drift from the run it is scoring.
    """
    neurons = offsets.size - 1
    generator = np.random.default_rng(seed)
    if mode == "zero":
        return source.copy(), 0
    if mode != "shuffle":
        raise SystemExit(f"Unknown control mode: {mode}")
    destination = np.repeat(np.arange(neurons, dtype=np.int64), np.diff(offsets))
    shuffled = source.copy()
    generator.shuffle(shuffled)
    # A global shuffle is a strong randomisation but produces a few self-loops and
    # duplicate edges, and the sparse backend refuses to coalesce topology. Repair
    # them with targeted source swaps, which keep the out-degree multiset exact.
    return repair(shuffled, destination, neurons, generator)


def rewire(brain, mode, seed):
    """Replace the graph in place and return a receipt of what moved."""
    # .cpu() is a no-op for a CPU tensor and .numpy() then shares its storage, so the
    # in-place copy_ below would rewrite these arrays and the receipt would compare the
    # new graph against itself. Take real copies.
    offsets = brain.w_offsets.cpu().numpy().copy()
    source = brain.w_indices.cpu().numpy().copy()
    values = brain.w_values.cpu().numpy().copy()
    neurons = offsets.size - 1

    before = {"in_degree": np.diff(offsets), "out_degree": np.bincount(source, minlength=neurons),
              "values_sha256": digest(values), "source_sha256": digest(source)}

    shuffled, repairs = control_indices(offsets, source, mode, seed)
    if mode == "zero":
        values = np.zeros_like(values)

    with torch.no_grad():
        brain.w_indices.copy_(torch.from_numpy(shuffled.astype(source.dtype)))
        brain.w_values.copy_(torch.from_numpy(values.astype(np.float32)))

    after = {"in_degree": np.diff(offsets),
             "out_degree": np.bincount(shuffled, minlength=neurons)}
    destination = np.repeat(np.arange(neurons), np.diff(offsets))
    unchanged = int((shuffled == source).sum())
    pairs = destination.astype(np.int64) * neurons + shuffled.astype(np.int64)

    if mode == "shuffle" and unchanged == source.size:
        raise SystemExit("Receipt says nothing moved; the source array was aliased to the tensor")

    receipt = {
        "mode": mode, "seed": seed, "neurons": int(neurons), "edges": int(source.size),
        "preserved": {
            "in_degree_identical": bool(np.array_equal(before["in_degree"], after["in_degree"])),
            "out_degree_identical": bool(np.array_equal(before["out_degree"], after["out_degree"])),
            "weight_multiset_identical": mode != "zero",
            "row_offsets_untouched": True,
        },
        "changed": {
            "repaired_slots": repairs,
            "edges_landing_on_their_original_source": unchanged,
            "fraction_rewired": 1 - unchanged / source.size,
            "self_loops_created": int((shuffled == destination).sum()),
            "duplicate_edges_created": int(pairs.size - np.unique(pairs).size),
        },
        "original_graph_sha256": {"w_indices": before["source_sha256"],
                                  "w_values": before["values_sha256"]},
        "control_graph_sha256": {"w_indices": digest(shuffled.astype(source.dtype)),
                                 "w_values": digest(values.astype(np.float32))},
    }
    if mode == "zero":
        receipt["preserved"]["out_degree_identical"] = True
        receipt["note"] = ("Topology untouched; every synaptic weight set to zero, so the "
                           "recurrence is removed and the neurons become independent leaky units.")
    return receipt


def install(mode, seed, output):
    """Patch the two seams where the reference graph enters the trainer."""
    import transformers
    original_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained
    original_load_groups = trainer.load_groups

    def patched_from_pretrained(*args, **kwargs):
        model = original_from_pretrained(*args, **kwargs)
        # Rewire before the trainer hashes the reference, so its own
        # "adaptation changed a reference buffer" check still guards the build.
        CONTROL["receipt"] = rewire(model.brain, mode, seed)
        CONTROL["hashes"] = trainer.frozen_hashes(model)
        print(json.dumps({"control": CONTROL["receipt"]}, indent=2), flush=True)
        output.mkdir(parents=True, exist_ok=True)
        (output / "graph-control.json").write_text(
            json.dumps({"created_at_utc": datetime.now(timezone.utc).isoformat(),
                        # The pinned trainer records its own sources, but it does not know
                        # about this wrapper, so the wrapper pins itself.
                        "control_source_sha256": file_hash(Path(__file__).resolve()),
                        **CONTROL["receipt"], "frozen_buffers_sha256": CONTROL["hashes"]},
                       indent=2) + "\n")
        return model

    def patched_load_groups(path):
        groups, metadata = original_load_groups(path)
        # The grouping pins the *original* graph's digests by design. The cell-type
        # assignment per neuron is still valid; only the edges moved.
        metadata = dict(metadata)
        metadata["frozen_buffers_sha256"] = CONTROL["hashes"]
        metadata["graph_control"] = CONTROL["receipt"]["mode"]
        return groups, metadata

    transformers.AutoModelForCausalLM.from_pretrained = patched_from_pretrained
    trainer.load_groups = patched_load_groups


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph-control", choices=("shuffle", "zero"), required=True)
    parser.add_argument("--control-seed", type=int, default=1729)
    known, rest = parser.parse_known_args()

    trainer_args = trainer.parser().parse_args(rest)
    install(known.graph_control, known.control_seed, trainer_args.output)
    print(f"[control] {known.graph_control} graph, seed {known.control_seed}", flush=True)
    trainer.run(trainer_args)


if __name__ == "__main__":
    main()
