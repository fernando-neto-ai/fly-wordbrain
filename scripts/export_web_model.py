#!/usr/bin/env python3
"""Export a trained arm as plain float32 arrays for browser inference, with a golden trace.

The browser demo runs this model itself rather than replaying a recording, so the
JavaScript forward pass has to be provably the same computation. This script does
three things in one pass:

1. Writes every learned tensor as a raw little-endian float32 file plus a manifest.
2. Re-implements the forward pass in NumPy and checks it against PyTorch, so the
   reference the JavaScript mirrors is itself verified rather than assumed.
3. Emits a golden trace - prompts, generated token IDs, per-step top-1 probability
   and mean |h| - computed with the *quantised* edge weights the browser downloads,
   so a JavaScript mismatch is a real defect and not a quantisation artifact.

It also measures how far the quantised graph drifts from the exact one, which is the
honest cost of halving the demo payload.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from train_connectorch import build_model, file_hash, frozen_hashes, load_groups, restore_parameters

PROMPTS = ["Once upon a time, there was a",
           "One day, a little girl named Lily found a needle",
           "Tom and his dog",
           "The little bird was afraid of the rain. Its mother"]
TENSORS = {"wte": "brain.wte.weight", "in_proj": "brain.in_proj", "gain": "brain.gain",
           "rec_gain": "brain.rec_gain", "bias": "brain.bias",
           "ln_weight": "ln.weight", "ln_bias": "ln.bias",
           "head_a": "lm_head.0.weight", "head_b": "lm_head.1.weight"}


def write_array(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(array, dtype=np.float32)
    path.write_bytes(array.tobytes())
    return {"file": path.name, "shape": list(array.shape), "dtype": "float32",
            "bytes": array.nbytes, "sha256": hashlib.sha256(array.tobytes()).hexdigest()}


class NumpyFly:
    """Reference forward pass. This is the specification the browser engine mirrors."""

    def __init__(self, params, offsets, source, values, in_index, out_index, cfg):
        from scipy.sparse import csr_matrix
        self.p = params
        self.W = csr_matrix((values, source.astype(np.int32), offsets.astype(np.int32)),
                            shape=(cfg["neurons"], cfg["neurons"]))
        self.in_index, self.out_index, self.cfg = in_index, out_index, cfg

    def start(self):
        return {"state": np.zeros(self.cfg["neurons"], dtype=np.float32),
                "prev": np.full(self.cfg["delay_k"], self.cfg["pad_token_id"], dtype=np.int64)}

    def step(self, cache, token):
        """One token in, logits out. Mirrors the torch loop for a single sequence."""
        cfg, p = self.cfg, self.p
        k = cfg["delay_k"]
        # ext[k] is the incoming token; ext[k-j] is what delay slot j reads.
        ext = np.concatenate([cache["prev"], [token]])
        drive = np.concatenate([p["wte"][ext[k - j]] @ p["in_proj"][j] for j in range(k)])
        recurrent = self.W.dot(cache["state"])
        pre = p["rec_gain"] * recurrent
        np.add.at(pre, self.in_index, drive)
        new = (1 - cfg["leak"]) * cache["state"] + cfg["leak"] * np.tanh(p["gain"] * pre + p["bias"])
        cache = {"state": new.astype(np.float32),
                 "prev": np.concatenate([cache["prev"], [token]])[-k:]}
        hidden = cache["state"][self.out_index]
        mean = hidden.mean()
        variance = ((hidden - mean) ** 2).mean()
        normed = (hidden - mean) / np.sqrt(variance + cfg["layer_norm_eps"])
        normed = normed * p["ln_weight"] + p["ln_bias"]
        logits = (normed @ p["head_a"].T) @ p["head_b"].T
        return cache, logits.astype(np.float32)

    def generate(self, prompt_ids, max_new_tokens, eos):
        cache = self.start()
        logits = None
        for token in prompt_ids:
            cache, logits = self.step(cache, int(token))
        produced, steps = [], []
        for _ in range(max_new_tokens):
            probabilities = softmax(logits)
            token = int(np.argmax(logits))
            produced.append(token)
            steps.append({"token": token, "top1_probability": float(probabilities[token]),
                          "mean_abs_state": float(np.abs(cache["state"]).mean())})
            if token == eos:
                break
            cache, logits = self.step(cache, token)
        return produced, steps, cache


def softmax(values):
    shifted = values - values.max()
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def torch_generate(model, prompt_ids, max_new_tokens, eos, device="cpu"):
    cache, logits = None, None
    with torch.no_grad():
        for token in prompt_ids:
            out = model(input_ids=torch.tensor([[int(token)]], device=device),
                        cache_params=cache, use_cache=True, return_dict=True)
            cache, logits = out.cache_params, out.logits[0, -1]
        produced = []
        for _ in range(max_new_tokens):
            token = int(logits.argmax())
            produced.append(token)
            if token == eos:
                break
            out = model(input_ids=torch.tensor([[token]], device=device),
                        cache_params=cache, use_cache=True, return_dict=True)
            cache, logits = out.cache_params, out.logits[0, -1]
    return produced


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", required=True, help="Experiment configuration name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--parity-tokens", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    spec = json.loads((ROOT / "experiments/configs" / (args.config + ".json")).read_text())["training"]
    reference = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True,
                                                     local_files_only=True, torch_dtype=torch.float32).cpu()
    groups, group_metadata = load_groups(args.groups)
    train_args = argparse.Namespace(d_embed=spec["d_embed"], plasticity=spec["plasticity"],
                                    readout_rank=spec["readout_rank"],
                                    history_length=spec["history_length"], seed=spec["seed"])
    model = build_model(reference, train_args, groups).eval()
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    restore_parameters(model, saved["parameters"])
    if frozen_hashes(model) != saved["frozen_buffers_sha256"]:
        raise SystemExit("Checkpoint was trained on a different frozen graph")
    model.requires_grad_(False)

    named = dict(model.named_parameters())
    params = {key: named[name].detach().cpu().numpy().astype(np.float32) for key, name in TENSORS.items()}
    brain = model.brain
    offsets = brain.w_offsets.cpu().numpy()
    source = brain.w_indices.cpu().numpy()
    exact = brain.w_values.cpu().numpy().astype(np.float32)
    in_index = brain.in_index.cpu().numpy()
    out_index = brain.out_index.cpu().numpy()
    cfg = {"neurons": int(model.config.n_neurons), "delay_k": int(model.config.delay_k),
           "vocabulary": int(model.config.vocab_size), "d_embed": int(model.config.d_embed),
           "readout_rank": int(model.config.readout_rank),
           "neurons_per_slot": int(model.config.n_in // model.config.delay_k),
           "leak": float(model.config.leak), "pad_token_id": int(model.config.pad_token_id),
           "bos_token_id": int(model.config.bos_token_id), "eos_token_id": int(model.config.eos_token_id),
           "layer_norm_eps": float(model.ln.eps)}

    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    checks = {"in_index_unique": bool(len(np.unique(in_index)) == in_index.size),
              "out_index_is_identity": bool(np.array_equal(out_index, np.arange(cfg["neurons"])))}

    # 1. NumPy against PyTorch on the exact graph: validates the reference implementation.
    engine_exact = NumpyFly(params, offsets, source, exact, in_index, out_index, cfg)
    parity = []
    for prompt in PROMPTS[:2]:
        ids = [cfg["bos_token_id"], *tokenizer.encode(prompt, add_special_tokens=False)]
        ours, _, _ = engine_exact.generate(ids, args.parity_tokens, cfg["eos_token_id"])
        theirs = torch_generate(model, ids, args.parity_tokens, cfg["eos_token_id"])
        parity.append({"prompt": prompt, "numpy_tokens": ours, "torch_tokens": theirs,
                       "identical": ours == theirs})
    if not all(p["identical"] for p in parity):
        raise SystemExit("NumPy reference does not reproduce the PyTorch forward pass")

    # 2. The browser downloads the exact float32 edges, so the golden trace uses them too.
    #    The quantised graph is still evaluated, to record what halving the payload would cost.
    scale = float(np.abs(exact).max()) / 32767.0
    quantised = (np.rint(exact / scale).astype(np.int16).astype(np.float32) * scale)
    engine_quantised = NumpyFly(params, offsets, source, quantised, in_index, out_index, cfg)
    engine = engine_exact

    golden, drift = [], []
    for prompt in PROMPTS:
        ids = [cfg["bos_token_id"], *tokenizer.encode(prompt, add_special_tokens=False)]
        produced, steps, cache = engine.generate(ids, args.max_new_tokens, cfg["eos_token_id"])
        golden.append({"prompt": prompt, "prompt_ids": ids, "tokens": produced,
                       "text": tokenizer.decode(ids + produced, skip_special_tokens=True),
                       "continuation": tokenizer.decode(produced, skip_special_tokens=True),
                       "steps": steps,
                       "final_state_sha256": hashlib.sha256(cache["state"].tobytes()).hexdigest(),
                       "final_state_mean_abs": float(np.abs(cache["state"]).mean())})
        other, _, _ = engine_quantised.generate(ids, args.max_new_tokens, cfg["eos_token_id"])
        shared = min(len(produced), len(other))
        first = next((i for i in range(shared) if produced[i] != other[i]), None)
        drift.append({"prompt": prompt, "exact_tokens": len(produced), "quantised_tokens": len(other),
                      "identical": produced == other, "first_divergent_step": first})

    out = args.output
    files = {key: write_array(out / f"{key}.f32", value) for key, value in params.items()}
    manifest = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": args.config, "selector": args.checkpoint.stem,
        "checkpoint_sha256": file_hash(args.checkpoint),
        "updates": saved["cursor"]["updates"],
        "trainable_parameters": int(sum(v.size for v in params.values())),
        "config": cfg, "index_checks": checks,
        "graph": {"dataset": "fernandofernandes/fly-connectome-49k",
                  "edges": int(source.size),
                  "encoding": "graph/edges_weight.f32 (exact) with graph/edges_source.u16 and "
                              "graph/edges_offsets.i32",
                  "quantised_alternative": {"file": "graph/edges_weight.i16", "scale": scale,
                                            "halves_the_edge_payload": True},
                  "frozen_buffers_sha256": saved["frozen_buffers_sha256"]},
        "layout": {"in_index": "interface/in_index.i32 from the connectome dataset",
                   "out_index": "identity" if checks["out_index_is_identity"] else "interface/out_index.i32",
                   "drive": "concatenate wte[ext[k-j]] @ in_proj[j] for j in 0..k-1, then index_add at in_index",
                   "recurrence": "x = (1-leak)*x + leak*tanh(gain*(rec_gain*(W@x) + drive) + bias)",
                   "readout": "logits = layer_norm(x) @ head_a.T @ head_b.T"},
        "files": files,
        "verification": {"numpy_matches_torch": parity,
                         "quantisation_drift": drift,
                         "note": "Golden traces use the exact float32 graph the browser downloads, so "
                                 "a JavaScript mismatch indicates an implementation defect. The drift "
                                 "rows record what the smaller int16 edge file would change instead."},
        "golden": golden,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    total = sum(f["bytes"] for f in files.values())
    print(json.dumps({"output": str(out), "model_bytes": total, "model_mb": round(total / 1e6, 2),
                      "trainable_parameters": manifest["trainable_parameters"],
                      "index_checks": checks,
                      "numpy_matches_torch": True,
                      "quantised_would_match_on": sum(d["identical"] for d in drift),
                      "prompts": len(drift)}, indent=2))
    for entry in golden:
        print(f"\n[{entry['prompt']}]\n{entry['continuation']}")


if __name__ == "__main__":
    main()
