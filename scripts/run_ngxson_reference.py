#!/usr/bin/env python3
"""Reproduce pinned ngxson inference; optionally compare CPU and MPS numerics."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_ngxson import DEFAULT_MODEL, FILES, REPO, REVISION, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--compare-mps", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prompt", action="append", help="Repeat to supply multiple prompts.")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.threads < 1:
        parser.error("token count and thread count must be positive")
    if args.output.exists():
        parser.error("Choose a new output path to preserve earlier results")
    if args.compare_mps and args.device != "cpu":
        parser.error("--compare-mps starts with the CPU reference; use --device cpu")
    # All downloaded code is verified before trust_remote_code executes it.
    files = {name: verify(args.model / name, spec) for name, spec in FILES.items()}
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
    ).eval()
    counts = {name: p.numel() for name, p in model.named_parameters()}
    assert sum(counts.values()) == 52756661
    assert counts["lm_head.weight"] == 50578432

    def frozen_hashes():
        return {name: hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                for name, value in model.named_buffers()}

    before = frozen_hashes()
    if args.device == "mps":
        from fly_wordbrain.ngxson_mps import enable_mps
        model = enable_mps(model)

    def sync(device):
        if device == "mps":
            torch.mps.synchronize()

    prompts = args.prompt or [
        "Once upon a time, there was a",
        "One day, a little girl named Lily found a needle",
        "Tom and his dog",
    ]
    samples = []
    with torch.no_grad():
        for prompt in prompts:
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            ids = torch.tensor([[tokenizer.bos_token_id, *encoded]], device=args.device)
            sync(args.device)
            started = time.perf_counter()
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            sync(args.device)
            elapsed = time.perf_counter() - started
            tokens = out[0].cpu().tolist()
            item = {"prompt": prompt, "prompt_ids": ids[0].cpu().tolist(), "all_ids": tokens,
                    "text": tokenizer.decode(tokens, skip_special_tokens=True), "seconds": elapsed,
                    "new_tokens": len(tokens) - ids.shape[1]}
            item["tokens_per_second_including_prefill"] = item["new_tokens"] / elapsed
            samples.append(item)
            print(item["text"], flush=True)

    report = {
        "task": "released_checkpoint_inference_reproduction", "training_run": False,
        "utc": datetime.now(timezone.utc).isoformat(), "repo": REPO, "revision": REVISION,
        "files": files, "device": args.device, "python": sys.version,
        "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__, "threads": args.threads,
        "trainable_parameters": sum(counts.values()), "parameter_breakdown": counts,
        "neurons": model.config.n_neurons, "stored_edges": model.config.n_edges,
        "injected_neurons": model.brain.in_index.numel(), "delay_tokens": model.config.delay_k,
        "samples": samples,
    }
    if args.compare_mps:
        # Same fixed token sequence on both backends, so sampling drift cannot
        # conceal a wrong sparse operation. Compare every position separately.
        from fly_wordbrain.ngxson_mps import enable_mps
        tokens = samples[0]["all_ids"]
        cpu_logits, cpu_states = [], []
        cache = None
        with torch.no_grad():
            for token in tokens:
                out = model(torch.tensor([[token]]), cache_params=cache, use_cache=True)
                cache = out.cache_params
                cpu_logits.append(out.logits[0, -1].clone())
                cpu_states.append(cache.state.clone())
        cpu_logits = torch.stack(cpu_logits)
        model = enable_mps(model)
        cache = None
        rows = []
        with torch.no_grad():
            for i, token in enumerate(tokens):
                out = model(torch.tensor([[token]], device="mps"), cache_params=cache, use_cache=True)
                cache = out.cache_params
                logits = out.logits[0, -1].cpu()
                state = cache.state.cpu()
                assert torch.isfinite(logits).all() and torch.isfinite(state).all()
                ref_probs = cpu_logits[i].softmax(-1)
                probs = logits.softmax(-1)
                rows.append({
                    "position": i, "logit_max_abs": (logits - cpu_logits[i]).abs().max().item(),
                    "state_max_abs": (state - cpu_states[i]).abs().max().item(),
                    "probability_tv": (probs - ref_probs).abs().sum().item() / 2,
                    "cpu_top1": int(cpu_logits[i].argmax()), "mps_top1": int(logits.argmax()),
                })
        ids = torch.tensor([samples[0]["prompt_ids"]], device="mps")
        with torch.no_grad():
            sync("mps")
            started = time.perf_counter()
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tokenizer.pad_token_id)
            sync("mps")
            elapsed = time.perf_counter() - started
        mps_ids = out[0].cpu().tolist()
        report["cpu_mps_comparison"] = {
            "fixed_sequence_positions": len(rows), "per_position": rows,
            "max_logit_abs": max(x["logit_max_abs"] for x in rows),
            "max_state_abs": max(x["state_max_abs"] for x in rows),
            "max_probability_tv": max(x["probability_tv"] for x in rows),
            "top1_disagreements": sum(x["cpu_top1"] != x["mps_top1"] for x in rows),
            "greedy_ids_equal": mps_ids == samples[0]["all_ids"],
            "mps_greedy_text": tokenizer.decode(mps_ids, skip_special_tokens=True),
            "mps_greedy_ids": mps_ids, "mps_seconds": elapsed,
            "mps_tokens_per_second_including_prefill": (len(mps_ids) - ids.shape[1]) / elapsed,
        }
    after = frozen_hashes()
    report["frozen_buffers_before"] = before
    report["frozen_buffers_after"] = after
    report["frozen_buffers_unchanged"] = before == after
    assert before == after, "Canonical graph/input/output buffers changed"
    if args.device == "mps" or args.compare_mps:
        from fly_wordbrain.ngxson_mps import mps_metadata
        report["mps_backend"] = mps_metadata(model, verify=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
