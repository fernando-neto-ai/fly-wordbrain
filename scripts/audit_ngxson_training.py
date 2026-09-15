#!/usr/bin/env python3
"""Check full-reference CPU/MPS losses and gradients before retraining."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_ngxson import DEFAULT_MODEL, FILES, verify
from train_ngxson import initialize_from_scratch, frozen_hashes, chunk_loss, batch_tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=ROOT / "data/ngxson-tinystories-v1/dataset.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh audit output")
    for name, spec in FILES.items():
        verify(args.model / name, spec)
    import torch
    from transformers import AutoModelForCausalLM
    from fly_wordbrain.ngxson_mps import enable_mps, mps_metadata

    torch.set_num_threads(4)
    rows = json.loads(args.data.read_text())["train"][:2]
    cpu = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    initialize_from_scratch(cpu, 42)
    before = frozen_hashes(cpu)
    inputs, targets, mask = batch_tensors(rows, "cpu")
    inputs, targets, mask = inputs[:, :32], targets[:, :32], mask[:, :32]
    started = time.perf_counter()
    cpu_loss, count, _, _ = chunk_loss(cpu, inputs, targets, mask)
    cpu_loss.backward()
    cpu_seconds = time.perf_counter() - started
    gpu = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    initialize_from_scratch(gpu, 42)
    enable_mps(gpu)
    torch.mps.synchronize()
    started = time.perf_counter()
    gpu_loss, gpu_count, _, _ = chunk_loss(gpu, inputs.to("mps"), targets.to("mps"), mask.to("mps"))
    gpu_loss.backward()
    torch.mps.synchronize()
    gpu_seconds = time.perf_counter() - started
    assert count == gpu_count == 64
    gradients = {}
    for (name, reference), (gpu_name, value) in zip(cpu.named_parameters(), gpu.named_parameters()):
        assert name == gpu_name and reference.grad is not None and value.grad is not None
        a, b = reference.grad, value.grad.cpu()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        difference = a - b
        norm = float(a.norm())
        rel = float(difference.norm()) / max(norm, 1e-12)
        gradients[name] = {"max_abs": float(difference.abs().max()), "relative_l2": rel,
                           "cpu_norm": norm, "mps_norm": float(b.norm()),
                           "cpu_nonzero": int(a.count_nonzero()), "mps_nonzero": int(b.count_nonzero())}
    pad = gpu.brain.wte.weight.grad[gpu.config.pad_token_id]
    unchanged = before == frozen_hashes(cpu) == frozen_hashes(gpu)
    loss_error = abs(cpu_loss.item() - gpu_loss.item())
    max_gradient_relative_l2 = max(row["relative_l2"] for row in gradients.values())
    passed = unchanged and loss_error < 1e-4 and max_gradient_relative_l2 < 1e-3 and int(pad.count_nonzero()) == 0
    result = {"passed": passed, "batch_size": 2, "chunk_size": 32, "targets": count,
              "initialization": "all trainable values reset from seed42", "torch": torch.__version__,
              "cpu_loss": cpu_loss.item(), "mps_loss": gpu_loss.item(), "loss_abs_difference": loss_error,
              "cpu_forward_backward_seconds": cpu_seconds, "mps_forward_backward_seconds": gpu_seconds,
              "gradient_comparison": gradients, "max_gradient_relative_l2": max_gradient_relative_l2,
              "criteria": {"loss_abs_lt": 1e-4, "all_parameter_gradient_relative_l2_lt": 1e-3,
                           "padding_gradient_exactly_zero": True, "canonical_buffers_unchanged": True},
              "frozen_buffers_unchanged": unchanged, "mps_backend": mps_metadata(gpu, verify=True),
              "scope": "One teacher-forced full-model forward/backward, 2 training stories x 32 tokens; not convergence or throughput benchmark"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit("CPU/MPS training parity gate failed")


if __name__ == "__main__":
    main()
