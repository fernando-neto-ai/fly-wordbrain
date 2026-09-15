#!/usr/bin/env python3
"""Export actual CPU-only generation and aligned recurrent-state replay.

No optimizer, backward pass, GPU device selection, or checkpoint mutation occurs.
The complete E graph is evaluated; the display samples neurons with known soma
coordinates. Local checkpoint relocation is allowed only with matching receipts.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import torch
import train_connectorch as trainer
from evaluate_connectorch_quality import require, verify_checkpoint, restore_model

SELECTORS = ("maximum_validation_accuracy", "minimum_validation_ce")
DEFAULT_STORIES = [
    {"id": "story-1", "title": "Once upon a time", "prompt": "Once upon a time, there was a"},
    {"id": "story-2", "title": "The little bird", "prompt": "The little bird was afraid of the rain. Its mother"},
    {"id": "story-3", "title": "Tom and his dog", "prompt": "Tom and his dog"},
]
ACTIVITY = {"encoding": "uint8", "quantity": "absolute_recurrent_state", "scale_min": 0,
            "scale_max": 1, "quantization": "round(255 * abs(h))",
            "note": "Brightness represents the magnitude of recurrent state, not excitation or inhibition.",
            "alignment": "State after consuming input_token_id, before predicting predicted_token_id."}


def read_json(path):
    return json.loads(Path(path).read_text())


def relocated_receipt(receipt):
    """Installation location is metadata; all package content hashes must match."""
    return {key: value for key, value in receipt.items() if key != "package_path"}


def require_cpu(model):
    require(all(value.device.type == "cpu" for value in list(model.parameters()) + list(model.buffers())),
            "Replay requires every parameter and buffer on CPU")


def load_verified_checkpoint(path, expected_sha256):
    # Hash and deserialize the same open file, then re-hash it to catch mutation.
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
        require(digest == expected_sha256, "Checkpoint SHA256 differs from selected native receipt")
        stream.seek(0)
        checkpoint = torch.load(stream, map_location="cpu", weights_only=False)
        stream.seek(0)
        require(hashlib.file_digest(stream, "sha256").hexdigest() == digest, "Checkpoint changed while loading")
    return checkpoint


def select_layout(layout, node_ids, count, seed):
    require(layout.get("coordinate_system", {}).get("type") in ("anatomical", "schematic"),
            "Layout must explicitly declare anatomical or schematic coordinates")
    source = layout.get("neurons", [])
    require(bool(source) and count > 0, "Neuron layout/sample must be nonempty")
    seen, checked = set(), []
    for neuron in source:
        index = neuron["node_index"]
        require(type(index) is int and 0 <= index < len(node_ids) and index not in seen,
                "Invalid or repeated layout node_index")
        require(str(neuron["body_id"]) == str(node_ids[index]), "Layout body ID differs from canonical graph node ID")
        position = neuron["position"]
        require(len(position) == 3 and all(np.isfinite(float(value)) for value in position), "Invalid neuron coordinates")
        seen.add(index)
        checked.append({**neuron, "body_id": str(neuron["body_id"]), "position": [float(value) for value in position]})
    checked.sort(key=lambda neuron: neuron["node_index"])
    if len(checked) > count:
        checked = sorted(random.Random(seed).sample(checked, count), key=lambda neuron: neuron["node_index"])
    return {"format_version": 1, "neurons": checked, "coordinate_system": layout["coordinate_system"],
            "full_neuron_count": len(node_ids), "neurons_with_coordinates": len(source),
            "sampled_neuron_count": len(checked), "missing_positions": len(node_ids) - len(source),
            "sampling": {"method": "uniform_without_replacement_from_known_positions", "seed": seed,
                         "order": "canonical_node_index"}}


def state_snapshot(state, indices):
    require(state.device.type == "cpu" and state.ndim == 2 and state.shape[0] == 1, "Unexpected recurrent-state device/shape")
    require(bool(torch.isfinite(state).all()), "Nonfinite actual neural state")
    absolute = state[0].abs()
    require(float(absolute.max()) <= 1.000001, "Recurrent state exceeds fixed tanh/leak display range")
    sampled = absolute[indices]
    return torch.round(sampled.clamp(0, 1) * 255).to(torch.uint8).tolist(), {
        "mean_absolute": float(absolute.mean()), "root_mean_square": float(state.square().mean().sqrt()),
        "minimum": float(state.min()), "maximum": float(state.max())}


@torch.no_grad()
def trace_story(model, tokenizer, story, neuron_indices, max_new_tokens=120, top_k=5):
    require(max_new_tokens > 0 and top_k > 0, "Generation lengths must be positive")
    require_cpu(model)
    prompt_ids = [tokenizer.bos_token_id, *tokenizer.encode(story["prompt"], add_special_tokens=False)]
    require(prompt_ids[0] is not None and bool(neuron_indices), "BOS and sampled neurons are required")
    indices = torch.tensor(neuron_indices, dtype=torch.long, device="cpu")
    cache, generated, frames = None, [], []
    inputs = iter(prompt_ids)
    token = next(inputs)
    position = 0
    started = time.perf_counter()
    while True:
        input_ids = torch.tensor([[token]], dtype=torch.long, device="cpu")
        output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), cache_params=cache,
                       use_cache=True, return_dict=True)
        cache = output.cache_params
        require(cache is not None and cache.state.shape[1] == model.config.n_neurons,
                "Missing full-graph recurrent cache")
        logits = output.logits[0, -1]
        require(logits.device.type == "cpu" and bool(torch.isfinite(logits).all()), "Invalid actual next-token logits")
        probabilities = torch.softmax(logits.float(), dim=-1)
        predicted = int(probabilities.argmax())
        values, ids = probabilities.topk(min(top_k, probabilities.numel()))
        activity, stats = state_snapshot(cache.state, indices)
        phase = "prompt" if position < len(prompt_ids) - 1 else "generation"
        frame = {"position": position, "phase": phase, "input_token_id": token,
                 "input_text": tokenizer.decode([token], skip_special_tokens=True),
                 "predicted_token_id": predicted, "predicted_text": tokenizer.decode([predicted], skip_special_tokens=True),
                 "probability": float(probabilities[predicted]),
                 "top_k": [{"id": int(index), "text": tokenizer.decode([int(index)], skip_special_tokens=True),
                            "probability": float(probability)} for index, probability in zip(ids, values)],
                 "state_u8": activity, "whole_brain_state": stats}
        if phase == "generation":
            generated.append(predicted)
            frame["generation_index"] = len(generated) - 1
            frame["continuation_so_far"] = tokenizer.decode(generated, skip_special_tokens=True)
        frames.append(frame)
        position += 1
        if phase == "prompt":
            token = next(inputs)
        elif predicted == tokenizer.eos_token_id or len(generated) >= max_new_tokens:
            break
        else:
            token = predicted
    require(bool(generated) and len(frames) == len(prompt_ids) - 1 + len(generated), "Empty or misaligned replay")
    require(all(len(frame["state_u8"]) == len(neuron_indices) for frame in frames), "Activity length differs from layout")
    require_cpu(model)
    return {"format_version": 1, "mode": "recorded_replay", **story, "prompt_ids": prompt_ids,
            "frames": frames, "generated_ids": generated, "continuation": tokenizer.decode(generated, skip_special_tokens=True),
            "full_text": tokenizer.decode(prompt_ids + generated, skip_special_tokens=True),
            "generation": {"method": "greedy", "max_new_tokens": max_new_tokens, "new_tokens": len(generated),
                           "stopped_on_eos": generated[-1] == tokenizer.eos_token_id, "fresh_cache": True,
                           "bos_once": True, "forward_calls": len(frames), "elapsed_seconds": time.perf_counter() - started},
            "activity": ACTIVITY}


def inspect_inputs(args):
    bindings = {}

    def bind(path, expected=None):
        path = Path(path).resolve()
        digest = trainer.file_hash(path)
        require(expected is None or digest == expected, "Input binding mismatch: " + str(path))
        bindings[str(path)] = digest
        return path

    manifest = read_json(bind(args.arm / "manifest.json"))
    accepted = read_json(bind(args.arm / "accepted-early-stop.json"))
    require(accepted.get("arm") == "E32rank128fixed" and accepted.get("status") == "accepted_early_stop",
            "Expected preserved E32rank128fixed accepted checkpoint")
    require(accepted["process_cessation"]["confirmed"] is True, "E checkpoint must be preserved after cessation")
    config = manifest["config"]
    expected = {"d_embed": 32, "readout_rank": 128, "history_length": 8, "plasticity": "fixed", "seed": 42}
    require(all(config.get(key) == value for key, value in expected.items()) and manifest.get("debug") is False,
            "Checkpoint is not the exact E width32/rank128/eight-delay architecture")
    require(manifest["reference_repository"] == "ngxson/fly-llm-hf" and manifest["reference_revision"] == trainer.REVISION,
            "Unexpected reference repository/revision")
    require(manifest["parameter_counts"]["total"] == 7183157, "Unexpected E parameter count")
    require(trainer.source_receipt() == manifest["trainer_sources_sha256"], "Bound trainer/model sources changed")
    for name, digest in manifest["trainer_sources_sha256"].items():
        bind(ROOT / name, digest)
    installed = trainer.connectorch_receipt()
    require(relocated_receipt(installed) == relocated_receipt(manifest["connectorch_source"]), "ConnecTorch package contents differ")
    require(trainer.verify_reference(args.model) == manifest["reference_files"], "Pinned local reference differs")
    for name, receipt in manifest["reference_files"].items():
        bind(args.model / name, receipt["sha256"])
    data = read_json(bind(args.data, manifest["data_sha256"]))
    require(all(len(data[split]) == manifest["split_sizes"][split] for split in ("train", "validation")), "Bound dataset split sizes differ")
    bind(args.groups, manifest["groups_sha256"])
    groups, metadata = trainer.load_groups(args.groups)
    require(metadata == manifest["groups_provenance"] and int(groups.max()) + 1 == manifest["node_types"], "Group provenance differs")
    with np.load(args.groups, allow_pickle=False) as archive:
        node_ids = archive["node_ids"].tolist()
    layout = select_layout(read_json(bind(args.layout)), node_ids, args.neurons, args.sample_seed)
    selectors = accepted["selected_checkpoints"]["selectors"]
    entry = selectors[args.selector]
    require(entry.get("checkpoint_contents_verified") is True and entry.get("frozen_buffers_preserved") is True
            and entry["frozen_buffers_sha256"] == manifest["frozen_buffers_sha256"], "Selected checkpoint graph/content receipt differs")
    saved = load_verified_checkpoint(args.checkpoint, entry["sha256"])
    bindings[str(args.checkpoint.resolve())] = entry["sha256"]
    arm = {"manifest": manifest, "groups": groups, "selectors": selectors}
    verify_checkpoint(saved, arm, args.selector)
    return arm, saved, entry, layout, bindings, installed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True, help="Directory with manifest.json and accepted-early-stop.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "demo/public/data")
    parser.add_argument("--selector", choices=SELECTORS, default=SELECTORS[0])
    parser.add_argument("--neurons", type=int, default=2048)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--story-limit", type=int, default=3)
    parser.add_argument("--prompts-json", type=Path, help="List of {id,title,prompt} records")
    args = parser.parse_args()
    require(args.threads > 0 and args.story_limit > 0 and args.max_new_tokens > 0 and args.neurons > 0, "Counts must be positive")
    require(not (args.output / "manifest.json").exists(), "Output already contains a published replay manifest; choose a fresh directory")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    print(json.dumps({"status": "verifying_inputs", "device": "cpu"}), flush=True)
    arm, saved, entry, layout, bindings, installed = inspect_inputs(args)
    for path in (Path(__file__), ROOT / "scripts/evaluate_connectorch_quality.py", ROOT / "scripts/evaluate_ngxson_quality.py"):
        bindings[str(path.resolve())] = trainer.file_hash(path)
    if args.prompts_json:
        bindings[str(args.prompts_json.resolve())] = trainer.file_hash(args.prompts_json)
    stories = (read_json(args.prompts_json) if args.prompts_json else DEFAULT_STORIES)[:args.story_limit]
    require(bool(stories) and len({story["id"] for story in stories}) == len(stories), "Story IDs must be unique and nonempty")
    for story in stories:
        require(all(isinstance(story.get(key), str) and story[key] for key in ("id", "title", "prompt")), "Invalid story input")
        require(story["id"].replace("-", "").replace("_", "").isalnum(), "Story ID must be a simple filename stem")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    require((tokenizer.pad_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id) == (0, 1, 2), "Tokenizer special IDs differ")
    reference = AutoModelForCausalLM.from_pretrained(str(args.model), trust_remote_code=True,
                                                    local_files_only=True, torch_dtype=torch.float32).cpu()
    model = restore_model(reference, arm, saved).eval().requires_grad_(False)
    del reference, saved
    gc.collect()
    require_cpu(model)
    before = trainer.parameter_audit(trainer.parameters_cpu(model))
    graph_before = trainer.frozen_hashes(model)
    # Warm lazy tensor/version caches outside inference_mode; generation uses no_grad.
    with torch.no_grad():
        model.brain.sparse_runtime()
        model.brain.effective_values()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.atomic_json(args.output / "neuron-layout.json", layout)
    summaries = []
    for story in stories:
        trace = trace_story(model, tokenizer, story, [neuron["node_index"] for neuron in layout["neurons"]],
                            args.max_new_tokens, args.top_k)
        trace["checkpoint_sha256"] = entry["sha256"]
        trace["selector"] = args.selector
        trace["checkpoint_updates"] = entry["cursor"]["updates"]
        filename = story["id"] + ".json"
        trainer.atomic_json(args.output / filename, trace)
        summaries.append({**story, "trace_url": filename, "sha256": trainer.file_hash(args.output / filename),
                          "new_tokens": trace["generation"]["new_tokens"], "frames": len(trace["frames"])})
        print(json.dumps({"status": "story_completed", "id": story["id"], **trace["generation"]}), flush=True)
    model.verify_frozen()
    require_cpu(model)
    after = trainer.parameter_audit(trainer.parameters_cpu(model))
    require(before == after and trainer.frozen_hashes(model) == graph_before, "Inference changed model parameters or graph")
    require(all(not value.requires_grad and value.grad is None for value in model.parameters()), "Inference created gradients")
    require(trainer.connectorch_receipt() == installed, "ConnecTorch installation changed during export")
    for path, digest in bindings.items():
        require(trainer.file_hash(path) == digest, "Export input changed: " + path)
    backend = model.backend_metadata()
    manifest = {"format_version": 1, "mode": "recorded_replay", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "model": {"label": "Fly brain storyteller · rank 128", "arm": "E32rank128fixed", "selector": args.selector,
                          "checkpoint_sha256": entry["sha256"], "checkpoint_updates": entry["cursor"]["updates"],
                          "validation": entry["validation"], "parameter_counts": arm["manifest"]["parameter_counts"],
                          "neurons": model.config.n_neurons, "edges": model.brain.w_values.numel(),
                          "encoder_width": 32, "readout_rank": 128, "history_length": 8},
                "neuron_layout_url": "neuron-layout.json", "neuron_layout_sha256": trainer.file_hash(args.output / "neuron-layout.json"),
                "activity": ACTIVITY, "stories": summaries,
                "provenance": {"device": "cpu", "host": platform.node(), "torch_version": torch.__version__,
                               "cpu_threads": args.threads, "training": False, "backward_passes": 0,
                               "checkpoint_and_graph_unchanged": True, "whole_graph_inference": True,
                               "displayed_neurons": len(layout["neurons"]), "input_sha256": bindings,
                               "parameter_audit": after, "frozen_buffers_sha256": graph_before,
                               "connectorch_saved": arm["manifest"]["connectorch_source"], "connectorch_local": installed,
                               "backend": backend},
                "notes": ["Recorded CPU inference from a saved trained checkpoint; this is not live generation or training.",
                          "The entire 49,393-neuron graph is evaluated; only a deterministic sample is displayed.",
                          "Brightness is absolute recurrent state. It does not indicate excitation or inhibition.",
                          "This model was trained on 1,000 TinyStories examples. These samples are a demonstration, not an independent quality benchmark."]}
    trainer.atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps({"status": "completed", "manifest": str(args.output / "manifest.json")}), flush=True)


if __name__ == "__main__":
    main()
