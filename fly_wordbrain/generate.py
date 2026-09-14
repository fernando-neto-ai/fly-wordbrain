"""Bounded qualitative samples from a frozen connectome and trained readout.

These illustrations are not evaluation metrics. The generation adapter, graph,
encoder and projection must match the feature-extraction manifest tied to the
decoder checkpoint. No fitting or parameter adjustment happens here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from .brain import FrozenBrain
from .data import BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, UNK_ID, words_from_text
from .decoder import Decoder


PROMPTS = ("once upon a time", "one day lily", "the little dog")
MAX_CONTEXT_WORDS = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampling_probabilities(logits: np.ndarray, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return constrained sampling probabilities and original model probabilities."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("Decoder must emit a finite logit vector over the full vocabulary")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    raw = np.exp(values - values.max())
    raw /= raw.sum()
    values = values / temperature
    values[[PAD_ID, BOS_ID]] = -np.inf
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    return probabilities, raw


def generate_sequence(brain: FrozenBrain, decoder: Decoder, prompt: str,
                      rng: np.random.Generator, *, max_words: int = 24,
                      temperature: float = 0.8) -> Dict[str, Any]:
    """Feed BOS/prompt; each sampled lexical token becomes the next neural input.

When the current context already contains 128 lexical words, adding a token
resets the brain, repeats its standard warmup, and replays BOS plus the latest
128 words. The discarded prefix therefore leaves no residual neural state.
"""
    if not 1 <= max_words <= 24:
        raise ValueError("This qualitative sampler permits 1--24 new words")
    vocabulary = decoder.vocabulary
    if list(vocabulary[:4]) != list(SPECIAL_TOKENS):
        raise ValueError("Expected reserved IDs PAD=0, UNK=1, BOS=2, EOS=3")
    lookup = {word: i for i, word in enumerate(vocabulary)}
    prompt_words = words_from_text(prompt)
    prompt_ids = [lookup.get(word, UNK_ID) for word in prompt_words]
    context = prompt_ids[-MAX_CONTEXT_WORDS:]
    before = brain.verify_frozen()
    brain.reset()
    features = brain.step(BOS_ID)
    for token in context:
        features = brain.step(token)
    generated, sampled, traces, replays = [], [], [], []
    stopped = "word_limit"
    for _ in range(max_words):
        probabilities, original = sampling_probabilities(decoder.logits(features), temperature)
        token = int(rng.choice(len(vocabulary), p=probabilities))
        sampled.append(token)
        top = np.argsort(-probabilities, kind="stable")[:min(5, len(vocabulary) - 2)]
        traces.append({"token_id": token, "token": vocabulary[token],
                       "sample_probability": float(probabilities[token]),
                       "model_probability": float(original[token]),
                       "top_sampling_tokens": [{"token_id": int(i), "token": vocabulary[int(i)],
                                                 "probability": float(probabilities[int(i)])}
                                                for i in top if probabilities[int(i)] > 0]})
        if token == EOS_ID:
            stopped = "eos"
            break
        generated.append(token)
        context.append(token)
        if len(context) > MAX_CONTEXT_WORDS:
            context = context[-MAX_CONTEXT_WORDS:]
            brain.reset()
            features = brain.step(BOS_ID)
            for word_id in context:
                features = brain.step(word_id)
            replays.append({"after_generated_words": len(generated),
                            "lexical_words_replayed": len(context), "bos_replayed": True})
        else:
            features = brain.step(token)
    after = brain.verify_frozen()
    if after != before:
        raise RuntimeError("Graph identity changed during qualitative generation")
    generated_words = [vocabulary[token] for token in generated]
    return {"prompt": prompt, "prompt_words": prompt_words, "prompt_ids": prompt_ids,
            "prompt_unknown_words": [word for word, token in zip(prompt_words, prompt_ids) if token == UNK_ID],
            "prompt_prefix_words_discarded": max(0, len(prompt_ids) - MAX_CONTEXT_WORDS),
            "generated_words": generated_words, "generated_ids": generated,
            "sampled_ids_including_eos": sampled, "continuation": " ".join(generated_words),
            "stopped": stopped, "requested_max_words": max_words,
            "final_context_lexical_words": len(context), "rolling_context_replays": replays,
            "samples": traces, "frozen_graph_sha256_before": before,
            "frozen_graph_sha256_after": after}


def check_provenance(dataset_path: Path, dataset: Dict[str, Any], decoder: Decoder,
                     manifest: Dict[str, Any], brain: FrozenBrain) -> Dict[str, Any]:
    """Fail closed when generation cannot be tied to the decoder's exact features."""
    if dataset["vocabulary"] != decoder.vocabulary:
        raise ValueError("Dataset vocabulary differs from decoder checkpoint")
    dataset_sha = sha256_file(dataset_path)
    if decoder.config.get("dataset_sha256") != dataset_sha or manifest.get("dataset_sha256") != dataset_sha:
        raise ValueError("Dataset hash differs from training/extraction provenance")
    if manifest.get("completed") is not True:
        raise ValueError("Generation requires a completed extraction manifest")
    hashes = decoder.config.get("feature_sha256")
    if not hashes or hashes != manifest.get("split_hashes"):
        raise ValueError("Feature manifest hashes do not match decoder training arrays")
    expected = manifest.get("brain")
    if not isinstance(expected, dict):
        raise ValueError("Feature manifest has no brain configuration")
    actual = brain.metadata()
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError("Generation brain differs from extraction: " + ", ".join(sorted(mismatches)))
    if len(decoder.scaler.mean) != actual.get("feature_dimensions"):
        raise ValueError("Decoder and brain feature dimensions differ")
    if manifest.get("final_graph_sha256") != expected.get("graph_sha256"):
        raise ValueError("Extraction did not preserve its initial frozen graph")
    source_sha = sha256_file(Path(__file__).with_name("brain.py"))
    if manifest.get("code_sha256", {}).get("brain.py") != source_sha:
        raise ValueError("Brain adapter source changed after feature extraction")
    return {"dataset_sha256": dataset_sha, "feature_split_sha256": hashes,
            "feature_manifest_identity": manifest.get("identity"),
            "brain_adapter_sha256": source_sha, "brain": actual}


def generate(dataset_path: Path, checkpoint: Path, output: Path, *,
             feature_manifest: Optional[Path] = None, word_ms: float = 20.,
             high: float = .02, warmup_ms: float = 100., bins: int = 128,
             encoder_seed: int = 1729, seed: int = 1729, temperature: float = .8,
             max_words: int = 24) -> Dict[str, Any]:
    dataset_path, checkpoint, output = Path(dataset_path), Path(checkpoint), Path(output)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "checkpoint.pt"
    if output.exists():
        raise ValueError("Refusing to overwrite existing generation output")
    if not 1 <= max_words <= 24 or seed < 0:
        raise ValueError("Require 1--24 new words and a nonnegative sampling seed")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    feature_manifest = Path(feature_manifest) if feature_manifest is not None else checkpoint.parent / "feature-manifest.json"
    if not feature_manifest.is_file():
        raise ValueError("Missing extraction provenance: pass --feature-manifest or copy it to checkpoint-directory/feature-manifest.json")
    dataset = json.loads(dataset_path.read_text())
    manifest = json.loads(feature_manifest.read_text())
    decoder = Decoder.load(checkpoint)
    # Reject cheaply detectable mismatches before allocating the full graph.
    if dataset["vocabulary"] != decoder.vocabulary or sha256_file(dataset_path) != decoder.config.get("dataset_sha256"):
        raise ValueError("Dataset differs from decoder training data")
    brain = FrozenBrain(len(dataset["vocabulary"]), word_ms=word_ms, warmup_ms=warmup_ms,
                        bins=bins, seed=encoder_seed, high=high)
    provenance = check_provenance(dataset_path, dataset, decoder, manifest, brain)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    samples = [generate_sequence(brain, decoder, prompt, rng, max_words=max_words,
                                 temperature=temperature) for prompt in PROMPTS]
    record = {"schema_version": 1, "purpose": "Qualitative illustration only; not an evaluation metric.",
              "prompts": list(PROMPTS), "sampling_seed": seed, "sampling_rng": "numpy.default_rng PCG64",
              "seed_policy": "One fixed RNG stream across the three prompts in their recorded order.",
              "temperature": temperature, "max_new_words": max_words,
              "max_context_lexical_words": MAX_CONTEXT_WORDS,
              "context_policy": "Reset/warmup/BOS at each prompt; if context would exceed 128 words, reset/warmup/replay BOS plus latest 128 words.",
              "sampling_policy": "PAD/BOS forbidden; UNK sampled literally; EOS stops without entering the next neural interval.",
              "frozen_graph": True, "checkpoint_sha256": sha256_file(checkpoint),
              "feature_manifest_sha256": sha256_file(feature_manifest),
              "generation_source_sha256": sha256_file(Path(__file__)),
              "decoder_source_sha256": sha256_file(Path(__file__).with_name("decoder.py")),
              "provenance": provenance, "samples": samples,
              "elapsed_seconds": time.perf_counter() - started}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New JSON output file")
    parser.add_argument("--feature-manifest", type=Path, help="Completed extraction manifest; defaults to checkpoint-directory/feature-manifest.json")
    parser.add_argument("--word-ms", type=float, default=20.)
    parser.add_argument("--high", type=float, default=.02)
    parser.add_argument("--warmup-ms", type=float, default=100.)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--encoder-seed", type=int, default=1729)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--words", type=int, default=24)
    args = parser.parse_args()
    result = generate(args.dataset, args.checkpoint, args.output, feature_manifest=args.feature_manifest,
                      word_ms=args.word_ms, high=args.high, warmup_ms=args.warmup_ms, bins=args.bins,
                      encoder_seed=args.encoder_seed, seed=args.seed, temperature=args.temperature,
                      max_words=args.words)
    print(json.dumps({"output": str(args.output.resolve()), "purpose": result["purpose"],
                      "samples": [{"prompt": row["prompt"], "continuation": row["continuation"],
                                   "stopped": row["stopped"]} for row in result["samples"]]}, indent=2))


if __name__ == "__main__":
    main()
