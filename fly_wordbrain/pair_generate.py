"""Qualitative generation with two independent future-word heads.

Both heads observe the same frozen-brain state. Their sampled lexical words
are then fed back in order through successive sliding bigram inputs. Sampling
head two never observes feedback from head one. These examples are not an
evaluation metric and do not test language understanding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .brain import array_digest
from .data import BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, UNK_ID, words_from_text


PROMPTS = ("once upon a time", "one day lily", "the little dog")
MAX_CONTEXT_WORDS = 128


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sampling_probabilities(logits: np.ndarray, temperature: float) -> Tuple[np.ndarray, np.ndarray]:
    """Keep model probabilities separate from temperature/masking for sampling."""
    values = np.asarray(logits, dtype=np.float64).copy()
    if values.ndim != 1 or len(values) < 4 or not np.isfinite(values).all():
        raise ValueError("Require finite full-vocabulary logits")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    original = np.exp(values - values.max())
    original /= original.sum()
    values /= temperature
    values[[PAD_ID, BOS_ID]] = -np.inf
    probabilities = np.exp(values - values.max())
    probabilities /= probabilities.sum()
    return probabilities, original


def _replay(brain, context):
    """Discard every previous state and previous-word slot when the window moves."""
    brain.reset()
    features = brain.step((BOS_ID, BOS_ID))
    previous = BOS_ID
    for current in context:
        features = brain.step((previous, current))
        previous = current
    return features


def generate_sequence(brain, decoder, prompt: str, rng, *, max_words: int = 24,
                      temperature: float = .8) -> Dict[str, Any]:
    """Sample a pair before feedback; replay only the latest 128 lexical words."""
    if not 1 <= max_words <= 24:
        raise ValueError("This qualitative sampler permits 1--24 new words")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive")
    vocabulary = list(decoder.vocabulary)
    if vocabulary[:4] != list(SPECIAL_TOKENS):
        raise ValueError("Expected reserved IDs PAD=0, UNK=1, BOS=2, EOS=3")
    lookup = {word: index for index, word in enumerate(vocabulary)}
    prompt_words = words_from_text(prompt)
    prompt_ids = [lookup.get(word, UNK_ID) for word in prompt_words]
    context = prompt_ids[-MAX_CONTEXT_WORDS:]
    before = brain.verify_frozen()
    features = _replay(brain, context)
    generated, sampled, traces, pair_predictions, replays = [], [], [], [], []
    stopped, eos_horizon = "word_limit", None
    while len(generated) < max_words:
        # One model call, copied before any further neural stimulation. Both
        # predictions are p(w[t+1]|state[t]) and p(w[t+2]|state[t]).
        logits = np.array(decoder.logits(features), dtype=np.float64, copy=True)
        if logits.shape != (2, len(vocabulary)) or not np.isfinite(logits).all():
            raise ValueError("Pair decoder must emit finite [2, vocabulary] logits")
        pair_index = len(pair_predictions)
        state_hash = array_digest([np.asarray(features)])
        capacity = min(2, max_words - len(generated))
        pending, pair_traces = [], []
        pair_record = {
            "pair_index": pair_index,
            "after_generated_words": len(generated),
            "context_lexical_words": len(context),
            "shared_feature_sha256": state_hash,
            "sampling_precedes_all_feedback": True,
        }
        for head in range(capacity):
            probabilities, original = sampling_probabilities(logits[head], temperature)
            token = int(rng.choice(len(vocabulary), p=probabilities))
            if token in (PAD_ID, BOS_ID) or not 0 <= token < len(vocabulary):
                raise RuntimeError("Sampler returned a forbidden or out-of-vocabulary token")
            sampled.append(token)
            top = np.argsort(-probabilities, kind="stable")[:min(5, len(vocabulary) - 2)]
            trace = {
                "pair_index": pair_index, "head": head, "future_word_offset": head + 1,
                "shared_feature_sha256": state_hash,
                "token_id": token, "token": vocabulary[token],
                "sample_probability": float(probabilities[token]),
                "model_probability": float(original[token]),
                "top_sampling_tokens": [
                    {"token_id": int(index), "token": vocabulary[int(index)],
                     "probability": float(probabilities[int(index)])}
                    for index in top if probabilities[int(index)] > 0],
            }
            pair_traces.append(trace)
            traces.append(trace)
            if token == EOS_ID:
                stopped, eos_horizon = "eos", head + 1
                break
            pending.append(token)
        pair_record.update(
            sampled_ids_including_eos=[row["token_id"] for row in pair_traces],
            sampled_heads=len(pair_traces),
            second_head_not_sampled_reason=("first_head_eos" if eos_horizon == 1
                                           else "word_limit" if capacity == 1 else None),
            independent_model_probability_product=math.prod(row["model_probability"] for row in pair_traces),
            independent_sampling_probability_product=math.prod(row["sample_probability"] for row in pair_traces),
        )
        pair_predictions.append(pair_record)
        # Only after both requested draws: consume the first and second lexical
        # predictions in order. EOS is never a neural input. If head two is EOS,
        # the first lexical prediction is still consumed before stopping.
        for token in pending:
            previous = context[-1] if context else BOS_ID
            generated.append(token)
            context.append(token)
            if len(context) > MAX_CONTEXT_WORDS:
                context = context[-MAX_CONTEXT_WORDS:]
                features = _replay(brain, context)
                replays.append({"after_generated_words": len(generated),
                                "lexical_words_replayed": len(context),
                                "initial_pair": [BOS_ID, BOS_ID],
                                "first_word_previous_id": BOS_ID})
            else:
                features = brain.step((previous, token))
        if eos_horizon is not None:
            break
    after = brain.verify_frozen()
    if after != before:
        raise RuntimeError("Frozen graph changed during pair generation")
    generated_words = [vocabulary[token] for token in generated]
    return {
        "prompt": prompt, "prompt_words": prompt_words, "prompt_ids": prompt_ids,
        "prompt_unknown_words": [word for word, token in zip(prompt_words, prompt_ids) if token == UNK_ID],
        "prompt_prefix_words_discarded": max(0, len(prompt_ids) - MAX_CONTEXT_WORDS),
        "generated_words": generated_words, "generated_ids": generated,
        "sampled_ids_including_eos": sampled, "continuation": " ".join(generated_words),
        "stopped": stopped, "eos_future_word_offset": eos_horizon,
        "requested_max_words": max_words,
        "final_context_lexical_words": len(context), "rolling_context_replays": replays,
        "samples": traces, "pair_predictions": pair_predictions,
        "frozen_graph_sha256_before": before, "frozen_graph_sha256_after": after,
    }


def check_provenance(dataset_path: Path, dataset: dict, decoder, manifest_path: Path,
                     manifest: dict, brain) -> dict:
    """Require the brain arm, exact feature receipts, and exact adapter sources."""
    if decoder.config.get("arm") != "brain":
        raise ValueError("Qualitative neural generation requires the brain arm")
    if dataset["vocabulary"] != decoder.vocabulary:
        raise ValueError("Dataset vocabulary differs from paired decoder")
    dataset_sha = sha256_file(dataset_path)
    if decoder.config.get("dataset_sha256") != dataset_sha or manifest.get("dataset_sha256") != dataset_sha:
        raise ValueError("Dataset hash differs from training/extraction provenance")
    if manifest.get("completed") is not True:
        raise ValueError("Generation requires completed feature extraction")
    feature_hashes = decoder.config.get("feature_sha256")
    if not feature_hashes or feature_hashes != manifest.get("split_hashes"):
        raise ValueError("Feature hashes differ from paired decoder training")
    if decoder.config.get("feature_manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Feature manifest bytes differ from paired decoder provenance")
    expected, actual = manifest.get("brain"), brain.metadata()
    if not isinstance(expected, dict) or decoder.config.get("brain") != expected:
        raise ValueError("Paired decoder and extraction brain configurations differ")
    if expected != actual:
        keys = sorted(key for key in set(expected) | set(actual) if expected.get(key) != actual.get(key))
        raise ValueError("Generation brain differs from extraction: " + ", ".join(keys))
    if manifest.get("final_graph_sha256") != expected.get("graph_sha256"):
        raise ValueError("Feature extraction did not preserve its original graph")
    dimensions = actual.get("feature_dimensions")
    if decoder.config.get("feature_dimensions") != dimensions or len(decoder.heads) != 2:
        raise ValueError("Require exactly two heads with the original feature dimensions")
    for number, head in enumerate(decoder.heads):
        if head.vocabulary != decoder.vocabulary or len(head.scaler.mean) != dimensions:
            raise ValueError("A future-word head differs in vocabulary or feature dimensions")
        if (head.config.get("arm") != "brain" or head.config.get("horizon") != number + 1
                or head.config.get("dataset_sha256") != dataset_sha
                or head.config.get("feature_sha256") != feature_hashes):
            raise ValueError("A future-word head has different arm/horizon/source provenance")
    adapter_sources = {filename: sha256_file(Path(__file__).with_name(filename))
                       for filename in ("brain.py", "pair_brain.py")}
    for filename, digest in adapter_sources.items():
        if manifest.get("code_sha256", {}).get(filename) != digest:
            raise ValueError("Adapter source changed after feature extraction: " + filename)
    return {"dataset_sha256": dataset_sha, "feature_split_sha256": feature_hashes,
            "feature_manifest_identity": manifest.get("identity"),
            "brain_adapter_sources_sha256": adapter_sources, "brain": actual,
            "decoder_arm": decoder.config["arm"], "decoder_seed": decoder.config.get("seed")}


def generate(dataset_path: Path, checkpoint: Path, output: Path, *,
             feature_manifest: Optional[Path] = None) -> dict:
    from .pair_brain import PairedFrozenBrain
    from .pair_decoder import PairDecoder

    dataset_path, checkpoint, output = Path(dataset_path), Path(checkpoint), Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite existing pair generation output")
    dataset = json.loads(dataset_path.read_text())
    decoder = PairDecoder.load(checkpoint)
    if decoder.config.get("arm") != "brain":
        raise ValueError("Pair generation supports only the frozen-brain arm")
    if dataset["vocabulary"] != decoder.vocabulary or sha256_file(dataset_path) != decoder.config.get("dataset_sha256"):
        raise ValueError("Dataset differs from paired decoder training data")
    checkpoint_dir = Path(decoder.checkpoint_dir)
    manifest_path = (Path(feature_manifest) if feature_manifest is not None
                     else checkpoint_dir / "feature-manifest.json")
    manifest = json.loads(manifest_path.read_text())
    brain = PairedFrozenBrain(len(dataset["vocabulary"]), word_ms=20., warmup_ms=100.,
                              high=.02, bins=128, seed=1729)
    provenance = check_provenance(dataset_path, dataset, decoder, manifest_path, manifest, brain)
    rng = np.random.default_rng(1729)
    started = time.perf_counter()
    samples = [generate_sequence(brain, decoder, prompt, rng, max_words=24, temperature=.8)
               for prompt in PROMPTS]
    record = {
        "schema_version": 1,
        "purpose": "Qualitative samples from two independent future-word heads; not an evaluation metric.",
        "prompts": list(PROMPTS), "sampling_seed": 1729,
        "sampling_rng": "numpy.default_rng PCG64", "temperature": .8,
        "seed_policy": "One fixed RNG stream across all three prompts and both heads in their recorded order.",
        "max_new_words": 24, "max_context_lexical_words": MAX_CONTEXT_WORDS,
        "prediction_policy": "Both independent heads observe one shared state; sample head1 and head2 before feeding either predicted word back. No conditioning of head2 on the sampled head1 word.",
        "feedback_policy": "Consume sampled lexical words in order through successive (previous,current) bigram inputs before the next paired prediction.",
        "context_policy": "Reset/warmup and (BOS,BOS) at each prompt. On overflow reset and replay (BOS,BOS), then sliding pairs over only the latest128 lexical words, using BOS as the first previous word.",
        "sampling_policy": "PAD/BOS forbidden; UNK sampled literally. First-head EOS stops without sampling head2. Second-head EOS consumes the first lexical word then stops. EOS is never fed into the brain.",
        "frozen_graph": True, "checkpoint_directory": str(checkpoint_dir.resolve()),
        "checkpoint_files_sha256": {name: sha256_file(checkpoint_dir / name)
                                    for name in ("pair.json", "head-0.pt", "head-1.pt")},
        "feature_manifest_sha256": sha256_file(manifest_path),
        "generation_source_sha256": sha256_file(Path(__file__)),
        "decoder_sources_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                   for name in ("pair_decoder.py", "decoder.py")},
        "provenance": provenance, "samples": samples,
        "elapsed_seconds": time.perf_counter() - started,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(output)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Output root, brain arm, exact seed directory, or pair.json")
    parser.add_argument("--output", type=Path, required=True, help="New qualitative JSON artifact")
    parser.add_argument("--feature-manifest", type=Path,
                        help="Defaults to the extraction manifest copied into the resolved seed directory")
    args = parser.parse_args()
    report = generate(args.dataset, args.checkpoint, args.output, feature_manifest=args.feature_manifest)
    print(json.dumps({"output": str(args.output.resolve()), "purpose": report["purpose"],
                      "samples": [{"prompt": row["prompt"], "continuation": row["continuation"],
                                   "stopped": row["stopped"]} for row in report["samples"]]}, indent=2))


if __name__ == "__main__":
    main()
