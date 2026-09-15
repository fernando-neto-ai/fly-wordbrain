# Paired quality audit: reference reproduction gate

The lowest-CE checkpoint does **not** match released FlyLLM quality. A later
accuracy-selected checkpoint meets the predefined accuracy tolerance but still
has substantially worse cross-entropy; both selectors are now tracked separately.
Apple GPU numerical parity and reference inference reproduction were already
verified; these are separate from reproducing training quality.

## Follow-up: retain both validation winners

The user requested both minimum validation CE and maximum validation accuracy.
The active trainer is unchanged: `best.pt` continues to retain the CE winner.
`scripts/watch_ngxson_accuracy.py` runs as a separate macm3 CPU observer and
retains `results/ngxson-dual-selection-v1/best_accuracy.pt`. It uses native kqueue
events, immutable hardlinks, CPU memory-mapped checkpoint inspection, source/data
bindings and parameter fingerprints. Its selection receipt distinguishes the
highest logged metric from the best actual retained weights. Some earlier
accuracy peaks were already overwritten, but a new all-time peak at update
20,600 was captured and frozen for comparison (SHA256
`a08c255338323295e20ebc5a38e300b4dc401e4a1efbc299412e9dca519b459c`).

| Checkpoint | Validation CE | Validation accuracy | Audit CE | Audit accuracy |
|---|---:|---:|---:|---:|
| Released reference | 3.6809 | 35.24% | 3.9882 | 31.38% |
| Min CE, update 3,500 | 4.6928 | 30.58% | 5.0362 | 26.58% |
| Max accuracy, update 20,600 | 5.8968 | 34.31% | 6.3744 | 30.14% |

The accuracy winner is 1.238pp below the release, with paired 95% CI
[−1.637, −0.844]pp, inside the predefined ±2pp practical accuracy margin.
Its CE gap is +2.3862 nats [2.3301, 2.4431], outside the +0.10 margin.
This supports close top-1 accuracy while retaining a substantial probability
modeling gap. It does not support full predictive-quality reproduction.

This follow-up reuses the previously revealed 200-story audit; checkpoint
selection uses validation alone. All output samples and the earlier available
accuracy candidate at update20,300 are preserved. The reserved final test remains
untouched. The comparison is in `results/ngxson-dual-selection-v1/comparison.md`;
live validation and retention continue on macm3. Saved validation CE replay for
the accuracy checkpoint differs by −9.94e-9 and its correct/token counts match
exactly. Twelve evaluator fixtures and twelve observer fixtures pass, including
native macOS event delivery through the actual observer CLI.

## Frozen inputs and results

The best validation checkpoint at update 3,500 was snapshotted on macm3 before
scoring. Its SHA256 is
`7c89ceadd6e0902b4b7677822f684e2f0c0692b09fe816ed6146b6797ad212fc`.
Both models use the exact pinned architecture, graph and 1,024-token BPE.
The release revision is `65c677b3d566a2e9793d5f72999cdb441c6c0a9f`.

The primary comparison uses 200 new official TinyStories validation stories,
45,059 targets, selected from rows 1,000–1,216 after excluding stories exceeding
320 tokens. Every existing train/validation/test ID and normalized text was
explicitly included in the exclusion population. Final ID, exact-text and
normalized-text intersections are zero; semantic overlap is not ruled out.
All source-page revisions match the original dataset revision. The audit data
SHA256 is `5fd63f448a2e09a001520ef77c3f45611c5c66600fa96ac95bf880afab112731`.

| Model | Audit cross-entropy ↓ | Perplexity ↓ | BPE-token accuracy ↑ |
|---|---:|---:|---:|
| Released ngxson checkpoint | 3.9882 | 53.96 | 31.38% |
| Our validation-selected checkpoint | 5.0362 | 153.88 | 26.58% |

Ours has 2.852× perplexity and 4.803 percentage points lower accuracy.
Paired whole-story bootstrap (10,000 resamples, seed 1729), preserving
token weighting, gives ΔCE +1.04794 [1.01483, 1.08128] and accuracy difference
−4.803pp [−5.214, −4.369] at 95% confidence. Both predefined operational margins
(±0.10 nats CE and ±2pp accuracy) fail, for equivalence and non-inferiority.
The margins were recorded before model scoring; they are a practical choice,
not a universal standard.

On the existing validation split (100 stories / 21,874 targets), the released
checkpoint scores CE3.680932 / accuracy35.2428%, versus ours
CE4.692816 / accuracy30.5797%. This split selected our checkpoint, so the new
audit is the primary comparison. The reserved final test was not evaluated.
This audit set is now revealed and must not be treated as fresh after tuning.

All six fixed 80-token greedy continuations, including early EOS, are in
`results/ngxson-quality-v1-mps-r2/report.md`. Both models produce imperfect
story-like text; surface resemblance does not establish comparable predictive
quality. The author's unpublished training sample IDs and exact trainer prevent
an exact training-reproduction claim even if a later quality comparison passes.

## Verification and execution

- Nine small CPU-only inference/statistical tests pass. They check padding,
  EOS, globally shifted targets at chunk boundaries, state carry/reset,
  token weighting, paired bootstrap, generation BOS and immutable parameters.
- Restored parameter hashes and exact graph hashes match the saved checkpoint.
  Replaying its validation CE differs from the saved score by only 1.32e-8.
- Both evaluated models retain unchanged parameter and graph hashes; native
  MPS metadata records zero backward calls and no CPU fallback.
- All model inference ran on macm3. The initial CPU audit was interrupted for
  speed; a GPU audit attempt failed before scoring on an audit-only hashing
  error, which was fixed. Artifacts from both attempts remain preserved.
- The successful GPU audit ran serially with the exact existing trainer
  suspended for about 61 seconds, then resumed by the controller's finally
  block at 2026-09-15T14:05:53.901166Z. No optimizer or training source changed.
- Local report files were copied with SHA256 verification. No local training
  or GPU evaluation was performed.

Implementation: `scripts/prepare_ngxson_quality_data.py`,
`scripts/evaluate_ngxson_quality.py`, `scripts/run_ngxson_quality_exclusive.py`.
The existing trainer and model sources were left unchanged.

## Consequence

Keep the already-running original training recipe intact through completion.
Hold the prepared Connectorch ablations until the training-quality gap is
reviewed. Completion of all epochs alone does not satisfy this quality gate.
The current evidence does not identify which unpublished recipe/data detail
causes the gap, and does not show a defect in the verified Apple GPU kernels.
