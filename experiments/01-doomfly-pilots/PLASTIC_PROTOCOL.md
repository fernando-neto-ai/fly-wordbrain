# Full-connectome fast-plasticity pilot

This arm keeps Doomfly’s complete directed graph: 166,700 neurons and 25,582,938 edges. It replaces spiking physiology with smooth, leaky tanh rates. It tests a particular learning rule on this wiring; it does not simulate a fly acquiring language.

## Fixed circuit and small interfaces

Word inputs use the existing fixed ordered-pair retinal codes. Every descending neuron contributes to a fixed signed 256-feature projection. The only decoder is one affine layer with two 1,024-word heads: 526,336 learned parameters.

The baseline weights and all endpoints remain fixed. Fast state modulates 347 existing, positive hDeltaB connections: 50 to hDeltaH, 90 to hDeltaA, 161 to hDeltaI, and 46 to hDeltaG. There are 19 presynaptic hDeltaB neurons. Each fast state multiplies its original weight by exp(log(2) tanh(H)), preserving the sign and bounding the multiplier between 0.5 and 2. Four groups each have five learned rule parameters, for 20 additional learned parameters. Fast state resets at every story boundary.

All selected sources are four directed hops from retinal inputs; selected targets are two hops from a descending neuron. Paths through these synapses therefore require at least seven updates. Initial timing and calibration use eight updates per observed word, leak 0.5, global weight scale 0.0008, and unchanged 0.02 retinal currents. A 40-step signed power iteration estimated a dominant eigenvalue of 1124.47 (relative residual 0.00107). The global scale places that estimated mode near 0.90; this is an initialization heuristic, not a stability proof. No row normalization, pruning, or new edges are used.

## Calibration and causality

Calibration uses only identified training stories. It fixes per-edge activity scales and the feature mean and standard deviation before optimization. Diagnostics check graph fingerprints, candidate activity, word-stream versus constant-input responses, the effect of fast state on the decoder features, gradients into the shared rule, and a real optimizer update. Nonzero effects establish computational wiring, not useful linguistic memory.

The initial `.0008` setting failed the repeat-controlled numerical visibility test. A training-prefix-only diagnostic tested the explicit scales `.0008, .0012, .002, .004, .008`; `.002` was the smallest setting whose fast effect exceeded `max(10 * same-input repeat difference, 1e-5)` in normalized feature units. The first GPU pipeline smoke run uses that setting. It preserves the relative baseline weights but is outside the estimated linear contraction regime. No validation/test losses entered this calibration choice.

At observed position p, the model sees only words p-1 and p. The heads predict p+1 and p+2 independently; neither receives the true future word as an input at that position. Context is at most 128 lexical words, excluding BOS. Full-story backpropagation preserves gradients within that boundary, and padding freezes both neural and fast state. Two-head average forecast loss is not ordinary sequence perplexity; only head 1 supplies standard next-word perplexity.

## Comparisons

The three matched rate-model arms are frozen synapses, a fixed fast-plasticity rule, and a learned fast-plasticity rule. They start from the same decoder and data-order seed and use the same circuit and calibration. Validation next-word loss selects each checkpoint. Every selection is locked before test evaluation. The earlier spiking model and direct-input baseline provide context, but the frozen rate arm is the matched control for this physiology.

Mid-story resets of fast state and neural activity are compared on aligned second-half predictions. A reset can damage ordinary dynamics, so its effect alone does not prove a separate memory mechanism. Geometry-specific advantage would require further matched randomized-connectome controls; these three arms alone cannot establish it.

## Hardware

macm3 has 128 GB unified memory. Stock PyTorch 2.8 MPS rejects sparse CSR construction, so this project supplies a custom Metal CSR kernel through `torch.mps.compile_shader`. The same kernel applies the explicitly transposed graph during backpropagation. State, fast weights, decoder, gradients, and optimizer updates stay on MPS; CPU fallback is disabled. The mathematical model and full graph are unchanged. Index arrays use compact int32 GPU caches verified against the canonical int64 indices.

CPU and GPU numerical tests compare asymmetric graphs, empty rows, long rows, signed and zero weights, strided states, batches of 1/2/8, and all twenty rule gradients. Full-graph timing and allocation samples are recorded separately from learning results. MPS exposes current and driver allocation, not an exact peak counter, so reports label stage-boundary samples accordingly. AMD also passes the corrected one-GPU sparse preflight using matching CSR index types and a fixed, hashed ROCm container.

The initial `.0008` rate scale produces extremely weak activity at the selected synapses. Its CPU fast-state intervention has zero observable readout effect; GPU rounding variation can produce differences around 1e-6 normalized units. Passing a finite-gradient check is insufficient. Calibration must demonstrate an effect above an explicitly measured same-input repeat floor before a learning pilot is launched. This numerical check still does not establish useful linguistic memory.
