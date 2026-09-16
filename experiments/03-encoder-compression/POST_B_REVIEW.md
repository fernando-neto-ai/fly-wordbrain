# Post-B review: preserve the reduced encoder with minimal brain adaptation

The user requested this assessment immediately after B32fixed. It precedes the
queued adaptive experiments and the later decoder-compression decision.
B has stopped under the standing plateau-stop authorization at 16,262 observed
updates / 16,200 durable updates after 13 completed epochs. Its dispatcher stays
paused; C/D wait. The completed assessment is in
[encoder32-post-B-assessment.md](encoder32-post-B-assessment.md).
The verified gate is stored at the continuation output's queue-review-gate.json.
Do not mistake the paused dispatcher for a stopped training worker or resume it
before assessment: its original queue would automatically launch C then D.

## Measure the loss before choosing plasticity

Keep the encoder at482,816 parameters, with the same decoder and eight explicit
input delays. Compare A and B at common recorded updates through12,400, alongside
both independent retained winners. A stopped early, so longer B training gives
unequal budgets. Do not equate missing historical checkpoint weights with an
available snapshot merely because its score remains in the logs.

Use matching prompts/decoding and inspect overlap with training text. Retain the
reserved test. Proposed practical review flags, chosen now and not statistical
significance claims, are more than0.10 CE or one accuracy percentage point of
loss, or clear deterioration in text coherence. Neither accuracy alone nor an
increase in parameter count establishes successful transfer of computation.

## Choose the smallest useful adaptation

| Tier | Additional edge-gain parameters | Total trained brain parameters | Status |
|---|---:|---:|---|
| Fixed base edges, per-neuron gain/rec_gain/bias trained | 0 | 148,179 | Current B |
| Source/destination cell-type gains, factors0.9–1.1 | 18,322 | 166,501 | Existing D32bounded candidate |
| Finer source/destination per-neuron gains, same bound initially | 98,786 | 246,965 | Untested proposal requiring implementation/parity checks |

If quality is maintained, no extra plasticity is required merely to reduce the
encoder. If it degrades, D32bounded is the first candidate. B's result quantifies
the deficit; it cannot determine how many extra parameters will recover it until
an adaptive experiment is trained. C128bounded remains the matched wide-encoder
adaptive control for distinguishing a general gain from compression recovery.

Keep all49,393 neurons and9,050,172 base-edge endpoints, stored values and signs.
Shared gain parameters may affect many edges simultaneously; their count is not
the fraction of synapses unfrozen. The ±10% constraint applies to new edge
multipliers; the original neuron gain/rec_gain values remain unconstrained.
The finer tier replaces18,322 values rather than adding98,786 on top.

## Diagnose before escalating

Verify source/graph identity, optimizer inclusion, gradients and actual updates.
Check input-drive coverage/variance, recurrent versus input scale, tanh saturation,
and whether learned gains affect centered logits and probabilities. Preserve
normalization when intervening and include magnitude-matched controls; simply
removing a pathway may measure a distribution shock.

If bounded gains already improve quality and saturate, investigate a modestly
wider bound. If coarse sharing is demonstrably limiting, investigate finer
sharing while keeping the original bound. Do not automatically unfreeze all
9.05M edges, restore a larger encoder, or launch an unbounded sweep. Select the
smallest verified deformation that recovers useful language quality.

All real training and post-training GPU inference run serially on macm3. When B
finishes, verify its own completion artifacts and process state: the dispatcher
is paused, so its wrapper status may be stale and its exited child may await
reaping. Perform inference only after the child has exited and released GPU
resources. Publish the assessment and concrete next configuration before the
adaptive queue is released.
