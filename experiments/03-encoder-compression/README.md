# Stage 3 — How wide does the input interface need to be?

**Objective.** The reference injects tokens through a 128-wide embedding fanned into
1,758 neurons per delay slot — 1,931,264 parameters of pure interface. Does the
connectome need that much bandwidth to be written into, or is the width arbitrary?

**Second question, run as a matched 2×2:** if we let the *synapses themselves* adapt
within ±10%, does that buy back anything a narrower encoder gives up?

## The design

| Arm | Encoder width | Encoder parameters | Edge adaptation | Total trainable | Status |
|---|---:|---:|---|---:|---|
| [A128fixed](../configs/A128fixed.json) | 128 | 1,931,264 | fixed base weights | 52,756,661 | accepted early stop |
| [B32fixed](../configs/B32fixed.json) | 32 | 482,816 | fixed base weights | 51,308,213 | accepted early stop |
| [C128bounded](../configs/C128bounded.json) | 128 | 1,931,264 | 18,322 shared gains, ±10% | 52,774,983 | deferred |
| [D32bounded](../configs/D32bounded.json) | 32 | 482,816 | 18,322 shared gains, ±10% | 51,326,535 | deferred |

The bounded arms do **not** rewire anything. Each edge keeps its endpoints, its sign and
its stored zero; only its magnitude is scaled:

```
W[e] = W0[e] · (1 + 0.1·tanh(theta_source[type(src)] + theta_destination[type(dst)]))
```

with 18,322 source/destination values over 9,161 cell-type groups, initialised to zero.
Independent type-*pair* gains would have cost 2,625,911 parameters and defeated the point.
Note that per-neuron gains stay unconstrained, so bounding the edge multiplier does **not**
bound the effective recurrence.

## Result: width 32 is enough; C and D were never needed

At the common 12,400-update budget B's best-through-budget CE was **0.21577 lower** than
A's, for **0.18744 points** less accuracy. Cutting the encoder by 75% cost nothing
measurable, so the bounded arms — which existed to buy back a loss that never appeared —
were deferred and have stayed deferred.

The [post-B assessment](encoder32-post-B-assessment.md), its
[generated texts](encoder32-post-B-texts.md) and
[validation curves](encoder32-post-B-validation.png) hold the detail;
[POST_B_REVIEW.md](POST_B_REVIEW.md) is the review that closed the stage.

Every later arm inherits **width 32**.

## The result this stage did not notice

Both arms keep the full 50,578,432-parameter readout, and both land *worse than the
released reference* on a held-out population — audit CE 5.050346 (A) and 4.912468 (B)
against the released model's 3.988231. Stage 4 is where that stops being about the
encoder at all.

## Superseded

[CONNECTORCH_NEXT_STAGE.md](CONNECTORCH_NEXT_STAGE.md) is the research plan written at the
end of this stage. Its proposals were overtaken by the Stage 4 results. It is kept as
history; **do not execute its queues.**
