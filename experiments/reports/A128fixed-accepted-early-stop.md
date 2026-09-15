# A128fixed: accepted plateau stop

The user asked to move to the next experiment if this run was plateauing.
The decision at update11,500 used validation results: no new CE best for7,900
updates and only0.192 percentage points of best-accuracy improvement over the
last2,000 updates. Training CE continued falling while validation CE worsened.
This is a practical diminishing-returns decision; later gains remain possible.

The worker was stopped at observed update12,442, after10 complete epochs.
The latest resumable checkpoint is update12,400. The original signal-exit
receipts remain intact, and a separate accepted-stop receipt records the user
instruction, preserved files, actual cursor and verified process cessation.
The planned44 epochs were not completed.

| Selector | Update | Validation CE | Next-token accuracy |
|---|---:|---:|---:|
| Minimum CE | 3,600 | 4.728279 | 30.4883% |
| Maximum accuracy | 12,200 | 5.546981 | 32.8975% |

Both checkpoint winners and the latest optimizer/RNG state are preserved with
SHA256 identities. Validation uses the same100 stories/21,874 targets. The
reserved test remains untouched. Graph buffers and bound source hashes passed
verification. No bound trainer or model code changed.

Continue with B32fixed, then C128bounded and D32bounded, through a separate
continuation controller. Use actual training budgets in the final report and
include comparisons at the same recorded optimizer update; retained-best
scores across unequal budgets alone are not a controlled architecture verdict.

[Verified stop record](../records/A128fixed-accepted-early-stop.json).
