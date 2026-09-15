# Rank64 partial progress — 2026-09-15 22:23 UTC

G32rank64fixed is running on macm3. Exact dispatcher/trainer identity, unchanged source/input/baseline bindings and exclusive GPU execution were verified at22:23:29UTC, observing2,719updates and2completeepochs. F remains stopped.

The coherent fetched checkpoint/validation snapshot at22:22:50UTC contains these matched2,500-update results on100stories/21,874next-BPE targets:

| Model | Validation CE | Validation accuracy | Decoder parameters |
|---|---:|---:|---:|
| E rank128 | 3.823163 | 27.1967% | 6,453,376 |
| G rank64 | 3.896398 | 26.0400% | 3,226,688 |

G trails by0.073235CE and1.1566percentagepoints. Over1,500→2,500updates, G improved0.506778CE and5.0608accuracy points. Continue training; the eight-completed-epoch plateau review threshold is not reached. Both retained G selectors in this snapshot point to2,500updates; its later live status reported a new minimumCE3.851774/accuracy26.0720% at2,700, which is outside the coherent comparison snapshot above.

These are partial results from one seed, not a final quality verdict. Rank changes also affect head initialization scale. The reserved test remains untouched. [Structured receipt](../records/G32rank64fixed-progress.json) binds the metric snapshot, checkpoint receipts and source identities. No training source/configuration was changed.
