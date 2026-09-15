# E32rank128fixed: accepted plateau stop and adaptive handoff

E stopped under the user's standing plateau-stop instruction at 15,918 observed updates, 15,900 durable updates and 13 completed epochs. The declared 30+14 schedule was not completed and the second learning-rate phase was not entered. Reserved test remains untouched.

| Retained selector | Update | Validation CE | Accuracy |
|---|---:|---:|---:|
| Minimum CE | 13,700 | 3.149704 | 36.4817% |
| Maximum accuracy | 14,300 | 3.159908 | 36.5822% |

At the final pre-signal check, the trailing 2,000 updates improved global best CE by zero and best accuracy by 0.086861 percentage points, below the predeclared thresholds of 0.10 CE and 0.5 points. Recent equal validation windows had nearly identical mean CE while complete-epoch training CE continued decreasing. This is an operational early-stop decision, not proof that longer training cannot improve.

The initial stop attempt sent no signal because it was outside the selected safe interval between validation saves. An event-based wait observed a suitable training interval; source/process identities, plateau and checkpoint coherence were rechecked. One SIGTERM to the exact dispatcher terminated and reaped its child. Both PIDs exited. Raw native signal-failure records are retained as intentional-stop history, not rewritten as successful schedule completion.

Both winner files and latest optimizer/RNG/cache state are preserved in the parent E stopped-checkpoints directory. Full CPU checkpoint-content audits verified manifests, cursors, parameter hashes and unchanged canonical graph hashes. No extra GPU inference or training was performed for the stop audit.

At a common 15,900-update budget, B's logged best CE was 4.512509 and best accuracy 33.3135%. E's best CE was 3.149704 and best accuracy 36.5822%, with 7,183,157 total parameters versus B's 51,308,213. Generated-text and independent quality checks for the final E/F winners remain pending; checkpoint-selection validation alone does not establish anatomical advantage.

F32rank128bounded launched from scratch in the fresh sibling output `results/connectorch-decoder-v1-continuation`. It retains the same rank128 head, width32 encoder, data, seed and schedule, adding 18,322 bounded cell-type source/destination gains. The separate handoff revalidated E acceptance, parent source/data/graph identities, rank128 parity and both original optimizer smokes. No smoke was rerun. Its manifest binds the reused evidence and new handoff source. F total parameters: 7,201,479.

The progress helper now combines B, preserved E and live F. The later rank64 comparison remains authorized after the E/F quality assessment.

- [Accepted E stop](../records/E32rank128fixed-accepted-early-stop.json)
- [Exact stop/process receipt](../records/E32rank128fixed-stop-receipt.json)
- [F continuation launch](../runs/decoder-v1-continuation-launch.json)
