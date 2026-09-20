# Launchers

The scripts that actually start a run. They lived only in a scratchpad until now, which
meant the configs, evaluators and registry were all version-controlled while the thing that
reads them and launches training was not. A launcher that disagrees with a config has
already caused two problems in this project:

- `launch_task.sh` hardcoded `--readout-rank 64`, so a rank-128 arm would have trained at
  64 while its config declared 128, and every evaluator rebuilds from the config — the
  checkpoint would have failed to restore after fifteen hours.
- `launch_task_n.sh` hardcoded `--toxicity-weight 1.0` while reading every other weight
  from the config. For the language-reweighted arm that would have been silent and fatal:
  the config declares 0.6, the arm trains 1.0, the task weights sum to 4.4 rather than 4.0,
  and a controlled test of *allocation* would instead have measured a larger total gradient.

Both classes are invisible until scoring, and the second would never surface at all.

| file | role |
|---|---|
| `launch_arm.sh` | starts one arm, reading rank, plasticity, task set, corpora **and loss weights** from `experiments/configs/<arm>.json` |
| `chain_audit.py` | waits for a run, gates on it reaching its budget, then runs the language audit and the router control |
| `queue_arms.py` | runs several arms in turn, each gated on the previous finishing |

## The rule

**A launcher reads every knob from the arm's config and hardcodes none of them.** If a knob
cannot be read from the config, it is not a knob — it is a constant, and it belongs in the
trainer rather than the launcher.

`tests/test_launchers.py` asserts that no launcher passes a literal value for a flag the
config also declares.
