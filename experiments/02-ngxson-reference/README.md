# Stage 2 — Reproducing the ngxson Fly LLM exactly

**Objective.** Stop inventing architectures and reproduce a working one. Take
[ngxson/fly-llm-hf](https://huggingface.co/ngxson/fly-llm-hf) at revision
`65c677b3d566a2e9793d5f72999cdb441c6c0a9f`, verify our implementation is numerically the
same model, then retrain it from scratch so that later changes have an honest baseline.

## The architecture we adopted

A **49,393-neuron central-brain subset with 9,050,172 stored directed edges**, rate
dynamics and a released 1,024-token byte-level BPE. One recurrent step per token:

```
x = 0.1·x + 0.9·tanh(gain · (rec_gain · W·x + input) + bias)
```

Eight explicit token-delay slots each project into 1,758 neurons, supplying the current
token and the seven before it — 14,064 injected neurons. The output LayerNorm and readout
see all 49,393 states.

| Trainable component | Parameters | Share |
|---|---:|---:|
| Token embedding | 131,072 | 0.2% |
| Eight input projections | 1,800,192 | 3.4% |
| Neuron gain, recurrent gain, bias | 148,179 | 0.3% |
| Output LayerNorm | 98,786 | 0.2% |
| **Bias-free full readout** | **50,578,432** | **95.9%** |
| **Total** | **52,756,661** | |

The graph itself is a frozen buffer, not a parameter. **95.9% of everything the model
learns is the output matrix.** That observation is what Stages 3 and 4 exist to test.

The delay line supplies recent history *externally*. It is not learned eight-token
retention, and nothing here demonstrates the 128-word context the project originally
aspired to.

## What was verified, and what was not

[NGXSON_REPLICATION.md](NGXSON_REPLICATION.md) records the numerical work:

- Released-checkpoint inference matched the author's sample continuations: all 80
  generated token IDs identical on CPU, 89 teacher-forced positions with maximum logit
  difference 2.48e-5 and probability TV 2.32e-6.
- Full-model CPU/Apple-GPU gradient parity passed. A PyTorch 2.11 MPS `padding_idx` bug
  gave the PAD embedding a gradient; an instance-scoped hook restores CPU semantics.
  **Keep that safeguard when changing adapters.**

[NGXSON_QUALITY_AUDIT.md](NGXSON_QUALITY_AUDIT.md) records the honest gap. The author
released neither the training story IDs nor the full trainer. We reconstructed the
documented settings and made every remaining choice explicit. On a fresh 200-story
population the released model scored CE 3.9882 / 31.38% and our reconstruction scored
5.0362 / 26.58%. **Numerical parity is not training-quality reproduction, and we did not
achieve the latter.**

Fluent continuations from the reconstruction sometimes reproduce training passages
verbatim — 55/55 and 51/51 words in recorded cases. Read generated text as recall until
proven otherwise.

## Receipts

Configuration [`configs/ngxson-reference.json`](../configs/ngxson-reference.json);
accepted baseline with full selector hashes
[`records/ngxson-reference/accepted-baseline.json`](../records/ngxson-reference/accepted-baseline.json).
Retained winners are CE **4.692816 @ 3,500** and accuracy **35.1605% @ 24,100**; those are
two independent checkpoints, never one.
