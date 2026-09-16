# Shared-population comparison: released reference and preserved arms

Inference only on Fernandos-MacBook-Pro-2.local (mps). No optimizer, no backward pass.
The reserved 100-story test was not opened.

Both populations are scored for every arm so one table is comparable. The
validation split selected our checkpoints, so it flatters our arms and not the
released model; the audit split selected none of them. Read the audit column
for the selection-unbiased comparison.

| Arm | Trainable parameters | Validation CE | Validation acc | Audit CE | Audit PPL | Audit acc |
|---|---:|---:|---:|---:|---:|---:|
| released-reference | 52,756,661 | 3.680932 | 35.2428% | 3.988231 | 53.96 | 31.3833% |
| reconstruction/min-ce | 52,756,661 | 4.692816 | 30.5797% | 5.036168 | 153.88 | 26.5807% |
| A128fixed/min-ce | 52,756,661 | 4.728279 | 30.4883% | 5.050346 | 156.08 | 27.4795% |
| B32fixed/min-ce | 51,308,213 | 4.512509 | 31.5260% | 4.912468 | 135.97 | 27.7237% |
| E32rank128fixed/min-ce | 7,183,157 | 3.149704 | 36.4817% | 3.373062 | 29.17 | 32.6350% |
| G32rank64fixed/min-ce | 3,956,469 | 3.093705 | 36.3674% | 3.280159 | 26.58 | 33.3740% |
| G32rank64fixed/max-accuracy | 3,956,469 | 3.103450 | 36.8885% | 3.272451 | 26.38 | 33.4961% |
| H32rank32fixed/min-ce | 2,343,125 | 3.137654 | 35.4713% | 3.309095 | 27.36 | 32.5795% |

## Paired story bootstrap against the released reference

Direction is arm minus released; 10,000 paired whole-story resamples, seed 1729.

| Arm | Split | ΔCE | ΔCE 95% CI | Δaccuracy | Δaccuracy 95% CI |
|---|---|---:|---|---:|---|
| reconstruction/min-ce | validation | +1.011885 | [+0.9660, +1.0590] | -4.6631 pp | [-5.238, -4.094] pp |
| reconstruction/min-ce | audit | +1.047937 | [+1.0148, +1.0813] | -4.8026 pp | [-5.214, -4.369] pp |
| A128fixed/min-ce | validation | +1.047348 | [+0.9954, +1.0992] | -4.7545 pp | [-5.423, -4.104] pp |
| A128fixed/min-ce | audit | +1.062115 | [+1.0268, +1.0980] | -3.9038 pp | [-4.314, -3.499] pp |
| B32fixed/min-ce | validation | +0.831578 | [+0.7899, +0.8726] | -3.7167 pp | [-4.372, -3.054] pp |
| B32fixed/min-ce | audit | +0.924237 | [+0.8896, +0.9586] | -3.6596 pp | [-4.005, -3.306] pp |
| E32rank128fixed/min-ce | validation | -0.531227 | [-0.5761, -0.4840] | +1.2389 pp | [+0.697, +1.778] pp |
| E32rank128fixed/min-ce | audit | -0.615169 | [-0.6427, -0.5877] | +1.2517 pp | [+0.864, +1.658] pp |
| G32rank64fixed/min-ce | validation | -0.587227 | [-0.6355, -0.5372] | +1.1246 pp | [+0.474, +1.770] pp |
| G32rank64fixed/min-ce | audit | -0.708072 | [-0.7379, -0.6787] | +1.9907 pp | [+1.613, +2.368] pp |
| G32rank64fixed/max-accuracy | validation | -0.577481 | [-0.6279, -0.5262] | +1.6458 pp | [+1.033, +2.263] pp |
| G32rank64fixed/max-accuracy | audit | -0.715780 | [-0.7464, -0.6857] | +2.1128 pp | [+1.697, +2.535] pp |
| H32rank32fixed/min-ce | validation | -0.543278 | [-0.5931, -0.4921] | +0.2286 pp | [-0.421, +0.870] pp |
| H32rank32fixed/min-ce | audit | -0.679137 | [-0.7108, -0.6471] | +1.1962 pp | [+0.777, +1.616] pp |

A negative ΔCE means the arm assigns higher probability to held-out text than the
released 52,756,661-parameter reference. That is a measurement on this corpus, not
evidence of an anatomical prior: the released model's own training stories are
unknown and may overlap either population.

