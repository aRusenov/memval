| capacity | metric | AHN | theta | tPC | recirc | EP | DTS-ESN |
|---|---|---|---|---|---|---|---|
| Continual retention | ΔMRR on A after B (0 = none) ↑ | +0.000 | +0.000 | +0.000 | +0.000 | -0.333 | +0.000 |
|  | chain ACC (6 tasks) ↑ | 1.000 | 1.000 | 0.667 | 0.667 | 0.250 | 1.000 |
|  | chain avg forgetting ↓ | 0.000 | 0.000 | 0.400 | 0.400 | 0.900 | 0.000 |
|  | AB retention after AC ↑ | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
|  | cue-competition cost ↓ | 1.000 | 1.000 | 1.000 | 1.000 | 0.800 | 1.000 |
|  | reversal trials to criterion ↓ | 2.0 | 2.0 | 8.0 | 1.0 | 3.0 | 12.0 |
|  | final perseveration ↓ | 0.000 | 0.000 | 0.000 | 0.375 | 0.000 | 0.625 |
| One-shot learning | baseline exposure (epochs) ↓ | 1 | 1 | 38 | 7 | 12 | 1 |
|  | epochs to convergence ↓ | 1 | 1 | 64 | 8 | 16 | 1 |
|  | MRR at convergence ↑ | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
|  | MRR, one streamed pass ↑ | — | 1.000 | 0.000 | 0.167 | 0.333 | 1.000 |
|  | MRR, 20 streamed passes ↑ | — | 1.000 | 0.833 | 1.000 | 1.000 | 1.000 |
|  | schema-consistency x speed (corr) ↓ | -0.199 | -0.199 | — | — | +0.153 | — |
| Pattern completion | σ-tolerance ↑ | 0.800 | 0.800 | 0.800 | 0.400 | 0.200 | 0.400 |
|  | mask tolerance (random) ↑ | 0.917 | 0.917 | 0.917 | 0.000 | 0.833 | 0.917 |
|  | mask tolerance (identity) ↑ | 0.417 | 0.417 | 0.417 | 0.000 | 0.417 | 0.417 |
|  | mask tolerance (category) ↑ | 0.917 | 0.917 | 0.917 | 0.000 | 0.833 | 0.917 |
|  | route coverage from a fragment ↑ | 0.818 | 0.727 | 1.000 | 1.000 | 0.909 | 0.909 |
| Sequence disambiguation | confusable episodes above chance ↑ | 8 | 8 | 0 | 4 | 8 | 8 |
|  | discriminator-similarity crossing ↑ | 0.223 | 0.223 | 0.104 | 0.104 | 0.104 | — |
|  | delay the discriminator survives ↑ | 0 | 0 | 0 | — | — | 6 |
|  | accuracy from carried state alone ↑ | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 | 0.750 |
|  | overlap cost vs orthogonal control ↓ | 0.500 | 0.500 | 0.000 | -0.125 | 0.250 | 0.000 |
|  | exposure cost at high similarity ↓ | 36.0x | 36.0x | 6.7x | censored | 8.5x | 3.0x |
|  | T-maze branch accuracy ↑ | 1.000 | 1.000 | 0.750 | 0.688 | 0.562 | 1.000 |
| Serial order | max memory span (rollout) ↑ | 6.0 | 6.0 | 5.0 | 2.0 | 2.0 | 4.0 |
|  | max span, quantized feedback ↑ | 10.0 | 10.0 | 10.0 | 2.0 | 10.0 | 10.0 |
|  | unrolling gap at criterion ↓ | 0.310 | 0.310 | 0.652 | 0.672 | 0.672 | 0.468 |
|  | unrolling exposure ratio ↓ | 6.6x | 6.6x | 2.8x | 1.9x | 1.0x | 1.0x |
|  | length at which order breaks ↑ | 11 | 11 | 11 | 5 | 11 | 11 |
