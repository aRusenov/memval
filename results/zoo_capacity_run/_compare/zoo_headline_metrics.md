| capacity | metric | AHN | theta | EP | spiking EP | tPC | DTS-ESN | Vieth STDP | BCPNN (columnar) |
|---|---|---|---|---|---|---|---|---|---|
| Continual retention | ΔMRR on A after B (0 = none) ↑ | +0.000 | +0.000 | -0.333 | +0.000 | +0.000 | +0.000 | -0.206 | -0.583 |
|  | chain ACC (6 tasks) ↑ | 1.000 | 1.000 | 0.250 | 0.042 | 0.667 | 1.000 | 0.042 | 0.708 |
|  | chain avg forgetting ↓ | 0.000 | 0.000 | 0.900 | 0.000 | 0.400 | 0.000 | 0.027 | 0.350 |
|  | AB retention after AC ↑ | 0.000 | 0.000 | 0.000 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 |
|  | cue-competition cost ↓ | 1.000 | 1.000 | 0.800 | 0.000 | 1.000 | 1.000 | — | 0.000 |
|  | reversal trials to criterion ↓ | 2.0 | 2.0 | 3.0 | 1.0 | 9.0 | 12.0 | 1.0 | 1.0 |
|  | final perseveration ↓ | 0.000 | 0.000 | 0.000 | 1.000 | 0.125 | 0.625 | 0.000 | 0.250 |
| One-shot learning | baseline exposure (epochs) ↓ | 1 | 1 | 12 | 512 | 38 | 1 | 512 | 5 |
|  | epochs to convergence ↓ | 1 | 1 | 16 | — | 64 | 1 | — | — |
|  | MRR at convergence ↑ | 1.000 | 1.000 | 1.000 | 0.333 | 1.000 | 1.000 | 0.028 | 0.794 |
|  | MRR, one streamed pass ↑ | — | 1.000 | 0.333 | 0.167 | 0.000 | 1.000 | 0.000 | — |
|  | MRR, 20 streamed passes ↑ | — | 1.000 | 1.000 | 0.167 | 0.833 | 1.000 | 0.000 | — |
|  | schema-consistency x speed (corr) ↓ | -0.199 | -0.199 | +0.253 | — | — | — | — | — |
| Pattern completion | σ-tolerance ↑ | 0.800 | 0.800 | 0.200 | 0.000 | 0.800 | 0.400 | 0.000 | — |
|  | mask tolerance (random) ↑ | 0.917 | 0.917 | 0.833 | 0.000 | 0.917 | 0.917 | 0.000 | — |
|  | mask tolerance (identity) ↑ | 0.417 | 0.417 | 0.417 | 0.000 | 0.417 | 0.417 | 0.000 | — |
|  | mask tolerance (category) ↑ | 0.917 | 0.917 | 0.833 | 0.000 | 0.917 | 0.917 | 0.000 | — |
|  | route coverage from a fragment ↑ | 0.818 | 0.727 | 0.909 | 0.000 | 0.909 | 0.909 | 0.000 | — |
| Sequence disambiguation | confusable episodes above chance ↑ | 8 | 8 | 8 | 0 | 0 | 8 | 0 | — |
|  | discriminator-similarity crossing ↑ | 0.223 | 0.223 | 0.104 | 0.104 | 0.104 | — | 0.104 | — |
|  | delay the discriminator survives ↑ | 0 | 0 | — | — | 0 | 6 | — | — |
|  | accuracy from carried state alone ↑ | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 | 0.750 | 0.250 | — |
|  | overlap cost vs orthogonal control ↓ | 0.500 | 0.500 | 0.250 | 0.000 | 0.000 | 0.000 | 0.000 | — |
|  | exposure cost at high similarity ↓ | 36.0x | 36.0x | 8.5x | censored | 6.7x | 3.0x | censored | — |
|  | T-maze branch accuracy ↑ | 1.000 | 1.000 | 0.500 | 0.500 | 1.000 | 1.000 | 0.500 | — |
| Serial order | max memory span (rollout) ↑ | 6.0 | 6.0 | 2.0 | 2.0 | 5.0 | 4.0 | 1.0 | 2.0 |
|  | max span, quantized feedback ↑ | 10.0 | 10.0 | 10.0 | 2.0 | 10.0 | 10.0 | 1.0 | 2.0 |
|  | unrolling gap at criterion ↓ | 0.310 | 0.310 | 0.672 | — | 0.652 | 0.468 | 0.500 | 0.000 |
|  | unrolling exposure ratio ↓ | 6.6x | 6.6x | 1.0x | censored | 2.8x | 1.0x | 1.1x | — |
|  | length at which order breaks ↑ | 11 | 11 | 11 | 0 | 11 | 11 | 0 | — |
