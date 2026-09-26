# Continual-Learning & Sequence-Memory Benchmark Landscape

> Literature survey compiled 2026-07 to keep the MemVal intro reviewer-proof.
> Five slices: (1) canonical classification CL, (2) sequential/temporal CL,
> (3) bio-inspired/hippocampal CL, (4) episodic/associative-memory evaluation,
> (5) CL libraries & metric standards. **Verify every citation against the
> reference library before submission** — confidence flags are noted per row.

## The one-sentence takeaway

The continual-learning evaluation ecosystem is **classification-centric by
construction** — its metrics (ACC/BWT/FWT/forgetting), its benchmarks
(Split-MNIST/CIFAR/Tiny-ImageNet, CORe50, CLEAR), its bio-inspired models (scored
on those same benchmarks), and its libraries (Avalanche, Continuum, Sequoia,
Mammoth, Renate) all measure **label prediction under distribution shift**, not
**cued recall of ordered content**. Sequence-memory evaluation, where it exists,
is **fragmented and per-model-class**. No existing benchmark scores *heterogeneous*
memory models on *pattern completion + disambiguation + one-shot + continual
retention* together under *controlled sequence structure*.

---

## 1. Canonical / classification-based CL benchmarks

| Benchmark | Origin (cite) | Models | Assays | Metrics | Datasets |
|---|---|---|---|---|---|
| Permuted-MNIST | Goodfellow et al. 2013 (arXiv 1312.6211); popularized by EWC, Kirkpatrick et al. 2017 PNAS | MLP→CNN; EWC/SI/GEM/replay | Classify 10 digits; per-task pixel permutation (Domain-IL) | ACC, sometimes BWT | MNIST, permuted |
| Rotated-MNIST | Lopez-Paz & Ranzato 2017 (GEM, NeurIPS) | MLP/CNN; GEM/A-GEM | Classify 10 digits; per-task rotation (Domain-IL) | ACC, BWT, FWT | MNIST, rotated |
| Split-MNIST | Zenke et al. 2017 (SI, ICML); scenarios: van de Ven & Tolias | MLP; SI/EWC/VCL/replay | 5 binary tasks; Task/Domain/Class-IL | ACC, forgetting | MNIST |
| Split-CIFAR-10/100 | Zenke et al. 2017 (SI); Rebuffi et al. 2017 (iCaRL) | ResNet; iCaRL/DER | Class splits, Task/Class-IL | ACC, forgetting, BWT/FWT | CIFAR-10/100 |
| Split-Tiny-ImageNet | **No canonical origin** — community standard | ResNet-18; replay/distillation | Class splits, Task/Class-IL | ACC, forgetting | Tiny-ImageNet |
| CORe50 | Lomonaco & Maltoni 2017 (CoRL) | CNN; latent/streaming replay | Object recognition, NI/NC/NIC scenarios | Run-averaged accuracy | CORe50 (128² RGB video) |
| CLEAR | Lin et al. 2021 (NeurIPS D&B) | CNN; supervised + semi-sup | Classification with real temporal concept drift | Accuracy; transfer matrix | CLEAR-10/100 (from YFCC100M) |
| Stream-51 | Roady et al. 2020 (CVPRW) | Streaming learners (ExStream, SLDA) | Streaming classification **+ novelty detection** | Streaming top-1; novelty AUROC | Stream-51 video |
| ImageNet-based CL | Rebuffi et al. 2017 (iCaRL) protocol | Deep CNN; iCaRL/LwF/BiC/DER | Large-scale Class-IL classification | Avg incremental accuracy; forgetting | ImageNet-100/1000 |

**Taxonomy/metrics:** Three scenarios (TIL/DIL/CIL) — van de Ven & Tolias 2019 →
van de Ven, Tuytelaars & Tolias 2022 (*Nat. Mach. Intell.* 4:1185–1197).
ACC/BWT/FWT — Lopez-Paz & Ranzato 2017 (GEM). Forgetting/Intransigence —
Chaudhry et al. 2018 (Riemannian Walk, ECCV). Survey literally scoped to
classification: De Lange et al. 2021, *"…Defying Forgetting in Classification
Tasks"* (TPAMI). **Every row above is classification-only** (Stream-51 adds
novelty detection — still not recall).

## 2. Sequential / temporal / recurrent CL

| Benchmark | Origin (cite) | Models | Assays | Metrics | Datasets | Structure varied |
|---|---|---|---|---|---|---|
| SSC / Quickdraw / PMNIST / SMNIST | Cossu, Carta, Lomonaco & Bacciu 2021, *Neural Networks* 143:607–627 (arXiv 2103.07492) | LSTM (+MLP baseline) | Whole-**sequence classification** | ACC | Synthetic Speech Commands (audio), Quickdraw (strokes), MNIST-as-sequence | **YES — sequence length** (28 vs 196 steps); single/multi-head |
| WESAD-CL / ASCERTAIN-CL | Matteoni, Cossu et al. 2022 (ESANN; arXiv 2207.00010) | GRU (2×18) | Affect/stress **sequence classification** | Avg accuracy | Wearable physiological time-series | Domain-IL by subject; none |
| **sino7 (20-trajectory)** | Annabi, Pitti & Quoy 2022 (*Front. Neurorobotics*) | ESN, Conceptors, PC-RNN, EWC, ESN+replay | **RECALL / generation** of ordered trajectory from index cue | Prediction MSE | 20 continuous trajectories (mocap, handwriting), len 60 | Fixed; recurrent vs input vs output weights |
| Lifelong Language Learning | d'Autume et al. 2019 (NeurIPS; arXiv 1906.01076) | Transformer + episodic memory | Text **classification** stream | Accuracy | 5 text datasets → 33 classes | Task order only |
| LAMOL | Sun, Ho & Lee 2020 (ICLR; arXiv 1909.03329) | GPT-2 | Classification/QA as **next-token generation** | Per-task accuracy | 5 text sets; decaNLP | Task order |
| HAR CL | Jha et al. 2021 (*Info. Sciences*; arXiv 2104.09396) | Deep sequence nets | HAR **sequence classification** | Accuracy/forgetting | Wearable/IMU | Class/task splits |
| Online CL SLR (landscape) | Rezaee et al. 2025 (arXiv 2501.04897) | ResNet18 (62/81); RNN nearly absent | Online **classification/detection** | 83 datasets reviewed | ~80–90% static image; OAK egocentric video | Temporal structure not a variable |

**Verdict:** sequential CL is overwhelmingly **sequence classification** (label
out). **sino7 (Annabi 2022) is the lone recall exception.** Only **Cossu 2021**
systematically varies a structural property (**length**); none varies inter-event
interval, noise, or sub-sequence overlap.

## 3. Bio-inspired / hippocampal / dual-memory CL

| Model | Origin (cite) | Architecture | Evaluation assays | Metrics | Datasets | Bio-assay in eval? |
|---|---|---|---|---|---|---|
| HiCL | Kapoor et al. 2025 (AAAI 39(27):22518; arXiv 2508.16651) | Grid→DG (top-k sep)→CA3 autoassoc; DG-gated MoE | **Class/Task-IL classification** | Accuracy, forgetting, BWT | Split-MNIST/CIFAR, Tiny-ImageNet | **No** |
| H2C | Wu et al. 2025 (arXiv 2508.01158) | DG-"separation" + CA3-"completion" replay buffers on trajectory predictor | **Trajectory prediction (regression)** | FDE, Miss Rate, FDE-BWT | INTERACTION driving | **No** |
| CH-HNN | Shi et al. 2025 (*Nat. Commun.* 16, 56405) | ANN(mPFC-CA1)+SNN(DG-CA3) hybrid | **Task/Class-IL classification** | Avg accuracy; inter-episode disparity | Split-MNIST/CIFAR-100, Tiny-ImageNet, DVS-Gesture | **No** |
| FearNet | Kemker & Kanan 2018 (ICLR; arXiv 1711.10563) | HC store + mPFC autoencoder + BLA gate; pseudorehearsal | **Class-incremental classification** | Ω_base/new/all | CIFAR-100, CUB-200, AudioSet | **No** |
| GDM | Parisi et al. 2018 (*Front. Neurorobotics*; arXiv 1805.10966) | Growing dual self-organizing (episodic→semantic) | **Object-recognition classification** (NI/NC/NIC) | Instance/category accuracy | CORe50 | **No** |
| Brain-inspired replay | van de Ven et al. 2020 (*Nat. Commun.* 11:4069) | Generative hidden-representation replay | **Task/Class-IL classification** | Final accuracy | Split-MNIST, Permuted-MNIST, Split-CIFAR-100 | **No** |
| Deep Generative Replay | Shin et al. 2017 (NeurIPS; arXiv 1705.08690) | Generator + solver pseudo-rehearsal | **Sequential classification** | Accuracy | Permuted-MNIST, MNIST↔SVHN | **No** |
| CLS theory (foundation) | Kumaran, Hassabis & McClelland 2016 (*TiCS* 20:512) | Theory (HC fast / neocortex slow) | — | — | — | N/A (concepts live here) |
| Humans vs NNs interference | Braun et al. 2025 (*Nat. Hum. Behav.* 02318-y) | Behavioral study (humans + linear ANN) | **Behavioral rule-learning** (assays ARE the eval) | Transfer; interference (von Mises) | Behavioral stimuli | **Yes** — but cognitive-science study, not a CL model |

**Verdict:** biology reaches the **architecture** (DG separation, CA3
completion, dual memory, generative replay) but **not the evaluation** — every
model is scored on Split-MNIST/CIFAR/Tiny-ImageNet (or trajectory regression).
Behavioral assays appear only as *motivation* or in *cognitive-science* papers.
**Watch:** Jun et al. 2025 (arXiv 2507.11393, "A Neural Network Model of CLS:
Pattern Separation and Completion for Continual Learning") is the closest
candidate to a bio-assay-in-eval model — **read the PDF directly before citing**;
its datasets/whether completion is *scored* were not verifiable.

## 4. Episodic / associative / sequence-memory evaluation (closest to MemVal)

| Name | Origin (cite) | Model class | Assays | Metrics | Datasets/stimuli |
|---|---|---|---|---|---|
| Hopfield capacity | Hopfield 1982; Amit-Gutfreund-Sompolinsky 1985 | Binary attractor | Storage + retrieval from corrupted cue | Capacity (~0.138N); basin size; overlap | Synthetic random ±1 patterns |
| Modern Hopfield | Ramsauer et al. 2021 (ICLR; arXiv 2008.02217) | Continuous Hopfield = attention | Exp-capacity storage; one-step retrieval; downstream tasks | Capacity, retrieval error, separation Δ | Synthetic + real ML datasets |
| Dense Associative Memory | Krotov & Hopfield 2016 (NeurIPS); 2021 | Higher-order associative | Error-free storage; noisy recall | Capacity scaling; recall from bit-flips | Synthetic binary; MNIST demos |
| Long Sequence Hopfield Memory | Chaudhry et al. 2023 (NeurIPS; arXiv 2306.04532) | Sequential Hopfield | Sequence storage + ordered recall; correlated patterns | Sequence capacity; recall error | Synthetic (incl. correlated) sequences |
| DG/CA3 separation-completion | Neunuebel & Knierim 2014 (Neuron); circuit models | Biophysical/rate circuits | Pattern separation & completion | Input-output correlation/overlap curves | Synthetic overlapping patterns; place data |
| Info-theoretic separation | Bird, Cuntz & Jedlicka 2024 (PLoS CB; PMC10906873) | DG models + Hopfield | Separation quantification | Sparsity-weighted MI; transfer entropy | Controlled-correlation spike ensembles |
| Disambiguation paradigm (A-B-X) | Agster, Fortin & Eichenbaum 2002 (J Neurosci) | Behavioral (rats) + models | Disambiguate overlapping sequences at shared item | Choice accuracy at ambiguous item | Structured overlapping odor sequences |
| CSCG / CHMM | George et al. 2021 (*Nat. Commun.* 12:2392) | Cloned HMM | Disambiguation of aliased obs; transitive inference | Next-obs prediction; graph recovery | Synthetic action-obs on structured graphs |
| TEM | Whittington et al. 2020 (Cell) | HC-EC relational net | Structural generalization; next-state prediction | Prediction accuracy; cell-type match | Synthetic random walks on graphs |
| MANN one-shot | Santoro et al. 2016 (ICML; arXiv 1605.06065) | External-memory net | One/few-shot bind-and-recall | nth-shot accuracy | Omniglot; synthetic sequences |
| NTM / DNC | Graves et al. 2014/2016 (Nature) | Differentiable memory | Copy, associative recall, bAbI QA | Bit/sequence error; QA accuracy | Synthetic algorithmic; bAbI |
| MQAR / Long Range Arena | Arora et al. 2023 (MQAR); Tay et al. 2021 (LRA, ICLR) | Transformers/SSMs/RNNs | Key→value recall; long-range classification | Recall accuracy; LRA per-task | Synthetic key-value streams (controlled); LRA mix |
| LLM Episodic Memory Benchmark | Huet et al. 2025 (ICLR; arXiv 2501.13121) | LLMs | Event + spatio-temporal recall | QA accuracy vs #events/complexity | LLM-generated synthetic narratives (controlled) |

**Verdict — the novelty crux:** evaluation here is **fragmented per model-class**.
Pattern completion/capacity → synthetic random patterns, Hopfield-only.
Separation/completion → DG/CA3 overlap curves, circuit-only. Disambiguation →
behavioral paradigms + bespoke per-model graphs (CSCG, TEM). One-shot → Omniglot /
NTM-DNC / bAbI. Controlled *synthetic sequence* benchmarks **do exist** (MQAR,
LRA, Long Sequence Hopfield, LLM Episodic Memory) — but each scores a **single
competence family** on a **single model class**. **None unifies completion +
disambiguation + one-shot + continual retention across heterogeneous models.**

## 5. CL libraries & metric standards

| Library/Standard | Origin (cite) | Assays supported | Metrics | Modalities | Recall? |
|---|---|---|---|---|---|
| Avalanche | Lomonaco et al. 2021 (CVPRW); JMLR 24 (2023) | Classification (core) + detection/segmentation; RL separate | ACC, forgetting, BWT, FWT, efficiency | Vision-dominant | **No** |
| Avalanche RL | Lucchesi et al. 2022 (arXiv 2202.13657) | Continual RL | Reward/return | Gym, Habitat | **No** |
| Continuum | Douillard & Lesort 2021 (arXiv 2102.06253) | Data loaders: NI/NC/NIC classification | accuracy/BWT/FWT helper | Image | **No** |
| Sequoia | Normandin et al. 2021 (arXiv 2108.01005) | Continual SL + Continual RL settings | SL accuracy; RL reward | Vision + RL | **No** |
| Mammoth | Buzzega et al. 2020 (DER, NeurIPS); Boschini et al. 2022 (TPAMI) | Class/Task/Domain/General-IL classification | Accuracy, forgetting, transfer | Image only | **No** |
| Renate | Wistuba et al. 2023 (Amazon Science) | Production retraining, classification | Accuracy/loss; forgetting | Vision + tabular/transformer | **No** |
| ACC/BWT/FWT | Lopez-Paz & Ranzato 2017 (GEM) | Classification task sequences | ACC/BWT/FWT | — | **No** |
| "Don't forget…" metrics | Díaz-Rodríguez et al. 2018 (arXiv 1810.13166) | Adds efficiency axes | +MS/SSS/compute | Vision | **No** |
| CLEVA-Compass | Mundt et al. 2022 (ICLR; arXiv 2110.03331) | Reporting standard | Prescribes reporting | Agnostic | **No** |

**Terminology trap for the paper:** in this literature "episodic memory" (GEM,
A-GEM, episodic replay) means a **rehearsal buffer mechanism**, NOT an
associative-recall *task*. The infrastructure operationalizes memory as *storage
for replay*, never as *recall/completion to be scored*. That distinction is the
crux of MemVal's gap claim.

---

## How to state the gap (reviewer-proof)

**Safe claims (well-supported):**
- CL metrics (ACC/BWT/FWT/forgetting) exist and are standard — MemVal *adopts*
  them, does not claim to invent them.
- Canonical CL benchmarks and all major libraries are classification-centric;
  none offers sequence recall / pattern completion / associative memory as an
  evaluable task type.
- Sequential CL is ~all sequence *classification*; only Cossu 2021 varies a
  structural property (length); sino7 (Annabi 2022) is the lone recall exception.
- Bio-inspired CL models put biology in the architecture, not the evaluation —
  scored on Split-MNIST/CIFAR/Tiny-ImageNet.
- Episodic/associative-memory evaluation is fragmented and per-model-class.

**Do NOT claim:**
- "First synthetic / structurally-controlled sequence benchmark" — MQAR, LRA,
  Long Sequence Hopfield, LLM Episodic Memory Benchmark already use controlled
  synthetic sequences.
- "No one measures forgetting gracefully" — BWT/forgetting are standard.

**MemVal's defensible novelty = the intersection nobody occupies:**
(a) **model-agnostic** scoring across *heterogeneous* classes (attractor / EqProp
/ online / spiking) on one footing, and (b) **joint** coverage of *pattern
completion + disambiguation + one-shot + continual retention* under *controlled
sequence structure* (length, ITI, noise, overlap). Each piece exists in a silo;
the unification does not.
