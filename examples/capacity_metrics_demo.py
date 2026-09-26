"""
Minimal, isolated demo of the capacity / stable-state metrics in
``memval.metrics.capacity`` on a tiny OriginalEqProp associator.

It trains one small model on two category-pure sequences (fruits, animals),
gives the encoder a handful of *untrained* distractor words, and then produces a
single 2x2 figure:

    (0,0) behavioural crosstalk matrix        -> which memories collide
    (0,1) synaptic interference matrix        -> why they collide (shared synapses)
    (1,0) channel mutual information vs noise  -> how many bits survive
    (1,1) attractor scan norm traces          -> where recall goes instead

Run:  python examples/capacity_metrics_demo.py
Output: examples/capacity_metrics_demo.png  (+ a text summary on stdout)
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from memval.encoders.symbolic import SymbolicDecoder, SymbolicEncoder
from memval.metrics.capacity import (
    attractor_scan,
    category_information,
    channel_mi,
    crosstalk_matrix,
    diagonal_dominance,
    recall_confusion,
    synaptic_interference_matrix,
)
from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork

SEED = 7
DIM = 48
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "capacity_metrics_demo.png")


def build_world():
    """Vocab with two trained categories plus untrained distractors."""
    vocab = {
        # trained
        "apple": "fruit", "banana": "fruit", "orange": "fruit", "grape": "fruit", "pear": "fruit",
        "cat": "animal", "dog": "animal", "cow": "animal", "horse": "animal", "sheep": "animal",
        # never trained -> landing here during recall == spurious
        "truck": "vehicle", "car": "vehicle", "red": "color", "blue": "color",
        "hammer": "tool", "one": "number",
    }
    encoder = SymbolicEncoder(vocab, embedding_dim=DIM, category_variance=0.2, seed=SEED)
    decoder = SymbolicDecoder(encoder)
    seq_fruit = ["apple", "banana", "orange", "grape", "pear"]
    seq_animal = ["cat", "dog", "cow", "horse", "sheep"]
    return vocab, encoder, decoder, seq_fruit, seq_animal


def train():
    vocab, encoder, decoder, seq_fruit, seq_animal = build_world()
    model = OriginalEqPropSequenceNetwork(
        n_features=DIM, n_hidden=96, learning_rate=0.08, beta=0.5,
        n_settle_steps=20, n_epochs=250, dt=0.5, seed=SEED,
    )
    # sequential exposure: fruits, then animals (a mild forgetting setup)
    model.fit_sequence(encoder.encode(seq_fruit))
    model.fit_sequence(encoder.encode(seq_animal))

    stored_words = seq_fruit + seq_animal
    transitions = (
        [(seq_fruit[i], seq_fruit[i + 1]) for i in range(len(seq_fruit) - 1)]
        + [(seq_animal[i], seq_animal[i + 1]) for i in range(len(seq_animal) - 1)]
    )
    return dict(
        vocab=vocab, encoder=encoder, decoder=decoder, model=model,
        stored_words=stored_words, transitions=transitions,
    )


def main():
    rng = np.random.default_rng(SEED)
    W = train()
    encoder, decoder, model = W["encoder"], W["decoder"], W["model"]
    transitions = W["transitions"]
    cues = [c for c, _ in transitions]
    targets = [t for _, t in transitions]
    labels = [f"{c}→{t}" for c, t in transitions]

    # --- a) crosstalk (behavioural) ---
    S = crosstalk_matrix(model, cues, targets, encoder)
    dom = diagonal_dominance(S)

    # --- a) synaptic interference (mechanistic) ---
    J = synaptic_interference_matrix(model, transitions, encoder)
    n_fruit = len(W["transitions"]) // 2  # split index for the two category blocks
    off_block = np.mean(np.abs(J[:n_fruit, n_fruit:]))

    # --- c) channel MI vs noise ---
    noise_levels = np.linspace(0.0, 1.0, 9)
    mi_item, mi_cat, eff = [], [], []
    for ns in noise_levels:
        conf = recall_confusion(model, transitions, encoder, decoder,
                                noise_scale=float(ns), n_trials=60, rng=rng)
        info = category_information(conf, W["vocab"])
        overall = channel_mi(conf["counts"])
        mi_item.append(info["mi_item_bits"])
        mi_cat.append(info["mi_category_bits"])
        eff.append(overall["efficiency"])
    mi_item, mi_cat, eff = map(np.asarray, (mi_item, mi_cat, eff))
    mi_max = mi_item[0] if mi_item[0] > 0 else 1.0
    crit_idx = np.argmax(mi_item < 0.5 * mi_max)
    crit_noise = noise_levels[crit_idx] if np.any(mi_item < 0.5 * mi_max) else np.nan

    # --- b) attractor scan ---
    seeds = []
    for w in W["stored_words"]:
        base = encoder.encode([w])[0]
        for _ in range(6):
            seeds.append(base + rng.normal(0.0, 0.25, size=DIM))
    seeds = np.asarray(seeds)
    scan = attractor_scan(model, seeds, decoder, W["stored_words"], max_steps=40)

    # ------------------------------------------------------------------ plots
    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Capacity / stable-state metrics — OriginalEqProp (fruits + animals)",
                 fontsize=15, fontweight="bold")

    # (0,0) crosstalk heatmap
    a = ax[0, 0]
    im = a.imshow(S, cmap="magma", vmin=-0.2, vmax=1.0, aspect="auto")
    a.set_xticks(range(len(labels))); a.set_xticklabels(labels, rotation=90, fontsize=8)
    a.set_yticks(range(len(labels))); a.set_yticklabels(labels, fontsize=8)
    a.axhline(n_fruit - 0.5, color="cyan", lw=1); a.axvline(n_fruit - 0.5, color="cyan", lw=1)
    a.set_title(f"a) Behavioural crosstalk  (diag-dominance = {dom:+.2f})")
    a.set_xlabel("scored against target"); a.set_ylabel("cue")
    fig.colorbar(im, ax=a, fraction=0.046, label="cosine")

    # (0,1) synaptic interference heatmap
    a = ax[0, 1]
    im = a.imshow(J, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    a.set_xticks(range(len(labels))); a.set_xticklabels(labels, rotation=90, fontsize=8)
    a.set_yticks(range(len(labels))); a.set_yticklabels(labels, fontsize=8)
    a.axhline(n_fruit - 0.5, color="k", lw=1); a.axvline(n_fruit - 0.5, color="k", lw=1)
    a.set_title(f"a) Synaptic interference  (mean |cross-block| = {off_block:.2f})")
    fig.colorbar(im, ax=a, fraction=0.046, label="cos(ΔW)")

    # (1,0) channel MI vs noise
    a = ax[1, 0]
    a.plot(noise_levels, mi_item, "o-", label="I(item) bits", color="#1f77b4")
    a.plot(noise_levels, mi_cat, "s--", label="I(category) bits", color="#ff7f0e")
    a.axhline(0.5 * mi_max, color="gray", ls=":", lw=1, label="½·I_max")
    if not np.isnan(crit_noise):
        a.axvline(crit_noise, color="red", ls=":", lw=1,
                  label=f"σ_c ≈ {crit_noise:.2f}")
    a.set_xlabel("cue noise σ"); a.set_ylabel("mutual information (bits)")
    a.set_title("c) Retained information vs noise")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # (1,1) attractor norm traces (colour = lands in the dominant basin or not)
    a = ax[1, 1]
    dom_w = scan["dominant_word"]
    for rec, trace in zip(scan["records"], scan["norm_traces"]):
        in_dom = rec["endpoint_word"] == dom_w
        a.plot(trace, color=("#d62728" if in_dom else "#2ca02c"), alpha=0.5, lw=1)
    a.axhline(1.0, color="k", ls="--", lw=1, label="unit sphere (stored norm)")
    a.plot([], [], color="#d62728", label=f"collapses to '{dom_w}'")
    a.plot([], [], color="#2ca02c", label="other endpoint")
    a.set_xlabel("recall step k"); a.set_ylabel("‖x_k‖")
    a.set_title(f"b) Attractor scan — {scan['n_distinct_attractors']} attractor(s); "
                f"{scan['dominant_basin_frac']:.0%} of chains collapse to '{dom_w}' "
                f"(spurious {scan['spurious_rate']:.0%})")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=130)
    print(f"\nsaved figure -> {OUT_PNG}")

    # ------------------------------------------------------------------ text
    print("\n=== summary ===")
    print(f"transitions stored     : {len(transitions)}")
    print(f"diagonal dominance     : {dom:+.3f}   (>0 => separable memories)")
    print(f"mean |cross-block ΔW|  : {off_block:.3f}   (interference DG should shrink)")
    print(f"clean I(item)          : {mi_item[0]:.2f} bits  of max {np.log2(len(set(targets))):.2f}")
    print(f"clean I(category)      : {mi_cat[0]:.2f} bits")
    print(f"critical noise σ_c     : {crit_noise:.2f}")
    print(f"spurious attractor rate: {scan['spurious_rate']:.0%}")
    print(f"distinct attractors    : {scan['n_distinct_attractors']} "
          f"({scan['n_spurious_attractors']} spurious)")
    print(f"dominant basin         : '{scan['dominant_word']}' swallows "
          f"{scan['dominant_basin_frac']:.0%} of chains (black-hole collapse)")


if __name__ == "__main__":
    main()
