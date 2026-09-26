"""
Stage 1: is catastrophic forgetting in the original EP caused by weight overlap?

Train list a (A -> B -> C) on a small offline `OriginalEqPropSequenceNetwork`,
then train list b (D -> E -> F) on the SAME weights (no reset). Expect:
  * behavioural CF   : recall of list a degrades after training list b;
  * mechanistic cause: the EP weight-update directions for list-b transitions
    OVERLAP those for list-a transitions, i.e. learning b rotates the very
    synapses a relies on.

The mechanistic side uses `synaptic_interference_matrix` from
`memval.metrics.capacity`: for each transition it recomputes the free/nudge EP
update it *would* apply at the current weights (`ep_update_signature`), flattens
(dW_ih, dW_ho) into a direction in weight space, and takes pairwise cosines.
High off-diagonal in the a<->b block == the two lists compete for synapses.
(The `memval.diagnostics.weights.update_interference` tier is the framework
version of the same computation, split per parameter group.)

Notes
-----
* This is a mechanistic *correlate* of CF, not proof of causation; the causal
  test is to REDUCE the a<->b block (DG separation) and show CF drops with it.
* `OriginalEqPropSequenceNetwork._settle` clips state to [0, 1], so we use
  non-negative unit codes (mixed-sign embeddings can't be represented at the
  output). The interference matrix lives in weight-update space, unaffected.
* Read J_init (all transitions un-learned, common reference) not J_post: once a
  transition is learned its update shrinks to ~0 and its direction is meaningless.
"""

import numpy as np

from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork
from memval.metrics.capacity import synaptic_interference_matrix


class ToyEncoder:
    """Non-negative unit codes, representable under the [0,1] state clip."""
    def __init__(self, symbols, dim, seed):
        rng = np.random.default_rng(seed)
        raw = np.abs(rng.standard_normal((len(symbols), dim)))
        self.codes = raw / np.linalg.norm(raw, axis=1, keepdims=True)
        self.symbols = list(symbols)
        self.idx = {s: i for i, s in enumerate(self.symbols)}

    def encode(self, seq):
        return np.stack([self.codes[self.idx[s]] for s in seq])

    def decode(self, vec):
        v = vec / (np.linalg.norm(vec) + 1e-12)
        return self.symbols[int(np.argmax(self.codes @ v))]


def one_step_fidelity(model, enc, transition):
    """Cosine between the model's one-step prediction and the true successor."""
    x, y = transition
    pred = model.predict_next(enc.encode([x])[0])
    pred = pred / (np.linalg.norm(pred) + 1e-12)
    tgt = enc.encode([y])[0]
    tgt = tgt / (np.linalg.norm(tgt) + 1e-12)
    return float(pred @ tgt)


def recall_accuracy(model, enc, seq):
    """Autoregressive recall from the first symbol; fraction of successors correct."""
    codes = enc.encode(seq)
    recalled = model.recall(codes[0:1], length=len(seq) - 1)
    decoded = [enc.decode(r) for r in recalled]
    correct = [d == t for d, t in zip(decoded, seq[1:])]
    return float(np.mean(correct)), decoded


def block_mean(J, rows, cols):
    return float(np.mean(J[np.ix_(rows, cols)]))


def main():
    symbols = ["A", "B", "C", "D", "E", "F"]
    list_a, list_b = ["A", "B", "C"], ["D", "E", "F"]
    trans_a = [("A", "B"), ("B", "C")]
    trans_b = [("D", "E"), ("E", "F")]
    transitions = trans_a + trans_b            # rows/cols 0,1 = a ; 2,3 = b
    a_idx, b_idx = [0, 1], [2, 3]

    dim = 64
    enc = ToyEncoder(symbols, dim=dim, seed=0)
    model = OriginalEqPropSequenceNetwork(
        n_features=dim, n_hidden=32, learning_rate=0.1, beta=0.5,
        n_settle_steps=50, n_epochs=250, seed=1)

    # structural overlap, before any training (order-independent, all un-learned)
    J_init = synaptic_interference_matrix(model, transitions, enc)

    # train list a, then list b on the SAME weights (no reset)
    model.fit_sequence(enc.encode(list_a))
    acc_a_after_a, dec_a1 = recall_accuracy(model, enc, list_a)
    fid_a_before = {t: one_step_fidelity(model, enc, t) for t in trans_a}

    model.fit_sequence(enc.encode(list_b))
    acc_a_after_b, dec_a2 = recall_accuracy(model, enc, list_a)
    acc_b_after_b, dec_b = recall_accuracy(model, enc, list_b)
    fid_a_after = {t: one_step_fidelity(model, enc, t) for t in trans_a}

    J_post = synaptic_interference_matrix(model, transitions, enc)
    labels = [f"{x}->{y}" for x, y in transitions]

    def show_matrix(name, J):
        print(f"\n{name} (cosine of EP update directions)")
        print("            " + "  ".join(f"{l:>6}" for l in labels))
        for i, l in enumerate(labels):
            print(f"  {l:>7}  " + "  ".join(f"{J[i, j]:6.2f}" for j in range(len(labels))))
        print(f"  within-a block (a<->a off-diag)  : {J[0,1]:.3f}")
        print(f"  within-b block (b<->b off-diag)  : {J[2,3]:.3f}")
        print(f"  a<->b block mean |cos|           : "
              f"{np.mean(np.abs(J[np.ix_(a_idx, b_idx)])):.3f}   "
              f"(signed mean {block_mean(J, a_idx, b_idx):+.3f})")

    print("=" * 68)
    print("BEHAVIOURAL — catastrophic forgetting")
    print("=" * 68)
    print(f"  list a recall after training a : {acc_a_after_a:.2f}   decoded {dec_a1}")
    print(f"  list a recall after training b : {acc_a_after_b:.2f}   decoded {dec_a2}   <-- CF")
    print(f"  list b recall after training b : {acc_b_after_b:.2f}   decoded {dec_b}")

    print("\n" + "=" * 68)
    print("MECHANISTIC — synaptic interference (weight-update overlap)")
    print("=" * 68)
    show_matrix("J_init  [at initialisation, order-independent]", J_init)
    show_matrix("J_post  [after training a then b]", J_post)
    n_params = 2 * model.n_hidden * dim
    print(f"\n  random-direction baseline |cos| ~ {np.sqrt(2.0/(np.pi*n_params)):.3f}  "
          f"(so a<->b overlap is many x chance)")

    print("\n" + "=" * 68)
    print("LINK — per-transition A forgetting vs its overlap with list b")
    print("=" * 68)
    print(f"  {'transition':>10}  {'fid_before':>10}  {'fid_after':>9}  "
          f"{'drop':>6}  {'b-overlap(init)':>15}")
    for k, t in enumerate(trans_a):
        drop = fid_a_before[t] - fid_a_after[t]
        b_ov = np.mean(np.abs(J_init[a_idx[k], b_idx]))
        print(f"  {f'{t[0]}->{t[1]}':>10}  {fid_a_before[t]:10.3f}  "
              f"{fid_a_after[t]:9.3f}  {drop:6.3f}  {b_ov:15.3f}")


if __name__ == "__main__":
    main()
