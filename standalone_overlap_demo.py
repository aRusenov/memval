"""What "overlap" between two sequences means, and why it is the forgetting lever.

A linear associator learns W by  W += lr * (target - W x) x^T.  A training step on
a list-B item x_B disturbs list-A's recall by exactly

    dW x_A = lr * err_B * (x_B . x_A)

so the damage B does to A is proportional to the INNER PRODUCT of their inputs.
Orthogonal inputs cannot interfere, whatever the budget. Three views of that:
(1) the A x B cosine matrix for the shipped pair vs a same-category pair;
(2) the fraction of each B item that lies inside span(A);
(3) a synthetic dial: rotate B into A's subspace by alpha and watch A forget.
"""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from memval.benchmarks.symbolic_pipeline import load_vocab, measure_recall_associative, mean_recall_rate
from memval.benchmarks.probe import measure_recall_margin, mean_margin
from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.models.baselines import ThetaPhaseSequenceNetwork as T

vocab = load_vocab(); fruit = [w for w, c in vocab.items() if c == "fruit"]; animal = [w for w, c in vocab.items() if c == "animal"]
DIM, CV = 100, 0.2

def geometry(A, B):
    enc = SymbolicEncoder({w: vocab[w] for w in A + B}, embedding_dim=DIM, category_variance=CV, seed=42)
    XA, XB = enc.encode(A), enc.encode(B)
    nA, nB = XA / np.linalg.norm(XA, axis=1, keepdims=True), XB / np.linalg.norm(XB, axis=1, keepdims=True)
    cos = nA @ nB.T                                   # (|A|, |B|)
    Q, _ = np.linalg.qr(XA.T)                         # orthonormal basis of span(A)
    in_span = np.linalg.norm(Q.T @ XB.T, axis=0) / np.linalg.norm(XB, axis=1)   # per B item
    return enc, XA, XB, cos, in_span

def forget_with(enc, A, XA, XB, ep=4, epB=4):
    dec = SymbolicDecoder(enc)
    m = T(n_features=DIM, learning_rate=0.1, n_epochs=ep); m.fit_sequence(XA, epochs=ep); m.reset_context()
    p0 = mean_recall_rate(measure_recall_associative(m, A, enc, dec, n_trials=1, noise_scale=0.0)); g0 = mean_margin(measure_recall_margin(m, A, enc))
    m.fit_sequence(XB, epochs=epB); m.reset_context()
    p1 = mean_recall_rate(measure_recall_associative(m, A, enc, dec, n_trials=1, noise_scale=0.0)); g1 = mean_margin(measure_recall_margin(m, A, enc))
    return p0, p1, g0, g1

print("=" * 78); print("(1) & (2): the two real pairs"); print("=" * 78)
pairs = {"disjoint categories (shipped)": (fruit[:7], animal[:7]), "same category": (fruit[:6], fruit[6:12])}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (label, (A, B)) in zip(axes[:2], pairs.items()):
    enc, XA, XB, cos, in_span = geometry(A, B)
    p0, p1, g0, g1 = forget_with(enc, A, XA, XB, epB=40)
    print(f"\n{label}:  A={A}\n{'':30s} B={B}")
    print(f"  mean |cos(A_i, B_j)| = {np.abs(cos).mean():.2f}   max = {np.abs(cos).max():.2f}")
    print(f"  fraction of each B item inside span(A): {np.round(in_span, 2)}   mean {in_span.mean():.2f}")
    print(f"  theta, A then B at 10x:  P(A) {p0:.2f} -> {p1:.2f}   margin {g0:.3f} -> {g1:.3f}")
    im = ax.imshow(cos, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(B))); ax.set_xticklabels(B, rotation=60, fontsize=8); ax.set_yticks(range(len(A))); ax.set_yticklabels(A, fontsize=8)
    ax.set_title(f"{label}\ncos(A_i, B_j); mean|cos|={np.abs(cos).mean():.2f}, in-span={in_span.mean():.2f}\n"
                 f"P(A) after B: {p0:.2f}->{p1:.2f}   margin {g0:.2f}->{g1:.2f}", fontsize=9)
plt.colorbar(im, ax=axes[:2], fraction=0.02)

print("\n" + "=" * 78); print("(3): overlap as a dial -- rotate B into span(A) by alpha"); print("=" * 78)
A, B = fruit[:7], animal[:7]
enc, XA, XB, _, _ = geometry(A, B)
Q, _ = np.linalg.qr(XA.T)
alphas = np.linspace(0, 1, 11); N_DRAWS = 8
Ps, Gs, OV, Plo, Phi, Glo, Ghi = [], [], [], [], [], [], []
print(f"{'alpha':>5s} {'in-span(B)':>10s} | {'P(A) after B: mean':>18s} {'[min,max]':>12s} | {'margin after: mean':>18s} {'[min,max]':>14s}")
for a in alphas:
    ps, gs, ovs = [], [], []
    for d in range(N_DRAWS):
        rng = np.random.default_rng(d)
        # B' = sqrt(1-a) * (B's component outside span A) + sqrt(a) * (a random direction inside span A), norm-matched
        outside = XB - (Q @ (Q.T @ XB.T)).T; outside /= np.linalg.norm(outside, axis=1, keepdims=True)
        inside = (Q @ rng.normal(size=(Q.shape[1], len(B)))).T; inside /= np.linalg.norm(inside, axis=1, keepdims=True)
        XBp = (np.sqrt(1 - a) * outside + np.sqrt(a) * inside) * np.linalg.norm(XB, axis=1, keepdims=True)
        ovs.append(float(np.mean(np.linalg.norm(Q.T @ XBp.T, axis=0) / np.linalg.norm(XBp, axis=1))))
        p0, p1, g0, g1 = forget_with(enc, A, XA, XBp, epB=40); ps.append(p1); gs.append(g1)
    Ps.append(np.mean(ps)); Plo.append(min(ps)); Phi.append(max(ps)); Gs.append(np.mean(gs)); Glo.append(min(gs)); Ghi.append(max(gs)); OV.append(np.mean(ovs))
    print(f"{a:5.1f} {OV[-1]:10.2f} | {Ps[-1]:18.2f} [{min(ps):.2f},{max(ps):.2f}] | {Gs[-1]:18.3f} [{min(gs):+.2f},{max(gs):+.2f}]")
ax = axes[2]; ax.plot(OV, Ps, "-o", color="red", label="P(A recalled) after B"); ax.fill_between(OV, Plo, Phi, color="red", alpha=0.15); ax.set_ylim(-0.05, 1.05)
ax.set_xlabel("overlap: fraction of B inside span(A)"); ax.set_ylabel("P(A) after B", color="red")
ax2 = ax.twinx(); ax2.plot(OV, Gs, "-s", color="purple", label="margin(A) after B"); ax2.fill_between(OV, Glo, Ghi, color="purple", alpha=0.15); ax2.axhline(0, color="k", ls=":", lw=0.8)
ax2.set_ylabel("margin(A) after B", color="purple")
ax.set_title(f"(3) overlap dial: same B budget (10x), only its direction\nrelative to A changes; mean of {N_DRAWS} rotations, band = min..max", fontsize=9)
fig.suptitle("Overlap = how much of list B's input lies in list A's input subspace.  dW x_A = lr * err_B * (x_B . x_A)", fontsize=10)
plt.tight_layout(); plt.savefig("standalone_overlap_demo.png", dpi=140); print("\nfigure: standalone_overlap_demo.png")
