"""
Standalone Theta-gated Equilibrium Propagation -- two-population prototype.

Bootstrap for the "theta rhythm orchestrates EP's two phases" idea
(docs/ep_theta_phase_exploration.md). We feed the *raw dwelling stream*

        A A A B B B C C C

and ask whether clean transition learning (A->B->C) emerges WITHOUT any
hand-cued phase orchestration or segment boundaries.

Two weight populations (the symmetric/asymmetric split -- doc section 3c)
------------------------------------------------------------------------
An earlier single-loop version merged two jobs into one matrix and could not
separate "A auto-associates to A" from "A transitions to B"; dwell updates never
zeroed and C leaked to B. This version splits them:

  * SYMMETRIC EP net (W_ih / W_ho, energy-based):  AUTO-ASSOCIATIVE MEMORY.
    Its nudge target is the *current* symbol x_from (reconstruct what is here),
    so its equilibria become the attractors A, B, C. Trained by the EP
    contrastive rule. Because the target is the currently-clamped state, dwell
    steps settle to a match and their EP update collapses to ~0 -- the symmetric
    matrix converges to clean state storage and stops chasing transitions.

  * ASYMMETRIC matrix (W_asym, directed s_t -> s_{t+1}):  TRANSITIONS.
    A separate population, NOT part of the EP energy (so EP stays valid --
    resolves the section-3c tension). Trained by a delta rule gated by an
    emergent novelty signal g = |x_to - retrieved_attractor|: ~0 on dwell (the
    arriving symbol matches the held attractor) and ~1 at a boundary (novel).
    So only real transitions are written; dwell is suppressed.

Theta orchestration (unchanged from the single-loop version)
------------------------------------------------------------
One continuous settling loop; a global theta phase drives beta:

        beta(t) = beta0 * max(0, cos phi)      # peak -> encode, trough -> retrieve

One theta cycle == one stream step; phi sweeps pi/2 -> 5pi/2 so the order is
retrieve (trough, beta=0) then encode (peak, beta=beta0), with the dynamical
state carrying across the seam. The free/nudge split and the update schedule are
emergent from phi, and the input is the raw un-segmented dwelling stream.

Read-outs
---------
Successor prediction now comes from W_asym applied to the auto-associatively
cleaned current state; the traveling sweep A->B->C is an autonomous rollout that
alternates EP cleaning with an asymmetric step.

Deliberately deferred (documented next layers): recurrent hidden<->hidden (CA3)
placement of W_asym (this file uses the minimal state-space operator, "Option
B"), eligibility traces for behavioural-time / one-shot-dwell bootstrapping, and
global inhibition for emergent boundaries. Per-unit adaptation is wired in but
off by default.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

SYMBOLS = ["A", "B", "C"]
STREAM = list("AAABBBCCC")
# one-hot codes: orthogonal, already inside EP's [0, 1] output range, trivial to
# decode (argmax). Keeps the bootstrap legible -- no 100-dim symbolic encoder.
ONEHOT = {s: np.eye(len(SYMBOLS))[i] for i, s in enumerate(SYMBOLS)}


def decode(vec):
    """Nearest symbol (argmax over the one-hot basis) + its activation value."""
    i = int(np.argmax(vec))
    return SYMBOLS[i], float(vec[i])


class ThetaEqProp:
    """Energy-based auto-associative net (symmetric, EP-trained) plus a separate
    asymmetric transition operator, both orchestrated by a theta oscillator."""

    def __init__(self, n_features, n_hidden=32, lr=0.05, lr_asym=0.3, beta0=1.0,
                 dt=0.2, steps_per_cycle=48, nov_scale=0.7, k_adapt=0.0,
                 tau_adapt=20.0, seed=0):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.lr = lr                    # EP (symmetric, auto-associative) rate
        self.lr_asym = lr_asym          # delta rate for the transition operator
        self.beta0 = beta0
        self.dt = dt
        self.steps_per_cycle = steps_per_cycle
        self.nov_scale = nov_scale      # novelty-gate scale for W_asym writes
        self.k_adapt = k_adapt          # spike-frequency adaptation gain (0 = off)
        self.tau_adapt = tau_adapt
        rng = np.random.default_rng(seed)

        lim_ih = np.sqrt(6.0 / (n_features + n_hidden))
        lim_ho = np.sqrt(6.0 / (n_hidden + n_features))
        self.W_ih = rng.uniform(-lim_ih, lim_ih, (n_hidden, n_features))
        self.W_ho = rng.uniform(-lim_ho, lim_ho, (n_features, n_hidden))
        self.b_h = np.zeros(n_hidden)
        self.b_o = np.zeros(n_features)

        # asymmetric transition operator: directed s_t -> s_{t+1}, starts blank
        self.W_asym = np.zeros((n_features, n_features))

        self.reset_state()

    def reset_state(self):
        """Clear the transient dynamical state (weights untouched)."""
        self.s_h = np.zeros(self.n_hidden)
        self.s_o = np.zeros(self.n_features)
        self.a_h = np.zeros(self.n_hidden)      # per-unit adaptation current

    # -- one integration step of the (continuous) settling dynamics ------------
    def _micro_step(self, x_in, target, beta):
        current_h = self.W_ih @ x_in + self.W_ho.T @ self.s_o + self.b_h - self.a_h
        self.s_h = np.clip(self.s_h + self.dt * (-self.s_h + current_h), 0.0, 1.0)

        current_o = self.W_ho @ self.s_h + self.b_o
        nudge = 2.0 * beta * (target - self.s_o) if (target is not None and beta > 0) else 0.0
        self.s_o = np.clip(self.s_o + self.dt * (-self.s_o + current_o + nudge), 0.0, 1.0)

        # adaptation: active units accrue a self-inhibiting current that decays
        self.a_h += self.dt * (self.k_adapt * self.s_h - self.a_h / self.tau_adapt)

    def _settle_free(self, x_in):
        """Clamp `x_in`, free-settle one theta-length, return the cleaned output
        (auto-associative completion toward the nearest stored attractor)."""
        self.reset_state()
        for _ in range(self.steps_per_cycle):
            self._micro_step(x_in, target=None, beta=0.0)
        return self.s_o.copy()

    # -- one theta cycle == one stream step ------------------------------------
    def step_cycle(self, x_from, x_to, learn=True):
        """Run a single theta cycle. `x_from` is clamped to the input; the
        symmetric EP net is nudged toward `x_from` (auto-association). The
        asymmetric operator learns the directed x_from -> x_to edge, gated by
        novelty. Returns a small diagnostics dict.
        """
        free_snap = nudge_snap = None
        best_trough_cos, best_peak_cos = 1.0, -1.0

        for i in range(self.steps_per_cycle):
            phi = 0.5 * np.pi + 2.0 * np.pi * (i + 0.5) / self.steps_per_cycle
            c = np.cos(phi)
            beta = self.beta0 * max(0.0, c)
            # auto-associative target: reconstruct the CURRENT symbol, not the next
            self._micro_step(x_from, x_from, beta)
            if c < best_trough_cos:
                best_trough_cos = c
                free_snap = (self.s_h.copy(), self.s_o.copy())
            if c > best_peak_cos:
                best_peak_cos = c
                nudge_snap = (self.s_h.copy(), self.s_o.copy())

        s_h_free, s_o_free = free_snap
        s_h_nud, s_o_nud = nudge_snap
        s_clean = s_o_free                       # retrieved current attractor

        dW_sym = dW_asym = g = 0.0
        if learn:
            # --- symmetric EP update (auto-associative storage) ---------------
            inv_beta = 1.0 / self.beta0
            d_ho = inv_beta * (np.outer(s_o_nud, s_h_nud) - np.outer(s_o_free, s_h_free))
            d_ih = inv_beta * (np.outer(s_h_nud, x_from) - np.outer(s_h_free, x_from))
            self.W_ho += self.lr * d_ho
            self.W_ih += self.lr * d_ih
            self.b_h += self.lr * inv_beta * (s_h_nud - s_h_free)
            self.b_o += self.lr * inv_beta * (s_o_nud - s_o_free)
            dW_sym = float(np.sqrt((d_ho ** 2).sum() + (d_ih ** 2).sum()))

            # --- asymmetric transition update (novelty-gated delta) -----------
            # g ~ 0 when the arriving symbol matches the held attractor (dwell),
            # ~ 1 when it is novel (boundary). Emergent, not hand-cued.
            g = float(np.clip(np.linalg.norm(x_to - s_clean) / self.nov_scale, 0.0, 1.0))
            err = x_to - self.W_asym @ s_clean
            d_asym = g * np.outer(err, s_clean)
            self.W_asym += self.lr_asym * d_asym
            dW_asym = float(np.sqrt((d_asym ** 2).sum())) * self.lr_asym

        pred_next = np.maximum(0.0, self.W_asym @ s_clean)
        return dict(dW_sym=dW_sym, dW_asym=dW_asym, g=g,
                    trough=decode(pred_next)[0])

    # -- read-outs -------------------------------------------------------------
    def predict_next(self, symbol):
        """Clean the clamped symbol (EP auto-assoc), then apply the asymmetric
        operator -> predicted successor vector."""
        clean = self._settle_free(ONEHOT[symbol])
        return np.maximum(0.0, self.W_asym @ clean)

    def rollout(self, start_symbol, n_steps, thresh=0.3):
        """Autonomous sweep: alternate EP cleaning with an asymmetric step,
        feeding each prediction back in. Shows whether the transition operator
        drives A->B->C on its own."""
        v = ONEHOT[start_symbol].copy()
        out = [start_symbol]
        for _ in range(n_steps):
            clean = self._settle_free(v)
            nxt = np.maximum(0.0, self.W_asym @ clean)
            sym, val = decode(nxt)
            out.append(sym if val > thresh else ".")
            v = np.clip(nxt, 0.0, 1.0)
        return out


def train(net, n_passes, verbose=True):
    """Stream the raw dwelling sequence `n_passes` times. Each pass presents the
    8 within-stream pairs; dynamical state is reset between passes (fresh
    re-entry) while weights persist. No pair spans the C...->...A seam."""
    log = []
    for p in range(n_passes):
        net.reset_state()
        for t in range(len(STREAM) - 1):
            x_from, x_to = STREAM[t], STREAM[t + 1]
            d = net.step_cycle(ONEHOT[x_from], ONEHOT[x_to])
            d.update(p=p, t=t, frm=x_from, to=x_to, boundary=(x_from != x_to))
            log.append(d)
    if verbose:
        _print_report(net, log, n_passes)
    return log


def _print_report(net, log, n_passes):
    print("=" * 70)
    print(" Theta EP + asymmetric transitions  --  raw stream  '%s'" % " ".join(STREAM))
    print("=" * 70)

    last = [r for r in log if r["p"] == n_passes - 1]

    def split_mean(key):
        d = np.mean([r[key] for r in last if not r["boundary"]])
        b = np.mean([r[key] for r in last if r["boundary"]])
        return d, b

    ds, bs = split_mean("dW_sym")
    da, ba = split_mean("dW_asym")
    print("\nfinal-pass mean update magnitude (dwell | boundary):")
    print(f"    symmetric EP (auto-assoc):  {ds:.4f} | {bs:.4f}   "
          f"(both -> 0 = states are stored, no transition chasing)")
    print(f"    asymmetric  (transitions):  {da:.4f} | {ba:.4f}   "
          f"(boundary >> dwell = change-gated edge writing)")

    print("\nnovelty gate g across the final pass (0 dwell / 1 boundary):")
    print("    " + "  ".join(f"{r['frm']}>{r['to']}:{r['g']:.2f}" for r in last))

    print("\ntransition read-out  (clamp -> EP clean -> W_asym -> predicted next):")
    print(f"    {'from':>6} | {'A':>6} {'B':>6} {'C':>6} | predicted")
    for s in SYMBOLS:
        o = net.predict_next(s)
        pred, val = decode(o)
        tag = pred if val > 0.3 else "(none)"
        print(f"    {s:>6} | {o[0]:>6.3f} {o[1]:>6.3f} {o[2]:>6.3f} |   {tag}")
    print("    (target chain:  A->B,  B->C,  C->none )")

    print("\nautonomous sweep from A (EP-clean + asymmetric step, fed back):")
    print("    " + " -> ".join(net.rollout("A", 4)))


def plot(log, n_passes, net, path="./results/theta_ep_bootstrap.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    # panel 1: asymmetric |dW| per cycle, dwell vs boundary -> change-gating
    xs = np.arange(len(log))
    dW = np.array([r["dW_asym"] for r in log])
    is_b = np.array([r["boundary"] for r in log])
    ax = axes[0]
    ax.plot(xs, dW, color="0.8", lw=0.7, zorder=1)
    ax.scatter(xs[~is_b], dW[~is_b], s=16, c="tab:blue", label="dwell (A->A ...)", zorder=2)
    ax.scatter(xs[is_b], dW[is_b], s=30, c="tab:red", marker="^",
               label="boundary (A->B, B->C)", zorder=3)
    for p in range(1, n_passes):
        ax.axvline(p * (len(STREAM) - 1) - 0.5, color="0.93", lw=0.8)
    ax.set_title("Change-gating on the asymmetric transition matrix (|dW_asym| per cycle)")
    ax.set_xlabel("theta cycle (concatenated passes)")
    ax.set_ylabel("|dW_asym|")
    ax.legend(loc="upper right", fontsize=8)

    # panel 2: learned transition read-out heatmap (via W_asym on cleaned states)
    ax = axes[1]
    M = np.array([net.predict_next(s) for s in SYMBOLS])
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SYMBOLS)))
    ax.set_xticklabels(SYMBOLS)
    ax.set_yticks(range(len(SYMBOLS)))
    ax.set_yticklabels(SYMBOLS)
    ax.set_xlabel("predicted next symbol")
    ax.set_ylabel("clamped symbol")
    ax.set_title("Transition read-out  (A->B, B->C bright; C->none = leak fixed)")
    for i in range(len(SYMBOLS)):
        for j in range(len(SYMBOLS)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="w" if M[i, j] < 0.6 else "k", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


def run(n_hidden=32, n_passes=60, lr=0.05, lr_asym=0.3, beta0=1.0, k_adapt=0.0, seed=0):
    net = ThetaEqProp(n_features=len(SYMBOLS), n_hidden=n_hidden, lr=lr,
                      lr_asym=lr_asym, beta0=beta0, k_adapt=k_adapt, seed=seed)
    log = train(net, n_passes=n_passes)
    plot(log, n_passes, net)
    return net, log


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Theta-gated EP + asymmetric transitions.")
    ap.add_argument("--n-hidden", type=int, default=32)
    ap.add_argument("--n-passes", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr-asym", type=float, default=0.3)
    ap.add_argument("--beta0", type=float, default=1.0)
    ap.add_argument("--k-adapt", type=float, default=0.0, help="adaptation gain (0=off)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(n_hidden=args.n_hidden, n_passes=args.n_passes, lr=args.lr,
        lr_asym=args.lr_asym, beta0=args.beta0, k_adapt=args.k_adapt, seed=args.seed)
