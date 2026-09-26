"""
Standalone true-online place-traversal learner -- Stage 1.

A mouse walks A -> B -> C -> D -> E once. Each location is dwelt for a variable
number of seconds; one stream item == one second of experience:

        A A A B B C C C C C C D D D E E E

The model is TRUE ONLINE: its only training entry point is `observe(frame)`,
called once per second in stream order. It never sees the whole stream, never
sees the future, and there is NO outer training loop (no `n_passes`). This is
the biologically grounded API -- the sensory cortex only ever has the current
frame.

Why the mechanisms changed from standalone_theta_ep.py
------------------------------------------------------
The earlier prototypes leaned on repeated passes (`n_passes`) because EP's
contrastive rule learns *gradually*. A single traversal has no repetition to
exploit, so Stage 1 replaces that gradient with what biology actually uses for
one-shot place-field formation:

  * ONE-SHOT ATTRACTOR IMPRINT (BTSP-style, doc section 2.2). On the first
    second at a novel location, a single Hebbian imprint writes it into the
    symmetric auto-associative store W_auto -- the attractor exists after ~1s,
    so a 2-second dwell (B, D) is plenty. EP's contrastive refinement is the
    slow complement, deferred to a later replay-driven stage.

  * ELIGIBILITY (doc section 3b). The transition A->B must be bound when B
    arrives, even though A was last seen seconds earlier. A persisted tag of the
    previous committed location bridges that behavioural-time gap, so variable
    dwell does not misbind the edge. Stage 1 uses the simplest robust form -- a
    latch holding the last committed attractor, refreshed at each onset and held
    (unchanged) across the whole dwell. A graded *decaying* trace is the
    generalisation needed for non-adjacent / skip bindings; deferred.

  * CHANGE-GATING. A novelty signal (current cleaned state vs the previously
    committed attractor) fires an "onset" only when the location actually
    changes. Writes happen on onsets; dwell writes nothing. Emergent, not
    hand-cued -- the model is never told where the boundaries are.

Two populations are retained (doc section 3c): symmetric W_auto stores the
locations (auto-associative), asymmetric W_asym stores the directed transitions.
The theta rhythm is retained as the read-then-write scaffold of each second:
retrieve (predict the next location) then encode (high-plasticity write).

Stage 2 (deferred): overlapping place fields -- set `overlap>0` so each frame is
a blend (current strong, neighbours weak). That is the realistic hard case that
stresses the novelty gate and makes W_auto's cleaning load-bearing. The `frame`
builder already supports it so Stage 2 is a one-line change.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

LOCS = ["A", "B", "C", "D", "E"]
STREAM = list("AAABBCCCCCCDDDEEE")     # variable dwell: 3,2,6,3,3 seconds
IDX = {l: i for i, l in enumerate(LOCS)}


def make_frame(loc, overlap=0.0):
    """Sensory frame at a location. overlap=0 -> orthogonal one-hot (Stage 1).
    overlap>0 -> current strong + immediate spatial neighbours weak (Stage 2)."""
    i = IDX[loc]
    v = np.zeros(len(LOCS))
    v[i] = 1.0
    if overlap > 0.0:
        if i - 1 >= 0:
            v[i - 1] = overlap
        if i + 1 < len(LOCS):
            v[i + 1] = overlap
    return v


def decode(vec):
    i = int(np.argmax(vec))
    return LOCS[i], float(vec[i])


class OnlinePlaceLearner:
    """True-online, single-pass place/transition learner. All state below is
    persistent across `observe` calls -- it is the animal's ongoing memory."""

    def __init__(self, n=len(LOCS), eta_auto=1.0, lr_asym=1.0,
                 nov_thresh=0.5, clean_iters=5):
        self.n = n
        self.eta_auto = eta_auto        # one-shot attractor imprint strength
        self.lr_asym = lr_asym          # transition write rate (1.0 = one-shot)
        self.nov_thresh = nov_thresh    # onset (novelty) threshold
        self.clean_iters = clean_iters

        self.W_auto = np.zeros((n, n))          # symmetric auto-associative store
        self.W_asym = np.zeros((n, n))          # asymmetric transition operator
        self.elig = None                        # eligibility latch: last committed loc
        self.prev_committed = None              # last committed attractor

    # -- auto-associative cleaning (input-anchored settle toward an attractor) --
    def _clean(self, frame):
        s = frame.copy()
        for _ in range(self.clean_iters):
            drive = self.W_auto @ s
            s = np.clip(0.5 * s + 0.5 * drive + 0.5 * frame, 0.0, 1.0)
        return s

    @staticmethod
    def _cos(a, b):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(a @ b / (na * nb)) if na > 1e-9 and nb > 1e-9 else 0.0

    # -- the ONLY training entry point: one second of experience ----------------
    def observe(self, frame):
        # ---- RETRIEVE (theta trough): clean the input, predict the successor --
        s_clean = self._clean(frame)
        pred = np.maximum(0.0, self.W_asym @ s_clean)

        # ---- novelty / change detection vs the held attractor -----------------
        if self.prev_committed is None:
            novelty = 1.0
        else:
            novelty = 1.0 - self._cos(s_clean, self.prev_committed)
        onset = novelty > self.nov_thresh

        # ---- ENCODE (theta peak): high-plasticity writes, only on an onset ----
        wrote_edge = False
        if onset:
            # one-shot attractor imprint for the newly entered location (BTSP)
            self.W_auto += self.eta_auto * np.outer(s_clean, s_clean)
            # bind previous -> current using the eligibility latch (the location
            # we are leaving, held unchanged across its whole dwell)
            if self.elig is not None:
                pre = self.elig / (np.linalg.norm(self.elig) + 1e-9)
                err = s_clean - self.W_asym @ pre
                self.W_asym += self.lr_asym * np.outer(err, pre)
                wrote_edge = True
            # refresh the latch to the location just entered; held until next onset
            self.elig = s_clean.copy()
            self.prev_committed = s_clean.copy()

        return dict(pred=decode(pred), novelty=novelty, onset=onset,
                    wrote_edge=wrote_edge)

    # -- read-outs (evaluation only; no learning) ------------------------------
    def predict_next(self, loc, overlap=0.0):
        clean = self._clean(make_frame(loc, overlap))
        return np.maximum(0.0, self.W_asym @ clean)

    def rollout(self, start, n_steps, thresh=0.3):
        v = make_frame(start)
        out = [start]
        for _ in range(n_steps):
            clean = self._clean(v)
            nxt = np.maximum(0.0, self.W_asym @ clean)
            sym, val = decode(nxt)
            out.append(sym if val > thresh else ".")
            v = np.clip(nxt, 0.0, 1.0)
        return out


def train_online(model, overlap=0.0, verbose=True):
    """Single pass over the raw stream -- one observe() per second, in order."""
    log = []
    for t, loc in enumerate(STREAM):
        info = model.observe(make_frame(loc, overlap))
        info.update(t=t, loc=loc)
        log.append(info)
    if verbose:
        _report(model, log, overlap)
    return log


def _report(model, log, overlap):
    print("=" * 70)
    print(" True-online place learner (Stage 1)  --  single pass, no outer loop")
    print(" stream:  %s   (%d seconds)" % (" ".join(STREAM), len(STREAM)))
    print("=" * 70)

    onsets = [r for r in log if r["onset"]]
    edges = [r for r in log if r["wrote_edge"]]
    print(f"\nonsets detected: {len(onsets)} at seconds "
          f"{[r['t'] for r in onsets]}  -> locations {[r['loc'] for r in onsets]}")
    print(f"transition writes: {len(edges)} (one per boundary; dwell wrote nothing)")

    print("\nnovelty per second (onset = change detected, no hand-cued boundary):")
    print("    sec:  " + " ".join(f"{r['t']:>4}" for r in log))
    print("    loc:  " + " ".join(f"{r['loc']:>4}" for r in log))
    print("    nov:  " + " ".join(f"{r['novelty']:>4.1f}" for r in log))
    print("    onset:" + " ".join(("  ^ " if r["onset"] else "  . ") for r in log))

    print("\ntransition read-out (clamp location -> clean -> W_asym -> predict):")
    header = "    " + "from |" + "".join(f"{l:>6}" for l in LOCS) + " | predicted"
    print(header)
    for l in LOCS:
        o = model.predict_next(l, overlap)
        pred, val = decode(o)
        tag = pred if val > 0.3 else "(none)"
        print("    " + f"{l:>4} |" + "".join(f"{v:>6.2f}" for v in o) + f" |   {tag}")
    print("    (target chain:  A->B->C->D->E,  E->none)")

    print("\nautonomous sweep from A (clean + asymmetric step, fed back):")
    print("    " + " -> ".join(model.rollout("A", 5)))


def plot(log, model, overlap, path="./results/theta_online_stage1.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))
    ts = [r["t"] for r in log]

    ax = axes[0]
    nov = [r["novelty"] for r in log]
    ax.plot(ts, nov, color="0.5", lw=1.2, marker="o", ms=4, label="novelty")
    on_t = [r["t"] for r in log if r["onset"]]
    on_v = [r["novelty"] for r in log if r["onset"]]
    ax.scatter(on_t, on_v, s=90, c="tab:red", marker="^", zorder=3, label="onset (edge write)")
    ax.axhline(model.nov_thresh, color="tab:orange", ls="--", lw=1, label="onset threshold")
    ax.set_xticks(ts)
    ax.set_xticklabels([r["loc"] for r in log])
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel("second of experience (labelled by location)")
    ax.set_ylabel("novelty")
    ax.set_title("Change-gating on a single pass: onsets fire only at boundaries")
    ax.legend(loc="center right", fontsize=8)

    ax = axes[1]
    M = np.array([model.predict_next(l, overlap) for l in LOCS])
    im = ax.imshow(M, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(LOCS)))
    ax.set_xticklabels(LOCS)
    ax.set_yticks(range(len(LOCS)))
    ax.set_yticklabels(LOCS)
    ax.set_xlabel("predicted next location")
    ax.set_ylabel("clamped location")
    ax.set_title("Learned transitions after ONE pass  (bright super-diagonal = A->B->C->D->E)")
    for i in range(len(LOCS)):
        for j in range(len(LOCS)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="w" if M[i, j] < 0.6 else "k", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {path}")


def run(overlap=0.0, **kw):
    model = OnlinePlaceLearner(**kw)
    log = train_online(model, overlap=overlap)
    plot(log, model, overlap)
    return model, log


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="True-online place-traversal learner (Stage 1).")
    ap.add_argument("--overlap", type=float, default=0.0,
                    help="place-field overlap (0 = orthogonal / Stage 1; >0 = Stage 2)")
    ap.add_argument("--nov-thresh", type=float, default=0.5)
    args = ap.parse_args()
    run(overlap=args.overlap, nov_thresh=args.nov_thresh)
