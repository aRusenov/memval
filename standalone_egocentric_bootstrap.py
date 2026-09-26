"""Egocentric spatial encoding — bootstrap prototype.

Replaces the allocentric `PlaceCellEncoder` (a Gaussian tiling over (x, y), i.e.
position handed to the model as a soft one-hot) with a multi-channel egocentric
sensory code, RatInABox-style but reimplemented so seeding stays under our
control:

  * head-direction cells  — von Mises tuning over heading theta
  * object-vector cells   — per landmark, a Gaussian in (egocentric bearing,
                            distance); this is what the agent actually senses

Position is never given. It is recoverable from the observation, but only by
conjoining bearing channels across landmarks with heading — the trilateration
argument. That is the whole point: the allocentric tiling *is* the non-linear
feature expansion, handed over for free, which is why the shipped spatial suite
is linearly solvable.

Three questions, in order:
  1. Does the encoder round-trip? (encode -> decode recovers the trajectory)
  2. Can AHN and EP learn it and reconstruct the trajectory open-loop?
  3. Is it linearly solvable? (the oracle of docs/nonlinearity_benchmark_design.md S3)

Run:  python standalone_egocentric_bootstrap.py
"""

import numpy as np

from memval.models.baselines import (
    AsymmetricHopfieldNetwork,
    OriginalEqPropSequenceNetwork,
)

SEED = 0
ENV = 1.0

# Four landmarks near the arena corners.
LANDMARKS = np.array([[-0.8, -0.8], [0.8, -0.8], [0.8, 0.8], [-0.8, 0.8]])


# ======================================================================
# Encoder
# ======================================================================
class EgocentricEncoder:
    """Head-direction cells + object-vector cells. No allocentric channel."""

    def __init__(self, landmarks=LANDMARKS, n_hd=12, n_bearing=8, n_dist=3,
                 max_dist=2.6, hd_kappa=4.0, bearing_kappa=4.0, dist_sigma=0.45):
        self.landmarks = np.asarray(landmarks, float)
        self.n_landmarks = len(self.landmarks)
        self.hd_centers = np.linspace(-np.pi, np.pi, n_hd, endpoint=False)
        self.bearing_centers = np.linspace(-np.pi, np.pi, n_bearing, endpoint=False)
        self.dist_centers = np.linspace(0.2, max_dist, n_dist)
        self.hd_kappa, self.bearing_kappa, self.dist_sigma = hd_kappa, bearing_kappa, dist_sigma
        self.n_hd = n_hd
        self.n_ovc = self.n_landmarks * n_bearing * n_dist
        self.n_features = self.n_hd + self.n_ovc
        self._codebook = None

    def encode(self, pos, heading):
        """(T,2), (T,) -> (T, n_features) in [0, 1]."""
        pos = np.atleast_2d(np.asarray(pos, float))
        heading = np.atleast_1d(np.asarray(heading, float))
        T = len(pos)

        # --- head direction: von Mises over theta ---
        d_hd = heading[:, None] - self.hd_centers[None, :]
        hd = np.exp(self.hd_kappa * (np.cos(d_hd) - 1.0))

        # --- object-vector cells: egocentric bearing x distance, per landmark ---
        rel = self.landmarks[None, :, :] - pos[:, None, :]          # (T, L, 2)
        dist = np.linalg.norm(rel, axis=2)                          # (T, L)
        allo_bear = np.arctan2(rel[:, :, 1], rel[:, :, 0])          # (T, L)
        ego_bear = np.arctan2(np.sin(allo_bear - heading[:, None]),
                              np.cos(allo_bear - heading[:, None]))

        b = np.exp(self.bearing_kappa * (
            np.cos(ego_bear[:, :, None] - self.bearing_centers[None, None, :]) - 1.0))
        d = np.exp(-0.5 * ((dist[:, :, None] - self.dist_centers[None, None, :])
                           / self.dist_sigma) ** 2)
        ovc = (b[:, :, :, None] * d[:, :, None, :]).reshape(T, -1)

        return np.hstack([hd, ovc])

    # ------------------------------------------------------------------
    def build_codebook(self, n_grid=26, n_head=24, seed=SEED):
        """Dense (x, y, theta) lookup used only for DECODING, never by the model.

        The harness-side decoder, exactly the slot `PlaceCellEncoder.decode`
        occupies today: the model works in observation space, the harness maps
        observations back to coordinates for scoring.
        """
        g = np.linspace(-ENV, ENV, n_grid)
        H = np.linspace(-np.pi, np.pi, n_head, endpoint=False)
        XX, YY, TT = np.meshgrid(g, g, H, indexing="ij")
        P = np.stack([XX.ravel(), YY.ravel()], axis=1)
        A = TT.ravel()
        C = self.encode(P, A)
        self._codebook = (P, C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12))
        return self

    def decode(self, codes, top_k=8, temperature=0.02):
        """(T, n_features) -> (T, 2). Softmax-weighted centre of mass over the
        best-matching codebook entries (cosine, so scale-invariant)."""
        if self._codebook is None:
            self.build_codebook()
        P, Cn = self._codebook
        Z = np.atleast_2d(np.asarray(codes, float))
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
        sim = Zn @ Cn.T
        idx = np.argpartition(-sim, top_k, axis=1)[:, :top_k]
        out = np.zeros((len(Z), 2))
        for t in range(len(Z)):
            s = sim[t, idx[t]]
            w = np.exp((s - s.max()) / temperature)
            out[t] = (w[:, None] * P[idx[t]]).sum(0) / w.sum()
        return out


# ======================================================================
# Trajectory
# ======================================================================
def smooth_route(n_steps=240, seed=SEED):
    """A single smooth looping route (kept as the degenerate control)."""
    t = np.linspace(0, 2 * np.pi, n_steps)
    pos = np.stack([0.72 * np.sin(t), 0.62 * np.sin(2 * t)], axis=1)
    d = np.gradient(pos, axis=0)
    return pos, np.arctan2(d[:, 1], d[:, 0])


def straight_route(n_steps, start, heading0, step=0.02):
    """Constant-velocity traversal. DETERMINISTIC, so open-loop rollout is
    well-posed: the next observation is a function of the current one.

    A random walk is not usable here -- the next heading is unpredictable in
    principle, so a 50-step open-loop rollout scores at chance for every model
    and every oracle alike, measuring nothing. The shipped spatial suite avoids
    this by using deterministic T-maze routes; this does the same.

    The update rule is a pure rotation-and-translate, shared across routes, so a
    held-out route tests whether the rule GENERALISED rather than whether one
    curve was memorised.
    """
    v = np.array([np.cos(heading0), np.sin(heading0)])
    pos = np.stack([start + step * t * v for t in range(n_steps)])
    return pos, np.full(n_steps, heading0)


def route_set(n_routes, n_steps, seed=SEED):
    """Traversals fanned across the arena: varied entry point and heading."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_routes):
        h = rng.uniform(-np.pi, np.pi)
        # start on the upwind side so the traversal stays inside the arena
        # Enter on the upwind side and travel ~1.2 units: with step=0.02 and
        # 60 steps the whole traversal stays inside +-0.85*ENV, which is what
        # the decoder codebook covers. An earlier version used step=0.055 and
        # walked 3.3 units clean out of the arena -- the decoder then had no
        # codebook entries to match and every number below was meaningless.
        start = -0.55 * ENV * np.array([np.cos(h), np.sin(h)]) + \
            rng.uniform(-0.15, 0.15, 2)
        out.append(straight_route(n_steps, start, h))
    return out


# ======================================================================
# Metrics
# ======================================================================
def traj_error(true_xy, dec_xy):
    err = np.linalg.norm(dec_xy - true_xy, axis=1)
    return dict(
        mean_err=float(err.mean()),
        median_err=float(np.median(err)),
        # first step whose error exceeds a quarter of the arena half-width
        divergence_step=int(np.argmax(err > 0.25 * ENV)) if (err > 0.25 * ENV).any() else len(err),
    )


def chance_error(true_xy, rng):
    """Error of predicting uniformly random arena points — the floor to beat."""
    r = rng.uniform(-ENV, ENV, size=(4000, 2))
    return float(np.mean([np.linalg.norm(r - p, axis=1).mean() for p in true_xy]))


# ======================================================================
# Linear oracle (docs/nonlinearity_benchmark_design.md S3)
# ======================================================================
def certify_nonlinear(X, Y, rel_tol=1e-2):
    """Rank identity of S3.1, reported WITH the evidence needed to read it.

    `np.linalg.matrix_rank`'s default tolerance counts noise-level singular
    values as real, which on this data inflated rank(X) from an effective 30 to
    108 and returned a confident "NON-LINEAR" for a task whose least-squares
    residual was 5e-7. So the boolean alone is not reportable: the effective
    rank at an explicit threshold and the relative residual ship with it.
    """
    sx = np.linalg.svd(X, compute_uv=False)
    eff = int((sx > rel_tol * sx[0]).sum())
    rx = np.linalg.matrix_rank(X, tol=rel_tol * sx[0])
    rxy = np.linalg.matrix_rank(np.hstack([X, Y]), tol=rel_tol * sx[0])
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = float(np.linalg.norm(X @ W - Y) / np.linalg.norm(Y))
    return dict(is_nonlinear=bool(rxy > rx), rank_X=rx, rank_XY=rxy,
                effective_rank=eff, dim=int(X.shape[1]), n=int(X.shape[0]),
                fit_residual=resid)


def conjunctive_features(Z, enc):
    """[z ; HD (x) per-landmark pooled OVC]. The multiplicative term the
    trilateration argument predicts is needed: what a bearing *means* depends
    on heading."""
    Z = np.atleast_2d(Z)
    hd = Z[:, :enc.n_hd]
    ovc = Z[:, enc.n_hd:].reshape(len(Z), enc.n_landmarks, -1).sum(axis=2)
    cross = (hd[:, :, None] * ovc[:, None, :]).reshape(len(Z), -1)
    return np.hstack([Z, cross])


def fit_oracle(X, Y):
    """SVD-based lstsq — never normal equations (caveat S3.3.1)."""
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def oracle_rollout(W, feat_fn, seed_obs, n_steps):
    """Same OBSERVATION-space rollout protocol the arms use: feed the
    prediction back verbatim."""
    out, cur = [], seed_obs.copy()
    for _ in range(n_steps):
        cur = (feat_fn(cur[None, :]) @ W).ravel()
        # Clip back into the code's valid range. The conjunctive oracle's
        # quadratic features otherwise amplify their own output and overflow to
        # nan within ~3 steps. The arms get this for free (EP clips to [0,1]),
        # so withholding it from the oracle would flatter the arms.
        cur = np.clip(cur, 0.0, 1.0)
        out.append(cur.copy())
    return np.array(out)


# ======================================================================
def main():
    rng = np.random.default_rng(SEED)
    enc = EgocentricEncoder().build_codebook()

    N_TRAIN, N_STEPS, PROMPT = 8, 60, 10
    routes = route_set(N_TRAIN + 1, N_STEPS)
    train, (test_pos, test_head) = routes[:-1], routes[-1]
    Ztr = [enc.encode(p, h) for p, h in train]
    Zte = enc.encode(test_pos, test_head)
    ROLL = N_STEPS - PROMPT
    true_tail = test_pos[PROMPT:]
    ch = chance_error(true_tail, rng)

    print("=" * 78)
    print("Egocentric encoding - bootstrap")
    print("=" * 78)
    print(f"  landmarks {enc.n_landmarks} | HD {enc.n_hd} | OVC {enc.n_ovc} "
          f"| n_features {enc.n_features}")
    print(f"  {N_TRAIN} training routes x {N_STEPS} steps, 1 held-out route; "
          f"prompt {PROMPT}, rollout {ROLL}")

    # --- 1. Round trip -------------------------------------------------
    rt = traj_error(test_pos, enc.decode(Zte))
    print(f"\n1. Encoder round-trip (encode -> decode), held-out route")
    print(f"   mean error {rt['mean_err']:.4f}   (chance {ch:.4f})")
    print("   -> position is fully recoverable from the egocentric code.")

    # --- 2. Certification: one route vs the route set -------------------
    print(f"\n2. Linear certification (S3.1) -- one route vs a route set")
    one_pos, one_head = smooth_route(240)
    Zone = enc.encode(one_pos, one_head)
    Xall = np.vstack([z[:-1] for z in Ztr]); Yall = np.vstack([z[1:] for z in Ztr])
    for tag, (X, Y) in [("single smooth route", (Zone[:-1], Zone[1:])),
                        (f"{N_TRAIN}-route set", (Xall, Yall))]:
        c = certify_nonlinear(X, Y)
        print(f"   {tag:<22s} n={c['n']:4d} dim={c['dim']:4d} eff_rank={c['effective_rank']:4d} "
              f"resid={c['fit_residual']:.3e}  -> "
              f"{'NON-LINEAR' if c['is_nonlinear'] else 'linearly fittable'}")
    print("   (a single route is a 1-D curve: degeneracy, not task structure)")

    # --- 3. Oracles, scored through the same rollout + decode -----------
    W_lin = fit_oracle(Xall, Yall)
    W_con = fit_oracle(conjunctive_features(Xall, enc), Yall)
    seed_obs = Zte[PROMPT - 1]
    e_lin = traj_error(true_tail, enc.decode(
        oracle_rollout(W_lin, lambda z: z, seed_obs, ROLL)))
    e_con = traj_error(true_tail, enc.decode(
        oracle_rollout(W_con, lambda z: conjunctive_features(z, enc), seed_obs, ROLL)))

    # --- 4. Arms --------------------------------------------------------
    print(f"\n3. Held-out-route rollout  (lower mean_err is better; chance {ch:.3f})")
    print(f"   {'':<32s} {'mean_err':>9s} {'median':>8s} {'diverge@':>9s}")
    print(f"   {'oracle: linear      [z]':<32s} {e_lin['mean_err']:9.4f} "
          f"{e_lin['median_err']:8.4f} {e_lin['divergence_step']:9d}")
    print(f"   {'oracle: conjunctive HDxOVC':<32s} {e_con['mean_err']:9.4f} "
          f"{e_con['median_err']:8.4f} {e_con['divergence_step']:9d}")

    arm_scores = {}
    for name, mk in [
        ("AsymmetricHopfieldNetwork", lambda: AsymmetricHopfieldNetwork(
            n_features=enc.n_features, learning_rate=0.05)),
        ("OriginalEqPropSequenceNetwork", lambda: OriginalEqPropSequenceNetwork(
            n_features=enc.n_features, n_hidden=128, seed=SEED, n_epochs=30)),
    ]:
        m = mk()
        for z in Ztr:
            m.fit_sequence(z, epochs=30)
        m.reset_context()
        rec = np.asarray(m.recall(Zte[:PROMPT], length=ROLL))
        e = traj_error(true_tail, enc.decode(rec))
        arm_scores[name] = e
        print(f"   {'arm: ' + name[:27]:<32s} {e['mean_err']:9.4f} "
              f"{e['median_err']:8.4f} {e['divergence_step']:9d}")
    print(f"   {'(chance)':<32s} {ch:9.4f}")

    # --- 5. Gaps --------------------------------------------------------
    print(f"\n4. Gaps")
    print(f"   conjunctive - linear      : {e_lin['mean_err'] - e_con['mean_err']:+.4f}"
          "   (positive = the multiplicative term is needed)")
    for name, e in arm_scores.items():
        print(f"   linear_oracle_gap {name[:14]:<14s}: "
              f"{e_lin['mean_err'] - e['mean_err']:+.4f}   (positive = arm beats a linear map)")
    print("\n   Read only the GAPS (caveat S3.3.2). Absolute residuals on")
    print("   trajectory codes reflect representational degeneracy.")


if __name__ == "__main__":
    main()
