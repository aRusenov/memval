#!/usr/bin/env python
"""Size each arm's fixed exposure budget by criterion, per suite, over seeds.

    python bin/size_budgets.py --out results/budget_sizing.json [--arms ahn,theta]

Protocol (settled 2026-09-23):

* **Material, load-matched across suites.** Symbolic: ONE list of 10 items (9
  scored transitions). Spatial: ONE route of 10 transitions (11 points). Same
  number of transitions, so a per-suite difference is about the modality and not
  about sequence length.
* **Criterion.** Symbolic: cued one-step MRR >= 0.95 under the clean single probe
  -- quantised to k/9, so in practice 9/9. Spatial: fraction of one-step
  predictions landing within ``eps`` of the true next position >= 0.95, i.e.
  10/10, with ``eps`` RELATIVE to the route's step (``relative_eps``, k=1.05),
  because an absolute eps silently changes difficulty with route length.
* **Seeds.** 5 model seeds; the budget is ``2 x mean(epochs to criterion over the
  seeds that REACHED)``, rounded up. Seeds that do not reach are excluded from
  the mean and counted; if none reaches, no budget is derived for that suite and
  the arm runs there at its other suite's budget with ``criterion_reached=False``
  recorded -- a result, not a tuning target.
* **Per suite.** Symbolic and spatial get separate budgets (registry ``n_epochs``
  and ``modality_kwargs["spatial"]["n_epochs"]``). The online suite inherits the
  symbolic budget: it is the same material through a different ingestion path.

**Seeds are a no-op for two arms.** ``AsymmetricHopfieldNetwork`` and
``ThetaPhaseSequenceNetwork`` initialise ``W`` to ZERO, so their trajectory is a
deterministic function of the data: all 5 seeds give identical numbers at 5x the
cost. The script detects this (``seed_invariant``) by comparing freshly
initialised weights across seeds, reports it, and runs a single seed for those
arms. Their zero variance is a property of the arm, never a finding.
"""
from __future__ import annotations
import argparse, importlib.util, json, math, os, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memval.benchmarks.exposure import epochs_to_criterion, DEFAULT_CRITERION
from memval.benchmarks.symbolic_pipeline import load_vocab, measure_recall_associative, mean_recall_rate
from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.encoders.spatial import PlaceCellEncoder, relative_eps, route_step, decode_floor
from memval.generators.t_maze import TMazeGenerator

N_ITEMS, N_TRANSITIONS, N_SEEDS, BUFFER = 10, 10, 5, 2.0


def _registry():
    spec = importlib.util.spec_from_file_location(
        "rb", os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_benchmark.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _weight_arrays(m):
    if hasattr(m, "named_parameters"):
        try: return {k: np.asarray(v) for k, v in m.named_parameters().items()}
        except Exception: pass
    st = m.get_latent_state()
    return {k: np.asarray(v) for k, v in (st if isinstance(st, dict) else {"W": st}).items()
            if hasattr(v, "shape")}


def _all_arrays(m):
    """Every float array on the instance -- NOT just the trainable ones.

    Trainable arrays alone are the wrong test: DTS-ESN's learnable readout
    (`W_out`) starts at zero like AHN's `W`, but its FROZEN reservoir (`W_in`,
    `W`) is drawn from the seed and fully determines the dynamics. Comparing only
    `named_parameters()` reported it seed-invariant, which is false.
    """
    out = {}
    for k, v in vars(m).items():
        a = np.asarray(v) if isinstance(v, np.ndarray) else None
        if a is not None and a.size and a.dtype.kind == "f":
            out[k] = a
    return out


def seed_invariant(cls, kw, n_features) -> bool:
    """True when NOTHING about the freshly built arm depends on `seed`.

    Holds for the two zero-init linear arms (AHN, theta): their trajectory is a
    deterministic function of the data, so extra seeds cost time and return
    identical numbers. Does NOT hold for an arm with frozen random structure.
    """
    try:
        a = _all_arrays(cls(n_features=n_features, **{**kw, "seed": 1}))
        b = _all_arrays(cls(n_features=n_features, **{**kw, "seed": 2}))
    except Exception:
        return False
    if a.keys() != b.keys() or not a:
        return False
    return all(np.array_equal(a[k], b[k]) for k in a)


def symbolic_material():
    fruits = [w for w, c in load_vocab().items() if c == "fruit"][:N_ITEMS]
    assert len(fruits) == N_ITEMS, f"vocab has too few fruits for a {N_ITEMS}-item list"
    enc = SymbolicEncoder({w: "fruit" for w in fruits}, embedding_dim=100,
                          category_variance=0.2, seed=42)
    return fruits, enc, SymbolicDecoder(enc), enc.encode(fruits)


def spatial_material():
    tm = np.asarray(TMazeGenerator(seed=42).generate(
        n_sequences=1, sequence_length=N_TRANSITIONS + 1, turn_direction="right"))
    traj = tm[0] if tm.ndim == 3 else tm
    # 20x20 = 400 cells: the grid every spatial section uses (fixed per suite
    # 2026-09-23). Sizing material must match what the suite runs.
    enc = PlaceCellEncoder(seed=42, n_cells_per_dim=20, sigma_scale=1.0)
    X = enc.encode(traj)
    eps = relative_eps(traj, k=1.05)
    return traj, enc, X, eps


def size_arm(name, entry, materials, n_seeds=N_SEEDS, max_epochs=512, verbose=True):
    cls = entry["class"]
    base = {k: v for k, v in entry["default_kwargs"].items() if k != "n_epochs"}
    (words, senc, sdec, Xs), (traj, penc, Xp, eps) = materials
    out = {"class": cls.__name__}
    inv = seed_invariant(cls, base, 100)
    out["seed_invariant"] = inv
    seeds = [42] if inv else list(range(42, 42 + n_seeds))
    out["seeds"] = seeds
    if inv and verbose:
        print(f"  {name}: seed-invariant init (zero weights) -> 1 seed, not {n_seeds}", flush=True)

    def score_sym(m):
        return mean_recall_rate(measure_recall_associative(
            m, words, senc, sdec, n_trials=1, noise_scale=0.0))

    def score_spa(m):
        m.reset_context(); hits = 0
        for t in range(len(Xp) - 1):
            m.current_t = t
            xy = penc.decode(np.asarray([m.predict_next(Xp[t])]))[0]
            hits += float(np.linalg.norm(xy - traj[t + 1]) <= eps)
        return hits / (len(Xp) - 1)

    for suite, X, nfeat, score in (("symbolic", Xs, 100, score_sym),
                                   ("spatial", Xp, penc.n_cells, score_spa)):
        runs = []
        for sd in seeds:
            kw = {**base, "seed": sd}
            t0 = time.time()
            def mk(ep, _kw=kw, _n=nfeat): return cls(n_features=_n, n_epochs=ep, **_kw)
            def fit(m, ep, _X=X):
                if hasattr(m, "n_epochs"): m.n_epochs = ep
                m.fit_sequence(_X, epochs=ep)
            try:
                r = epochs_to_criterion(lambda: mk(1), fit, score, max_epochs=max_epochs)
                runs.append(dict(seed=sd, epochs=int(r["epochs"]), reached=bool(r["reached"]),
                                 score=round(float(r["score"]), 3), secs=round(time.time() - t0, 1)))
            except Exception as e:
                runs.append(dict(seed=sd, error=f"{type(e).__name__}: {str(e)[:90]}",
                                 secs=round(time.time() - t0, 1)))
            if verbose: print(f"    {name:26s} {suite:9s} {runs[-1]}", flush=True)
        ok = [r for r in runs if r.get("reached")]
        rec = dict(runs=runs, n_reached=len(ok), n_seeds=len(seeds))
        if ok:
            mean_ep = float(np.mean([r["epochs"] for r in ok]))
            rec.update(mean_epochs_reached=round(mean_ep, 2),
                       sd_epochs=round(float(np.std([r["epochs"] for r in ok])), 2),
                       budget=int(math.ceil(BUFFER * mean_ep)))
        else:
            rec.update(mean_epochs_reached=None, budget=None,
                       note="criterion unreached on every seed; no budget derived for this suite")
        out[suite] = rec
    out["online"] = {"inherits": "symbolic",
                     "budget": out["symbolic"].get("budget")}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=None, help="comma-separated registry keys (default: all active)")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--max-epochs", type=int, default=512)
    args = ap.parse_args()

    rb = _registry()
    R = rb.MODEL_REGISTRY
    names = [rb.resolve_model_name(n) for n in args.arms.split(",")] if args.arms else list(R)
    materials = (symbolic_material(), spatial_material())
    (words, _, _, _), (traj, penc, _, eps) = materials
    floor = decode_floor(penc, traj)
    header = dict(n_items=N_ITEMS, n_transitions=N_TRANSITIONS, n_seeds=args.seeds,
                  buffer=BUFFER, criterion=DEFAULT_CRITERION, max_epochs=args.max_epochs,
                  symbolic_items=words, spatial_route_step=round(route_step(traj), 4),
                  spatial_eps=round(eps, 4), spatial_eps_k=1.05,
                  spatial_decode_floor_max=round(float(np.max(floor)), 4),
                  spatial_decode_floor_frac_of_eps=round(float(np.max(floor) / eps), 3))
    print("Sizing protocol:", json.dumps(header, indent=1), flush=True)
    out = {"protocol": header, "arms": {}}
    for n in names:
        if n not in R:
            print(f"  [skip] {n}: not in the active registry", flush=True); continue
        print(f"\n{n}:", flush=True)
        out["arms"][n] = size_arm(n, R[n], materials, n_seeds=args.seeds, max_epochs=args.max_epochs)
        json.dump(out, open(args.out, "w"), indent=1)
    print(f"\n{'arm':26s} {'symbolic':>10s} {'spatial':>10s}  (budget; — = unreached)")
    for n, a in out["arms"].items():
        f = lambda s: (str(a[s]["budget"]) if a[s]["budget"] is not None else "—")
        inv = "  [seed-invariant]" if a.get("seed_invariant") else ""
        print(f"{n:26s} {f('symbolic'):>10s} {f('spatial'):>10s}{inv}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
