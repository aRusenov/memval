#!/usr/bin/env python
"""Why does the Vieth arm forget list A when list B is trained?

Starts from the sanity configuration that scores 1.00 on a 6-item list
(bin/vieth_stdp_sanity.py) and adds a second 6-item list, then walks toward the
pipeline's `multiple_sequences` configuration one factor at a time:

    ingestion   continuous (upstream) -> fenced fit_sequence (MemVal)
    substrate   one-hot -> SymbolicEncoder(category_variance=0.2)
    target_act  1/L -> 1/15 (registry default)
    n_exc       300 -> 500 (registry: 0.5 * codec population)

For each row: cued recall of A before B, of A and B after B, the rollout span
of A after B, and two mechanism read-outs -- how many excitatory units each
list's assemblies share, and the homeostatic budget (units per assembly x
assemblies vs units available).
"""
import argparse
import numpy as np

from memval.encoders.symbolic import SymbolicEncoder
from memval.models.baselines.vieth_gaba_stdp import CodecViethNetwork

FRUIT = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']
ANIMAL = ['cat', 'dog', 'cow', 'horse', 'sheep', 'pig', 'lion']


def material(kind, L, seed, cat_var=0.2):
    if kind == "onehot":
        E = np.eye(2 * L)
        return E[:L], E[L:], E
    vocab = {w: 'fruit' for w in FRUIT[:L]}
    vocab.update({w: 'animal' for w in ANIMAL[:L]})
    enc = SymbolicEncoder(vocab, embedding_dim=100, category_variance=cat_var, seed=42)
    A, B = enc.encode(FRUIT[:L]), enc.encode(ANIMAL[:L])
    return A, B, np.vstack([A, B])


def build(D, n_exc, target_act, seed):
    return CodecViethNetwork(n_features=D, n_per_feature=5, window_steps=50,
                             n_exc=n_exc, target_activity=target_act, seed=seed)


def train(arm, X, passes, ingestion):
    if ingestion == "fenced":
        arm.fit_sequence(X, epochs=passes)
    else:                       # continuous: no seams, wrap-around included
        for _ in range(passes):
            for x in X:
                arm.fit_event(x)


def cued(arm, X, codebook, noise, rng):
    """Per-position hit: cue x_t (+ noise), nearest codebook row must be x_{t+1}."""
    hits = []
    for t in range(len(X) - 1):
        cue = X[t] + rng.normal(0, noise, X.shape[1])
        pred = arm.predict_next(cue)
        if not np.any(pred):
            hits.append(False); continue
        sims = codebook @ pred / (np.linalg.norm(codebook, axis=1) * np.linalg.norm(pred) + 1e-12)
        hits.append(bool(np.allclose(codebook[int(np.argmax(sims))], X[t + 1])))
    return np.array(hits)


def span(arm, X, codebook):
    out = arm.recall(X[0], len(X) - 1)          # wrapper rollout in R^D
    k = 0
    for t, pred in enumerate(out):
        sims = codebook @ pred / (np.linalg.norm(codebook, axis=1) * np.linalg.norm(pred) + 1e-12)
        if np.any(pred) and np.allclose(codebook[int(np.argmax(sims))], X[t + 1]):
            k += 1
        else:
            break
    return k


def assemblies(arm, X, frac=0.5):
    """Excitatory units an item recruits: top-responding units through ES."""
    inner, enc = arm.inner, arm.encoder
    out = []
    for x in X:
        active = enc.encode(x, rng=np.random.default_rng(0)).any(axis=1)
        drive = inner.W_es[active].sum(axis=0)
        k = max(1, int(round(inner.target_activity * inner.n_exc)))
        out.append(set(np.argsort(-drive)[:k]))
    return out


def shared_fraction(asm_A, asm_B):
    A = set().union(*asm_A); B = set().union(*asm_B)
    return len(A & B) / max(1, len(A | B)), len(A), len(B)


def row(label, kind, ingestion, target_act, n_exc, L, passes, noise, seeds, cat_var=0.2,
        driven=None):
    """driven=None: fixed target_act. 'A': calibrate on list A. 'AB': on A then B."""
    print(f"\n=== {label} ===")
    print(f"    {kind}, {ingestion}, target_activity="
          f"{'driven('+driven+')' if driven else f'{target_act:.3f}'}, n_exc={n_exc}, "
          f"{passes} passes, probe noise {noise}")
    for seed in seeds:
        A, B, book = material(kind, L, seed, cat_var)
        arm = build(book.shape[1], n_exc, target_act, seed)
        rng = np.random.default_rng(100 + seed)
        if driven:
            enc = arm.encoder
            cal = [enc.encode(x, rng=np.random.default_rng(7 + i)) for i, x in enumerate(A if driven == "A" else np.vstack([A, B]))]
            r = arm.inner.calibrate_target_activity(cal, rounds=4, passes=200)
            print(f"  seed {seed}  driven rate measured {r:.3f} (was {target_act:.3f}); "
                  f"assembly ~{r*n_exc:.0f} units, budget {r*n_exc*2*L:.0f} slots / {n_exc}")
        train(arm, A, passes, ingestion)
        a_before = cued(arm, A, book, noise, rng)
        asm_A0 = assemblies(arm, A)
        train(arm, B, passes, ingestion)
        a_after = cued(arm, A, book, noise, rng)
        b_after = cued(arm, B, book, noise, rng)
        asm_A1, asm_B1 = assemblies(arm, A), assemblies(arm, B)
        sh, nA, nB = shared_fraction(asm_A1, asm_B1)
        drift = np.mean([len(a0 & a1) / max(1, len(a0 | a1)) for a0, a1 in zip(asm_A0, asm_A1)])
        inner = arm.inner
        budget = inner.target_activity * inner.n_exc * 2 * L
        print(f"  seed {seed}  A before {a_before.mean():.2f} [{''.join('1' if h else '.' for h in a_before)}]"
              f"   A after {a_after.mean():.2f} [{''.join('1' if h else '.' for h in a_after)}]"
              f"   B after {b_after.mean():.2f} [{''.join('1' if h else '.' for h in b_after)}]"
              f"   span(A) {span(arm, A, book)}/{L-1}")
        print(f"           A/B assemblies share {sh:.0%} of units ({nA} vs {nB});"
              f" A's assemblies kept {drift:.0%} of their units after B;"
              f" budget {budget:.0f} unit-slots for {inner.n_exc} units"
              f" ({inner.target_activity*inner.n_exc:.0f}/assembly x {2*L})")
    return


def mechanism(L=6, seeds=(0, 1)):
    """The direct reads behind docs/vieth_stdp_port.md 'Why the forgetting curve'."""
    A, B, book = material("onehot", L, 0)
    print("\n=== mechanism 1: does B re-label A's positional assemblies? ===")
    for seed in seeds:
        arm = build(book.shape[1], 300, 1 / L, seed); inner, enc = arm.inner, arm.encoder
        train(arm, A, 400, "continuous"); asmA = assemblies(arm, A)
        W = inner.W_ee
        ee0 = np.mean([W[np.ix_(sorted(asmA[t]), sorted(asmA[t + 1]))].mean() for t in range(L - 1)])
        train(arm, B, 400, "continuous"); asmB = assemblies(arm, B)
        ee1 = np.mean([W[np.ix_(sorted(asmA[t]), sorted(asmA[t + 1]))].mean() for t in range(L - 1)])
        actA = [enc.encode(A[t], rng=np.random.default_rng(0)).any(axis=1) for t in range(L)]
        actB = [enc.encode(B[t], rng=np.random.default_rng(0)).any(axis=1) for t in range(L)]
        same = np.mean([len(asmA[t] & asmB[t]) / len(asmA[t] | asmB[t]) for t in range(L)])
        other = np.mean([len(asmA[t] & asmB[u]) / len(asmA[t] | asmB[u]) for t in range(L) for u in range(L) if u != t])
        mA = np.mean([inner.W_es[actA[t]][:, sorted(asmA[t])].sum(0).mean() for t in range(L)])
        mB = np.mean([inner.W_es[actB[t]][:, sorted(asmA[t])].sum(0).mean() for t in range(L)])
        out = inner.recall(enc.encode(A[0], rng=np.random.default_rng(0)), L - 1)
        walk = []
        for t in range(L - 1):
            rec = out[t, :, 0]
            sc = [rec[a].sum() for a in actA] + [rec[b].sum() for b in actB]
            k = int(np.argmax(sc)); walk.append(("A" if k < L else "B") + str(k % L))
        print(f"  seed {seed}: A-chain EE {ee0:.4f} -> {ee1:.4f};  B_t on A_t Jaccard {same:.2f} (other positions {other:.2f});"
              f"  ES mass onto assembly_t: A_t {mA:.3f} vs B_t {mB:.3f};  free-run from A0: {' '.join(walk)}")

    print("\n=== mechanism 2: homeostatic drift when target_activity < driven rate ===")
    for tgt, name in ((1 / L, "1/L"), (1 / (2 * L), "1/(2L)")):
        for seed in seeds:
            arm = build(book.shape[1], 300, tgt, seed); inner = arm.inner
            rates = []; orig = inner._iterate
            def spy(inp, plastic, rng=None, _o=orig, _r=rates):
                o = _o(inp, plastic, rng); _r.append(o.mean()); return o
            inner._iterate = spy
            done, cells = 0, []
            for chk in (200, 400, 800, 1600):
                for _ in range(chk - done):
                    for x in A: arm.fit_event(x)
                done = chk
                inner._iterate = orig
                a = cued(arm, A, book, 0.0, np.random.default_rng(1)).mean()
                inner._iterate = spy
                cells.append(f"@{chk} rate {np.mean(rates[-L*50:]):.3f} sens {inner.sensitivity.mean():+.2f} cued {a:.2f}")
            inner._iterate = orig
            print(f"  target {name:6s} seed {seed}: " + " | ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", type=int, default=6)
    ap.add_argument("--passes", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--mechanism", action="store_true",
                    help="run the direct mechanism reads instead of the factor ladder")
    ap.add_argument("--driven", choices=("A", "AB"), default=None,
                    help="re-site target_activity to the fixed-point driven rate before training")
    a = ap.parse_args()
    if a.mechanism:
        return mechanism(a.items, range(a.seeds))
    if a.driven:
        L, P, S, N, D = a.items, a.passes, range(a.seeds), a.noise, a.driven
        row(f"R1. onehot, continuous [driven {D}]", "onehot", "continuous", 1 / L, 300, L, P, N, S, driven=D)
        row(f"R2. onehot, fenced [driven {D}]", "onehot", "fenced", 1 / L, 300, L, P, N, S, driven=D)
        row(f"R3. onehot, continuous, registry 1/15 start [driven {D}]", "onehot", "continuous", 1 / 15, 300, L, P, N, S, driven=D)
        row(f"R4. symbolic var=0.2, continuous [driven {D}]", "symbolic", "continuous", 1 / L, 300, L, P, N, S, driven=D)
        row(f"R5. pipeline config: symbolic, fenced, n_exc 500 [driven {D}]", "symbolic", "fenced", 1 / 15, 500, L, P, N, S, driven=D)
        return
    L, P, S, N = a.items, a.passes, range(a.seeds), a.noise

    row("1. sanity config + second list", "onehot", "continuous", 1 / L, 300, L, P, N, S)
    row("2. ...fenced fit_sequence", "onehot", "fenced", 1 / L, 300, L, P, N, S)
    row("3. ...continuous, registry target_activity 1/15", "onehot", "continuous", 1 / 15, 300, L, P, N, S)
    row("4. ...continuous, target 1/L, SymbolicEncoder var=0.2", "symbolic", "continuous", 1 / L, 300, L, P, N, S)
    row("5. ...continuous, target 1/L, SymbolicEncoder var=2.0 (near-orthogonal)", "symbolic", "continuous", 1 / L, 300, L, P, N, S, cat_var=2.0)
    row("6. pipeline config: symbolic var=0.2, fenced, target 1/15, n_exc 500", "symbolic", "fenced", 1 / 15, 500, L, P, N, S)


if __name__ == "__main__":
    main()
