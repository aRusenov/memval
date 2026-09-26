"""
Minimal standalone T-maze reversal, straight from Hasselmo, Bodelon & Wyble
(2002) section 2.3. No benchmark machinery -- four basis vectors and three
stages, so every number is inspectable.

The paper's protocol and its stated assumptions:

    stage 1  initial learning    n trials   L -> F_L
    stage 2  erroneous reversal  e trials   L -> 0      <- a_EC = 0, no food
    stage 3  correct reversal    c trials   R -> F_R

    a_CA3(n) . a_CA3(e) = 1     same left-arm place cells in stages 1 and 2
    a_CA3(n) . a_CA3(c) = 0     left and right place codes orthogonal
    a_EC(left) . a_EC(right) = 0

Place and food live in orthogonal subspaces of one vector, so W is the
CA3->CA1 map: cue with a place vector, read out a food vector.

Because L and R are orthogonal, learning R -> F_R cannot disturb L -> F_L on
its own. **Extinction is the only thing that can remove the stale association**,
which is exactly why the paper's error trials carry the whole argument.

Run:  python examples/theta_reversal_demo.py
"""

import numpy as np

from memval.models.baselines import ThetaPhaseSequenceNetwork

DIM = 8
L, R, F_L, F_R = (np.eye(DIM)[i] for i in range(4))  # place L/R, food left/right
N_ACQ, N_ERR, N_REV = 20, 10, 20
LR = 0.1


def probe(model):
    """The paper's retrieval test at the choice point: how much retrieval
    resembles the new correct food location, versus the old incorrect one."""
    stale = float(model.predict_next(L) @ F_L)
    correct = float(model.predict_next(R) @ F_R)
    return stale, correct, correct - stale


def run_protocol(model, lesion_after_acquisition=False):
    """Three stages; returns the probe after each."""
    trace = {}
    for _ in range(N_ACQ):
        model.fit_sequence(np.stack([L, F_L]))
    trace["1. acquisition"] = probe(model)

    if lesion_after_acquisition:
        model.modulation_depth = 0.0
        model._build_phase_profiles()

    for _ in range(N_ERR):
        model.fit_sequence(np.stack([L, np.zeros(DIM)]))  # a_EC = 0
    trace["2. error trials"] = probe(model)

    for _ in range(N_REV):
        model.fit_sequence(np.stack([R, F_R]))
    trace["3. reversal"] = probe(model)
    return trace


def theta(**over):
    kw = dict(n_features=DIM, learning_rate=LR)
    kw.update(over)
    return ThetaPhaseSequenceNetwork(**kw)


def show(title, trace):
    print(f"  {title}")
    print(f"    {'stage':<18}{'stale (L->F_L)':>16}{'correct (R->F_R)':>18}{'M':>10}")
    for stage, (stale, correct, m) in trace.items():
        print(f"    {stage:<18}{stale:>16.4f}{correct:>18.4f}{m:>10.4f}")


def main():
    print("=" * 74)
    print("1. Intact theta")
    print("=" * 74)
    show("X=1, phases at the paper's optimum", run_protocol(theta()))
    print("    Acquisition converges to 1-(1-lr)^n = 0.878, not 1.0 -- the delta")
    print("    rule approaching its fixed point. Error trials then decay the stale")
    print("    association geometrically by (1-lr) per trial, 0.878 -> 0.306 over")
    print("    10 trials, because depotentiation is proportional to what is")
    print("    retrieved. Extinction rate and learning rate are the same constant.")

    print()
    print("=" * 74)
    print("2. Fornix lesion (X=0), applied after acquisition")
    print("=" * 74)
    show("theta abolished from the first error trial",
         run_protocol(theta(), lesion_after_acquisition=True))
    print("    The stale association survives error trials untouched, and the")
    print("    new one is never acquired. NB this is the *equations'* lesion")
    print("    (eq 2.14 at X=0): with the gates flat and mu_LTP zero-mean the")
    print("    cycle integral vanishes, so the model freezes. Figure 6 instead")
    print("    describes retrieval actively *strengthening* the stale link,")
    print("    which needs mu_LTP flattened too -- not implemented.")

    print()
    print("=" * 74)
    print("3. Detuning the CA3 phase: the paper's claim, made behavioural")
    print("=" * 74)
    print("    The paper requires pi/2 < phi_LTP - phi_CA3 < 3pi/2 for the old")
    print("    association to decay. Below, only phase_ca3 moves.")
    print()
    print("    Detuning also changes how much is acquired, so the isolated")
    print("    effect of extinction is the ratio stale_after_err / stale_after_acq:")
    print("    <1 extinguished, =1 untouched, >1 reinforced.")
    print()
    print(f"    {'phi_CA3':<10}{'retrieve':>10}{'after acq':>12}{'after err':>12}"
          f"{'ratio':>9}{'M final':>10}")
    for frac in (1.0, 0.75, 0.5, 0.25, 0.0):
        m = theta(phase_ca3=frac * np.pi)
        for _ in range(N_ACQ):
            m.fit_sequence(np.stack([L, F_L]))
        after_acq = probe(m)[0]
        for _ in range(N_ERR):
            m.fit_sequence(np.stack([L, np.zeros(DIM)]))
        after_err = probe(m)[0]
        for _ in range(N_REV):
            m.fit_sequence(np.stack([R, F_R]))
        coef = m.phase_coefficients()["retrieve"]
        ratio = after_err / after_acq if after_acq else float("nan")
        print(f"    {frac:>4.2f}pi    {coef:>10.3f}{after_acq:>12.4f}"
              f"{after_err:>12.4f}{ratio:>9.3f}{probe(m)[2]:>10.4f}")
    print()
    print("    1.00pi is the paper's optimum: retrieval subtracts, ratio 0.349.")
    print("    0.50pi zeroes the retrieval coefficient, so error trials do")
    print("    nothing (ratio 1.000) -- and acquisition runs unchecked Hebbian,")
    print("    overshooting to 2.0 because nothing subtracts the prediction.")
    print("    0.00pi flips its sign: error trials *reinforce* the stale link,")
    print("    so the wrong association grows on exactly the trials meant to")
    print("    extinguish it. Only this arm can be detuned this way.")


if __name__ == "__main__":
    main()
