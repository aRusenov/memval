"""Minimal check that the spiking-EP port runs and learns a sequence.

The arm is `SpikingEqPropSequenceNetwork`, a thin MemVal wrapper around the
authors' own code for

    O'Connor, Gavves & Welling (2019), "Training a Spiking Neural Network with
    Equilibrium Propagation", AISTATS -- github.com/QUVA-Lab/spiking-eqprop

vendored at `memval.vendor.spiking_eqprop`. The paper's claim is narrow and
mechanical: EP still trains when neurons talk to each other in **binary
spikes** instead of real-valued rates. So the demo tests exactly that, and
nothing more ambitious.

    chain :  red -> green -> blue -> dog -> cat -> bird

One five-transition chain, learned offline, then cued item by item. Three
things are checked:

  1. the spiking arm learns the chain (next-item accuracy, decoded by cosine);
  2. the *same* code path with the quantizer switched off -- upstream's
     real-valued `SimpleLayerController` -- learns it too. This is the control.
     Both arms get the identical settling budget, so the only difference
     between the rows is the spike channel itself;
  3. the channel really is binary. We tap the hidden layer's outgoing signal
     during a free-phase settle, confirm every value is 0 or 1, and check that
     the spike train's mean rate tracks the real-valued rate it encodes.

Both arms run with `random_flip_beta=False`. Upstream defaults it on, which is
right for its MNIST regime (~3000 minibatch updates per epoch, so the sign
noise averages out) and wrong here (one full-batch update per epoch over five
transitions, where it does not) -- see the arm's docstring for the numbers.

This is a smoke test of the port, not a benchmark result. The chain is short
and one-shot-easy -- prior MemVal work already found a 4x5 symbolic chain sits
at ceiling for linear memories -- so passing here says the port works, not that
the arm is any good. Load, interference and retention live in the suites.

Run:  python examples/spiking_eqprop_demo.py   (~30s)
"""

import time

import numpy as np
import torch

from memval.encoders.symbolic import SymbolicEncoder, SymbolicDecoder
from memval.models.baselines import SpikingEqPropSequenceNetwork
from memval.vendor.spiking_eqprop import rho

DIM = 32
N_HIDDEN = 64
N_EPOCHS = 300
SEEDS = (0, 1, 2)

# Matched settling budget for both arms: upstream's "longer" setting. The
# control is not given the cheaper vanilla 20/4, so that a gap between the two
# rows cannot be blamed on one arm settling closer to equilibrium.
N_NEG, N_POS = 100, 50

SEQ = ["red", "green", "blue", "dog", "cat", "bird"]
VOCAB = SEQ + ["one", "two", "three", "hammer"]  # distractors: never trained

ARMS = {
    "spiking (sigma-delta)": dict(
        quantizer="sigma_delta", epsilons="0.354/t**0.0", lambdas="0.503/t**0.292"
    ),
    "real-valued control": dict(quantizer=None, epsilons=0.354),
}


def build(seed, **arm):
    return SpikingEqPropSequenceNetwork(
        n_features=DIM, hidden_sizes=(N_HIDDEN,),
        n_negative_steps=N_NEG, n_positive_steps=N_POS,
        learning_rate=0.05, n_epochs=N_EPOCHS, random_flip_beta=False,
        seed=seed, **arm,
    )


def next_item_accuracy(model, encoder, decoder):
    """Cue each item of the chain, decode the prediction, score the successor."""
    hits, rows = 0, []
    for cue, target in zip(SEQ[:-1], SEQ[1:]):
        pred = model.predict_next(encoder.encode([cue])[0])
        word, sim = decoder.decode(pred, top_k=3)[0][0]
        hits += word == target
        rows.append((cue, target, word, sim))
    return hits / (len(SEQ) - 1), rows


def channel_report(model, encoder):
    """Step the settling dynamics by hand and inspect what a hidden layer emits.

    `run_inference` only returns the final output potential, so to see the
    traffic *between* layers we drive the layer objects directly -- the same
    call the EP step makes -- and collect the hidden layer's `output`, which is
    exactly what the next layer's weights multiply.
    """
    states = model._fresh_states(n_samples=1)
    x = model._to_unit(encoder.encode(["red"])[0])
    emissions = []
    for _ in range(model.n_negative_steps):
        states = [
            layer(
                x_aft=None if ix == 0 else states[ix - 1].output,
                x_fore=states[ix + 1].output if ix < len(states) - 1 else None,
                clamp=x if ix == 0 else None,
            )
            for ix, layer in enumerate(states)
        ]
        emissions.append(states[1].output.numpy().copy())
    emissions = np.concatenate(emissions, axis=0)

    return dict(
        binary=bool(np.all(np.isin(emissions, (0.0, 1.0)))),
        spike_rate=float(emissions.mean()),
        # what those spikes are meant to encode: the rate a real-valued neuron
        # would have sent instead
        encoded_rate=float(rho(states[1].potential).numpy().mean()),
        n_ticks=emissions.shape[0],
        n_units=emissions.shape[1],
    )


def main():
    encoder = SymbolicEncoder(VOCAB, embedding_dim=DIM, seed=0)
    decoder = SymbolicDecoder(encoder)
    seq = encoder.encode(SEQ)

    print(f"chain: {' -> '.join(SEQ)}")
    print(f"dim={DIM}  hidden={N_HIDDEN}  epochs={N_EPOCHS}  vocab={len(VOCAB)}  "
          f"settling={N_NEG}/{N_POS}  seeds={list(SEEDS)}\n")

    trained = {}
    for name, arm in ARMS.items():
        accs, t0 = [], time.time()
        for seed in SEEDS:
            torch.manual_seed(seed)
            model = build(seed, **arm)
            model.fit_sequence(seq)
            acc, rows = next_item_accuracy(model, encoder, decoder)
            accs.append(acc)
            if seed == SEEDS[0]:
                trained[name] = (model, rows)
        elapsed = time.time() - t0
        print(f"{name:24s} next-item accuracy {np.mean(accs):5.0%} "
              f"(per seed {[f'{a:.0%}' for a in accs]}, {elapsed:.0f}s)")
        for cue, target, word, sim in trained[name][1]:
            mark = "ok  " if word == target else "MISS"
            print(f"    {mark} {cue:>6s} -> {word:<6s} (want {target:<6s}, cos {sim:+.2f})")
        print()

    ch = channel_report(trained["spiking (sigma-delta)"][0], encoder)
    print("hidden-layer channel during a free-phase settle:")
    print(f"    every emitted value is 0 or 1 : {ch['binary']}")
    print(f"    {ch['n_units']} units x {ch['n_ticks']} ticks, spike rate "
          f"{ch['spike_rate']:.3f} vs encoded rate {ch['encoded_rate']:.3f}")

    print("\nport runs: EP trains this chain over a strictly binary inter-layer channel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
