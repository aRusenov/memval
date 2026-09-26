"""
Demo: the memval.diagnostics framework on OriginalEqPropSequenceNetwork.

Runs the unified A->B forgetting report (behavioural + geometry + weight tiers)
through the public API — the same numbers the session's hand-rolled standalone
scripts produced, now behind one call. Saves the paired diagnostic panel.
"""

import numpy as np

from memval.models.baselines.original_eqprop import OriginalEqPropSequenceNetwork
from memval.diagnostics import ab_forgetting_report, format_report, plot_report


class ToyEncoder:
    """Non-negative unit codes (representable under the [0,1] state clip)."""
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


def main():
    list_a = list("ABCDEF")
    list_b = list("GHIJKL")
    dim = 64
    enc = ToyEncoder(list_a + list_b, dim=dim, seed=0)

    model = OriginalEqPropSequenceNetwork(
        n_features=dim, n_hidden=128, learning_rate=0.1, beta=0.5,
        n_settle_steps=50, n_epochs=300, seed=0)

    report = ab_forgetting_report(model, enc, list_a, list_b)

    print("capabilities:", report["capabilities"])
    print()
    print(format_report(report))

    out = "/private/tmp/claude-501/-Users-atanas-Documents-workspace-memval/" \
          "5f5bf02a-2fe4-40b1-89ea-d6db9416340e/scratchpad/diagnostics_panel.png"
    plot_report(report, out, title="Original EP — A→B forgetting diagnostics")
    print(f"\npanel saved to {out}")


if __name__ == "__main__":
    main()
