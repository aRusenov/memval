"""Render the 5-sequence x 10-item retention matrices (Original / DG / XdG,
use_output_bias=False) produced by standalone_scale_matrix.py. Values are the
seed-averaged noise-free MRR R[j,i] = recall of list i after training through
list j. Regenerates docs/figures/ep_retention_matrices.png."""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NL = 5
NAN = np.nan
MATS = {
    "Original EP\n(diag 1.00 · final 0.26)": [
        [1.00, NAN, NAN, NAN, NAN],
        [0.51, 1.00, NAN, NAN, NAN],
        [0.37, 0.43, 1.00, NAN, NAN],
        [0.16, 0.16, 0.49, 1.00, NAN],
        [0.21, 0.19, 0.28, 0.37, 1.00],
    ],
    "DG-EP\n(diag 0.67 · final 0.06)": [
        [0.60, NAN, NAN, NAN, NAN],
        [0.13, 0.66, NAN, NAN, NAN],
        [0.05, 0.05, 0.73, NAN, NAN],
        [0.05, 0.04, 0.09, 0.73, NAN],
        [0.04, 0.05, 0.06, 0.10, 0.63],
    ],
    "XdG-EP + no bias\n(diag 0.89 · final 0.46)": [
        [0.91, NAN, NAN, NAN, NAN],
        [0.53, 0.88, NAN, NAN, NAN],
        [0.31, 0.56, 0.96, NAN, NAN],
        [0.26, 0.44, 0.76, 0.89, NAN],
        [0.22, 0.33, 0.58, 0.72, 0.82],
    ],
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
cmap = plt.cm.viridis.copy()
cmap.set_bad("#eeeeee")            # masked (upper-triangle) cells

im = None
for col, (ax, (title, M)) in enumerate(zip(axes, MATS.items())):
    A = np.ma.masked_invalid(np.array(M, dtype=float))
    im = ax.imshow(A, cmap=cmap, vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_title(title, fontsize=10)
    ax.set_xticks(range(NL)); ax.set_xticklabels([f"L{i}" for i in range(NL)])
    ax.set_yticks(range(NL))
    ax.set_yticklabels([f"after L{j}" for j in range(NL)] if col == 0 else [])
    ax.set_xlabel("recall of list i")
    for j in range(NL):
        for i in range(NL):
            v = M[j][i]
            if not np.isnan(v):
                ax.text(i, j, f"{v:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if v < 0.6 else "black")
axes[0].set_ylabel("after training through list j")

fig.suptitle("Continual-learning retention matrix  R[j,i] = recall MRR of list i after training through list j\n"
             "5 sequences x 10 items (read DOWN a column = decay of that list as later lists are added)",
             fontsize=11)
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("noise-free recall MRR")
out = os.path.join(os.path.dirname(__file__), "ep_retention_matrices.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved", out)
