"""
Theta-phase encode/retrieve sequence memory.

Reference
---------
Hasselmo, Bodelon & Wyble, "A proposed function for hippocampal theta rhythm:
separate phases of encoding and retrieval enhance reversal of prior learning",
Neural Computation 14(4):793-817 (2002). Ported from the published equations
(2.2, 2.3, 2.5, 2.7, 2.14); no public implementation of the 2002 model exists.
The biophysical instantiation of the same theory --- Cutsuridis, Cobb & Graham,
Hippocampus 20(3):423-446 (2010), ModelDB 123815 --- is a NEURON CA1 microcircuit
and is *not* what this ports.

The mechanism
-------------
Three oscillatory variables gate one theta cycle. Two scale synaptic
transmission between 1 and (1-X); the third, plasticity, is zero-mean and so
changes sign within the cycle:

    mu_CA3(t) = (X/2) sin(t + phi_CA3) + (1 - X/2)                     (2.2)
    mu_EC(t)  = (X/2) sin(t + phi_EC)  + (1 - X/2)                     (2.3)
    mu_LTP(t) = sin(t + phi_LTP)                                       (2.5)

    a_CA1(t)  = mu_EC(t) a_EC(t) + mu_CA3(t) W a_CA3(t)                (2.4)
    dW        = integral of  mu_LTP(t) a_CA1(t) a_CA3(t)^T  dt         (2.7)

Under the sequence mapping ``a_CA3 = x_t`` (the cue) and ``a_EC = x_{t+1}`` (the
target; the paper sets ``W_EC`` to identity), the two half-cycles do opposite
things --- encode the target while recurrent retrieval is suppressed, then
subtract what is retrieved while plasticity is depotentiating.

Because ``mu_LTP`` is zero-mean the DC term of each gate drops out of the cycle
integral, leaving two coefficients:

    coef_EC  = (X/4) cos(phi_LTP - phi_EC)
    coef_CA3 = (X/4) cos(phi_LTP - phi_CA3)

which is the paper's own performance measure (2.14), and which ``M()`` returns.

Phases are independent
----------------------
``phi_EC`` and ``phi_CA3`` are **separate** parameters, as in the paper: its
Figure 4 is a two-dimensional surface over ``phi_LTP - phi_EC`` and
``phi_LTP - phi_CA3``, with two independent conditions ---

    correct association grows for   -pi/2 < phi_LTP - phi_EC  < pi/2
    old association decays for       pi/2 < phi_LTP - phi_CA3 < 3pi/2

Locking CA3 to the exact antiphase of EC collapses that surface onto the one
diagonal where the two coefficients always cancel, which makes the update an
exact delta rule at *every* offset and the paper's central claim untestable by
construction. Antiphase is the paper's **optimum**, not its constraint, and the
defaults here sit at that optimum while leaving the surface free.

At the defaults (``X=1``, ``phi_EC = phi_LTP = 0``, ``phi_CA3 = pi``) the cycle
integral is exactly ``(x_next - W x_cur) x_cur^T`` --- the delta rule --- so this
arm and ``AsymmetricHopfieldNetwork`` take *identical* steps at the same
``learning_rate``. That makes them a controlled pair: the same rule, with the
error derived by oscillatory phase separation instead of explicit subtraction.
The one remaining difference is the readout --- AHN rectifies, equation 2.4 does
not --- so ``rectify_output=True`` is needed to match them on *predictions* as
well as on weights.
(Ketz, Morkonda & O'Reilly, PLoS Comp Biol 9(6):e1003067, 2013 independently
published the theta-yields-error-driven-learning claim, implemented as
contrastive Hebbian learning in Leabra --- EP's family rather than this one.)

Reversal, and the fornix-lesion control
---------------------------------------
The paper's task is T-maze reversal, and ``modulation_depth`` (X) is its lesion
control: at ``X=0`` both gates go constant, the zero-mean ``mu_LTP`` integrates
them away, no weight change occurs during reversal, and the originally learned
association persists unopposed. That is the paper's account of why fornix
lesions impair reversal.

Error trials need no special path here. The paper represents an unrewarded trial
as ``a_EC = 0`` ("Because the rat does not encounter food reward on erroneous
trials, the EC activity vector representing food reward is zero"), so the cycle
reduces to ``coef_CA3 (W x_cur) x_cur^T`` --- pure depotentiation, proportional
to whatever is currently retrieved. ``SpatialReversalBenchmark`` produces that
by zeroing the reward block of the target vector, which *is* ``a_EC = 0`` given
that EC carries reward and CA3 carries place.

Deviations from the paper, to carry into any writeup
-----------------------------------------------------
- The paper's ``a_CA3`` and ``a_EC`` elements "assume only two values, zero or
  q". Ours are continuous vectors.
- No modification threshold. Section 2.6 extends the model with thresholded
  plasticity plus somatic shunting (``a_CA1^s = mu_soma a_CA1^d``, eq 2.15),
  yielding "a performance measure very similar to equation 2.14, with a maximum
  only slightly smaller". Without it this arm is linear, so it stays on the
  linear-oracle list.
- Rate-coded and clocked by event ordinal, like every arm except the DTS-ESN.
  It is **rate x online**: one theta cycle per transition, so ``fit_event`` is
  the primary path. Making arrival *time* load-bearing (the ``elapsed=``
  convention ``DTSESNSequenceNetwork`` establishes) needs a presentation-window
  decision first --- with an item clamped across the whole cycle the update is
  invariant to the phase at which the cycle starts.
"""

from typing import Optional

import numpy as np

from ..base import HippocampalModel
from ..capabilities import OnlineTrainable, RolloutMode


class ThetaPhaseSequenceNetwork(HippocampalModel, OnlineTrainable):
    """Sequence memory whose error signal arises from theta phase separation."""

    #: `recall` feeds the raw `predict_next` output straight back as the next cue.
    rollout_mode = RolloutMode.OBSERVATION

    #: both paths apply the same ``_cycle_delta`` per transition.
    online_equivalent = True

    def __init__(
        self,
        n_features: int,
        learning_rate: float = 0.1,
        n_phase_steps: int = 256,
        phase_ec: float = 0.0,
        phase_ca3: float = np.pi,
        phase_ltp: float = 0.0,
        modulation_depth: float = 1.0,
        rectify_output: bool = False,
        n_epochs: int = 1,
        **kwargs,
    ):
        """
        Parameters
        ----------
        n_features : int
            Dimensionality of each observation vector.
        learning_rate : float
            Step size on the cycle-integrated update. At the default phases and
            ``modulation_depth=1`` this is exactly the LMS step of
            ``AsymmetricHopfieldNetwork``.
        n_phase_steps : int
            Points on the phase grid used to discretise the cycle integral.
        phase_ec, phase_ca3, phase_ltp : float
            Phases (radians) of the entorhinal, CA3 and plasticity oscillations,
            equations 2.3, 2.2 and 2.5. Independent, as in the paper --- its
            Figure 4 sweeps ``phi_LTP - phi_EC`` and ``phi_LTP - phi_CA3``
            separately. The defaults sit at the reported optimum: EC in phase
            with LTP, CA3 180 degrees out.
        modulation_depth : float
            The paper's ``X``, in [0, 1]: how deeply theta modulates synaptic
            transmission. 1.0 is intact theta. **0.0 is the fornix-lesion
            control** --- gates go constant, the cycle integral vanishes, and no
            reversal learning occurs.
        rectify_output : bool
            Apply a ReLU to ``predict_next``. False (the default) is faithful to
            equation 2.4, which is linear. Set True to match
            ``AsymmetricHopfieldNetwork``, whose readout *is* rectified --- the
            two arms compute identical weight updates, so the readout is the
            only thing separating them, and this flag makes the controlled pair
            exact on predictions as well as on weights.
        n_epochs : int
            Passes over a sequence in ``fit_sequence``. The model is naturally
            online, so 1 is the honest default.
        """
        super().__init__(**kwargs)
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.n_phase_steps = int(n_phase_steps)
        self.phase_ec = float(phase_ec)
        self.phase_ca3 = float(phase_ca3)
        self.phase_ltp = float(phase_ltp)
        self.modulation_depth = float(modulation_depth)
        self.rectify_output = bool(rectify_output)
        self.n_epochs = int(n_epochs)

        self.W = np.zeros((n_features, n_features))
        self.current_state = np.zeros(n_features)
        # online-streaming buffer, so `fit_event` can form a (prev -> current)
        # transition. None means "start of stream".
        self._prev_event: Optional[np.ndarray] = None

        self._build_phase_profiles()

    # ------------------------------------------------------------------
    # Phase machinery
    # ------------------------------------------------------------------
    def _build_phase_profiles(self) -> None:
        """Discretise the three theta oscillations and collapse the cycle
        integral.

        ``a_CA1`` is linear in the gates, so equation 2.7 reduces to two scalar
        coefficients --- the correlation of the plasticity profile with each
        transmission profile. The profiles are kept as arrays so a caller can
        substitute their own and the coefficients follow.
        """
        phi = np.linspace(0.0, 2.0 * np.pi, self.n_phase_steps, endpoint=False)
        x = self.modulation_depth
        self.phi = phi
        self.mu_ec = (x / 2.0) * np.sin(phi + self.phase_ec) + (1.0 - x / 2.0)
        self.mu_ca3 = (x / 2.0) * np.sin(phi + self.phase_ca3) + (1.0 - x / 2.0)
        self.mu_ltp = np.sin(phi + self.phase_ltp)

        self._coef_ec = float(np.mean(self.mu_ltp * self.mu_ec))
        self._coef_ca3 = float(np.mean(self.mu_ltp * self.mu_ca3))
        # Fixed reference: the encode coefficient at X=1 with EC aligned to LTP,
        # i.e. 1/4. Deliberately NOT a function of X --- normalising by X would
        # cancel the modulation depth and destroy the lesion control.
        self._norm = float(np.mean(np.sin(phi) * (0.5 * np.sin(phi) + 0.5)))

    def phase_coefficients(self) -> dict:
        """The two cycle-integral coefficients, normalised so that the paper's
        optimum gives ``encode=+1, retrieve=-1``.

        ``residual`` is their sum: zero means the update is exactly
        error-correcting, and it departs from zero as the phase relationship
        moves off the paper's configuration.
        """
        return {
            "encode": self._coef_ec / self._norm,
            "retrieve": self._coef_ca3 / self._norm,
            "residual": (self._coef_ec + self._coef_ca3) / self._norm,
        }

    def M(self, K: float = 1.0) -> float:
        """The paper's performance measure, equation 2.14::

            M = (X/2) pi cos(phi_LTP - phi_EC) - K - (X/2) pi cos(phi_LTP - phi_CA3)

        Higher M means more retrieval of the correct post-reversal association
        and less of the stale one. ``K`` is the strength of the association
        carried in from initial learning; the paper plots Figure 4 at ``X=K=1``.
        Analytic, so this reproduces that figure exactly without simulating.
        """
        half_x_pi = (self.modulation_depth / 2.0) * np.pi
        return float(
            half_x_pi * np.cos(self.phase_ltp - self.phase_ec)
            - K
            - half_x_pi * np.cos(self.phase_ltp - self.phase_ca3)
        )

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------
    def _cycle_delta(self, x_cur: np.ndarray, x_next: np.ndarray) -> np.ndarray:
        """One theta cycle's integrated weight change (eq 2.7) for a
        (x_cur -> x_next) transition. Returns the un-scaled delta for W.

        Pass ``x_next`` as zeros for the paper's unrewarded error trial
        (``a_EC = 0``): the encoding term vanishes and what remains is pure
        depotentiation proportional to the retrieved association.
        """
        a_eff = self._coef_ec * x_next + self._coef_ca3 * (self.W @ x_cur)
        return np.outer(a_eff, x_cur) / self._norm

    def fit_sequence(
        self,
        sequence_data: np.ndarray,
        context_data: Optional[np.ndarray] = None,
        epochs: Optional[int] = None,
        **kwargs,
    ):
        """Train on one sequence: one theta cycle per transition, per epoch.

        ``epochs`` overrides ``self.n_epochs`` for this call, matching how the
        symbolic pipeline drives its sweeps.
        """
        sequence_data = np.asarray(sequence_data, dtype=float)
        seq_len = sequence_data.shape[0]
        if seq_len < 2:
            return

        # Fence the batch path with the same seam semantics streaming uses, so
        # `fit_sequence` is exactly `[fit_event(x) for x in seq]` and no stale
        # carried state leaks in or out (online_equivalent = True).
        self.on_event_boundary()

        for _epoch in range(self.n_epochs if epochs is None else int(epochs)):
            for t in range(seq_len - 1):
                self.W += self.learning_rate * self._cycle_delta(
                    sequence_data[t], sequence_data[t + 1]
                )
        self.on_event_boundary()

    def fit_event(self, event: np.ndarray, context: Optional[np.ndarray] = None, **kwargs):
        """Consume one streamed event; run one theta cycle for the
        (prev -> event) transition. The first event of a stream only primes the
        buffer. Call ``on_event_boundary`` at a sequence seam so no spurious
        transition forms across it."""
        x = np.asarray(event, dtype=float).ravel()
        if self._prev_event is not None:
            self.W += self.learning_rate * self._cycle_delta(self._prev_event, x)
        self._prev_event = x

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_next(
        self, current_event: np.ndarray, current_context: Optional[np.ndarray] = None, **kwargs
    ) -> np.ndarray:
        """Retrieval-phase readout. A pure function of the cue --- no internal
        state is read or mutated, which is what ``measure_recall_associative``
        requires of every stateful arm. ``current_context`` is accepted for
        interface parity and unused."""
        x = np.asarray(current_event, dtype=float).ravel()
        pred = self.W @ x
        if self.rectify_output:
            pred = np.maximum(0.0, pred)
        self.current_state = pred.copy()
        return pred

    def recall(
        self,
        prompt_event: np.ndarray,
        length: int,
        prompt_context: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """Autoregressive rollout from the last vector of the prompt."""
        prompt_event = np.asarray(prompt_event, dtype=float)
        current = prompt_event[-1].copy() if prompt_event.ndim > 1 else prompt_event.copy()

        recalled = np.zeros((length, self.n_features))
        for t in range(length):
            current = self.predict_next(current)
            recalled[t] = current
        return recalled

    def get_latent_state(self) -> dict:
        return {
            "W": self.W.copy(),
            "phase_ec": self.phase_ec,
            "phase_ca3": self.phase_ca3,
            "phase_ltp": self.phase_ltp,
            "modulation_depth": self.modulation_depth,
        }

    def reset_context(self):
        """Clear transient state and the streaming buffer; weights untouched."""
        self.current_state = np.zeros(self.n_features)
        self._prev_event = None

    # ------------------------------------------------------------------
    # Diagnostics capability interface (consumed by memval.diagnostics)
    # ------------------------------------------------------------------
    def named_representations(self, x: np.ndarray) -> dict:
        x = np.asarray(x, dtype=float).ravel()
        return {"input": x, "output": self.W @ x}

    def named_parameters(self) -> dict:
        return {"W": self.W}

    def transition_grads(self, x_t: np.ndarray, x_next: np.ndarray) -> dict:
        return {
            "W": self._cycle_delta(
                np.asarray(x_t, dtype=float).ravel(),
                np.asarray(x_next, dtype=float).ravel(),
            )
        }
