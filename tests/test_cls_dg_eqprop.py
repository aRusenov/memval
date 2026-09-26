import numpy as np
import pytest
from memval.models.baselines import CLSDGEqPropNetwork

def test_cls_dg_eqprop_buffer():
    # Setup sequences of length 3, 2 features
    seq1 = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    seq2 = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    seq3 = np.array([[1.0, 1.0], [0.0, 0.0], [0.1, 0.9]])

    # Initialize model with max_buffer_size = 2
    model = CLSDGEqPropNetwork(n_features=2, max_buffer_size=2, auto_consolidate=False)
    
    # Fit first sequence
    model.fit_sequence(seq1)
    assert len(model._replay_buffer) == 1
    assert np.allclose(model._replay_buffer[0][0], seq1)
    
    # Fit second sequence
    model.fit_sequence(seq2)
    assert len(model._replay_buffer) == 2
    assert np.allclose(model._replay_buffer[0][0], seq1)
    assert np.allclose(model._replay_buffer[1][0], seq2)

    # Fit third sequence: seq1 should be evicted (FIFO)
    model.fit_sequence(seq3)
    assert len(model._replay_buffer) == 2
    assert np.allclose(model._replay_buffer[0][0], seq2)
    assert np.allclose(model._replay_buffer[1][0], seq3)


def test_cls_dg_eqprop_blending():
    seq = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    
    # Init model
    model = CLSDGEqPropNetwork(n_features=2, blend_alpha=0.5, seed=42)
    model.fit_sequence(seq)

    # Predict with blend_alpha=1.0 (HPC only)
    model.reset_context()
    pred_hpc = model.predict_next(seq[0], blend_alpha=1.0)

    # Predict with blend_alpha=0.0 (CTX only)
    model.reset_context()
    pred_ctx = model.predict_next(seq[0], blend_alpha=0.0)

    # Predict with blend_alpha=0.5 (Blended)
    model.reset_context()
    pred_blended = model.predict_next(seq[0], blend_alpha=0.5)

    # Verification: blended prediction should be a weighted sum
    expected = 0.5 * pred_hpc + 0.5 * pred_ctx
    assert np.allclose(pred_blended, expected, atol=1e-6)


def test_cls_dg_eqprop_consolidation():
    # Setup simple sequence in 5D to prevent low-dimensional coincidences
    seq = np.array([
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0]
    ])

    # Initialize model with auto_consolidate=False so CTX is untrained initially
    model = CLSDGEqPropNetwork(
        n_features=5, 
        n_dg=100, 
        n_hidden=32, 
        learning_rate=0.15, 
        ctx_epochs=20, 
        auto_consolidate=False, 
        seed=42
    )

    # Fit sequence online (only HPC learns)
    model.fit_sequence(seq)

    # CTX should be untrained: it should have high error on predicting seq[1] from seq[0]
    model.reset_context()
    pred_ctx_untrained = model.predict_next(seq[0], blend_alpha=0.0)
    error_untrained = np.mean((pred_ctx_untrained - seq[1]) ** 2)

    # Run consolidation
    model.consolidate(n_cycles=5)

    # CTX should now be trained: error on prediction should be reduced significantly
    model.reset_context()
    pred_ctx_trained = model.predict_next(seq[0], blend_alpha=0.0)
    error_trained = np.mean((pred_ctx_trained - seq[1]) ** 2)

    # Verify that consolidation improved the neocortical recall
    assert error_trained < error_untrained


def test_cls_dg_eqprop_reset_context():
    model = CLSDGEqPropNetwork(n_features=2, seed=42)
    
    # Store initial weights
    hpc_W_ho_init = model.hpc.ep_net.W_ho.copy()
    ctx_W_ho_init = model.ctx.ep_net.W_ho.copy()

    # Trigger some states
    seq = np.array([[1.0, 0.0], [0.0, 1.0]])
    model.predict_next(seq[0])

    # Ensure context has changed transient state
    assert not np.allclose(model.hpc.ep_net.current_state, 0.0)
    assert not np.allclose(model.ctx.ep_net.current_state, 0.0)

    # Reset context
    model.reset_context()

    # Transient states should be reset to 0
    assert np.allclose(model.hpc.ep_net.current_state, 0.0)
    assert np.allclose(model.ctx.ep_net.current_state, 0.0)

    # Learned weights must remain untouched
    assert np.allclose(model.hpc.ep_net.W_ho, hpc_W_ho_init)
    assert np.allclose(model.ctx.ep_net.W_ho, ctx_W_ho_init)


def test_cls_dg_eqprop_novelty_gate():
    model = CLSDGEqPropNetwork(
        n_features=2, 
        blend_alpha=0.5, 
        use_novelty_gate=True, 
        novelty_scale=1.5, 
        seed=42
    )

    # Initial state: last_pred_hpc is None
    assert model.last_pred_hpc is None

    # Step 1: Predict next event
    event1 = np.array([1.0, 0.0])
    pred_hpc_1 = model.hpc.predict_next(event1)
    pred_ctx_1 = model.ctx.predict_next(event1)
    
    # First prediction should fall back to blend_alpha because last_pred_hpc was None
    model.reset_context()
    pred_blended_1 = model.predict_next(event1)
    assert np.allclose(pred_blended_1, 0.5 * pred_hpc_1 + 0.5 * pred_ctx_1)
    assert np.allclose(model.last_pred_hpc, pred_hpc_1)

    # Step 2: Predict from next event with novelty gate active
    event2 = np.array([0.0, 1.0])
    
    # Manually compute expected alpha
    error = np.linalg.norm(event2 - pred_hpc_1)
    expected_alpha = 1.0 - np.exp(-error / 1.5)
    
    pred_hpc_2 = model.hpc.predict_next(event2)
    pred_ctx_2 = model.ctx.predict_next(event2)
    expected_blended = expected_alpha * pred_hpc_2 + (1.0 - expected_alpha) * pred_ctx_2

    # Run the model's blended prediction
    # We do NOT reset context here so that last_pred_hpc is preserved from the first step
    # Note: to prevent the underlying model state from advancing twice (since we called predict_next
    # on hpc/ctx manually), we can reset context, set last_pred_hpc back, and then run it.
    model.reset_context()
    model.last_pred_hpc = pred_hpc_1
    pred_blended_2 = model.predict_next(event2)
    
    assert np.allclose(pred_blended_2, expected_blended, atol=1e-6)

    # Verify override still works
    model.reset_context()
    model.last_pred_hpc = pred_hpc_1
    pred_override = model.predict_next(event2, blend_alpha=0.8)
    assert np.allclose(pred_override, 0.8 * pred_hpc_2 + 0.2 * pred_ctx_2, atol=1e-6)

    # Verify reset_context clears last_pred_hpc
    model.reset_context()
    assert model.last_pred_hpc is None

