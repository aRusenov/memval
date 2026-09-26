import numpy as np
import pytest
from memval.models.baselines import EWCDGEqPropSequenceNetwork


def test_ewc_init():
    # Valid initialization
    model = EWCDGEqPropSequenceNetwork(n_features=2, ewc_lambda=50.0, ewc_method="lr")
    assert model.ewc_lambda == 50.0
    assert model.ewc_method == "lr"
    assert model.fisher is None
    assert model.anchor_params is None

    # Case insensitivity test
    model_upper = EWCDGEqPropSequenceNetwork(n_features=2, ewc_method="LR")
    assert model_upper.ewc_method == "lr"

    # Invalid method initialization should fail
    with pytest.raises(ValueError):
        EWCDGEqPropSequenceNetwork(n_features=2, ewc_method="invalid")


def test_ewc_fisher_estimation():
    seq = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    
    for method in ["canonical", "lr", "mas"]:
        model = EWCDGEqPropSequenceNetwork(n_features=2, n_dg=10, n_hidden=5, ewc_method=method, seed=42)
        model.consolidate(seq)
        
        # Verify fisher and anchors are created and have correct shapes
        assert model.fisher is not None
        assert model.anchor_params is not None
        for key in ["W_ih", "W_ho", "b_h", "b_o"]:
            assert key in model.fisher
            assert key in model.anchor_params
            assert model.fisher[key].shape == getattr(model.ep_net, key).shape
            assert model.anchor_params[key].shape == getattr(model.ep_net, key).shape
            # Fisher entries should be non-negative
            assert np.all(model.fisher[key] >= 0.0)


def test_ewc_reset():
    seq = np.array([[1.0, 0.0], [0.0, 1.0]])
    model = EWCDGEqPropSequenceNetwork(n_features=2, n_dg=10, n_hidden=5, seed=42)
    model.consolidate(seq)
    assert model.fisher is not None
    assert model.anchor_params is not None
    
    model.reset_ewc()
    assert model.fisher is None
    assert model.anchor_params is None


def test_ewc_weight_preservation():
    # Verify that EWC penalizes weight drift when training on a new sequence.
    seq1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    seq2 = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    
    # Model A: Trained on seq1, consolidated, then trained on seq2 WITH EWC
    model_ewc = EWCDGEqPropSequenceNetwork(
        n_features=3, n_dg=32, n_hidden=16, ewc_lambda=1.0, ewc_method="lr", n_epochs=5, seed=42
    )
    model_ewc.fit_sequence(seq1)
    model_ewc.consolidate(seq1)
    anchor_W_ho = model_ewc.ep_net.W_ho.copy()
    
    # Fit seq2 with EWC
    model_ewc.fit_sequence(seq2)
    W_ho_ewc = model_ewc.ep_net.W_ho.copy()
    
    # Model B: Trained on seq1, then trained on seq2 WITHOUT EWC (ewc_lambda=0)
    model_no_ewc = EWCDGEqPropSequenceNetwork(
        n_features=3, n_dg=32, n_hidden=16, ewc_lambda=0.0, ewc_method="lr", n_epochs=5, seed=42
    )
    model_no_ewc.fit_sequence(seq1)
    # Fit seq2 without EWC
    model_no_ewc.fit_sequence(seq2)
    W_ho_no_ewc = model_no_ewc.ep_net.W_ho.copy()
    
    # The distance to anchor should be smaller under EWC compared to no EWC
    dist_ewc = np.linalg.norm(W_ho_ewc - anchor_W_ho)
    dist_no_ewc = np.linalg.norm(W_ho_no_ewc - anchor_W_ho)
    
    assert dist_ewc < dist_no_ewc
