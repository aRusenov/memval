import numpy as np
from memval.models.baselines import KNNEpisodicModel

def test_knn_episodic_basic():
    # 3 features, sequence of length 4
    seq1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0]
    ])
    
    # Another sequence
    seq2 = np.array([
        [0.5, 0.5, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.1, 0.2, 0.3]
    ])
    
    model = KNNEpisodicModel(n_features=3)
    model.fit_sequence(seq1)
    model.fit_sequence(seq2)
    
    # Test step-by-step prediction for seq1
    model.reset_context()
    for i in range(3):
        model.current_t = i
        pred = model.predict_next(seq1[i])
        # The prediction should match seq1[i+1]
        assert np.allclose(pred, seq1[i+1])
        
    # Test step-by-step prediction for seq2
    model.reset_context()
    for i in range(3):
        model.current_t = i
        pred = model.predict_next(seq2[i])
        # The prediction should match seq2[i+1]
        assert np.allclose(pred, seq2[i+1])

    # Test autoregressive recall
    model.reset_context()
    recalled = model.recall(seq1[0], length=3)
    assert recalled.shape == (3, 3)
    assert np.allclose(recalled[0], seq1[1])
    assert np.allclose(recalled[1], seq1[2])
    assert np.allclose(recalled[2], seq1[3])

def test_knn_episodic_context():
    # Identical events, different contexts
    seq1 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    ctx1 = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0]
    ])

    seq2 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.1, 0.2, 0.3] # different termination
    ])
    ctx2 = np.array([
        [0.0, 1.0],
        [0.0, 1.0],
        [0.0, 1.0]
    ])

    model = KNNEpisodicModel(n_features=3)
    model.fit_sequence(seq1, ctx1)
    model.fit_sequence(seq2, ctx2)

    # Predict with context 1
    model.reset_context()
    pred1 = model.predict_next(seq1[0], current_context=ctx1[0]) # at t=0
    pred2 = model.predict_next(seq1[1], current_context=ctx1[1]) # at t=1
    assert np.allclose(pred2, seq1[2])

    # Predict with context 2
    model.reset_context()
    pred1_alt = model.predict_next(seq2[0], current_context=ctx2[0]) # at t=0
    pred2_alt = model.predict_next(seq2[1], current_context=ctx2[1]) # at t=1
    assert np.allclose(pred2_alt, seq2[2])
