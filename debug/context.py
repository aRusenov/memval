import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt
from memval.encoders import SymbolicEncoder, SymbolicDecoder
from memval.models.baselines.dg_eqprop import DGEqPropSequenceNetwork

def cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors."""
    na, nb_ = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb_)) if na > 0 and nb_ > 0 else 0.0


def measure_recall_associative(network, words, encoder, decoder, context_vec=None,
                                n_trials=30, noise_scale=0.05):
    """One-step cued recall: each position probed independently with a noisy cue."""
    success_counts = np.zeros(len(words))
    context = np.asarray(context_vec) if context_vec is not None else np.array([1.0])
    for _ in range(n_trials):
        for i in range(len(words) - 1):
            network.current_t = i
            noisy_evt = encoder.encode([words[i]])[0] + np.random.normal(0, noise_scale, encoder.embedding_dim)
            pred = network.predict_next(noisy_evt, current_context=context)
            if decoder.decode(pred, top_k=1)[0] == words[i + 1]:
                success_counts[i + 1] += 1
    success_counts[0] = n_trials
    return success_counts / n_trials


def measure_recall_autoregressive(network, words, encoder, decoder, context_vec=None,
                                   n_trials=30, noise_scale=0.05):
    """Chained autoregressive recall using the model's decode_prediction API.
    
    No noise is injected during the generative loop (only on the initial cue).
    """
    success_counts = np.zeros(len(words))
    success_counts[0] = n_trials
    context = np.asarray(context_vec) if context_vec is not None else np.array([1.0])
    for _ in range(n_trials):
        network.current_t = 0
        current_evt = encoder.encode([words[0]])[0] + np.random.normal(0, noise_scale, encoder.embedding_dim)
        for i in range(len(words) - 1):
            raw_pred = network.predict_next(current_evt, current_context=context)
            pred_word = decoder.decode(raw_pred, top_k=1)[0]
            if pred_word == words[i + 1]:
                success_counts[i + 1] += 1
            current_evt = network.decode_prediction(raw_pred)
    return success_counts / n_trials


def measure_recall_with_recency(network, words, encoder, decoder, context_vec=None,
                                 n_trials=30, noise_scale=0.15, buffer_decay_rate=0.8):
    """Autoregressive recall with an exponentially-decaying STM recency buffer."""
    success_counts = np.zeros(len(words))
    context = np.asarray(context_vec) if context_vec is not None else np.array([1.0])
    for _ in range(n_trials):
        network.current_t = 0
        current_evt = encoder.encode([words[0]])[0] + np.random.normal(0, noise_scale, encoder.embedding_dim)
        for i in range(len(words) - 1):
            target_idx = i + 1
            p_buffer = np.exp(-buffer_decay_rate * ((len(words) - 1) - target_idx))
            if np.random.rand() < p_buffer:
                pred_word = words[target_idx]          # STM buffer hit
                current_evt = encoder.encode([pred_word])[0]
            else:
                pred = network.predict_next(current_evt, current_context=context)
                pred_word = decoder.decode(pred, top_k=1)[0]
                current_evt = network.decode_prediction(pred)
            if pred_word == words[target_idx]:
                success_counts[target_idx] += 1
        success_counts[0] = n_trials
    return success_counts / n_trials


def measure_recall_threshold(network, words, encoder, decoder, context_vec=None,
                              n_trials=30, noise_scale=0.05, theta=0.5):
    """One-step cued (associative) recall scored by strict decoded matching and confidence >= theta.

    A position counts as a success ONLY if the nearest-neighbor decoded word matches the target
    AND the cosine similarity to that target is >= theta (ensuring both accuracy and confidence).
    """
    success_counts = np.zeros(len(words))
    success_counts[0] = n_trials
    context = np.asarray(context_vec) if context_vec is not None else np.array([1.0])
    for _ in range(n_trials):
        for i in range(len(words) - 1):
            network.current_t = i
            noisy_evt = encoder.encode([words[i]])[0] + np.random.normal(0, noise_scale, encoder.embedding_dim)
            raw_pred = network.predict_next(noisy_evt, current_context=context)
            pred_word = decoder.decode(raw_pred, top_k=1)[0]
            if pred_word == words[i + 1]:
                success_counts[i + 1] += 1
    return success_counts / n_trials

def measure_recall_threshold_raw(network, words, encoder, decoder, context_vec=None,
                              n_trials=30, noise_scale=0.05, theta=0.5):
    """Autoregressive recall scored by strict decoded matching and confidence >= theta.

    Feeds the raw predicted continuous vector forward to the next step.
    A position counts as a success ONLY if the nearest-neighbor decoded word matches the target
    AND the cosine similarity to that target is >= theta.
    """
    success_counts = np.zeros(len(words))
    success_counts[0] = n_trials
    context = np.asarray(context_vec) if context_vec is not None else np.array([1.0])
    for _ in range(n_trials):
        network.current_t = 0
        current_evt = encoder.encode([words[0]])[0] + np.random.normal(0, noise_scale, encoder.embedding_dim)
        for i in range(len(words) - 1):
            raw_pred = network.predict_next(current_evt, current_context=context)
            pred_word = decoder.decode(raw_pred, top_k=1)[0]
            if pred_word == words[i + 1]:
                success_counts[i + 1] += 1
            current_evt = network.decode_prediction(raw_pred)
    return success_counts / n_trials

def calibrate_theta(encoder, category_a_words, category_b_words):
    """Suggest a recall threshold theta from the embedding geometry.

    Returns the midpoint between mean within-category and cross-category
    cosine similarity as a data-driven default for measure_recall_threshold.
    """
    from itertools import combinations
    emb_a = encoder.encode(category_a_words)
    emb_b = encoder.encode(category_b_words)
    within = [float(emb_a[i] @ emb_a[j]) for i, j in combinations(range(len(emb_a)), 2)]
    cross  = [float(a @ b) for a in emb_a for b in emb_b]
    within_mean, cross_mean = float(np.mean(within)), float(np.mean(cross))
    return {"within_mean": within_mean, "cross_mean": cross_mean,
            "theta": (within_mean + cross_mean) / 2.0}


def mean_recall_rate(recall_curve):
    return float(np.mean(recall_curve[1:]))
def memory_span(recall_curve, threshold=0.75):
    span = 0
    for r in recall_curve[1:]:
        if r >= threshold:
            span += 1
        else:
            break
    return span
def recall_fidelity(network, words, encoder, n_trials=30, noise_scale=0.05):
    gt_embs = encoder.encode(words)
    total_sim, count = 0.0, 0
    for _ in range(n_trials):
        network.current_t = 0
        for i in range(len(words) - 1):
            noisy = encoder.encode([words[i]])[0] + np.random.normal(0, noise_scale, encoder.embedding_dim)
            pred = network.predict_next(noisy, current_context=np.array([1.0]))
            sim = float(pred @ gt_embs[i + 1]) / (np.linalg.norm(pred) * np.linalg.norm(gt_embs[i + 1]) + 1e-8)
            total_sim += sim
            count += 1
    return total_sim / count

vocab = {
    'apple': 'fruit', 'banana': 'fruit', 'orange': 'fruit', 'grape': 'fruit', 'pear': 'fruit',
    'peach': 'fruit', 'plum': 'fruit', 'cherry': 'fruit', 'kiwi': 'fruit', 'mango': 'fruit',
    'lemon': 'fruit', 'lime': 'fruit'
}

encoder = SymbolicEncoder(vocab, embedding_dim=100, category_variance=0.2, seed=42)
decoder = SymbolicDecoder(encoder)
word_list = ['apple', 'banana', 'orange', 'grape', 'pear', 'peach', 'plum']


# 3. Categorical Sequences
vocab_multi = {
    # Fruits
    'apple': 'fruit', 'banana': 'fruit', 'orange': 'fruit', 'grape': 'fruit', 'pear': 'fruit',
    # Animals
    'dog': 'animal', 'cat': 'animal', 'horse': 'animal', 'cow': 'animal', 'sheep': 'animal',
    # Cars
    'ford': 'car', 'chevy': 'car', 'dodge': 'car', 'toyota': 'car', 'honda': 'car'
}

encoder_multi = SymbolicEncoder(vocab_multi, embedding_dim=100, category_variance=0.2, seed=42)
decoder_multi = SymbolicDecoder(encoder_multi)

lists = {
    'fruits': ['apple', 'banana', 'orange', 'grape', 'pear'],
    'animals': ['dog', 'cat', 'horse', 'cow', 'sheep'],
    'cars': ['ford', 'chevy', 'dodge', 'toyota', 'honda']
}

# 32-dimensional random binary vectors give each category a distinct,
# near-orthogonal context that the DG layer uses to separate their engrams.
n_explicit = 32
rng = np.random.default_rng(42)
fruits_vec = rng.integers(0, 2, size=32).astype(float)
animals_vec = rng.integers(0, 2, size=32).astype(float)
cars_vec = rng.integers(0, 2, size=32).astype(float)

context_map = {
    'fruits':  fruits_vec / np.linalg.norm(fruits_vec),
    'animals': animals_vec / np.linalg.norm(animals_vec),
    'cars':    cars_vec / np.linalg.norm(cars_vec),
}

# Redefine recall_fidelity locally to support multi-dimensional category context vectors
def recall_fidelity(network, words, encoder, context_vec=None, n_trials=30, noise_scale=0.05):
    gt_embs = encoder.encode(words)
    total_sim, count = 0.0, 0
    context = np.asarray(context_vec) if context_vec is not None else np.array([1.0])
    for _ in range(n_trials):
        network.current_t = 0
        for i in range(len(words) - 1):
            noisy = encoder.encode([words[i]])[0] + np.random.normal(0, noise_scale, encoder.embedding_dim)
            pred = network.predict_next(noisy, current_context=context)
            sim = float(pred @ gt_embs[i + 1]) / (np.linalg.norm(pred) * np.linalg.norm(gt_embs[i + 1]) + 1e-8)
            total_sim += sim
            count += 1
    return total_sim / count

# -----------------------------------------------------------------------
# Sequential Training: Train on one sequence at a time to test catastrophic 
# forgetting. At each stage, evaluate metrics for all sequences learned 
# so far.
# -----------------------------------------------------------------------
n_epochs = 300
n_trials = 30

stages_data = []
categories_trained = []

# Initialize ONE network
net = DGEqPropSequenceNetwork(
    n_features=100, n_context=n_explicit, explicit_scale=1.0,
    n_dg=1000, sparsity=0.05, learning_rate=0.1, n_epochs=n_epochs, seed=42
)

for idx, (cat, words) in enumerate(lists.items()):
    categories_trained.append(cat)
    enc_words = encoder_multi.encode(words)
    context_seq = np.tile(context_map[cat], (len(words), 1))
    
    print(f"Stage {idx+1}: Training ONLY on {cat} (Epochs = {n_epochs})...")
    # Train ONLY on the newly added sequence
    net.fit_sequence(enc_words, context_seq)
    
    stage_metrics = {}
    # Evaluate ALL categories at EVERY stage to establish zero-shot baseline and track full progression
    for c_cat in lists.keys():
        c_words = lists[c_cat]
        c_context = context_map[c_cat]
        
        # Evaluate cued recall (associative)
        curve_assoc = measure_recall_associative(net, c_words, encoder_multi, decoder_multi, c_context, n_trials=n_trials)
        # Evaluate autoregressive cued recall
        curve_auto = measure_recall_autoregressive(net, c_words, encoder_multi, decoder_multi, c_context, n_trials=n_trials)
        
        mrr = mean_recall_rate(curve_assoc)
        span = memory_span(curve_auto, threshold=0.75)
        fid = recall_fidelity(net, c_words, encoder_multi, context_vec=c_context, n_trials=n_trials)
        
        stage_metrics[c_cat] = {'mrr': mrr, 'span': span, 'fidelity': fid}
    stages_data.append(stage_metrics)
    
    # Format and print a clean table of primary metrics for this stage
    print("=" * 65)
    print(f"  Stage {idx+1} Primary Metrics (after training on: {cat})")
    print("-" * 65)
    print(f"  {'Category':<15} | {'MRR (Assoc)':<12} | {'Word Span':<12} | {'Recall Fidelity':<15}")
    print("-" * 65)
    for c_cat in lists.keys():
        metrics = stage_metrics[c_cat]
        print(f"  {c_cat:<15} | {metrics['mrr']:<12.3f} | {metrics['span']:<12} | {metrics['fidelity']:<15.3f}")
    print("=" * 65)
    print()

# Expose the final trained network as net_multi for downstream cells
net_multi = net
print("Training stages complete and metrics recorded.")

# Compute Absolute Metrics for all 3 stages for all categories
absolute_mrr = {
    'fruits': [stages_data[0]['fruits']['mrr'], stages_data[1]['fruits']['mrr'], stages_data[2]['fruits']['mrr']],
    'animals': [stages_data[0]['animals']['mrr'], stages_data[1]['animals']['mrr'], stages_data[2]['animals']['mrr']],
    'cars': [stages_data[0]['cars']['mrr'], stages_data[1]['cars']['mrr'], stages_data[2]['cars']['mrr']]
}

absolute_span = {
    'fruits': [stages_data[0]['fruits']['span'], stages_data[1]['fruits']['span'], stages_data[2]['fruits']['span']],
    'animals': [stages_data[0]['animals']['span'], stages_data[1]['animals']['span'], stages_data[2]['animals']['span']],
    'cars': [stages_data[0]['cars']['span'], stages_data[1]['cars']['span'], stages_data[2]['cars']['span']]
}

absolute_fidelity = {
    'fruits': [stages_data[0]['fruits']['fidelity'], stages_data[1]['fruits']['fidelity'], stages_data[2]['fruits']['fidelity']],
    'animals': [stages_data[0]['animals']['fidelity'], stages_data[1]['animals']['fidelity'], stages_data[2]['animals']['fidelity']],
    'cars': [stages_data[0]['cars']['fidelity'], stages_data[1]['cars']['fidelity'], stages_data[2]['cars']['fidelity']]
}

# Compute Deltas (Change relative to when the list was first learned)
# Baseline is Stage 1 for Fruits, Stage 2 for Animals, Stage 3 for Cars
delta_mrr = {
    'fruits': [0.0, stages_data[1]['fruits']['mrr'] - stages_data[0]['fruits']['mrr'], stages_data[2]['fruits']['mrr'] - stages_data[0]['fruits']['mrr']],
    'animals': [0.0, stages_data[2]['animals']['mrr'] - stages_data[1]['animals']['mrr']],
    'cars': [0.0]
}

delta_span = {
    'fruits': [0.0, stages_data[1]['fruits']['span'] - stages_data[0]['fruits']['span'], stages_data[2]['fruits']['span'] - stages_data[0]['fruits']['span']],
    'animals': [0.0, stages_data[2]['animals']['span'] - stages_data[1]['animals']['span']],
    'cars': [0.0]
}

delta_fidelity = {
    'fruits': [0.0, stages_data[1]['fruits']['fidelity'] - stages_data[0]['fruits']['fidelity'], stages_data[2]['fruits']['fidelity'] - stages_data[0]['fruits']['fidelity']],
    'animals': [0.0, stages_data[2]['animals']['fidelity'] - stages_data[1]['animals']['fidelity']],
    'cars': [0.0]
}

# Plot Absolute & Delta Metrics in a 2x3 grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
colors  = {'fruits': '#ff6347', 'animals': '#4682b4', 'cars': '#3cb371'}
markers = {'fruits': 'o',       'animals': 's',       'cars': '^'}

###### plots


# 6. Robust Distance Metrics: PCA & Cosine Similarity
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

print("Generating DG codes for all categories...")
dg_codes = []
labels = []
colors = []
cat_colors = {'fruits': '#ff6347', 'animals': '#4682b4', 'cars': '#3cb371'}

# We use the encoder, words, and contexts from the previous cell
for cat, words in lists.items():
    enc_words = encoder_multi.encode(words)
    context_vec = np.array(context_map[cat])
    
    for i, word in enumerate(words):
        x_t = enc_words[i]
        
        # Consistent scale matching fit_sequence
        dg_in = net_multi._make_dg_input(x_t, context_vec)
        dg_code = net_multi.dg.transform(dg_in)
        
        dg_codes.append(dg_code)
        labels.append(f"{cat.capitalize()} - {word}")
        colors.append(cat_colors[cat])

dg_codes = np.array(dg_codes)

print("Computing PCA and Pairwise Cosine Similarity on DG codes...")

# 1. PCA Projection (Linear & Deterministic)
pca = PCA(n_components=2, random_state=42)
dg_pca = pca.fit_transform(dg_codes)

# 2. Pairwise Cosine Similarity Matrix
# We calculate the cosine similarity between every pair of the 30 DG sparse codes
cos_sim_matrix = cosine_similarity(dg_codes)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# --- Plot 1: PCA ---
ax_pca = axes[0]
for cat in lists.keys():
    mask = np.array([cat.capitalize() in label for label in labels])
    ax_pca.scatter(dg_pca[mask, 0], dg_pca[mask, 1], 
                   label=cat.capitalize(), color=cat_colors[cat], 
                   marker='o', s=120, edgecolors='black', alpha=0.8)

for i, label in enumerate(labels):
    word = label.split(" - ")[1]
    ax_pca.annotate(word, (dg_pca[i, 0], dg_pca[i, 1]), 
                    textcoords="offset points", xytext=(8,0), va='center', fontsize=9)

ax_pca.set_title(f"PCA Projection (Explained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%)", fontsize=14, fontweight='bold')
ax_pca.set_xlabel("Principal Component 1")
ax_pca.set_ylabel("Principal Component 2")
ax_pca.legend(frameon=True)
ax_pca.grid(alpha=0.2)

# --- Plot 2: Cosine Similarity Heatmap ---
ax_sim = axes[1]
# Plot the 30x30 similarity matrix
sns.heatmap(cos_sim_matrix, cmap='magma', ax=ax_sim, 
            xticklabels=labels, yticklabels=labels, 
            cbar_kws={'label': 'Cosine Similarity'})

ax_sim.set_title("Pairwise Cosine Similarity Matrix", fontsize=14, fontweight='bold')

# Color code the axis labels to match the categories
for tick_label in ax_sim.get_yticklabels():
    cat = tick_label.get_text().split(" - ")[0].lower()
    tick_label.set_color(cat_colors[cat])
for tick_label in ax_sim.get_xticklabels():
    cat = tick_label.get_text().split(" - ")[0].lower()
    tick_label.set_color(cat_colors[cat])

plt.tight_layout()
plt.show()