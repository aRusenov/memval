import numpy as np
import matplotlib.pyplot as plt
from memval.encoders import SymbolicEncoder, SymbolicDecoder
from memval.models.baselines.dg_eqprop import DGEqPropSequenceNetwork
from memval.models.baselines.ewc_dg_eqprop import EWCDGEqPropSequenceNetwork

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


# 1. Vocab and Categories definition
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

n_epochs = 300
n_trials = 30

def run_experiment(name, ewc_method=None, ewc_lambda=0.0):
    print(f"\n--- Running Experiment: {name} (method={ewc_method}, lambda={ewc_lambda}) ---")
    if ewc_method is None:
        net = DGEqPropSequenceNetwork(
            n_features=100, n_context=n_explicit, explicit_scale=1.0,
            n_dg=1000, sparsity=0.05, learning_rate=0.1, n_epochs=n_epochs, seed=42
        )
    else:
        net = EWCDGEqPropSequenceNetwork(
            n_features=100, n_context=n_explicit, explicit_scale=1.0,
            n_dg=1000, sparsity=0.05, learning_rate=0.1, n_epochs=n_epochs, seed=42,
            ewc_lambda=ewc_lambda, ewc_method=ewc_method
        )

    stages_data = []
    for idx, (cat, words) in enumerate(lists.items()):
        enc_words = encoder_multi.encode(words)
        context_seq = np.tile(context_map[cat], (len(words), 1))
        
        # Train only on the current category
        net.fit_sequence(enc_words, context_seq)
        
        # Consolidate if EWC is active
        if ewc_method is not None:
            net.consolidate(enc_words, context_seq)
            
        stage_metrics = {}
        for c_cat in lists.keys():
            c_words = lists[c_cat]
            c_context = context_map[c_cat]
            
            curve_assoc = measure_recall_associative(net, c_words, encoder_multi, decoder_multi, c_context, n_trials=n_trials)
            curve_auto = measure_recall_autoregressive(net, c_words, encoder_multi, decoder_multi, c_context, n_trials=n_trials)
            
            mrr = mean_recall_rate(curve_assoc)
            span = memory_span(curve_auto, threshold=0.75)
            fid = recall_fidelity(net, c_words, encoder_multi, context_vec=c_context, n_trials=n_trials)
            
            stage_metrics[c_cat] = {'mrr': mrr, 'span': span, 'fidelity': fid}
        stages_data.append(stage_metrics)
        
        print(f"Stage {idx+1} (after training on {cat}) evaluation:")
        for c_cat in lists.keys():
            m = stage_metrics[c_cat]
            print(f"  {c_cat:<10} -> MRR: {m['mrr']:.3f}, Span: {m['span']}, Fidelity: {m['fidelity']:.3f}")
            
    return stages_data, net

# Run experiments
results = {}
models = {}

# 1. No EWC Baseline
results['No EWC'], models['No EWC'] = run_experiment('No EWC', ewc_method=None)

# 2. Canonical EWC
results['Canonical EWC'], models['Canonical EWC'] = run_experiment('Canonical EWC', ewc_method='canonical', ewc_lambda=150.0)

# 3. EWC Done Right (lr)
results['EWC Done Right (LR)'], models['EWC Done Right (LR)'] = run_experiment('EWC Done Right (LR)', ewc_method='lr', ewc_lambda=150.0)

# 4. MAS
results['MAS'], models['MAS'] = run_experiment('MAS', ewc_method='mas', ewc_lambda=150.0)

# Print a final comparative table of retention at Stage 3
print("\n" + "=" * 80)
print(f"  FINAL STAGE 3 COMPARISON TABLE (Learned sequence retention)")
print("=" * 80)
print(f"  {'Model':<22} | {'Fruits MRR':<12} | {'Animals MRR':<12} | {'Cars MRR':<12}")
print("-" * 80)
for name, data in results.items():
    final_stage = data[2]
    print(f"  {name:<22} | {final_stage['fruits']['mrr']:<12.3f} | {final_stage['animals']['mrr']:<12.3f} | {final_stage['cars']['mrr']:<12.3f}")
print("=" * 80)

# Plotting metrics
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
colors  = {'No EWC': '#ff6347', 'Canonical EWC': '#4682b4', 'EWC Done Right (LR)': '#3cb371', 'MAS': '#9370db'}
markers = {'No EWC': 'o',       'Canonical EWC': 's',       'EWC Done Right (LR)': '^',       'MAS': 'd'}

# 1. Fruits MRR retention over stages
ax = axes[0]
stages = [1, 2, 3]
for name, data in results.items():
    vals = [data[0]['fruits']['mrr'], data[1]['fruits']['mrr'], data[2]['fruits']['mrr']]
    ax.plot(stages, vals, marker=markers[name], color=colors[name], label=name, linewidth=2, markersize=8)
ax.set_title("Fruits Recall MRR (Stability on Task 1)")
ax.set_xlabel("Training Stage")
ax.set_ylabel("Mean Recall Rate")
ax.set_xticks(stages)
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(alpha=0.2)

# 2. Animals MRR retention over stages
ax = axes[1]
stages_anim = [2, 3]
for name, data in results.items():
    vals = [data[1]['animals']['mrr'], data[2]['animals']['mrr']]
    ax.plot(stages_anim, vals, marker=markers[name], color=colors[name], label=name, linewidth=2, markersize=8)
ax.set_title("Animals Recall MRR (Stability on Task 2)")
ax.set_xlabel("Training Stage")
ax.set_ylabel("Mean Recall Rate")
ax.set_xticks(stages_anim)
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(alpha=0.2)

# 3. Cars MRR over stages
ax = axes[2]
stages_cars = [3]
for name, data in results.items():
    vals = [data[2]['cars']['mrr']]
    ax.scatter(stages_cars, vals, marker=markers[name], color=colors[name], label=name, s=100, zorder=5)
ax.set_title("Cars Recall MRR (Plasticity on Task 3)")
ax.set_xlabel("Training Stage")
ax.set_ylabel("Mean Recall Rate")
ax.set_xticks(stages_cars)
ax.set_ylim(-0.05, 1.05)
ax.legend()
ax.grid(alpha=0.2)

plt.suptitle("Elastic Weight Consolidation (EWC) Comparison on DG-EqProp", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("debug/ewc_comparison.png", dpi=150)
print("Saved comparative metrics plot to debug/ewc_comparison.png")

# Now let's plot PCA of Dentate Gyrus codes for EWC Done Right to confirm representation protection
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

print("\nGenerating DG codes for all categories using EWC Done Right model...")
net_multi = models['EWC Done Right (LR)']
dg_codes = []
labels = []
cat_colors = {'fruits': '#ff6347', 'animals': '#4682b4', 'cars': '#3cb371'}

for cat, words in lists.items():
    enc_words = encoder_multi.encode(words)
    context_vec = np.array(context_map[cat])
    
    for i, word in enumerate(words):
        x_t = enc_words[i]
        dg_in = net_multi._make_dg_input(x_t, context_vec)
        dg_code = net_multi.dg.transform(dg_in)
        dg_codes.append(dg_code)
        labels.append(f"{cat.capitalize()} - {word}")

dg_codes = np.array(dg_codes)
pca = PCA(n_components=2, random_state=42)
dg_pca = pca.fit_transform(dg_codes)
cos_sim_matrix = cosine_similarity(dg_codes)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))

# PCA Plot
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

ax_pca.set_title(f"PCA Projection of DG representations (EWC Done Right)\nExplained Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%", fontsize=12, fontweight='bold')
ax_pca.set_xlabel("Principal Component 1")
ax_pca.set_ylabel("Principal Component 2")
ax_pca.legend(frameon=True)
ax_pca.grid(alpha=0.2)

# Cosine Similarity Heatmap
ax_sim = axes[1]
sns.heatmap(cos_sim_matrix, cmap='magma', ax=ax_sim, 
            xticklabels=labels, yticklabels=labels, 
            cbar_kws={'label': 'Cosine Similarity'})
ax_sim.set_title("Pairwise Cosine Similarity Matrix (EWC Done Right)", fontsize=12, fontweight='bold')

for tick_label in ax_sim.get_yticklabels():
    cat = tick_label.get_text().split(" - ")[0].lower()
    tick_label.set_color(cat_colors[cat])
for tick_label in ax_sim.get_xticklabels():
    cat = tick_label.get_text().split(" - ")[0].lower()
    tick_label.set_color(cat_colors[cat])

plt.suptitle("EWC Done Right (LR) Representational Spaces", fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig("debug/ewc_pca_similarity.png", dpi=150)
print("Saved PCA & Similarity plot to debug/ewc_pca_similarity.png")
print("Verification run completed successfully.")
