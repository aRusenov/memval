import numpy as np
from typing import List, Dict, Tuple, Optional, Union

class SymbolicEncoder:
    """
    Sensory encoder that projects discrete symbols (e.g., words) into a 
    high-dimensional semantic vector space.
    
    This uses a synthetic embedding approach where random vectors on a 
    hypersphere represent distinct concepts. It supports "categories" 
    to mathematically enforce higher cosine similarity between semantically 
    related words, avoiding the need to download large pre-trained models.
    """

    def __init__(self, vocab: Union[List[str], Dict[str, str]], embedding_dim: int = 100,
                 category_variance: float = 0.5, seed: Optional[int] = None,
                 between_category_cosine: Optional[float] = None):
        """
        Initialize the Symbolic Encoder.

        Args:
            vocab: Either a list of unique words (assumed independent), or a
                   dictionary mapping words to their semantic category
                   (e.g., {'apple': 'fruit', 'dog': 'animal'}).

                   **Iteration order is load-bearing.** One RNG sample is drawn
                   per word, in this mapping's order, so two vocabularies with
                   the same members in a different order produce different
                   embeddings from the same ``seed``. Callers must therefore
                   pass an order-stable mapping: build it from a list, or dedup
                   with ``dict.fromkeys``, never with ``set()`` -- set iteration
                   order depends on ``PYTHONHASHSEED`` and is randomised per
                   process. Category base vectors are safe either way
                   (``self.categories`` is sorted); it is the per-word draws
                   that move.
            embedding_dim: Dimensionality of the generated semantic vectors.
            category_variance: If using categories, how much noise to add to the 
                               base category vector for each word. Smaller means 
                               higher within-category cosine similarity.
            seed: Random seed for reproducibility.
            between_category_cosine: Cosine between any two category prototypes,
                ``rho_b``. This is the second overlap dial -- ``category_variance``
                sets how similar items are *within* a category, this sets how
                similar the categories themselves are.

                ``None`` (default) keeps the historical construction: prototypes
                are independent random unit vectors, quasi-orthogonal with a
                per-pair cosine of ``~N(0, 1/d)`` that is fixed by the seed and
                not controllable. Every saved result was produced this way and
                the RNG stream is unchanged, so ``None`` is bit-identical to
                the code before this argument existed.

                A float in ``[0, 1)`` builds the prototypes **exactly** at that
                cosine: ``c_k = sqrt(rho_b) * g + sqrt(1 - rho_b) * u_k`` with
                ``u_k`` orthonormal (QR of the same Gaussian draws) and ``g`` a
                shared unit direction orthogonal to all of them. Then
                ``<c_k, c_l> = rho_b`` to machine precision for every pair, and
                at the word level ``E<e_w, e_w'> = rho * rho_b`` across
                categories (``rho`` the within-category cosine), so the
                cross-category cosine can never exceed the within-category one.
                Pass ``0.0`` explicitly for exactly orthogonal categories.
                Requires ``n_categories + 1 <= embedding_dim``. For a fixed
                seed the per-word noise draws are identical for every value
                of ``rho_b``, so a sweep over it moves only the prototypes.
        """
        self.embedding_dim = embedding_dim
        self.rng = np.random.default_rng(seed)
        self.category_variance = category_variance
        if between_category_cosine is not None:
            between_category_cosine = float(between_category_cosine)
            if not (0.0 <= between_category_cosine < 1.0):
                raise ValueError(
                    f"between_category_cosine must be in [0, 1), got {between_category_cosine}")
        self.between_category_cosine = between_category_cosine
        
        self.word_to_idx = {}
        self.idx_to_word = []
        
        if isinstance(vocab, list):
            # No categories, all words get independent random orthogonal vectors
            vocab_dict = {word: word for word in vocab}
        else:
            vocab_dict = vocab
            
        self.categories = sorted(list(set(vocab_dict.values())))
        
        # 1. Generate base vectors for each category
        # In high dimensions, random vectors are approximately orthogonal (cosine sim ~ 0)
        category_vectors = {}
        for cat in self.categories:
            vec = self.rng.standard_normal(embedding_dim)
            category_vectors[cat] = vec / np.linalg.norm(vec)

        if between_category_cosine is not None:
            # Controlled between-category geometry. Same Gaussian draws as
            # above (so the RNG stream up to here is shared with the legacy
            # path), orthonormalised so the prototype cosine is *exactly*
            # rho_b rather than rho_b plus a seed-dependent O(1/sqrt(d)) offset.
            K = len(self.categories)
            if K + 1 > embedding_dim:
                raise ValueError(
                    f"between_category_cosine needs n_categories + 1 <= embedding_dim, "
                    f"got {K} categories in d={embedding_dim}")
            G = np.stack([category_vectors[c] for c in self.categories], axis=1)   # (d, K)
            Q, Rq = np.linalg.qr(G)                                                # Q: (d, K)
            Q = Q * np.sign(np.diag(Rq))[None, :]        # keep each u_k aligned with its draw
            g = self.rng.standard_normal(embedding_dim)
            g = g - Q @ (Q.T @ g)                        # shared direction, orthogonal to every u_k
            g = g / np.linalg.norm(g)
            a, b = np.sqrt(between_category_cosine), np.sqrt(1.0 - between_category_cosine)
            for k, cat in enumerate(self.categories):
                category_vectors[cat] = a * g + b * Q[:, k]
        self.category_vectors = category_vectors
            
        # 2. Generate word embeddings
        self.embeddings = np.zeros((len(vocab_dict), embedding_dim))
        
        for idx, (word, category) in enumerate(vocab_dict.items()):
            self.word_to_idx[word] = idx
            self.idx_to_word.append(word)
            
            base_vec = category_vectors[category]
            noise = self.rng.standard_normal(embedding_dim) * self.category_variance
            
            word_vec = base_vec + noise
            # Normalize to unit length for cosine similarity
            self.embeddings[idx] = word_vec / np.linalg.norm(word_vec)

    def encode(self, sequence: List[str]) -> np.ndarray:
        """
        Translates a sequence of words into semantic vectors.

        Args:
            sequence: List of strings (words).
        
        Returns:
            np.ndarray: Array of shape (seq_len, embedding_dim).
        """
        encoded = np.zeros((len(sequence), self.embedding_dim))
        for i, word in enumerate(sequence):
            if word not in self.word_to_idx:
                raise ValueError(f"Word '{word}' not in vocabulary.")
            idx = self.word_to_idx[word]
            encoded[i] = self.embeddings[idx]
        return encoded

class SymbolicDecoder:
    """
    Decodes high-dimensional semantic vectors back into discrete symbols (words)
    using cosine similarity.
    """
    
    def __init__(self, encoder: SymbolicEncoder):
        """
        Initialize with a reference to the encoder's embedding matrix.
        """
        self.encoder = encoder
        self.embeddings = encoder.embeddings
        self.idx_to_word = encoder.idx_to_word
        
    def decode(self, activations: np.ndarray, top_k: int = 1) -> Union[List[str], List[List[Tuple[str, float]]]]:
        """
        Decodes sequence of activations back to words.
        
        Args:
            activations: Array of shape (seq_len, embedding_dim).
            top_k: Number of top matches to return per timestep.
            
        Returns:
            If top_k == 1, returns a flat list of words.
            If top_k > 1, returns a list of lists, each containing (word, similarity_score) tuples.
        """
        # Ensure activations is 2D
        if activations.ndim == 1:
            activations = activations[np.newaxis, :]
            
        # Normalize activations for cosine similarity
        norms = np.linalg.norm(activations, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        normalized_acts = activations / norms
        
        # Calculate cosine similarity matrix (seq_len, vocab_size)
        # Since self.embeddings is also normalized, dot product = cosine similarity
        similarities = np.dot(normalized_acts, self.embeddings.T)
        
        results = []
        for i in range(len(activations)):
            # Sort indices by similarity descending
            top_indices = np.argsort(similarities[i])[::-1][:top_k]
            
            if top_k == 1:
                results.append(self.idx_to_word[top_indices[0]])
            else:
                step_results = [(self.idx_to_word[idx], similarities[i][idx]) for idx in top_indices]
                results.append(step_results)
                
        return results

    def decode_with_score(self, activation: np.ndarray) -> Tuple[str, float]:
        """Decode a single activation vector and return (word, cosine_similarity).

        Unlike ``decode(..., top_k=1)`` this always surfaces the similarity
        score of the top hit, which is needed by threshold-gated evaluation
        functions to distinguish high-confidence correct predictions from
        low-confidence top-1 guesses.

        Args:
            activation: Shape ``(embedding_dim,)``.

        Returns:
            ``(word, cosine_similarity)`` for the nearest vocabulary item.
        """
        norm = np.linalg.norm(activation)
        if norm == 0.0:
            norm = 1.0
        normalized = activation / norm
        sims = self.embeddings @ normalized          # (vocab_size,) cosine similarities
        best_idx = int(np.argmax(sims))
        return self.idx_to_word[best_idx], float(sims[best_idx])
