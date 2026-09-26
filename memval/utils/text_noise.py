import string
from typing import List, Optional
import numpy as np

def edit_distance(s1: str, s2: str) -> int:
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return edit_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]

def edit_distance_nearest(word: str, vocab: List[str]) -> str:
    """Find the word in vocabulary with the minimum edit distance."""
    if not vocab:
        raise ValueError("Vocabulary cannot be empty.")
    
    # Sort ties alphabetically for stability
    return min(vocab, key=lambda v: (edit_distance(word, v), v))

def corrupt_word(word: str, noise_type: str = "substitute", noise_level: float = 0.2, rng=None) -> str:
    """
    Corrupt a word by randomly modifying individual characters.

    Args:
        word: The original word string.
        noise_type: One of 'substitute', 'delete', 'insert', 'transpose', or 'mixed'.
        noise_level: Probability (0-1) that each character position is affected.
        rng: Optional random number generator.

    Returns:
        The corrupted word string.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not word:
        return word

    letters = list(string.ascii_lowercase)

    if noise_type == "substitute":
        result = []
        for char in word:
            if rng.random() < noise_level:
                result.append(rng.choice(letters))
            else:
                result.append(char)
        return "".join(result)

    elif noise_type == "delete":
        indices_to_keep = []
        for idx in range(len(word)):
            if rng.random() >= noise_level:
                indices_to_keep.append(idx)
        if not indices_to_keep:
            indices_to_keep.append(rng.integers(0, len(word)))
        return "".join(word[idx] for idx in sorted(indices_to_keep))

    elif noise_type == "insert":
        result = []
        for char in word:
            if rng.random() < noise_level:
                result.append(rng.choice(letters))
            result.append(char)
        if rng.random() < noise_level:
            result.append(rng.choice(letters))
        return "".join(result)

    elif noise_type == "transpose":
        chars = list(word)
        i = 0
        while i < len(chars) - 1:
            if rng.random() < noise_level:
                chars[i], chars[i+1] = chars[i+1], chars[i]
                i += 2
            else:
                i += 1
        return "".join(chars)

    elif noise_type == "mixed":
        chars = list(word)
        result = []
        i = 0
        deleted_all = True
        
        while i < len(chars):
            if rng.random() < noise_level:
                action = rng.choice(["substitute", "delete", "insert", "transpose"])
                if action == "substitute":
                    result.append(rng.choice(letters))
                    deleted_all = False
                    i += 1
                elif action == "delete":
                    i += 1
                elif action == "insert":
                    result.append(rng.choice(letters))
                    result.append(chars[i])
                    deleted_all = False
                    i += 1
                elif action == "transpose":
                    if i < len(chars) - 1:
                        result.append(chars[i+1])
                        result.append(chars[i])
                        deleted_all = False
                        i += 2
                    else:
                        result.append(rng.choice(letters))
                        deleted_all = False
                        i += 1
            else:
                result.append(chars[i])
                deleted_all = False
                i += 1
                
        if deleted_all:
            idx = rng.integers(0, len(word))
            result.append(word[idx])
            
        return "".join(result)
    
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
