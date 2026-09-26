"""Two-modality item code: a symbolic embedding beside an audio (pure-tone) code.

The suite's cue-degradation sections remove *dimensions* of a cue (`cue_masking`)
or corrupt all of them (`noise_invariance`). The third probe property in the
paper's figure is **availability**: a whole modality is absent from the cue.
That needs an input in which the modalities are separate blocks, so that a cue
can carry one and not the other and the read-out can be taken per block.

``MultimodalEncoder`` concatenates, for every word in a vocabulary,

    [ symbolic embedding (d_sym, unit)  |  audio_gain * tone code (n_bins, unit) ]

and renormalises the whole vector, so a full cue sits on the unit sphere like a
plain symbolic embedding and every arm sees the input scale it was built for.
The tone is a property of the *item* (one pure tone per studied word, encoded by
`PureToneEncoder` into Gaussian frequency bins), so the audio block is a second,
independent code for the same item and either block alone can in principle
name it. Until 2026-09-26 the tone belonged to the *sequence* (constant within
it); that made an audio-only cue carry no position, so it could be scored on
membership only and could not launch a rollout. Words that belong to no studied
sequence carry a zero audio block: they are decoding distractors only.

Partial cues (`cue(word, modality)`) zero one block and, by default, renormalise
the remainder to unit length, exactly as `cue_masking.masked_cue` does, so an
arm whose read-out is scale-sensitive is not additionally penalised by a
shorter input.

Decoding is per block: `item_decoder` ranks the symbolic slice of a prediction
against the symbolic embeddings of the WHOLE vocabulary (so chance stays
1/|vocab| and an audio-only cue cannot be flattered by a tiny candidate set),
and `tone_of_prediction` ranks the audio slice against the studied items' tones.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .audio import PureToneEncoder
from .symbolic import SymbolicDecoder, SymbolicEncoder

MODALITIES = ("both", "symbolic", "audio")


class _SliceView:
    """The minimal encoder surface `SymbolicDecoder` needs, over one block."""

    def __init__(self, embeddings: np.ndarray, idx_to_word: List[str]):
        self.embeddings = embeddings
        self.idx_to_word = idx_to_word


class MultimodalEncoder:
    def __init__(
        self,
        symbolic: SymbolicEncoder,
        item_tones: Dict[str, float],
        sequences: Dict[str, Sequence[str]],
        n_bins: int = 100,
        min_freq: float = 200.0,
        max_freq: float = 600.0,
        tone_sigma: float = 4.0,
        audio_gain: float = 1.0,
    ):
        """
        Args:
            symbolic: the symbolic encoder over the full vocabulary.
            item_tones: studied word -> its pure-tone frequency (Hz). Every word
                of every sequence needs one.
            sequences: sequence name -> its words, in study order.
            n_bins, min_freq, max_freq, tone_sigma: `PureToneEncoder` settings.
                The defaults (100 bins, sigma 4 Hz over 200-600 Hz) keep 30
                evenly spaced item tones at pairwise cosine <= 0.06, below the
                symbolic code's own within-category cosine (~0.19). At the old
                50 bins / 15 Hz, adjacent tones overlapped at 0.82 and an
                audio-only cue would have scored tone discrimination rather
                than modality absence.
            audio_gain: relative weight of the audio block before the joint
                renormalisation. 1.0 gives the two blocks equal norm.
        """
        self.symbolic = symbolic
        self.tone_encoder = PureToneEncoder(n_bins=n_bins, min_freq=min_freq,
                                            max_freq=max_freq, sigma=tone_sigma)
        self.audio_gain = float(audio_gain)
        self.d_sym = int(symbolic.embedding_dim)
        self.d_audio = int(n_bins)
        self.embedding_dim = self.d_sym + self.d_audio
        self.sym_slice = slice(0, self.d_sym)
        self.audio_slice = slice(self.d_sym, self.embedding_dim)
        self.word_to_idx = dict(symbolic.word_to_idx)
        self.idx_to_word = list(symbolic.idx_to_word)
        self.sequences = {k: list(v) for k, v in sequences.items()}
        self.sequence_of: Dict[str, str] = {}
        for name, words in self.sequences.items():
            for w in words:
                if w in self.sequence_of:
                    raise ValueError(f"word {w!r} appears in two sequences; the "
                                     f"availability section needs disjoint material")
                self.sequence_of[w] = name
        missing = [w for w in self.sequence_of if w not in item_tones]
        if missing:
            raise ValueError(f"studied words without a tone: {missing}")
        # One unit tone code per studied item.
        self.tone_names = list(item_tones)
        self.item_tones = {w: float(f) for w, f in item_tones.items()}
        codes = []
        for name in self.tone_names:
            c = self.tone_encoder.encode(np.array([[item_tones[name]]]))[0]
            codes.append(c / (np.linalg.norm(c) + 1e-12))
        self.tone_codes = np.stack(codes) if codes else np.zeros((0, self.d_audio))
        self.tone_index = {n: i for i, n in enumerate(self.tone_names)}
        # Full embeddings: [sym | gain * tone], jointly unit-normalised.
        E = np.zeros((len(self.idx_to_word), self.embedding_dim))
        for w, i in self.word_to_idx.items():
            E[i, self.sym_slice] = symbolic.embeddings[i]
            if w in self.tone_index:
                E[i, self.audio_slice] = self.audio_gain * self.tone_codes[self.tone_index[w]]
            E[i] /= (np.linalg.norm(E[i]) + 1e-12)
        self.embeddings = E
        self.item_decoder = SymbolicDecoder(_SliceView(symbolic.embeddings, self.idx_to_word))

    # -- encoding ----------------------------------------------------------
    def encode(self, sequence: Iterable[str]) -> np.ndarray:
        return np.stack([self.embeddings[self.word_to_idx[w]] for w in sequence])

    def cue(self, word: str, modality: str = "both", renormalize: bool = True) -> np.ndarray:
        """The cue for ``word`` with one modality absent (zeroed)."""
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}, got {modality!r}")
        v = self.embeddings[self.word_to_idx[word]].copy()
        if modality == "symbolic":
            v[self.audio_slice] = 0.0
        elif modality == "audio":
            v[self.sym_slice] = 0.0
        if renormalize:
            n = float(np.linalg.norm(v))
            if n > 1e-12:
                v = v / n
        return v

    # -- per-block read-outs ------------------------------------------------
    def item_similarities(self, prediction: np.ndarray) -> np.ndarray:
        """Cosine of the symbolic slice against every word's symbolic embedding."""
        p = np.asarray(prediction, dtype=float)[self.sym_slice]
        p = p / (np.linalg.norm(p) + 1e-12)
        return self.symbolic.embeddings @ p

    def decode_item(self, prediction: np.ndarray) -> str:
        return self.idx_to_word[int(np.argmax(self.item_similarities(prediction)))]

    def tone_similarities(self, prediction: np.ndarray) -> np.ndarray:
        """Cosine of the audio slice against every studied item's tone code."""
        p = np.asarray(prediction, dtype=float)[self.audio_slice]
        p = p / (np.linalg.norm(p) + 1e-12)
        return self.tone_codes @ p

    def tone_of_prediction(self, prediction: np.ndarray) -> Optional[str]:
        if not self.tone_names:
            return None
        return self.tone_names[int(np.argmax(self.tone_similarities(prediction)))]

    def describe(self) -> Dict[str, object]:
        tc = self.tone_codes
        off = (tc @ tc.T)[~np.eye(len(tc), dtype=bool)] if len(tc) > 1 else np.array([])
        return {
            "d_sym": self.d_sym, "d_audio": self.d_audio, "audio_gain": self.audio_gain,
            "n_sequences": len(self.sequences), "n_tones": len(self.tone_names),
            "tone_sigma": float(self.tone_encoder.sigma),
            "tone_cosine_max": float(off.max()) if off.size else None,
        }
