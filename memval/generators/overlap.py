"""Modality-agnostic assembly of overlapping episodes.

The sequence-disambiguation capacity has one structure, and it is not spatial:

    prefix        unique to each episode
    shared        IDENTICAL observations across every episode
      zone        the sub-interval of `shared` where a discriminator is available
    suffix        divergent again -- the branch the model must choose

`OverlapConfig` already resolves those indices without reference to any
modality. What was spatial-only was the *assembly*: building the discriminator
time-series, concatenating it to the content channel, balancing the two blocks,
and reporting the geometry. That work is identical for place cells and for
symbol embeddings, so it lives here and both generators call it.

Every episode's input is laid out as a content block followed by a
discriminator block::

    [ content (n_content dims) | discriminator (n_discriminator dims) ]

which is the layout `BifurcatingRouteGenerator` already used for MEC place
cells and the LEC odour, generalised to N episodes and any content encoder.

Why N episodes rather than two: with two episodes sharing one stretch the task
sits far below any associator's capacity, and the load finding on the symbolic
chain was that load rather than overlap is what separates arms. N confusable
episodes sharing a stretch is the disambiguation analogue of the load sweep.
See docs/disambiguation_design.md sec 3, axis C.
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..benchmarks.overlap_config import OverlapConfig


class SharedStretchViolation(ValueError):
    """The episodes do not actually agree over the shared stretch.

    Raised rather than warned because it silently destroys the benchmark: if the
    content differs inside `shared`, the model can read the episode's identity
    off the observations and the task stops being a disambiguation task at all.
    A run that measured nothing would still produce plausible-looking numbers.
    """


def assemble_overlapping_episodes(
    contents: Sequence[np.ndarray],
    discriminators: np.ndarray,
    geom: Dict[str, int],
    total_length: int,
    n_content: int,
    disc_on_suffix: bool = False,
    balance_modalities: bool = False,
    disc_scale: float = 1.0,
    atol: float = 1e-9,
) -> Dict[str, Any]:
    """Assemble N episodes that share a stretch, into model-ready inputs.

    Parameters
    ----------
    contents
        N arrays of shape ``(total_length, n_content)``. They MUST be equal over
        ``[shared_start, shared_end)``; that is the premise of the capacity and
        it is checked, not assumed.
    discriminators
        ``(N, n_discriminator)``. Episode i's discriminator, written into the
        zone. Their mutual similarity is axis B, and it is the caller's job --
        a spatial caller rotates a basis vector, a symbolic caller draws them
        from one category at a chosen variance.
    geom
        Output of ``OverlapConfig.resolve()``.
    n_content
        Width of the content block. Passed explicitly rather than inferred: a
        place-cell encoder rounds its cell count up to a square grid, so the
        requested width and the real one differ, and slicing on the requested
        one splits the blocks in the wrong place.
    disc_on_suffix
        Carry the discriminator past ``shared_end`` into the divergent suffix.
        This is the DEGENERATE case and exists only as a reference row: with the
        discriminator present at the decision step the task is solved by
        concurrent binding and needs no memory of the corridor. Leave False for
        any row meant to test carrying a cue across a delay.
    balance_modalities
        Normalise each block to a common per-timestep norm (content -> 1.0,
        discriminator -> ``disc_scale``). Without it the content block can
        outweigh the discriminator it is supposed to be overridden by, and an
        arm then fails for a scaling reason rather than a memory one.
    disc_scale
        Per-timestep norm of the discriminator block under balancing. Note this
        is a configuration choice and not a task property: for a converged
        linear map it is absorbed into the weights.

    Returns
    -------
    dict
        ``inputs`` (list of N arrays), the resolved geometry, the block widths,
        and ``delay``. For N == 2, ``input_A`` / ``input_B`` alias
        ``inputs[0]`` / ``inputs[1]`` so existing spatial callers keep working.
    """
    contents = [np.asarray(c, dtype=float) for c in contents]
    discriminators = np.asarray(discriminators, dtype=float)

    n_episodes = len(contents)
    if n_episodes < 2:
        raise ValueError("Need at least 2 episodes to have anything to disambiguate.")
    if discriminators.shape[0] != n_episodes:
        raise ValueError(
            f"Got {n_episodes} episodes but {discriminators.shape[0]} discriminators."
        )

    for i, c in enumerate(contents):
        if c.shape != (total_length, n_content):
            raise ValueError(
                f"contents[{i}] has shape {c.shape}, expected "
                f"({total_length}, {n_content})."
            )

    ss, se = geom["shared_start"], geom["shared_end"]
    zs, ze = geom["zone_start"], geom["zone_end"]
    n_disc = discriminators.shape[1]

    # The premise. If this fails the task is not a disambiguation task.
    ref = contents[0][ss:se]
    for i, c in enumerate(contents[1:], start=1):
        if not np.allclose(ref, c[ss:se], atol=atol):
            worst = float(np.max(np.abs(ref - c[ss:se])))
            raise SharedStretchViolation(
                f"Episode {i} differs from episode 0 over the shared stretch "
                f"[{ss}, {se}) by up to {worst:.3g}. The shared stretch must be "
                f"identical across episodes, otherwise the episode's identity is "
                f"readable from the content channel and the benchmark measures "
                f"nothing."
            )

    inputs: List[np.ndarray] = []
    for c, d in zip(contents, discriminators):
        disc_series = np.zeros((total_length, n_disc), dtype=float)
        if ze > zs:
            disc_series[zs:ze, :] = d
        if disc_on_suffix:
            disc_series[se:, :] = d

        X = np.concatenate([c, disc_series], axis=-1)

        if balance_modalities:
            content_block = X[:, :n_content]
            disc_block = X[:, n_content:]
            c_norm = np.linalg.norm(content_block, axis=1, keepdims=True)
            d_norm = np.linalg.norm(disc_block, axis=1, keepdims=True)
            c_norm[c_norm == 0] = 1.0
            d_norm[d_norm == 0] = 1.0
            X[:, :n_content] = content_block / c_norm
            X[:, n_content:] = disc_block / d_norm * disc_scale

        inputs.append(X)

    out: Dict[str, Any] = {
        "inputs": inputs,
        "n_episodes": n_episodes,
        "n_content": n_content,
        "n_discriminator": n_disc,
        "shared_start": ss,
        "shared_end": se,
        "zone_start": zs,
        "zone_end": ze,
        "prefix_end": ss,
        "suffix_start": se,
        # Shared steps over which NO discriminator is available. 0 reproduces
        # the fully-cued corner; > 0 requires carried state.
        "delay": int(se - ze),
    }
    if n_episodes == 2:
        out["input_A"], out["input_B"] = inputs[0], inputs[1]
    return out


def discriminator_similarity(discriminators: np.ndarray) -> Dict[str, float]:
    """Realised mutual cosine similarity of a discriminator set.

    Axis B is a *property of the produced vectors*, not of the knob that asked
    for it: a spatial rotation hits its requested cosine exactly, while drawing
    from a category at a given variance does not. Report what was achieved so a
    sweep is labelled by its realised similarity rather than by its dial.
    """
    D = np.asarray(discriminators, dtype=float)
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = D / norms
    C = U @ U.T
    iu = np.triu_indices(len(D), k=1)
    if len(iu[0]) == 0:
        return {"mean": 1.0, "min": 1.0, "max": 1.0}
    vals = C[iu]
    return {"mean": float(vals.mean()),
            "min": float(vals.min()),
            "max": float(vals.max())}


def resolve_geometry(
    total_length: int,
    shared_fraction: float,
    shared_position: float,
    zone_fraction: float,
    zone_offset: float,
    encounter_similarity: float = 0.0,
) -> Dict[str, int]:
    """Convenience wrapper so callers need not import OverlapConfig directly."""
    return OverlapConfig(
        total_length=total_length,
        shared_fraction=shared_fraction,
        shared_position=shared_position,
        zone_fraction=zone_fraction,
        zone_offset=zone_offset,
        encounter_similarity=encounter_similarity,
    ).resolve()


def zone_params_for(
    total_length: int,
    shared_fraction: float,
    shared_position: float,
    duration: int,
    delay: int,
) -> Dict[str, float]:
    """Solve for the ``zone_fraction`` / ``zone_offset`` giving a target
    (duration, delay) pair.

    Sweeping ``zone_fraction`` alone with ``zone_offset`` pinned at 0 moves BOTH
    quantities at once -- the zone always starts at the corridor entrance, so
    shortening it shortens the cue and lengthens the gap to the decision in the
    same step. Every "how much cue does it need" number read off such a sweep is
    confounded with "how long can it hold one".

    This solves the two apart::

        duration = zone_end - zone_start     how much support was given
        delay    = shared_end - zone_end     how long it had to be held

    Raises
    ------
    ValueError
        If ``duration + delay`` exceeds the shared stretch, in which case the
        pair is not realisable at this geometry and the caller must widen
        ``shared_fraction`` rather than silently receive a clamped zone.
    """
    base = resolve_geometry(total_length, shared_fraction, shared_position, 1.0, 0.0)
    ss, se = base["shared_start"], base["shared_end"]
    shared_len = se - ss

    if duration < 1:
        raise ValueError(f"duration must be >= 1, got {duration}.")
    if delay < 0:
        raise ValueError(f"delay must be >= 0, got {delay}.")
    if duration + delay > shared_len:
        raise ValueError(
            f"duration={duration} + delay={delay} exceeds the shared stretch "
            f"({shared_len} steps at shared_fraction={shared_fraction}). Widen "
            f"shared_fraction or shorten the request; a clamped zone would "
            f"silently report a different geometry than the one asked for.")

    zone_fraction = duration / shared_len
    slack = shared_len - duration
    target_start_within = shared_len - delay - duration
    zone_offset = 0.0 if slack == 0 else target_start_within / slack
    return {"zone_fraction": zone_fraction, "zone_offset": zone_offset}
