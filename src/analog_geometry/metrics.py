"""Representational-geometry metrics: linear CKA and RSA (Spearman on RDMs).

Both operate on activation matrices X [N, Dx], Y [N, Dy] for the same N stimuli.
Gram-matrix formulation so that very wide layers (conv feature maps) stay cheap.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def _centered_gram(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    return X @ X.T


def stimcenter(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X - X.mean(axis=1, keepdims=True)


def cka_from_grams(K: np.ndarray, L: np.ndarray) -> float:
    hsic = float((K * L).sum())
    norm = float(np.sqrt((K * K).sum() * (L * L).sum()))
    return hsic / norm if norm > 0 else float("nan")


def linear_cka(X: np.ndarray, Y: np.ndarray, gram_x: np.ndarray | None = None) -> float:
    """Linear CKA (Kornblith et al., 2019). 1 = identical up to rotation/isotropic scaling.

    `gram_x` (= _centered_gram(X)) may be passed to avoid recomputing a fixed reference; result is identical.
    """
    K = _centered_gram(X) if gram_x is None else gram_x
    L = _centered_gram(Y)
    hsic = float((K * L).sum())
    norm = float(np.sqrt((K * K).sum() * (L * L).sum()))
    return hsic / norm if norm > 0 else float("nan")


def linear_cka_stimcentered(X: np.ndarray, Y: np.ndarray, gram_x: np.ndarray | None = None) -> float:
    """Linear CKA after removing each stimulus' mean over units (then the usual per-unit centering).

    Invariant to an additive offset that is constant across units for a given stimulus (e.g. from a uniform weight
    shift). Offset control for drift: if the CKA deviation vanishes here, it is an offset, not a reordering.
    `gram_x` (= _centered_gram(stimcenter(X))) may be passed for a fixed reference; result is identical.
    """
    return linear_cka(stimcenter(X), stimcenter(Y), gram_x=gram_x)


def rdm(X: np.ndarray) -> np.ndarray:
    """Correlation-distance RDM: 1 - Pearson r between stimulus patterns. Returns upper triangle (k=1)."""
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0  # constant patterns (e.g. all-zero ReLU rows) -> treated as zero vector
    Xn = X / nrm
    D = 1.0 - Xn @ Xn.T
    iu = np.triu_indices(D.shape[0], k=1)
    return D[iu]


def rsa(X: np.ndarray, Y: np.ndarray, rdm_x: np.ndarray | None = None) -> float:
    """Spearman correlation between the upper triangles of the two RDMs."""
    a = rdm(X) if rdm_x is None else rdm_x
    b = rdm(Y)
    return float(spearmanr(a, b).statistic)
