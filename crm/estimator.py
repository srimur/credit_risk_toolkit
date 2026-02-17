"""
crm.estimator — transition matrix estimation from observed data.

Implements the cohort method: count observed migrations between
consecutive rating snapshots and normalize to row-stochastic form.

This is Step 1 of the pipeline. The output is a RAW matrix that
likely has quality issues (non-monotonic PD, sparse cells, etc.)
which downstream modules fix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .types import D_IDX, N, RATING_IDX, RATING_SCALE, TransitionMatrix


def estimate_cohort(
    ratings_from: np.ndarray,
    ratings_to: np.ndarray,
) -> Tuple[TransitionMatrix, np.ndarray]:
    """
    Estimate a single transition matrix from paired rating observations.

    Parameters
    ----------
    ratings_from, ratings_to : array of str
        Rating labels at time t and t+1 for each borrower.

    Returns
    -------
    (TransitionMatrix, count_matrix)
    """
    counts = np.zeros((N, N), dtype=np.float64)
    for rf, rt in zip(ratings_from, ratings_to):
        if pd.notna(rf) and pd.notna(rt):
            i = RATING_IDX.get(str(rf).strip())
            j = RATING_IDX.get(str(rt).strip())
            if i is not None and j is not None:
                counts[i, j] += 1

    prob = _normalize(counts)

    tm = TransitionMatrix(
        matrix=prob,
        source="cohort",
        corrections=[],
    )
    tm.diagnostics["count_matrix"] = counts
    tm.diagnostics["min_row_count"] = int(counts.sum(axis=1).min())
    tm.diagnostics["total_transitions"] = int(counts.sum())

    return tm, counts


def estimate_multi_period(
    ratings_matrix: np.ndarray,
) -> Dict[str, TransitionMatrix]:
    """
    Estimate annual transition matrices from a full rating history.

    Parameters
    ----------
    ratings_matrix : array of shape (n_borrowers, n_years)
        Rating labels per borrower per year.

    Returns
    -------
    dict of {year_label: TransitionMatrix}
    """
    n_borrowers, n_years = ratings_matrix.shape
    results = {}

    for t in range(n_years - 1):
        label = f"Year {t + 1} -> {t + 2}"
        tm, _ = estimate_cohort(ratings_matrix[:, t], ratings_matrix[:, t + 1])
        results[label] = tm

    return results


def pool_matrices(matrices: List[TransitionMatrix]) -> TransitionMatrix:
    """
    Pool multiple annual matrices by averaging count matrices
    (not probability matrices — this correctly handles different
    sample sizes per year).
    """
    if not matrices:
        raise ValueError("No matrices to pool")

    # Sum count matrices if available, else average probabilities
    has_counts = all("count_matrix" in m.diagnostics for m in matrices)

    if has_counts:
        total_counts = sum(m.diagnostics["count_matrix"] for m in matrices)
        prob = _normalize(total_counts)
    else:
        prob = np.mean([m.matrix for m in matrices], axis=0)
        # Re-normalize
        for i in range(N):
            s = prob[i].sum()
            if s > 0:
                prob[i] /= s

    return TransitionMatrix(
        matrix=prob,
        source="pooled",
        corrections=[f"pooled from {len(matrices)} annual matrices"],
    )


def _normalize(counts: np.ndarray) -> np.ndarray:
    """Row-normalize a count matrix to probabilities."""
    prob = np.zeros_like(counts, dtype=np.float64)
    for i in range(N):
        s = counts[i].sum()
        if s > 0:
            prob[i] = counts[i] / s
        else:
            prob[i, i] = 1.0  # no observations → assume stay
    return prob
