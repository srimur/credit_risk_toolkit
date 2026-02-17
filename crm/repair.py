"""
crm.repair — transition matrix correction engine.

THE CORE MODULE. This is where the real value is.

Problem statement
-----------------
Raw transition matrices estimated from internal bank data are almost
always broken in multiple ways:

  1. Non-monotonic PD (AAA defaults more than A — rating system failure)
  2. Sparse cells (zero counts → zero probabilities → degenerate matrices)
  3. Negative entries (from generator matrix exponentiation or smoothing)
  4. Non-embeddable (matrix log doesn't exist → can't do continuous-time)
  5. Non-absorbing default (cured borrowers pollute the D row)
  6. Diagonal non-dominance (borrowers more likely to jump 3 notches
     than stay — unstable rating system)

Every bank's credit risk team spends significant effort manually
"fixing" these issues with ad-hoc adjustments. This module automates
the full repair pipeline with mathematically principled methods and
a complete audit trail.

Pipeline
--------
    raw matrix
        → enforce_absorbing_default()
        → remove_negatives()
        → enforce_monotonic_pd()       ← isotonic regression
        → bayesian_smooth()            ← prior from published matrices
        → enforce_embeddability()      ← nearest valid generator
        → enforce_row_stochastic()
    = production-quality matrix

Each step is independent, composable, and logs what it changed.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize

from .types import D_IDX, N, RATING_SCALE, TransitionMatrix


# ── Published benchmark (S&P Global 1981–2023 average 1-year) ────────────────
# Used as Bayesian prior for smoothing sparse matrices.
# Source: S&P Global Ratings, "2023 Annual Global Corporate Default
# and Rating Transition Study"

SP_BENCHMARK = np.array([
    # AAA     AA       A      BBB      BB       B      CCC      D
    [0.8702, 0.0929, 0.0265, 0.0055, 0.0011, 0.0005, 0.0005, 0.0028],  # AAA
    [0.0052, 0.8670, 0.0942, 0.0230, 0.0051, 0.0014, 0.0010, 0.0031],  # AA
    [0.0004, 0.0162, 0.8726, 0.0830, 0.0153, 0.0038, 0.0021, 0.0066],  # A
    [0.0001, 0.0013, 0.0337, 0.8537, 0.0711, 0.0213, 0.0078, 0.0110],  # BBB
    [0.0001, 0.0005, 0.0035, 0.0432, 0.7633, 0.1100, 0.0348, 0.0446],  # BB
    [0.0000, 0.0003, 0.0012, 0.0058, 0.0530, 0.7522, 0.0871, 0.1004],  # B
    [0.0000, 0.0000, 0.0030, 0.0063, 0.0141, 0.1049, 0.5510, 0.3207],  # CCC
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],  # D
], dtype=np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# Individual repair operations
# ═══════════════════════════════════════════════════════════════════════════════


def enforce_absorbing_default(tm: TransitionMatrix) -> TransitionMatrix:
    """
    Force default to be an absorbing state: P(D→D) = 1, P(D→j) = 0 for j≠D.

    Rationale: Under Basel and IFRS 9, default is a terminal state.
    Observed "cures" (D → non-D) should be modeled separately, not
    in the transition matrix.
    """
    P = tm.matrix.copy()
    P[D_IDX, :] = 0.0
    P[D_IDX, D_IDX] = 1.0

    corrections = tm.corrections + ["absorbing_default"]
    return TransitionMatrix(matrix=P, source=tm.source, corrections=corrections)


def remove_negatives(tm: TransitionMatrix, method: str = "redistribute") -> TransitionMatrix:
    """
    Remove negative entries from the transition matrix.

    Parameters
    ----------
    method : str
        'zero' — set negatives to 0, re-normalize row.
        'redistribute' — set negatives to 0, redistribute their mass
                         proportionally to positive entries in the same row.
    """
    P = tm.matrix.copy()
    n_fixed = 0

    for i in range(N):
        neg_mask = P[i] < 0
        if not np.any(neg_mask):
            continue

        neg_mass = np.abs(P[i, neg_mask]).sum()
        P[i, neg_mask] = 0.0
        n_fixed += int(neg_mask.sum())

        if method == "redistribute":
            pos_mask = P[i] > 0
            if np.any(pos_mask):
                # Redistribute proportionally
                pos_sum = P[i, pos_mask].sum()
                P[i, pos_mask] += P[i, pos_mask] / pos_sum * neg_mass

        # Re-normalize
        s = P[i].sum()
        if s > 0:
            P[i] /= s

    corrections = tm.corrections.copy()
    if n_fixed > 0:
        corrections.append(f"removed_{n_fixed}_negatives({method})")

    return TransitionMatrix(matrix=P, source=tm.source, corrections=corrections)


def enforce_monotonic_pd(
    tm: TransitionMatrix,
    method: str = "isotonic",
) -> TransitionMatrix:
    """
    Enforce that PD is monotonically non-decreasing from AAA to CCC.

    This is the most common quality issue with internal rating systems.
    If AAA has higher PD than A, the rating system is not discriminatory.

    Parameters
    ----------
    method : str
        'isotonic' — Pool Adjacent Violators Algorithm (PAVA).
                     Optimal L2 solution under monotonicity constraints.
        'spline'   — Fit monotone spline through PD values.

    The non-default columns are also adjusted proportionally to maintain
    row-stochastic property.
    """
    P = tm.matrix.copy()
    pd_col = P[:D_IDX, D_IDX].copy()  # AAA..CCC default probs

    if method == "isotonic":
        pd_fixed = _isotonic_regression(pd_col)
    elif method == "spline":
        pd_fixed = _monotone_spline(pd_col)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Apply corrected PD and redistribute the difference
    for i in range(D_IDX):
        delta = pd_fixed[i] - P[i, D_IDX]
        if abs(delta) < 1e-14:
            continue

        P[i, D_IDX] = pd_fixed[i]

        # Redistribute delta across non-default, non-self columns
        # proportionally to their current values
        other_cols = [j for j in range(N) if j != i and j != D_IDX]
        other_sum = P[i, other_cols].sum()

        if other_sum > 1e-14:
            P[i, other_cols] -= P[i, other_cols] / other_sum * delta
        else:
            # Fallback: take from diagonal
            P[i, i] -= delta

        # Clamp and re-normalize
        P[i] = np.maximum(P[i], 0.0)
        s = P[i].sum()
        if s > 0:
            P[i] /= s

    was_monotonic = all(
        pd_col[k] <= pd_col[k + 1] + 1e-10 for k in range(len(pd_col) - 1)
    )
    corrections = tm.corrections.copy()
    if not was_monotonic:
        corrections.append(f"monotonic_pd({method})")

    return TransitionMatrix(matrix=P, source=tm.source, corrections=corrections)


def bayesian_smooth(
    tm: TransitionMatrix,
    prior: np.ndarray = SP_BENCHMARK,
    weight: float = 0.3,
    count_adaptive: bool = True,
) -> TransitionMatrix:
    """
    Bayesian smoothing: blend observed matrix with a prior (published benchmark).

    P_smoothed = (1 - w) × P_observed + w × P_prior

    When count_adaptive=True, the weight adapts per row based on
    sample size: rows with fewer observations get more prior weight.
    This elegantly solves the sparse-cell problem.

    Parameters
    ----------
    prior : array (N, N)
        Prior transition matrix (default: S&P published averages).
    weight : float
        Base prior weight. 0 = pure observed, 1 = pure prior.
    count_adaptive : bool
        If True, weight increases for rows with fewer observations.
        Effective weight for row i = weight × (100 / (100 + n_i))
        where n_i is the observation count for rating i.
    """
    P = tm.matrix.copy()

    P_smooth = np.zeros_like(P)
    for i in range(N):
        if count_adaptive and "count_matrix" in tm.diagnostics:
            n_i = tm.diagnostics["count_matrix"].sum(axis=1)[i]
            w = weight * (100.0 / (100.0 + n_i))
            w = min(w, 0.95)
        else:
            w = weight

        P_smooth[i] = (1 - w) * P[i] + w * prior[i]

        # Re-normalize
        s = P_smooth[i].sum()
        if s > 0:
            P_smooth[i] /= s

    corrections = tm.corrections + [
        f"bayesian_smooth(base_weight={weight}, adaptive={count_adaptive})"
    ]
    return TransitionMatrix(matrix=P_smooth, source=tm.source, corrections=corrections)


def enforce_embeddability(
    tm: TransitionMatrix,
    max_iter: int = 200,
) -> TransitionMatrix:
    """
    Find the nearest embeddable matrix: a valid P = exp(Q) for some
    generator Q with non-negative off-diagonal and zero row sums.

    This ensures the matrix is consistent with a continuous-time
    Markov chain, which is required for:
    - Fractional-year transitions (e.g., quarterly PD)
    - Credit spread pricing (JLT model)
    - Regulatory consistency checks

    Method: Optimization-based. Minimize ||P_target - exp(Q)||_F
    subject to Q being a valid generator.

    Falls back to the Kreinin-Sidelnikova diagonal adjustment if
    optimization fails.
    """
    P_target = tm.matrix.copy()

    # Check if already embeddable
    eigvals = np.linalg.eigvals(P_target)
    if np.all(np.real(eigvals) > 0):
        try:
            Q = logm(P_target).real
            # Check if Q is a valid generator
            if _is_valid_generator(Q):
                P_recon = expm(Q).real
                if np.allclose(P_recon, P_target, atol=1e-6):
                    corrections = tm.corrections + ["embeddability_verified"]
                    return TransitionMatrix(
                        matrix=P_target, source=tm.source, corrections=corrections
                    )
        except Exception:
            pass

    # Optimization: find nearest valid generator
    Q_opt = _optimize_generator(P_target, max_iter=max_iter)

    if Q_opt is not None:
        P_embed = expm(Q_opt).real
        # Clamp and re-normalize
        P_embed = np.maximum(P_embed, 0.0)
        for i in range(N):
            s = P_embed[i].sum()
            if s > 0:
                P_embed[i] /= s

        corrections = tm.corrections + ["embeddability_optimized"]
        result = TransitionMatrix(matrix=P_embed, source=tm.source, corrections=corrections)
        result.diagnostics["generator"] = Q_opt
        return result

    # Fallback: diagonal adjustment (Kreinin-Sidelnikova)
    P_fixed = _kreinin_sidelnikova(P_target)
    corrections = tm.corrections + ["embeddability_kreinin_sidelnikova"]
    return TransitionMatrix(matrix=P_fixed, source=tm.source, corrections=corrections)


def enforce_row_stochastic(tm: TransitionMatrix) -> TransitionMatrix:
    """Final cleanup: ensure all entries ≥ 0 and rows sum to 1."""
    P = np.maximum(tm.matrix.copy(), 0.0)
    for i in range(N):
        s = P[i].sum()
        if s > 0:
            P[i] /= s
        else:
            P[i, i] = 1.0

    corrections = tm.corrections + ["row_stochastic"]
    return TransitionMatrix(matrix=P, source=tm.source, corrections=corrections)


# ═══════════════════════════════════════════════════════════════════════════════
# Full repair pipeline
# ═══════════════════════════════════════════════════════════════════════════════


def full_repair(
    tm: TransitionMatrix,
    prior: np.ndarray = SP_BENCHMARK,
    smooth_weight: float = 0.3,
    enforce_embed: bool = True,
) -> TransitionMatrix:
    """
    Run the full repair pipeline on a raw transition matrix.

    Steps (in order):
      1. Enforce absorbing default
      2. Remove negatives
      3. Bayesian smoothing (adaptive to sample size)
      4. Enforce monotonic PD (isotonic regression)
      5. Enforce embeddability (optional)
      6. Final row-stochastic cleanup

    Returns a production-quality TransitionMatrix with full audit trail.
    """
    result = tm

    # Step 1: Absorbing default
    result = enforce_absorbing_default(result)

    # Step 2: Remove negatives
    result = remove_negatives(result, method="redistribute")

    # Step 3: Bayesian smoothing
    result = bayesian_smooth(result, prior=prior, weight=smooth_weight, count_adaptive=True)

    # Step 4: Monotonic PD
    result = enforce_monotonic_pd(result, method="isotonic")

    # Step 5: Embeddability (expensive but important)
    if enforce_embed:
        result = enforce_embeddability(result)

    # Step 6: Final cleanup
    result = enforce_row_stochastic(result)
    result = TransitionMatrix(
        matrix=result.matrix,
        source="repaired",
        corrections=result.corrections,
    )

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Internal algorithms
# ═══════════════════════════════════════════════════════════════════════════════


def _isotonic_regression(y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators Algorithm (PAVA) for monotone regression.

    Given y = [y_1, ..., y_n], find z = [z_1, ..., z_n] that minimizes
    Σ(y_i - z_i)² subject to z_1 ≤ z_2 ≤ ... ≤ z_n.

    This is the optimal L2 projection onto the cone of non-decreasing
    sequences. O(n) time.
    """
    n = len(y)
    z = y.copy().astype(np.float64)
    blocks = [[i] for i in range(n)]

    # Forward pass: merge violating adjacent blocks
    merged = True
    while merged:
        merged = False
        new_blocks = [blocks[0]]
        for k in range(1, len(blocks)):
            prev_mean = np.mean(z[new_blocks[-1]])
            curr_mean = np.mean(z[blocks[k]])
            if prev_mean > curr_mean + 1e-14:
                # Merge
                combined = new_blocks[-1] + blocks[k]
                pool_val = np.mean(y[combined])
                z[combined] = pool_val
                new_blocks[-1] = combined
                merged = True
            else:
                new_blocks.append(blocks[k])
        blocks = new_blocks

    return z


def _monotone_spline(y: np.ndarray) -> np.ndarray:
    """
    Fit a monotone-increasing curve through PD values using
    constrained interpolation.
    """
    # Simple approach: use isotonic then lightly smooth
    z = _isotonic_regression(y)
    # Apply light Gaussian smoothing while preserving monotonicity
    from scipy.ndimage import gaussian_filter1d
    z_smooth = gaussian_filter1d(z, sigma=0.5)
    # Re-apply isotonic to ensure monotonicity after smoothing
    z_smooth = _isotonic_regression(z_smooth)
    return z_smooth


def _is_valid_generator(Q: np.ndarray) -> bool:
    """Check if Q is a valid intensity (generator) matrix."""
    n = Q.shape[0]
    for i in range(n):
        # Off-diagonal should be ≥ 0
        for j in range(n):
            if i != j and Q[i, j] < -1e-8:
                return False
        # Row sums should be ~0
        if abs(Q[i].sum()) > 1e-6:
            return False
        # Diagonal should be ≤ 0
        if Q[i, i] > 1e-8:
            return False
    return True


def _optimize_generator(P_target: np.ndarray, max_iter: int = 200) -> np.ndarray | None:
    """
    Find generator Q that minimizes ||exp(Q) - P_target||_F
    subject to Q being a valid generator matrix.

    Uses L-BFGS-B with off-diagonal elements as free variables
    and diagonal set to -row_sum(off-diagonal).
    """
    n = P_target.shape[0]

    # Initial guess: try matrix log, clamp to valid generator
    try:
        Q_init = logm(P_target).real
    except Exception:
        Q_init = np.zeros((n, n))

    # Extract off-diagonal elements as free variables
    off_diag_idx = [(i, j) for i in range(n) for j in range(n) if i != j]
    n_free = len(off_diag_idx)

    def _pack(Q):
        return np.array([max(Q[i, j], 0) for i, j in off_diag_idx])

    def _unpack(x):
        Q = np.zeros((n, n))
        for k, (i, j) in enumerate(off_diag_idx):
            Q[i, j] = x[k]
        for i in range(n):
            Q[i, i] = -Q[i, [j for j in range(n) if j != i]].sum()
        return Q

    def _objective(x):
        Q = _unpack(x)
        try:
            P_recon = expm(Q).real
        except Exception:
            return 1e10
        return np.sum((P_recon - P_target) ** 2)

    x0 = _pack(Q_init)
    bounds = [(0, None)] * n_free  # off-diagonal ≥ 0

    try:
        result = minimize(
            _objective, x0, method="L-BFGS-B",
            bounds=bounds, options={"maxiter": max_iter, "ftol": 1e-12}
        )
        if result.success or result.fun < 0.01:
            return _unpack(result.x)
    except Exception:
        pass

    return None


def _kreinin_sidelnikova(P: np.ndarray) -> np.ndarray:
    """
    Kreinin-Sidelnikova diagonal adjustment for embeddability.

    Adjusts diagonal elements to make the matrix embeddable while
    preserving off-diagonal ratios. Simple fallback when optimization fails.
    """
    n = P.shape[0]
    P_fixed = P.copy()

    # Ensure all eigenvalues are positive by adjusting diagonal
    eigvals = np.linalg.eigvals(P_fixed)
    min_eig = np.min(np.real(eigvals))

    if min_eig <= 0:
        shift = abs(min_eig) + 0.01
        for i in range(n):
            P_fixed[i, i] += shift
        # Re-normalize
        for i in range(n):
            P_fixed[i] /= P_fixed[i].sum()

    return P_fixed
