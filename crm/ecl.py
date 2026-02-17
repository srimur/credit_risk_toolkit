"""
crm.ecl — IFRS 9 Expected Credit Loss computation.

THE THIRD KEY PAIN POINT.

Under IFRS 9, banks must compute forward-looking, scenario-weighted
Expected Credit Losses with proper discounting.

    ECL = Σ_t  PD_marginal(t) × LGD × EAD(t) × DF(t)

This module computes:
  - 12-month ECL (Stage 1)
  - Lifetime ECL (Stages 2 & 3)
  - Stage allocation based on rating migration
  - Scenario-weighted ECL

The output is directly usable for IFRS 9 provisioning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .types import D_IDX, N, RATING_IDX, RATING_SCALE, TransitionMatrix


@dataclass
class ECLResult:
    """IFRS 9 ECL computation results per exposure."""

    rating: str
    stage: int              # 1, 2, or 3
    ead: float
    lgd: float
    ecl_12m: float          # Stage 1: 12-month ECL
    ecl_lifetime: float     # Stages 2/3: lifetime ECL
    ecl_reported: float     # actual provision (12m if Stage 1, lifetime if 2/3)
    pd_12m: float
    pd_lifetime: float
    discount_rate: float
    horizon: int            # years for lifetime calculation
    scenario_ecls: Dict[str, float] = field(default_factory=dict)


def compute_ecl(
    pit_matrix: TransitionMatrix,
    rating: str,
    ead: float,
    lgd: float,
    original_rating: Optional[str] = None,
    discount_rate: float = 0.05,
    lifetime_horizon: int = 10,
    scenario_matrices: Optional[Dict[str, TransitionMatrix]] = None,
) -> ECLResult:
    """
    Compute IFRS 9 ECL for a single exposure.

    Parameters
    ----------
    pit_matrix : TransitionMatrix
        PIT (or scenario-weighted) transition matrix.
    rating : str
        Current credit rating.
    ead : float
        Exposure at Default.
    lgd : float
        Loss Given Default (0–1).
    original_rating : str or None
        Rating at origination. Used for staging: if current rating is
        significantly worse, assign Stage 2.
    discount_rate : float
        Effective interest rate for discounting.
    lifetime_horizon : int
        Maximum years for lifetime ECL computation.
    scenario_matrices : dict or None
        If provided, compute scenario-level ECLs for audit trail.

    Returns
    -------
    ECLResult
    """
    r_idx = RATING_IDX[rating]
    P = pit_matrix.matrix

    # ── Stage allocation ─────────────────────────────────────────────
    stage = _allocate_stage(rating, original_rating)

    # ── 12-month ECL ─────────────────────────────────────────────────
    pd_12m = P[r_idx, D_IDX]
    ecl_12m = pd_12m * lgd * ead

    # ── Lifetime ECL (discounted) ────────────────────────────────────
    ecl_lifetime = _lifetime_ecl(
        P, r_idx, lgd, ead, discount_rate, lifetime_horizon
    )
    pd_lifetime = _lifetime_pd(P, r_idx, lifetime_horizon)

    # ── Reported ECL (depends on stage) ──────────────────────────────
    ecl_reported = ecl_12m if stage == 1 else ecl_lifetime

    # ── Scenario-level ECLs (for audit) ──────────────────────────────
    scenario_ecls = {}
    if scenario_matrices:
        for name, s_matrix in scenario_matrices.items():
            s_ecl = _lifetime_ecl(
                s_matrix.matrix, r_idx, lgd, ead, discount_rate, lifetime_horizon
            )
            scenario_ecls[name] = s_ecl

    return ECLResult(
        rating=rating,
        stage=stage,
        ead=ead,
        lgd=lgd,
        ecl_12m=ecl_12m,
        ecl_lifetime=ecl_lifetime,
        ecl_reported=ecl_reported,
        pd_12m=pd_12m,
        pd_lifetime=pd_lifetime,
        discount_rate=discount_rate,
        horizon=lifetime_horizon,
        scenario_ecls=scenario_ecls,
    )


def compute_portfolio_ecl(
    pit_matrix: TransitionMatrix,
    exposures: pd.DataFrame,
    discount_rate: float = 0.05,
    lifetime_horizon: int = 10,
    scenario_matrices: Optional[Dict[str, TransitionMatrix]] = None,
) -> pd.DataFrame:
    """
    Compute IFRS 9 ECL for an entire portfolio.

    Parameters
    ----------
    pit_matrix : TransitionMatrix
    exposures : DataFrame
        Must have columns: 'rating', 'ead', 'lgd'.
        Optional: 'original_rating' for staging.
    discount_rate : float
    lifetime_horizon : int
    scenario_matrices : dict or None

    Returns
    -------
    DataFrame with ECL results per exposure plus portfolio totals.
    """
    results = []
    for _, row in exposures.iterrows():
        orig = row.get("original_rating", None)
        ecl = compute_ecl(
            pit_matrix=pit_matrix,
            rating=row["rating"],
            ead=row["ead"],
            lgd=row["lgd"],
            original_rating=orig,
            discount_rate=discount_rate,
            lifetime_horizon=lifetime_horizon,
            scenario_matrices=scenario_matrices,
        )
        results.append({
            "rating": ecl.rating,
            "stage": ecl.stage,
            "ead": ecl.ead,
            "lgd": ecl.lgd,
            "pd_12m": ecl.pd_12m,
            "ecl_12m": ecl.ecl_12m,
            "ecl_lifetime": ecl.ecl_lifetime,
            "ecl_reported": ecl.ecl_reported,
            **{f"ecl_{k}": v for k, v in ecl.scenario_ecls.items()},
        })

    df = pd.DataFrame(results)
    return df


# ── Staging logic ────────────────────────────────────────────────────────────


def _allocate_stage(
    current_rating: str,
    original_rating: Optional[str],
    notch_threshold: int = 2,
) -> int:
    """
    IFRS 9 stage allocation.

    Stage 1: No significant increase in credit risk since origination.
    Stage 2: Significant increase (downgraded ≥ notch_threshold notches).
    Stage 3: Credit-impaired (rated D/default).
    """
    if current_rating == "D":
        return 3

    if original_rating is None:
        return 1

    current_idx = RATING_IDX.get(current_rating, 0)
    original_idx = RATING_IDX.get(original_rating, 0)

    if current_idx - original_idx >= notch_threshold:
        return 2
    return 1


# ── ECL internals ────────────────────────────────────────────────────────────


def _lifetime_ecl(
    P: np.ndarray,
    r_idx: int,
    lgd: float,
    ead: float,
    discount_rate: float,
    horizon: int,
) -> float:
    """
    Discounted lifetime ECL.

    ECL = Σ_{t=1}^{T}  marginal_PD(t) × LGD × EAD × DF(t)

    where marginal_PD(t) = cumulative_PD(t) - cumulative_PD(t-1)
    and DF(t) = 1 / (1 + r)^t
    """
    P_power = np.eye(N)
    prev_cum_pd = 0.0
    ecl = 0.0

    for t in range(1, horizon + 1):
        P_power = P_power @ P
        cum_pd = P_power[r_idx, D_IDX]
        marginal_pd = max(cum_pd - prev_cum_pd, 0.0)
        prev_cum_pd = cum_pd

        df = 1.0 / (1.0 + discount_rate) ** t
        ecl += marginal_pd * lgd * ead * df

    return ecl


def _lifetime_pd(P: np.ndarray, r_idx: int, horizon: int) -> float:
    """Cumulative PD over the full horizon."""
    P_power = np.linalg.matrix_power(P, horizon)
    return float(P_power[r_idx, D_IDX])
