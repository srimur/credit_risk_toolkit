"""
crm.pit — Through-the-Cycle (TTC) to Point-in-Time (PIT) PD conversion.


Approaches implemented
----------------------
1. **Vasicek scalar adjustment**: PIT_PD = Φ[(Φ⁻¹(TTC_PD) + √ρ × Z(t)) / √(1-ρ)]
   where Z(t) is the current macro state. Simple, closed-form

2. **Merton-style structural**: Map macro variables to the systematic
   factor via regression. More flexible, requires macro data.

3. **Scenario-weighted**: Compute PIT matrix under multiple macro scenarios
   and probability-weight them. Required by IFRS 9 for ECL.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm

from .types import D_IDX, N, RATING_SCALE, TransitionMatrix


@dataclass
class MacroScenario:
    """A macroeconomic scenario with associated probability."""
    name: str
    z_factor: float       # systematic factor realization
    probability: float    # scenario weight (must sum to 1 across scenarios)
    description: str = ""


# Standard IFRS 9 scenarios
IFRS9_SCENARIOS = [
    MacroScenario("Base",     z_factor=0.0,   probability=0.50, description="Current trajectory"),
    MacroScenario("Upside",   z_factor=1.0,   probability=0.15, description="Strong recovery"),
    MacroScenario("Downside", z_factor=-1.0,  probability=0.25, description="Mild recession"),
    MacroScenario("Severe",   z_factor=-2.33, probability=0.10, description="Deep recession"),
]


def ttc_to_pit_vasicek(
    ttc_matrix: TransitionMatrix,
    z: float,
    rho: float = 0.20,
) -> TransitionMatrix:
    """
    Convert TTC transition matrix to PIT using Vasicek scalar adjustment.

    For each row i (non-default), adjust the PD:
        PIT_PD_i = Φ[(Φ⁻¹(TTC_PD_i) + √ρ × z) / √(1-ρ)]

    The non-default transition probabilities are scaled proportionally
    to absorb the PD change, preserving the relative migration structure.

    Parameters
    ----------
    ttc_matrix : TransitionMatrix
        Through-the-cycle transition matrix.
    z : float
        Systematic factor. z < 0 → recession (higher PD).
        z = 0 → average conditions (PIT ≈ TTC).
        z > 0 → expansion (lower PD).
    rho : float
        Asset correlation (Basel corporate: 0.12–0.24).

    Returns
    -------
    TransitionMatrix with PIT probabilities.
    """
    P_ttc = ttc_matrix.matrix.copy()
    P_pit = P_ttc.copy()
    sqrt_rho = np.sqrt(rho)
    sqrt_1_rho = np.sqrt(1.0 - rho)

    for i in range(D_IDX):  # for each non-default rating
        ttc_pd = P_ttc[i, D_IDX]
        if ttc_pd <= 1e-10 or ttc_pd >= 1.0 - 1e-10:
            continue

        # Vasicek conditional PD:
        # P(default | Z) = Φ[(Φ⁻¹(PD) - √ρ × Z) / √(1-ρ)]
        # Z < 0 (recession) → subtraction of negative → higher PD. Correct.
        pit_pd = float(norm.cdf(
            (norm.ppf(ttc_pd) - sqrt_rho * z) / sqrt_1_rho
        ))
        pit_pd = np.clip(pit_pd, 1e-10, 1.0 - 1e-10)

        # Redistribute the PD change across non-default columns
        pd_delta = pit_pd - ttc_pd
        P_pit[i, D_IDX] = pit_pd

        non_d_cols = [j for j in range(N) if j != D_IDX]
        non_d_sum = P_pit[i, non_d_cols].sum()

        if non_d_sum > 1e-10:
            scale = (non_d_sum - pd_delta) / non_d_sum
            scale = max(scale, 0.01)
            P_pit[i, non_d_cols] *= scale

        # Re-normalize
        P_pit[i] = np.maximum(P_pit[i], 0.0)
        s = P_pit[i].sum()
        if s > 0:
            P_pit[i] /= s

    return TransitionMatrix(
        matrix=P_pit,
        source="pit_vasicek",
        corrections=ttc_matrix.corrections + [f"ttc_to_pit(z={z:.2f}, rho={rho})"],
    )


def scenario_weighted_pit(
    ttc_matrix: TransitionMatrix,
    scenarios: List[MacroScenario] = None,
    rho: float = 0.20,
) -> Tuple[TransitionMatrix, Dict[str, TransitionMatrix]]:
    """
    IFRS 9 scenario-weighted PIT matrix.

    Computes PIT matrix under each scenario, then probability-weights:
        P_PIT = Σ w_s × P_PIT(z_s)

    This is the standard approach for IFRS 9 forward-looking ECL.

    Parameters
    ----------
    ttc_matrix : TransitionMatrix
    scenarios : list of MacroScenario
        If None, uses standard IFRS 9 scenarios.
    rho : float

    Returns
    -------
    (weighted_pit_matrix, scenario_matrices_dict)
    """
    if scenarios is None:
        scenarios = IFRS9_SCENARIOS

    # Validate weights sum to 1
    total_weight = sum(s.probability for s in scenarios)
    if abs(total_weight - 1.0) > 0.01:
        raise ValueError(f"Scenario weights sum to {total_weight}, expected 1.0")

    scenario_matrices = {}
    P_weighted = np.zeros((N, N))

    for scenario in scenarios:
        pit = ttc_to_pit_vasicek(ttc_matrix, z=scenario.z_factor, rho=rho)
        scenario_matrices[scenario.name] = pit
        P_weighted += scenario.probability * pit.matrix

    # Normalize
    for i in range(N):
        s = P_weighted[i].sum()
        if s > 0:
            P_weighted[i] /= s

    weighted = TransitionMatrix(
        matrix=P_weighted,
        source="pit_scenario_weighted",
        corrections=ttc_matrix.corrections + [
            f"scenario_weighted({len(scenarios)} scenarios, rho={rho})"
        ],
    )
    weighted.diagnostics["scenario_pds"] = {
        s.name: float(scenario_matrices[s.name].pd_vector().mean())
        for s in scenarios
    }

    return weighted, scenario_matrices


def compute_z_from_macro(
    gdp_growth: float,
    unemployment: float,
    credit_spread: float,
    coefficients: Optional[Dict[str, float]] = None,
) -> float:
    """
    Map macro variables to the systematic factor Z via linear model.

    Z = β₁ × GDP_growth + β₂ × Unemployment + β₃ × Credit_spread + intercept

    Default coefficients calibrated to US data 2000–2023.
    Positive Z = good economy, negative Z = stress.

    Parameters
    ----------
    gdp_growth : float
        Real GDP growth rate (e.g., 0.02 for 2%).
    unemployment : float
        Unemployment rate (e.g., 0.05 for 5%).
    credit_spread : float
        Investment-grade credit spread in bps (e.g., 150).

    Returns
    -------
    float : Z factor (standardized, mean 0, std ~1 in normal conditions).
    """
    if coefficients is None:
        coefficients = {
            "gdp_growth": 15.0,      # higher GDP → higher Z
            "unemployment": -8.0,    # higher unemployment → lower Z
            "credit_spread": -0.005, # wider spreads → lower Z
            "intercept": 0.5,
        }

    z = (
        coefficients["gdp_growth"] * gdp_growth
        + coefficients["unemployment"] * unemployment
        + coefficients["credit_spread"] * credit_spread
        + coefficients["intercept"]
    )

    return float(z)


def lifetime_pd_curve(
    pit_matrix: TransitionMatrix,
    rating: str,
    horizons: List[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute IFRS 9 lifetime PD curve for a given rating.

    Returns marginal PD and cumulative PD at each horizon (in years).
    Uses matrix powers of the PIT matrix.

    Parameters
    ----------
    pit_matrix : TransitionMatrix
    rating : str
        Starting rating (e.g., "BBB").
    horizons : list of int
        Years to compute (default: 1..10).

    Returns
    -------
    dict with 'horizons', 'marginal_pd', 'cumulative_pd', 'survival'
    """
    from .types import RATING_IDX

    if horizons is None:
        horizons = list(range(1, 11))

    r_idx = RATING_IDX[rating]
    P = pit_matrix.matrix

    marginal_pds = []
    cum_pds = []
    survival = 1.0

    P_power = np.eye(N)
    prev_cum_pd = 0.0

    for h in horizons:
        P_power = P_power @ P
        cum_pd = P_power[r_idx, D_IDX]
        marginal_pd = cum_pd - prev_cum_pd
        prev_cum_pd = cum_pd

        marginal_pds.append(marginal_pd)
        cum_pds.append(cum_pd)

    return {
        "horizons": np.array(horizons),
        "marginal_pd": np.array(marginal_pds),
        "cumulative_pd": np.array(cum_pds),
        "survival": 1.0 - np.array(cum_pds),
    }
