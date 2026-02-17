"""
crm — Credit Risk Modelling Toolkit
=====================================

A production-grade toolkit for credit rating transition matrix
analytics, solving three key pain points in credit risk quant work:

1. **Matrix Repair**: Raw transition matrices from internal data are
   almost always broken (non-monotonic PD, sparse cells, non-embeddable).
   The repair module applies principled corrections with a full audit trail.

2. **TTC-to-PIT Conversion**: Banks need both Through-the-Cycle and
   Point-in-Time PD estimates. The pit module converts between them
   using Vasicek's model with scenario-weighting for IFRS 9.

3. **IFRS 9 ECL**: Forward-looking, scenario-weighted, properly
   discounted Expected Credit Loss with automatic stage allocation.

Pipeline
--------
    Raw data → Cohort estimation → Matrix repair → TTC matrix
                                                     ↓
                                            TTC-to-PIT conversion
                                                     ↓
                                            IFRS 9 ECL computation
"""

from .types import TransitionMatrix, RATING_SCALE, RATING_IDX, N, D_IDX
from .estimator import estimate_cohort, estimate_multi_period, pool_matrices
from .repair import (
    full_repair,
    enforce_absorbing_default,
    remove_negatives,
    enforce_monotonic_pd,
    bayesian_smooth,
    enforce_embeddability,
    SP_BENCHMARK,
)
from .pit import (
    ttc_to_pit_vasicek,
    scenario_weighted_pit,
    compute_z_from_macro,
    lifetime_pd_curve,
    MacroScenario,
    IFRS9_SCENARIOS,
)
from .ecl import compute_ecl, compute_portfolio_ecl, ECLResult

__version__ = "1.0.0"

__all__ = [
    # Types
    "TransitionMatrix", "RATING_SCALE", "RATING_IDX",
    # Estimation
    "estimate_cohort", "estimate_multi_period", "pool_matrices",
    # Repair
    "full_repair", "enforce_absorbing_default", "remove_negatives",
    "enforce_monotonic_pd", "bayesian_smooth", "enforce_embeddability",
    "SP_BENCHMARK",
    # PIT conversion
    "ttc_to_pit_vasicek", "scenario_weighted_pit", "compute_z_from_macro",
    "lifetime_pd_curve", "MacroScenario", "IFRS9_SCENARIOS",
    # ECL
    "compute_ecl", "compute_portfolio_ecl", "ECLResult",
]
