"""
crm.types — shared types and constants for credit risk modelling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

RATING_SCALE: List[str] = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
RATING_IDX: Dict[str, int] = {r: i for i, r in enumerate(RATING_SCALE)}
N: int = len(RATING_SCALE)
D_IDX: int = RATING_IDX["D"]


@dataclass
class TransitionMatrix:
    """
    Annotated transition matrix with provenance metadata.

    Every matrix in this system carries metadata about how it was
    produced, what corrections were applied, and quality diagnostics.
    This is critical for model audit trails.
    """

    matrix: np.ndarray                  # (N, N) stochastic matrix
    labels: List[str] = field(default_factory=lambda: list(RATING_SCALE))
    source: str = "raw"                 # raw | smoothed | generator | stressed
    corrections: List[str] = field(default_factory=list)
    diagnostics: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.diagnostics.update(self._compute_diagnostics())

    def _compute_diagnostics(self) -> Dict:
        P = self.matrix
        n = P.shape[0]
        diag = {}

        # Row sums
        row_sums = P.sum(axis=1)
        diag["max_row_sum_error"] = float(np.max(np.abs(row_sums - 1.0)))

        # Negative entries
        diag["has_negatives"] = bool(np.any(P < -1e-12))
        diag["min_entry"] = float(P.min())

        # PD monotonicity: PD should increase AAA → CCC
        pd_col = P[:D_IDX, D_IDX]
        diag["pd_monotonic"] = bool(
            all(pd_col[i] <= pd_col[i + 1] + 1e-10 for i in range(len(pd_col) - 1))
        )

        # Diagonal dominance: each state most likely stays
        diag["diagonal_dominant"] = bool(
            all(P[i, i] >= P[i, j] for i in range(n) for j in range(n) if i != j)
        )

        # Absorbing default
        diag["default_absorbing"] = bool(
            np.isclose(P[D_IDX, D_IDX], 1.0) and np.allclose(P[D_IDX, :D_IDX], 0.0)
        )

        # Embeddability: can this matrix come from a valid generator?
        # Necessary condition: det(P) > 0 and all eigenvalues are positive real
        eigvals = np.linalg.eigvals(P)
        diag["all_eigenvalues_positive"] = bool(np.all(np.real(eigvals) > 0))
        diag["determinant"] = float(np.real(np.linalg.det(P)))
        diag["embeddable"] = diag["all_eigenvalues_positive"] and diag["determinant"] > 0

        return diag

    def pd_vector(self) -> np.ndarray:
        """Default column: P(rating → D)."""
        return self.matrix[:, D_IDX].copy()

    def retention_rates(self) -> np.ndarray:
        """Diagonal: P(rating → same rating)."""
        return np.diag(self.matrix).copy()

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.matrix, index=self.labels, columns=self.labels)

    def quality_score(self) -> float:
        """
        0–100 quality score based on diagnostics.

        100 = perfect (monotonic PD, diagonal dominant, embeddable,
              absorbing default, no negatives).
        """
        d = self.diagnostics
        score = 0.0
        if not d["has_negatives"]:
            score += 20
        if d["pd_monotonic"]:
            score += 25
        if d["diagonal_dominant"]:
            score += 20
        if d["default_absorbing"]:
            score += 15
        if d["embeddable"]:
            score += 15
        if d["max_row_sum_error"] < 1e-10:
            score += 5
        return score

    def __repr__(self) -> str:
        return (
            f"TransitionMatrix(source={self.source!r}, "
            f"quality={self.quality_score():.0f}/100, "
            f"corrections={self.corrections})"
        )
