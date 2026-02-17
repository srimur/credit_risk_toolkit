# CRM — Credit Risk Modelling Toolkit

A production-grade toolkit for credit rating transition matrix analytics, solving three key pain points in credit risk quantitative work.

## Pain Points Addressed

### 1. Transition Matrix Repair (`crm.repair`)
Raw transition matrices estimated from internal bank data are almost always broken: non-monotonic PD, sparse cells, non-embeddable matrices. The repair module applies mathematically principled corrections—isotonic regression for PD monotonicity, Bayesian smoothing with S&P benchmarks for sparse cells, optimization-based embeddability enforcement—with a complete audit trail for model governance.

### 2. TTC-to-PIT Conversion (`crm.pit`)
Banks need both Through-the-Cycle PD (Basel RWA) and Point-in-Time PD (IFRS 9 ECL). This module converts between them using the Vasicek single-factor model with scenario-weighting, producing IFRS 9-compliant forward-looking PD curves.

### 3. IFRS 9 ECL Computation (`crm.ecl`)
Forward-looking, scenario-weighted, properly discounted Expected Credit Loss with automatic IFRS 9 stage allocation. Takes the transition matrix pipeline output and produces provision-ready numbers.

## Architecture

```
crm/
├── types.py       # TransitionMatrix with auto-diagnostics and quality scoring
├── estimator.py   # Cohort method estimation from rating histories
├── repair.py      # 6-step repair pipeline (absorbing → negatives → smooth → monotonic → embeddable → stochastic)
├── pit.py         # Vasicek TTC→PIT, scenario weighting, lifetime PD curves
├── ecl.py         # IFRS 9 ECL with staging, discounting, scenario decomposition
tests/
└── test_crm.py    # 30 tests covering all modules
```

## Key Technical Highlights

- **Isotonic regression (PAVA)** for PD monotonicity enforcement — optimal L2 projection onto the monotone cone
- **Bayesian smoothing** with adaptive weighting: sparse rating grades automatically get more prior (S&P benchmark)
- **Generator matrix estimation** via constrained optimization for continuous-time consistency
- **Vasicek conditional PD** with correct sign convention: `Φ[(Φ⁻¹(PD) - √ρ × Z) / √(1-ρ)]`
- **Quality scoring** (0–100) with automated diagnostic flags for model governance

## Quick Start

```python
from crm import estimate_cohort, full_repair, ttc_to_pit_vasicek, compute_ecl

# 1. Estimate raw transition matrix
raw_tm, counts = estimate_cohort(ratings_year1, ratings_year2)
print(raw_tm.quality_score())  # e.g., 35/100

# 2. Repair it
repaired = full_repair(raw_tm)
print(repaired.quality_score())  # e.g., 90/100
print(repaired.corrections)  # full audit trail

# 3. Convert TTC → PIT for recession scenario
pit = ttc_to_pit_vasicek(repaired, z=-2.33, rho=0.20)

# 4. Compute IFRS 9 ECL
ecl = compute_ecl(pit, rating="BBB", ead=1_000_000, lgd=0.45)
print(f"Stage: {ecl.stage}, 12m ECL: {ecl.ecl_12m:,.0f}, Lifetime: {ecl.ecl_lifetime:,.0f}")
```

## Tests

```bash
python tests/test_crm.py  # 30/30 passing
```

## Dependencies

- numpy, scipy, pandas (core computation)
- scikit-learn (not required for core modules)
