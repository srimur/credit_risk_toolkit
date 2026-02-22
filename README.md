# CRM — Credit Risk Modelling Toolkit

A toolkit for credit rating transition matrix analytics, solving key pain points in credit risk work.

### Transition Matrix Repair
Raw transition matrices estimated from internal bank data are sometimes broken: non-monotonic PD, sparse cells, non-embeddable matrices. The repair module applies mathematically principled corrections—isotonic regression for PD monotonicity, Bayesian smoothing with S&P benchmarks for sparse cells, optimization-based embeddability enforcement—with a complete audit trail for model governance.

- **Isotonic regression (PAVA)** for PD monotonicity enforcement — optimal L2 projection onto the monotone cone
- **Bayesian smoothing** with adaptive weighting: sparse rating grades automatically get more prior (S&P benchmark)
- **Generator matrix estimation** via constrained optimization for continuous-time consistency
- **Quality scoring** (0–100) with automated diagnostic flags for model governance

## Quick Start

```python
from crm import estimate_cohort, full_repair

# Estimate raw transition matrix
raw_tm, counts = estimate_cohort(ratings_year1, ratings_year2)
print(raw_tm.quality_score())  # e.g., 35/100

# Repair it
repaired = full_repair(raw_tm)
print(repaired.quality_score())  # e.g., 90/100
print(repaired.corrections)  # full audit trail


```

## Tests

```bash
python tests/test_crm.py  # 30/30 passing
```

## Dependencies

- numpy, scipy, pandas (core computation)
- scikit-learn
