# cvauc-ci
Confidence-intervals for cross-validated AUC scores based on influence curves.

Implementation of the algorithm described in
     LeDell et al. (2015). "Computationally efficient confidence intervals for
    cross-validated area under the ROC curve estimates." Electronic Journal of
    Statistics, 9(1), 1583-1607 (https://pubmed.ncbi.nlm.nih.gov/26279737/).

The code is a modification of sklearn's cross_val_score() (v1.7.2). Main API is cvauc.py's cross_val_auc() function. See also the author's  R code at https://github.com/ledell/cvAUC.

For a primer on influence curves, see e.g.
    Hampel et al. (1986/2011). "Robust Statistics: The Approach Based on Influence Functions." Wiley. DOI:10.1002/9781118186435

## Installation

```bash
uv sync
```

For development:

```bash
uv sync --extra dev
uv run pre-commit install
```

## Usage

```python
from cvauc import cross_val_auc
from sklearn.linear_model import LogisticRegression

scores, conf_int = cross_val_auc(
    LogisticRegression(),
    X, y,
    cv=5,
    confidence_level=0.95
)
print(f"AUC: {scores.mean():.3f} (95% CI: {conf_int[0]:.3f}-{conf_int[1]:.3f})")
```

## Features

- Drop-in replacement for sklearn's `cross_val_score` with AUC
- Influence curve-based confidence intervals (faster runtime compared to bootstrap)
- Subset evaluation via `eval_subset` parameter for subgroup/fairness analysis

## References

LeDell E, Petersen M, van der Laan M (2015). "Computationally efficient confidence intervals for cross-validated area under the ROC curve estimates." *Electronic Journal of Statistics*, 9(1), 1583-1607. [PubMed](https://pubmed.ncbi.nlm.nih.gov/26279737/)

## Note

The `cross_validate()` and `_fit_and_score()` functions are modified from scikit-learn 1.7.2. Requires scikit-learn >= 1.5.0.
