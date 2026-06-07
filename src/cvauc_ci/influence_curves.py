"""
Influence curve computation for (cross-validated) AUC confidence intervals.

This module implements the influence curve-based variance estimation from:

    LeDell et al. (2015). "Computationally efficient confidence intervals for
    cross-validated area under the ROC curve estimates." Electronic Journal of
    Statistics, 9(1), 1583-1607 (https://pubmed.ncbi.nlm.nih.gov/26279737/).

and matches the R cvAUC package implementation by the paper's authors:
www.github.com/ledell/cvAUC/
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def compute_auc_influence_curve(y_pred, y_true):
    """Compute influence curves for all samples in a single validation fold/test set.

    Vectorized implementation for efficiency.

    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).

    Returns
    -------
    ic : ndarray of shape (n_samples,)
        Influence curve values for each sample.

    Notes
    -----
    Per LeDell et al. (2015), the empirical probabilities Pn(Y=1) and Pn(Y=0)
    should be computed from the entire dataset, not the validation fold.
    The ranking comparisons P(ψ<w|Y=0) and P(ψ>w|Y=1) are computed within
    the validation fold.

    This implementation intentionally uses tie-aware pairwise comparisons
    (half credit for equal scores) to align influence-curve terms with
    sklearn.metrics.roc_auc_score. The original R cvAUC implementation
    uses strict inequalities, which can diverge from tie-aware AUC values
    when tied prediction scores are present.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    n = len(y_true)
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)

    if n1 == 0 or n0 == 0:
        # Cannot compute AUC with single class
        return np.zeros(n)

    # Use class proportions of the test set.
    # For CVAUC, these are fold-local, which is incorrect per the paper; correction is done in cross_validate().
    emp_prob_1 = n1 / n
    emp_prob_0 = n0 / n

    auc = roc_auc_score(y_true, y_pred)

    # Vectorized computation of P1 and P0 for all samples
    y_pred_1 = y_pred[y_true == 1]
    y_pred_0 = y_pred[y_true == 0]

    # For each sample, count how many negatives have lower scores and how many positives have higher scores.
    # Use tie-aware ranks that match roc_auc_score semantics:
    # strict wins + half credit for ties.
    p1 = (
        np.sum(y_pred_0[:, np.newaxis] < y_pred, axis=0)
        + 0.5 * np.sum(y_pred_0[:, np.newaxis] == y_pred, axis=0)
    ) / n0
    p0 = (
        np.sum(y_pred_1[:, np.newaxis] > y_pred, axis=0)
        + 0.5 * np.sum(y_pred_1[:, np.newaxis] == y_pred, axis=0)
    ) / n1

    # Compute influence curve for each sample
    is_positive = y_true == 1
    ic = np.where(is_positive, (p1 - auc) / emp_prob_1, (p0 - auc) / emp_prob_0)

    return ic


# TODO: implement fold weights properly. Currently: IC curves are weighted by fold weights, variance is computed without fold weights.
# TODO: Fix1: add fold weights if use_fold_weights=True
def compute_variance(ic_all):
    """Compute variance of the AUC estimate from influence curves.


    Note: we follow the R cvAUC package formula: variance = (1/n) * sum(IC_i^2).
    The paper's equation 4.11 shows variance = (1/(V*n) * sum(IC_i^2).
    The R implementation by the paper's authors uses (1/n) instead of (1/(V*n)),
    which gives more conservative (wider) CIs that account for this correlation.

    Parameters
    ----------
    ic_all : array-like of shape (n_samples,)
        Influence curve values for all samples.

    use_fold_weights : bool
        Whether to use fold weights. Useful for folds of different sizes.
        LeDell et al. (2015) do not use fold weights, and the R cvAUC package does not implement fold weights.

    Returns
    -------
    variance : float
        Variance estimate of the AUC.
    """
    n = len(ic_all)
    variance = np.sum(ic_all**2) / n
    return variance


def compute_confidence_interval(estimate, variance, n, confidence_level=0.95):
    """Compute confidence interval for the AUC estimate.

    CI = estimate +/- z * sigma / sqrt(n)
    where sigma = sqrt(variance) is the std of the influence curves.

    The CI is truncated to [0, 1].

    Parameters
    ----------
    estimate : float
        Point estimate of AUC.

    variance : float
        Variance estimate from influence curves (sigma^2).

    n : int
        Total number of samples.

    confidence_level : float, default=0.95
        Confidence level for the interval.

    Returns
    -------
    conf_int : tuple of (float, float)
        Lower and upper bounds of the confidence interval, truncated to [0, 1].
    """
    z_score = norm.ppf(0.5 + confidence_level / 2)
    bound = z_score * np.sqrt(variance) / np.sqrt(n)
    lower = max(0.0, estimate - bound)  # Truncate at 0
    upper = min(1.0, estimate + bound)  # Truncate at 1

    return (lower, upper)


def _compute_global_weights(y_global, y_test):
    """Construct sample weights.

    Per LeDell et al. (2015), weights for folds are defined by total sample distribution, not fold-local distribution.
    This requires correcting generic test set weights by the ratio of global to fold-local class proportions.
    """
    emp_prob_1_global = np.mean(y_global)
    emp_prob_0_global = 1 - emp_prob_1_global

    emp_prob_1_fold = np.mean(y_test)
    emp_prob_0_fold = 1 - emp_prob_1_fold

    weight_0 = emp_prob_0_fold / emp_prob_0_global if emp_prob_0_fold > 0 else 0
    weight_1 = emp_prob_1_fold / emp_prob_1_global if emp_prob_1_fold > 0 else 0

    # Construct weights array
    weights = np.where(y_test == 1, weight_1, weight_0)

    return weights
