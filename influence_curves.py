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


def _compute_influence_curve_single_fold(y_pred, y_true, emp_prob_1_global=None, emp_prob_0_global=None):
    """Compute influence curves for all samples in a single fold.
    
    Vectorized implementation for efficiency.
    
    Parameters
    ----------
    y_pred : array-like of shape (n_samples,)
        Predicted probabilities for the positive class.
    
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    
    emp_prob_1_global : float, optional
        Proportion of positive samples in the FULL dataset (Pn(Y=1)).
        If None, computed from the fold (not recommended, for backward compatibility).
    
    emp_prob_0_global : float, optional
        Proportion of negative samples in the FULL dataset (Pn(Y=0)).
        If None, computed from the fold (not recommended, for backward compatibility).
    
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
    """
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    
    if n1 == 0 or n0 == 0:
        # Cannot compute AUC with single class
        return np.zeros(n)
    
    # Use global class proportions if provided, otherwise fall back to fold-local
    # (fold-local is incorrect per the paper, but kept for backward compatibility)
    emp_prob_1 = emp_prob_1_global if emp_prob_1_global is not None else (n1 / n)
    emp_prob_0 = emp_prob_0_global if emp_prob_0_global is not None else (n0 / n)
    
    auc = roc_auc_score(y_true, y_pred)
    
    # Vectorized computation of P1 and P0 for all samples
    y_pred_1 = y_pred[y_true == 1]
    y_pred_0 = y_pred[y_true == 0]
    
    # For each sample, count how many negatives have lower scores
    # and how many positives have higher scores
    p1 = np.sum(y_pred_0[:, np.newaxis] < y_pred, axis=0) / n0
    p0 = np.sum(y_pred_1[:, np.newaxis] > y_pred, axis=0) / n1
    
    # Compute influence curve for each sample
    is_positive = (y_true == 1)
    ic = np.where(
        is_positive,
        (p1 - auc) / emp_prob_1,
        (p0 - auc) / emp_prob_0
    )
    
    return ic


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
    
    Returns
    -------
    variance : float
        Variance estimate of the AUC.
    """
    n = len(ic_all)
    variance = np.sum(ic_all ** 2) / n 
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