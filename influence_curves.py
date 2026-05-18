"""
Influence curve computation for AUC confidence intervals.

This module implements the influence curve-based variance estimation from:

    LeDell et al. (2015). "Computationally efficient confidence intervals for 
    cross-validated area under the ROC curve estimates." Electronic Journal of
    Statistics, 9(1), 1583-1607 (https://pubmed.ncbi.nlm.nih.gov/26279737/). 

Functions:
    - _compute_influence_curve_single_fold() - IC computation per fold
    - compute_variance() - Variance estimation from ICs
    - compute_confidence_interval() - CI computation from variance
"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


# New: added functionality for influence curve computation from here on
def _compute_influence_curve_single_fold(y_pred, y_true):
    """Compute influence curves for all samples in a single fold.
    
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
    """
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    
    if n1 == 0 or n0 == 0:
        # Cannot compute AUC with single class
        return np.zeros(n)
    
    emp_prob_1 = n1 / n
    emp_prob_0 = n0 / n
    
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


def compute_variance(ic_all, V):
    """Compute variance of the AUC estimate from influence curves.
    
    Following the paper's definition: variance = (1/Vn) * sum(IC_i^2)
    This assumes mean of IC should be 0, so we don't subtract it.
    
    Parameters
    ----------
    ic_all : array-like of shape (n_samples,)
        Influence curve values for all samples.
    
    V : int
        Number of cross-validation folds.
    
    Returns
    -------
    variance : float
        Variance estimate of the AUC.
    """
    n = len(ic_all)
    variance = np.sum(ic_all ** 2) / (n * V)
    return variance


def compute_confidence_interval(estimate, variance, n, confidence_level=0.95):
    """Compute confidence interval for the AUC estimate.
    
    CI = estimate +/- z * sigma / sqrt(n)
    where sigma = sqrt(variance) is the std of the influence curves.
    
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
        Lower and upper bounds of the confidence interval.
    """
    z_score = norm.ppf(0.5 + confidence_level / 2)
    bound = z_score * np.sqrt(variance) / np.sqrt(n)
    conf_int = (estimate - bound, estimate + bound)
    
    return conf_int