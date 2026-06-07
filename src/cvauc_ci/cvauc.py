"""
Cross-validated AUC with influence curve-based confidence intervals.

This module extends sklearn's cross_validate and _fit_and_score functions
to compute influence curve-based confidence intervals for AUC, following:

    LeDell et al. (2015). "Computationally efficient confidence intervals for
    cross-validated area under the ROC curve estimates." Electronic Journal of
    Statistics, 9(1), 1583-1607.

In addition, functionality is added for evaluating model performance on subsegments of the data
(e.g., specific categories in a categorical column) while training on the full dataset.

Usage:
    from cvauc import cross_val_auc

    scores, conf_int = cross_val_auc(
        estimator, X, y,
        scoring='roc_auc',
        cv=5,
        confidence_level=0.95
    )

    # Evaluate on subgroups (X must be a DataFrame with a categorical column):
    scores, conf_int = cross_val_auc(
        estimator, X, y,
        scoring='roc_auc',
        cv=5,
        confidence_level=0.95,
        eval_subset=('group', 'A')  # Evaluate only on group A
    )
    # Returns: scores['A'], conf_int['score_A']

Modified sklearn functions (changes marked with # NEW):
    - cross_validate() - Added return_influence_curves and eval_subset parameters
    - _fit_and_score() - Added influence curve computation after scoring

Note: cross_validate() and _fit_and_score() are based on sklearn 1.7.2. If sklearn's
internal APIs change in future versions, this code may need to be updated.

New functions:
    - cross_val_auc() - Main API for AUC with confidence intervals. Modification of
    sklearn's cross_val_score() with added parameters for CI and category-specific evaluation.
"""

import numbers
import time
import warnings
from traceback import format_exc

import numpy as np
import pandas as pd
from joblib import logger

# Import everything needed from sklearn (unchanged functions)
from sklearn.base import clone, is_classifier
from sklearn.exceptions import UnsetMetadataPassedError
from sklearn.metrics import check_scoring, roc_auc_score
from sklearn.metrics._scorer import _MultimetricScorer
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _score,
    _warn_or_raise_about_fit_failures,
    _insert_error_scores,
    _aggregate_score_dicts,
    _normalize_score_results,
)
from sklearn.utils import Bunch, indexable
from sklearn.utils._array_api import _convert_to_numpy, device, get_namespace
from sklearn.utils.metadata_routing import (
    MetadataRouter,
    MethodMapping,
    _routing_enabled,
    process_routing,
)
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _check_method_params, _num_samples

from .influence_curves import (
    compute_auc_influence_curve,
    compute_variance,
    compute_confidence_interval,
    _compute_global_weights,
)


# New: main API for usage.
# Equivalent to cross_val_score but returns AUC scores and influence curves for CI computation.
def cross_val_auc(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    params=None,
    pre_dispatch="2*n_jobs",
    error_score=np.nan,
    confidence_level=None,
    use_global_weights=True,  # NEW: whether to apply global weighting to influence curves. Per LeDell et al. (2015), global rather than fold-local weighting is needed to get correct CIs.
    eval_subset=None,
):
    """Run cross-validated ROC AUC with optional influence-curve confidence intervals.

    This function is a specialized variant of cross-validation that always
    evaluates ROC AUC (``scoring='roc_auc'``) and can return confidence intervals
    estimated from influence curves.

    Parameters
    ----------
    estimator : estimator object implementing ``fit``
        The object to use for fitting.

    X : array-like or pandas.DataFrame of shape (n_samples, n_features)
        Feature matrix.

        If ``eval_subset`` is not ``None``, ``X`` must be a pandas DataFrame and
        the subset column is removed from model training/prediction and used only
        for evaluation filtering.

    y : array-like of shape (n_samples,), default=None
        Binary target labels.

    groups : array-like of shape (n_samples,), default=None
        Group labels used by group-aware CV splitters.

    cv : int, CV splitter, or iterable of (train, test) indices, default=None
        Cross-validation splitting strategy.

    n_jobs : int, default=None
        Number of jobs to run in parallel over CV splits.

    verbose : int, default=0
        The verbosity level.

    params : dict, default=None
        Metadata/parameters forwarded to estimator fit, scorer, and splitter,
        following sklearn's metadata-routing conventions.

    pre_dispatch : int or str, default='2*n_jobs'
        Number of jobs pre-dispatched by joblib.

    error_score : 'raise' or numeric, default=np.nan
        Value assigned when fitting/scoring fails for a split.

    confidence_level : float or None, default=None
        Confidence level for interval estimation.

        - If ``None``, only fold AUC scores are returned.
        - If a float in (0, 1), influence curves are computed and confidence
          interval(s) are returned.

    use_global_weights : bool, default=True
        Whether to reweight fold influence curves to use global class proportions
        (LeDell et al., 2015) before variance estimation.

    eval_subset : tuple or None, default=None
        Optional subset evaluation mode ``(column_name, category_value)``.

        - ``None``: evaluate on each full test fold.
        - ``(col, value)``: evaluate only rows where ``X[col] == value``.
        - ``(col, None)``: evaluate each category in ``X[col]`` separately.

    Returns
    -------
    scores : ndarray or dict
        - If ``eval_subset is None``: ndarray of per-fold ROC AUC values.
        - If ``eval_subset is not None``: dict mapping category value to per-fold
          ROC AUC array.

    conf_int : tuple, dict, or None
        - ``None`` if ``confidence_level is None``.
        - If ``eval_subset is None``: ``(lower, upper)``.
        - If ``eval_subset is not None``: dict mapping score key
          (for example ``'score_A'``) to ``(lower, upper)``.
    """
    # Validate eval_subset
    if eval_subset is not None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("eval_subset requires X to be a pandas DataFrame")
        if not isinstance(eval_subset, tuple) or len(eval_subset) != 2:
            raise ValueError(
                "eval_subset must be a tuple of (column_name, category_value or None)"
            )
        col_name, col_value = eval_subset
        if col_name not in X.columns:
            raise ValueError(f"Column '{col_name}' not found in DataFrame")

    # To ensure multimetric format is not supported
    scoring = "roc_auc"
    scorer = check_scoring(estimator, scoring=scoring)

    # Determine if we need influence curves for confidence intervals
    compute_ic = confidence_level is not None

    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        groups=groups,
        scoring={"score": scorer},
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        params=params,
        pre_dispatch=pre_dispatch,
        error_score=error_score,
        return_estimator=False,  # Don't need estimators anymore
        return_indices=False,  # Don't need indices anymore
        return_influence_curves=compute_ic,  # NEW: compute ICs in the loop
        use_global_weights=use_global_weights,  # NEW: whether to apply global weighting to influence curves
        eval_subset=eval_subset,  # NEW: category-specific evaluation
        manual_roc_auc=scoring == "roc_auc",  # Avoid scorer tag edge-cases
    )

    conf_int = None
    if compute_ic and "influence_curves" in cv_results:
        # Influence curves were computed inside cross_validate
        X, y = indexable(X, y)
        n = len(y)
        ic_all = cv_results["influence_curves"]

        if isinstance(ic_all, dict):
            # Multi-category: compute CI per category
            conf_int = {}
            for cat_key, ic_cat in ic_all.items():
                # Get the test scores for this category
                test_key = f"test_{cat_key}"
                if test_key in cv_results:
                    test_scores_cat = cv_results[test_key]
                    n_cat = len(ic_cat)
                    if n_cat == 0:
                        continue
                    variance = compute_variance(ic_cat)
                    estimate = np.mean(test_scores_cat)
                    conf_int[cat_key] = compute_confidence_interval(
                        estimate, variance, n_cat, confidence_level
                    )
        else:
            # Single or no eval_subset
            variance = compute_variance(ic_all)
            estimate = np.mean(cv_results["test_score"])
            conf_int = compute_confidence_interval(
                estimate, variance, n, confidence_level
            )

    # Return scores based on eval_subset mode
    if eval_subset is None:
        return cv_results["test_score"], conf_int
    else:
        # Return dict of scores per category
        # We need to reconstruct the original category values (preserving their types)
        col_name, col_value = eval_subset

        # Determine which categories were evaluated
        if col_value is None:
            # All categories - get unique values from X with original types
            categories = sorted(X[col_name].unique())
        else:
            # Single category with original type
            categories = [col_value]

        scores_dict = {}
        for category in categories:
            # Look for the corresponding key in cv_results
            # The key format is "test_score_{category}"
            key = f"test_score_{category}"
            if key in cv_results:
                # Store with original category type as key
                scores_dict[category] = cv_results[key]

        return scores_dict, conf_int


def cross_validate(
    estimator,
    X,
    y=None,
    *,
    groups=None,
    scoring=None,
    cv=None,
    n_jobs=None,
    verbose=0,
    params=None,
    pre_dispatch="2*n_jobs",
    return_train_score=False,
    return_estimator=False,
    return_indices=False,
    error_score=np.nan,
    return_influence_curves=False,  # NEW: for AUC confidence intervals
    use_global_weights=True,  # NEW: whether to apply global weighting to influence curves. Per LeDell et al. (2015), global rather than fold-local weighting is needed to get correct CIs.
    eval_subset=None,  # NEW: for category-specific evaluation
    manual_roc_auc=False,
):
    """Evaluate metric(s) by cross-validation and also record fit/score times.

    This is sklearn's cross_validate with two additions:
    - return_influence_curves: compute influence curves for CI estimation
    - eval_subset: evaluate on categorical subsets of test data

    See sklearn.model_selection.cross_validate for full documentation.
    """
    _check_groups_routing_disabled(groups)

    X, y = indexable(X, y)
    params = {} if params is None else params
    cv = check_cv(cv, y, classifier=is_classifier(estimator))

    scorers = check_scoring(estimator, scoring=scoring)

    if _routing_enabled():
        router = (
            MetadataRouter(owner="cross_validate")
            .add(
                splitter=cv,
                method_mapping=MethodMapping().add(caller="fit", callee="split"),
            )
            .add(
                estimator=estimator,
                method_mapping=MethodMapping().add(caller="fit", callee="fit"),
            )
            .add(
                scorer=scorers,
                method_mapping=MethodMapping().add(caller="fit", callee="score"),
            )
        )
        try:
            routed_params = process_routing(router, "fit", **params)
        except UnsetMetadataPassedError as e:
            raise UnsetMetadataPassedError(
                message=str(e).replace("cross_validate.fit", "cross_validate"),
                unrequested_params=e.unrequested_params,
                routed_params=e.routed_params,
            )
    else:
        routed_params = Bunch()
        routed_params.splitter = Bunch(split={"groups": groups})
        routed_params.estimator = Bunch(fit=params)
        routed_params.scorer = Bunch(score={})

    indices = cv.split(X, y, **routed_params.splitter.split)
    if return_indices:
        indices = list(indices)

    parallel = Parallel(n_jobs=n_jobs, verbose=verbose, pre_dispatch=pre_dispatch)
    results = parallel(
        delayed(_fit_and_score)(
            clone(estimator),
            X,
            y,
            scorer=scorers,
            train=train,
            test=test,
            verbose=verbose,
            parameters=None,
            fit_params=routed_params.estimator.fit,
            score_params=routed_params.scorer.score,
            return_train_score=return_train_score,
            return_times=True,
            return_estimator=return_estimator,
            error_score=error_score,
            return_influence_curves=return_influence_curves,  # NEW
            use_global_weights=use_global_weights,  # NEW
            eval_subset=eval_subset,  # NEW
            manual_roc_auc=manual_roc_auc,
        )
        for train, test in indices
    )

    _warn_or_raise_about_fit_failures(results, error_score)

    if callable(scoring):
        _insert_error_scores(results, error_score)

    results = _aggregate_score_dicts(results)

    ret = {}
    ret["fit_time"] = results["fit_time"]
    ret["score_time"] = results["score_time"]

    if return_estimator:
        ret["estimator"] = results["estimator"]

    if return_indices:
        ret["indices"] = {}
        ret["indices"]["train"], ret["indices"]["test"] = zip(*indices)

    test_scores_dict = _normalize_score_results(results["test_scores"])
    if return_train_score:
        train_scores_dict = _normalize_score_results(results["train_scores"])

    for name in test_scores_dict:
        ret["test_%s" % name] = test_scores_dict[name]
        if return_train_score:
            key = "train_%s" % name
            ret[key] = train_scores_dict[name]

    # NEW: Aggregate influence curves if requested
    if return_influence_curves:
        if "influence_curve" in results and results["influence_curve"]:
            first_ic = results["influence_curve"][0]

            if isinstance(first_ic, dict):
                # Multi-category: aggregate per-category IC values only.
                ic_parts = {}
                ics = results["influence_curve"]

                for ic_dict in ics:
                    if ic_dict is None:
                        continue
                    for cat, ic_cat in ic_dict.items():
                        if ic_cat is None:
                            continue
                        ic_cat_np = _convert_to_numpy(ic_cat, xp=np)
                        ic_parts.setdefault(cat, []).append(ic_cat_np)

                ret["influence_curves"] = {
                    cat: np.concatenate(parts) if parts else np.array([], dtype=float)
                    for cat, parts in ic_parts.items()
                }
            else:
                # Single category or no eval_subset
                n_samples = len(y) if y is not None else X.shape[0]
                ic_all = np.zeros(n_samples)
                ic_indices = results["influence_curve_indices"]
                ics = results["influence_curve"]

                for ic, indices_fold in zip(ics, ic_indices):
                    if ic is not None and indices_fold is not None:
                        ic_all[indices_fold] = ic

                ret["influence_curves"] = ic_all

    return ret


def _fit_and_score(
    estimator,
    X,
    y,
    *,
    scorer,
    train,
    test,
    verbose,
    parameters,
    fit_params,
    score_params,
    return_train_score=False,
    return_parameters=False,
    return_n_test_samples=False,
    return_times=False,
    return_estimator=False,
    split_progress=None,
    candidate_progress=None,
    error_score=np.nan,
    return_influence_curves=False,  # NEW
    use_global_weights=True,  # NEW
    eval_subset=None,  # NEW
    manual_roc_auc=False,
):
    """Fit estimator and compute scores for a given dataset split.

    This is sklearn's _fit_and_score with two additions:
    - return_influence_curves: compute influence curves for AUC CI
    - eval_subset: evaluate on categorical subsets, excluding cat column from model

    See sklearn.model_selection._validation._fit_and_score for full documentation.
    """
    xp, _ = get_namespace(X)
    X_device = device(X)

    train, test = xp.asarray(train, device=X_device), xp.asarray(test, device=X_device)

    if not isinstance(error_score, numbers.Number) and error_score != "raise":
        raise ValueError(
            "error_score must be the string 'raise' or a numeric value. "
            "(Hint: if using 'raise', please make sure that it has been "
            "spelled correctly.)"
        )

    progress_msg = ""
    if verbose > 2:
        if split_progress is not None:
            progress_msg = f" {split_progress[0] + 1}/{split_progress[1]}"
        if candidate_progress and verbose > 9:
            progress_msg += f"; {candidate_progress[0] + 1}/{candidate_progress[1]}"

    if verbose > 1:
        if parameters is None:
            params_msg = ""
        else:
            sorted_keys = sorted(parameters)
            params_msg = ", ".join(f"{k}={parameters[k]}" for k in sorted_keys)
    if verbose > 9:
        start_msg = f"[CV{progress_msg}] START {params_msg}"
        print(f"{start_msg}{(80 - len(start_msg)) * '.'}")

    fit_params = fit_params if fit_params is not None else {}
    fit_params = _check_method_params(X, params=fit_params, indices=train)
    score_params = score_params if score_params is not None else {}
    score_params_train = _check_method_params(X, params=score_params, indices=train)
    score_params_test = _check_method_params(X, params=score_params, indices=test)

    if parameters is not None:
        estimator = estimator.set_params(**clone(parameters, safe=False))

    start_time = time.time()

    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    # NEW: Exclude categorical column from model training/prediction
    cat_col_name = None
    X_train_for_fit = X_train
    X_test_for_pred = X_test
    if eval_subset is not None:
        cat_col_name, _ = eval_subset
        if hasattr(X_train, "drop"):  # DataFrame
            X_train_for_fit = X_train.drop(columns=[cat_col_name])
            X_test_for_pred = X_test.drop(columns=[cat_col_name])
        else:
            raise ValueError("eval_subset requires X to be a pandas DataFrame")

    result = {}
    try:
        if y_train is None:
            estimator.fit(X_train_for_fit, **fit_params)
        else:
            estimator.fit(X_train_for_fit, y_train, **fit_params)

    except Exception:
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, _MultimetricScorer):
                test_scores = {name: error_score for name in scorer._scorers}
                if return_train_score:
                    train_scores = test_scores.copy()
            else:
                test_scores = error_score
                if return_train_score:
                    train_scores = error_score
        result["fit_error"] = format_exc()
    else:
        result["fit_error"] = None

        fit_time = time.time() - start_time

        # NEW: Determine categories to evaluate
        eval_categories = None
        col_name = None
        if eval_subset is not None:
            col_name, col_value = eval_subset
            if col_value is None:
                eval_categories = sorted(X_test[col_name].unique())
            else:
                eval_categories = [col_value]

        def _manual_roc_auc_score(X_score, y_score_true):
            if y_score_true is None:
                raise ValueError("roc_auc scoring requires y_true")

            if hasattr(estimator, "predict_proba"):
                y_score = estimator.predict_proba(X_score)
                if np.ndim(y_score) == 2:
                    y_score = y_score[:, 1] if y_score.shape[1] == 2 else y_score[:, -1]
            elif hasattr(estimator, "decision_function"):
                y_score = estimator.decision_function(X_score)
            else:
                y_score = estimator.predict(X_score)

            y_true_np = _convert_to_numpy(y_score_true, xp=np)
            y_score_np = _convert_to_numpy(y_score, xp=np)

            try:
                return float(roc_auc_score(y_true_np, y_score_np))
            except Exception:
                if error_score == "raise":
                    raise
                if isinstance(error_score, numbers.Number):
                    return float(error_score)
                raise

        # Score on test set
        if manual_roc_auc:
            if eval_categories is None:
                test_scores = {"score": _manual_roc_auc_score(X_test_for_pred, y_test)}
            else:
                test_scores = {}
                for category in eval_categories:
                    mask = X_test[col_name] == category
                    mask_np = mask.values if hasattr(mask, "values") else mask
                    X_test_cat = X_test_for_pred[mask_np]
                    y_test_cat = (
                        y_test[mask_np]
                        if hasattr(y_test, "__getitem__")
                        else y_test[mask_np]
                    )
                    test_scores[f"score_{category}"] = _manual_roc_auc_score(
                        X_test_cat, y_test_cat
                    )
        else:
            if eval_categories is None:
                test_scores = _score(
                    estimator,
                    X_test_for_pred,
                    y_test,
                    scorer,
                    score_params_test,
                    error_score,
                )
            else:
                # NEW: Score each category separately
                test_scores = {}
                for category in eval_categories:
                    mask = X_test[col_name] == category
                    mask_np = mask.values if hasattr(mask, "values") else mask
                    X_test_cat = X_test_for_pred[mask_np]
                    y_test_cat = (
                        y_test[mask_np]
                        if hasattr(y_test, "__getitem__")
                        else y_test[mask_np]
                    )

                    score_cat = _score(
                        estimator,
                        X_test_cat,
                        y_test_cat,
                        scorer,
                        score_params_test,
                        error_score,
                    )
                    if isinstance(score_cat, dict):
                        for metric_name, metric_value in score_cat.items():
                            test_scores[f"{metric_name}_{category}"] = metric_value
                    else:
                        test_scores[category] = score_cat

        score_time = time.time() - start_time - fit_time
        if return_train_score:
            if manual_roc_auc:
                train_scores = {
                    "score": _manual_roc_auc_score(X_train_for_fit, y_train)
                }
            else:
                train_scores = _score(
                    estimator,
                    X_train_for_fit,
                    y_train,
                    scorer,
                    score_params_train,
                    error_score,
                )

        # NEW: Compute influence curves if requested
        if return_influence_curves:
            influence_curve = None
            influence_curve_indices = None
            # Normalize once so boolean masking is safe for list-like targets.
            y_full_np = _convert_to_numpy(y, xp=np) if y is not None else None

            if hasattr(estimator, "predict_proba"):
                try:
                    if eval_categories is None:
                        y_pred_proba = estimator.predict_proba(X_test_for_pred)

                        if y_pred_proba.shape[1] == 2:
                            y_pred = y_pred_proba[:, 1]
                        else:
                            y_pred = y_pred_proba[:, -1]

                        y_test_np = _convert_to_numpy(y_test, xp=np)
                        y_pred_np = _convert_to_numpy(y_pred, xp=np)
                        test_np = _convert_to_numpy(test, xp=np)

                        influence_curve = compute_auc_influence_curve(
                            y_pred_np,
                            y_test_np,
                        )
                        if use_global_weights:
                            # Global rather than fold weighting per LeDell et al. (2015)
                            weights_global = _compute_global_weights(
                                y_full_np, y_test_np
                            )
                            influence_curve *= weights_global

                        influence_curve_indices = test_np
                    else:
                        # Compute ICs per category
                        influence_curve = {}
                        influence_curve_indices = {}
                        y_test_np = _convert_to_numpy(y_test, xp=np)

                        for category in eval_categories:
                            mask = X_test[col_name] == category
                            mask_np = (
                                mask.values
                                if hasattr(mask, "values")
                                else np.asarray(mask)
                            )

                            X_test_cat = X_test_for_pred[mask_np]
                            y_test_cat_np = y_test_np[mask_np]
                            test_cat = test[mask_np]

                            y_pred_proba_cat = estimator.predict_proba(X_test_cat)

                            if y_pred_proba_cat.shape[1] == 2:
                                y_pred_cat = y_pred_proba_cat[:, 1]
                            else:
                                y_pred_cat = y_pred_proba_cat[:, -1]

                            y_pred_cat_np = _convert_to_numpy(y_pred_cat, xp=np)
                            test_cat_np = _convert_to_numpy(test_cat, xp=np)

                            ic_cat = compute_auc_influence_curve(
                                y_pred_cat_np,
                                y_test_cat_np,
                            )
                            if use_global_weights:
                                # Global rather than fold weighting per LeDell et al. (2015)
                                full_mask = X[col_name] == category
                                full_mask_np = (
                                    full_mask.values
                                    if hasattr(full_mask, "values")
                                    else np.asarray(full_mask)
                                )
                                y_full_cat = y_full_np[full_mask_np]
                                weights_global = _compute_global_weights(
                                    y_full_cat, y_test_cat_np
                                )
                                ic_cat *= weights_global

                            if isinstance(scorer, _MultimetricScorer):
                                metric_name = list(scorer._scorers.keys())[0]
                                key = f"{metric_name}_{category}"
                            else:
                                key = category

                            influence_curve[key] = ic_cat
                            influence_curve_indices[key] = test_cat_np

                except Exception as e:
                    warnings.warn(
                        f"Influence curve computation failed: {e}", RuntimeWarning
                    )

            result["influence_curve"] = influence_curve
            result["influence_curve_indices"] = influence_curve_indices

    if verbose > 1:
        total_time = score_time + fit_time
        end_msg = f"[CV{progress_msg}] END "
        result_msg = params_msg + (";" if params_msg else "")
        if verbose > 2:
            if isinstance(test_scores, dict):
                for scorer_name in sorted(test_scores):
                    result_msg += f" {scorer_name}: ("
                    if return_train_score:
                        scorer_scores = train_scores[scorer_name]
                        result_msg += f"train={scorer_scores:.3f}, "
                    result_msg += f"test={test_scores[scorer_name]:.3f})"
            else:
                result_msg += ", score="
                if return_train_score:
                    result_msg += f"(train={train_scores:.3f}, test={test_scores:.3f})"
                else:
                    result_msg += f"{test_scores:.3f}"
        result_msg += f" total time={logger.short_format_time(total_time)}"

        end_msg += "." * (80 - len(end_msg) - len(result_msg))
        end_msg += result_msg
        print(end_msg)

    result["test_scores"] = test_scores
    if return_train_score:
        result["train_scores"] = train_scores
    if return_n_test_samples:
        result["n_test_samples"] = _num_samples(X_test)
    if return_times:
        result["fit_time"] = fit_time
        result["score_time"] = score_time
    if return_parameters:
        result["parameters"] = parameters
    if return_estimator:
        result["estimator"] = estimator
    return result


def _check_groups_routing_disabled(groups):
    """Stub for sklearn 1.5.1 compatibility (function exists in sklearn 1.6+)."""
    pass
