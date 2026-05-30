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

from influence_curves import (
    _compute_influence_curve_single_fold,
    compute_variance,
    compute_confidence_interval
)


# New: main API for usage. 
# Equivalent to cross_val_score but returns AUC scores and influence curves for CI computation.
def cross_val_auc(
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
    error_score=np.nan,
    confidence_level=None,
    eval_subset=None,
):
    """Evaluate a score by cross-validation.

    Read more in the :ref:`User Guide <cross_validation>`.

    Parameters
    ----------
    estimator : estimator object implementing 'fit'
        The object to use to fit the data.

    X : {array-like, sparse matrix} of shape (n_samples, n_features)
        The data to fit. Can be for example a list, or an array.

    y : array-like of shape (n_samples,) or (n_samples, n_outputs), \
            default=None
        The target variable to try to predict in the case of
        supervised learning.

    groups : array-like of shape (n_samples,), default=None
        Group labels for the samples used while splitting the dataset into
        train/test set. Only used in conjunction with a "Group" :term:`cv`
        instance (e.g., :class:`GroupKFold`).

        .. versionchanged:: 1.4
            ``groups`` can only be passed if metadata routing is not enabled
            via ``sklearn.set_config(enable_metadata_routing=True)``. When routing
            is enabled, pass ``groups`` alongside other metadata via the ``params``
            argument instead. E.g.:
            ``cross_val_score(..., params={'groups': groups})``.

    scoring : str or callable, default=None
        Strategy to evaluate the performance of the `estimator` across cross-validation
        splits.

        - str: see :ref:`scoring_string_names` for options.
        - callable: a scorer callable object (e.g., function) with signature
          ``scorer(estimator, X, y)``, which should return only a single value.
          See :ref:`scoring_callable` for details.
        - `None`: the `estimator`'s
          :ref:`default evaluation criterion <scoring_api_overview>` is used.

        Similar to the use of `scoring` in :func:`cross_validate` but only a
        single metric is permitted.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - `None`, to use the default 5-fold cross validation,
        - int, to specify the number of folds in a `(Stratified)KFold`,
        - :term:`CV splitter`,
        - An iterable that generates (train, test) splits as arrays of indices.

        For `int`/`None` inputs, if the estimator is a classifier and `y` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with `shuffle=False` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

        .. versionchanged:: 0.22
            `cv` default value if `None` changed from 3-fold to 5-fold.

    n_jobs : int, default=None
        Number of jobs to run in parallel. Training the estimator and computing
        the score are parallelized over the cross-validation splits.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : int, default=0
        The verbosity level.

    params : dict, default=None
        Parameters to pass to the underlying estimator's ``fit``, the scorer,
        and the CV splitter.

        .. versionadded:: 1.4

    pre_dispatch : int or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

        - ``None``, in which case all the jobs are immediately created and spawned. Use
          this for lightweight and fast-running jobs, to avoid delays due to on-demand
          spawning of the jobs
        - An int, giving the exact number of total jobs that are spawned
        - A str, giving an expression as a function of n_jobs, as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised.
        If a numeric value is given, FitFailedWarning is raised.

        .. versionadded:: 0.20
    
    eval_subset : tuple or None, default=None
        Evaluate model performance on categorical subsets of test data while
        training on the full dataset.
        
        - If None (default): evaluate on full test set (standard behavior)
        - If ('column_name', 'category_value'): evaluate only on test samples 
          where X[column_name] == category_value
        - If ('column_name', None): evaluate separately for each unique value 
          in X[column_name]
        
        Note: The categorical column is automatically excluded from model training
        and prediction. It is used only for filtering test data, ensuring the
        model cannot learn from it.
        
        Requirements:
        
        - X must be a pandas DataFrame
        - Category values must be hashable (strings, numbers, tuples)
        
        Return format when eval_subset is used:
        
        - scores: dict mapping category -> array
        - conf_int: dict mapping 'score_category' -> (lower, upper)

    Returns
    -------
    scores : ndarray of float of shape=(len(list(cv)),)
        Array of scores of the estimator for each run of the cross validation.

    See Also
    --------
    cross_validate : To run cross-validation on multiple metrics and also to
        return train scores, fit times and score times.

    cross_val_predict : Get predictions from each split of cross-validation for
        diagnostic purposes.

    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.

    Examples
    --------
    >>> from sklearn import datasets, linear_model
    >>> from sklearn.model_selection import cross_val_score
    >>> diabetes = datasets.load_diabetes()
    >>> X = diabetes.data[:150]
    >>> y = diabetes.target[:150]
    >>> lasso = linear_model.Lasso()
    >>> print(cross_val_score(lasso, X, y, cv=3))
    [0.3315057  0.08022103 0.03531816]
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
    scorer = check_scoring(estimator, scoring=scoring)

    # Determine if we need influence curves for confidence intervals
    compute_ic = confidence_level is not None and scoring == 'roc_auc'

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
        return_indices=False,     # Don't need indices anymore
        return_influence_curves=compute_ic,  # NEW: compute ICs in the loop
        eval_subset=eval_subset,  # NEW: category-specific evaluation
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
                    V = len(test_scores_cat)
                    variance = compute_variance(ic_cat, V)
                    estimate = np.mean(test_scores_cat)
                    conf_int[cat_key] = compute_confidence_interval(
                        estimate, variance, n, confidence_level
                    )
        else:
            # Single or no eval_subset
            V = len(cv_results["test_score"])  # number of folds
            variance = compute_variance(ic_all, V)
            estimate = np.mean(cv_results["test_score"])
            conf_int = compute_confidence_interval(estimate, variance, n, confidence_level)

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
    eval_subset=None,  # NEW: for category-specific evaluation
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
            eval_subset=eval_subset,  # NEW
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
                # Multi-category: aggregate per category
                n_samples = len(y) if y is not None else X.shape[0]
                categories = first_ic.keys()
                ic_all = {cat: np.zeros(n_samples) for cat in categories}
                
                ic_indices = results["influence_curve_indices"]
                ics = results["influence_curve"]
                
                for ic_dict, indices_dict in zip(ics, ic_indices):
                    if ic_dict is not None and indices_dict is not None:
                        for cat in categories:
                            if ic_dict[cat] is not None and indices_dict[cat] is not None:
                                ic_all[cat][indices_dict[cat]] = ic_dict[cat]
                
                ret["influence_curves"] = ic_all
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
    eval_subset=None,  # NEW
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
        if hasattr(X_train, 'drop'):  # DataFrame
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
        
        # Score on test set
        if eval_categories is None:
            test_scores = _score(
                estimator, X_test_for_pred, y_test, scorer, score_params_test, error_score
            )
        else:
            # NEW: Score each category separately
            test_scores = {}
            for category in eval_categories:
                mask = X_test[col_name] == category
                mask_np = mask.values if hasattr(mask, 'values') else mask
                X_test_cat = X_test_for_pred[mask_np]
                y_test_cat = y_test[mask_np] if hasattr(y_test, '__getitem__') else y_test[mask_np]
                
                score_cat = _score(
                    estimator, X_test_cat, y_test_cat, scorer, score_params_test, error_score
                )
                if isinstance(score_cat, dict):
                    for metric_name, metric_value in score_cat.items():
                        test_scores[f"{metric_name}_{category}"] = metric_value
                else:
                    test_scores[category] = score_cat
        
        score_time = time.time() - start_time - fit_time
        if return_train_score:
            train_scores = _score(
                estimator, X_train_for_fit, y_train, scorer, score_params_train, error_score
            )
        
        # NEW: Compute influence curves if requested
        if return_influence_curves:
            influence_curve = None
            influence_curve_indices = None
            
            # Compute global class proportions from the FULL dataset (Pn)
            # Per LeDell et al. (2015), these should NOT be fold-local
            y_full = y if not hasattr(y, 'values') else y.values
            n_total = len(y_full)
            emp_prob_1_global = np.sum(y_full == 1) / n_total
            emp_prob_0_global = np.sum(y_full == 0) / n_total
            
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
                        
                        influence_curve = _compute_influence_curve_single_fold(
                            y_pred_np, y_test_np, emp_prob_1_global, emp_prob_0_global
                        )
                        influence_curve_indices = test_np
                    else:
                        # Compute ICs per category
                        influence_curve = {}
                        influence_curve_indices = {}
                        
                        for category in eval_categories:
                            mask = X_test[col_name] == category
                            mask_np = mask.values if hasattr(mask, 'values') else np.asarray(mask)
                            
                            X_test_cat = X_test_for_pred[mask_np]
                            y_test_cat = y_test[mask_np]
                            test_cat = test[mask_np]
                            
                            y_pred_proba_cat = estimator.predict_proba(X_test_cat)
                            
                            if y_pred_proba_cat.shape[1] == 2:
                                y_pred_cat = y_pred_proba_cat[:, 1]
                            else:
                                y_pred_cat = y_pred_proba_cat[:, -1]
                            
                            y_test_cat_np = _convert_to_numpy(y_test_cat, xp=np)
                            y_pred_cat_np = _convert_to_numpy(y_pred_cat, xp=np)
                            test_cat_np = _convert_to_numpy(test_cat, xp=np)
                            
                            ic_cat = _compute_influence_curve_single_fold(
                                y_pred_cat_np, y_test_cat_np, emp_prob_1_global, emp_prob_0_global
                            )
                            
                            if isinstance(scorer, _MultimetricScorer):
                                metric_name = list(scorer._scorers.keys())[0]
                                key = f"{metric_name}_{category}"
                            else:
                                key = category
                            
                            influence_curve[key] = ic_cat
                            influence_curve_indices[key] = test_cat_np
                            
                except Exception as e:
                    warnings.warn(f"Influence curve computation failed: {e}", RuntimeWarning)
            
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