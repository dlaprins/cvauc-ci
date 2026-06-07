"""
Tests for cvauc: Cross-validated AUC with influence curve-based confidence intervals.

Test organization:
- TestInfluenceCurveMath: Unit tests for IC computation with hand-calculated example
- TestCrossValAUCPipeline: Integration tests for full cross_val_auc function
- TestEvalSubsetBasic: Basic eval_subset functionality
- TestCategoricalColumnExclusion: Verify categorical column is never used by model
- TestIntegerCategories: Integer (ordinal) categorical values

The hand-calculated example (used in TestInfluenceCurveMath):
- 6 samples: y_true = [0, 1, 0, 1, 0, 1]
- X = [[0], [1], [2], [3], [4], [5]] (indices as features)
- 2-fold CV: fold 0 tests [0,1,2], fold 1 tests [3,4,5]
- Mock estimator returns: y_pred = [0.6, 0.5, 0.4, 0.6, 0.3, 0.8]
- Global class proportions: Pn(Y=1) = Pn(Y=0) = 0.5

Fold 0: AUC=0.5, ICs=[-1.0, 0.0, 1.0] (using global proportions)
Fold 1: AUC=1.0, ICs=[0.0, 0.0, 0.0]
Combined: variance=1/3 (R-style, without /V), mean AUC=0.75
"""

import numpy as np
import pandas as pd
import pytest
from typing import cast
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from cvauc_ci import (
    cross_val_auc,
    compute_auc_influence_curve,
    compute_variance,
    compute_confidence_interval,
)


# =============================================================================
# Fixtures and Helpers
# =============================================================================


class MockClassifier(BaseEstimator, ClassifierMixin):
    """Mock classifier returning predetermined probabilities based on X values.

    X values are treated as indices into pred_proba_map.
    """

    def __init__(self, pred_proba_map):
        self.pred_proba_map = pred_proba_map
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        probs = []
        for x in X:
            idx = int(x[0])
            p1 = self.pred_proba_map[idx]
            probs.append([1 - p1, p1])
        return np.array(probs)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


@pytest.fixture
def hand_calculated_example():
    """The 6-sample hand-calculated example from the module docstring."""
    X = np.array([[0], [1], [2], [3], [4], [5]])
    y = np.array([0, 1, 0, 1, 0, 1])
    pred_proba_map = {0: 0.6, 1: 0.5, 2: 0.4, 3: 0.6, 4: 0.3, 5: 0.8}
    return X, y, pred_proba_map


@pytest.fixture
def simple_dataframe():
    """Simple DataFrame for eval_subset tests."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame(
        {
            "x1": np.random.randn(n),
            "x2": np.random.randn(n),
            "category": np.random.choice(["A", "B", "C"], n),
        }
    )


# =============================================================================
# TestInfluenceCurveMath: Unit tests for IC computation
# =============================================================================


class TestInfluenceCurveMath:
    """Unit tests for influence curve computation functions."""

    def test_compute_influence_curve_fold0(self):
        """Test IC computation for fold 0 (AUC=0.5) with fold-local proportions."""
        y_pred = np.array([0.6, 0.5, 0.4])
        y_true = np.array([0, 1, 0])
        ic = compute_auc_influence_curve(y_pred, y_true)

        # With fold-local proportions: P(Y=1)=1/3 and P(Y=0)=2/3.
        # Sample 0 (Y=0): p0 = 0, IC = (0-0.5)/(2/3) = -0.75
        # Sample 1 (Y=1): p1 = 0.5, IC = (0.5-0.5)/(1/3) = 0.0
        # Sample 2 (Y=0): p0 = 1, IC = (1-0.5)/(2/3) = 0.75
        expected_ic = np.array([-0.75, 0.0, 0.75])
        np.testing.assert_array_almost_equal(ic, expected_ic, decimal=10)

    def test_compute_influence_curve_fold1(self):
        """Test IC computation for fold 1 (perfect AUC=1.0) with global proportions."""
        y_pred = np.array([0.6, 0.3, 0.8])
        y_true = np.array([1, 0, 1])
        ic = compute_auc_influence_curve(y_pred, y_true)

        # With AUC=1.0, all ICs are 0 since p - AUC = 1 - 1 = 0
        expected_ic = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(ic, expected_ic, decimal=10)

    def test_compute_influence_curve_single_class(self):
        """IC returns zeros when only one class is present."""
        y_pred = np.array([0.6, 0.5, 0.4])
        y_true = np.array([1, 1, 1])  # All positive

        ic = compute_auc_influence_curve(y_pred, y_true)

        np.testing.assert_array_equal(ic, np.zeros(3))

    def test_compute_variance(self):
        """Test variance computation from ICs (R-style formula)."""
        # ICs with global proportions: [-1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ic_all = np.array([-1.0, 0.0, 1.0, 0.0, 0.0, 0.0])

        variance = compute_variance(ic_all)

        # R-style variance = sum(IC^2) / n = 2/6 = 1/3
        # (Note: paper formula would be 2/12 = 1/6, but R doesn't divide by V)
        expected_variance = 2.0 / 6.0  # = 1/3 ≈ 0.3333
        assert abs(variance - expected_variance) < 1e-10

    def test_compute_confidence_interval(self):
        """Test CI computation from variance."""
        estimate = 0.75
        variance = 1.0 / 3.0  # R-style variance (without /V)
        n = 6

        ci = compute_confidence_interval(estimate, variance, n, confidence_level=0.95)

        expected_bound = 1.96 * np.sqrt(variance) / np.sqrt(n)
        # CI is truncated to [0, 1] to match R's cvAUC package
        expected_lower = max(0.0, 0.75 - expected_bound)
        expected_upper = min(1.0, 0.75 + expected_bound)

        assert abs(ci[0] - expected_lower) < 1e-5
        assert abs(ci[1] - expected_upper) < 1e-5

    def test_compute_confidence_interval_different_levels(self):
        """Test CI at different confidence levels."""
        estimate = 0.8
        variance = 0.1
        n = 100

        ci_90 = compute_confidence_interval(
            estimate, variance, n, confidence_level=0.90
        )
        ci_95 = compute_confidence_interval(
            estimate, variance, n, confidence_level=0.95
        )
        ci_99 = compute_confidence_interval(
            estimate, variance, n, confidence_level=0.99
        )

        # Higher confidence -> wider interval
        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]

        assert width_90 < width_95 < width_99


# =============================================================================
# TestCrossValAUCPipeline: Integration tests
# =============================================================================


class TestCrossValAUCPipeline:
    """Integration tests for the full cross_val_auc function."""

    def test_full_pipeline_hand_calculated(self, hand_calculated_example):
        """Test cross_val_auc with hand-calculated expected outputs."""
        X, y, pred_proba_map = hand_calculated_example
        clf = MockClassifier(pred_proba_map)
        cv = KFold(n_splits=2, shuffle=False)

        scores, conf_int = cross_val_auc(clf, X, y, cv=cv, confidence_level=0.95)

        # Verify fold AUCs
        np.testing.assert_array_almost_equal(np.asarray(scores), [0.5, 1.0], decimal=10)

        # Verify confidence interval with R-style variance (no division by V)
        # variance = 2/6 = 1/3 (R-style, matches cvAUC package)
        # CI is truncated to [0, 1]
        expected_variance = 1.0 / 3.0
        expected_bound = 1.96 * np.sqrt(expected_variance) / np.sqrt(6)
        expected_lower = max(0.0, 0.75 - expected_bound)
        expected_upper = min(1.0, 0.75 + expected_bound)
        assert conf_int is not None
        assert abs(conf_int[0] - expected_lower) < 1e-4
        assert abs(conf_int[1] - expected_upper) < 1e-4

    def test_no_confidence_interval(self, hand_calculated_example):
        """Test cross_val_auc without CI computation."""
        X, y, pred_proba_map = hand_calculated_example
        clf = MockClassifier(pred_proba_map)
        cv = KFold(n_splits=2, shuffle=False)

        scores, conf_int = cross_val_auc(clf, X, y, cv=cv, confidence_level=None)

        np.testing.assert_array_almost_equal(np.asarray(scores), [0.5, 1.0], decimal=10)
        assert conf_int is None

    def test_with_real_estimator(self):
        """Test with a real sklearn estimator."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        scores, conf_int = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=5),
            confidence_level=0.95,
        )

        assert len(scores) == 5
        assert all(0 <= s <= 1 for s in scores)
        assert conf_int is not None
        assert conf_int[0] < conf_int[1]
        assert 0 <= conf_int[0] <= 1
        assert 0 <= conf_int[1] <= 1

    def test_global_vs_local_fold_weights_change_ci_not_scores(self):
        """Global weighting should affect CI (IC variance) but not fold AUC scores."""
        X = np.arange(12).reshape(-1, 1)
        y = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0])
        pred_proba_map = {
            0: 0.90,
            1: 0.80,
            2: 0.20,
            3: 0.70,
            4: 0.10,
            5: 0.20,
            6: 0.80,
            7: 0.60,
            8: 0.70,
            9: 0.40,
            10: 0.30,
            11: 0.20,
        }
        clf = MockClassifier(pred_proba_map)
        cv = KFold(n_splits=3, shuffle=False)

        scores_global, ci_global = cross_val_auc(
            clf,
            X,
            y,
            cv=cv,
            confidence_level=0.95,
            use_global_weights=True,
        )
        scores_local, ci_local = cross_val_auc(
            clf,
            X,
            y,
            cv=cv,
            confidence_level=0.95,
            use_global_weights=False,
        )

        np.testing.assert_allclose(np.asarray(scores_global), np.asarray(scores_local))
        assert ci_global is not None
        assert ci_local is not None
        assert not (
            np.isclose(ci_global[0], ci_local[0])
            and np.isclose(ci_global[1], ci_local[1])
        )


# =============================================================================
# TestEvalSubsetBasic: Basic eval_subset functionality
# =============================================================================


class TestEvalSubsetBasic:
    """Test basic eval_subset functionality."""

    def test_single_category(self, simple_dataframe):
        """Test evaluation on a single category."""
        X = simple_dataframe
        y = (X["x1"] + X["x2"] > 0).astype(int)

        scores, ci = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=3),
            eval_subset=("category", "A"),
            confidence_level=0.95,
        )

        assert "A" in scores
        assert len(scores["A"]) == 3
        assert ci is not None

    def test_all_categories(self, simple_dataframe):
        """Test evaluation on all categories separately."""
        X = simple_dataframe
        y = (X["x1"] > 0).astype(int)

        scores, ci = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=3),
            eval_subset=("category", None),
            confidence_level=0.95,
        )

        for cat in ["A", "B", "C"]:
            assert cat in scores
            assert len(scores[cat]) == 3
            assert ci is not None
            assert f"score_{cat}" in ci

    def test_without_confidence_interval(self, simple_dataframe):
        """Test eval_subset without CI computation."""
        X = simple_dataframe
        y = (X["x1"] > 0).astype(int)

        scores, ci = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=2),
            eval_subset=("category", None),
            confidence_level=None,
        )

        assert isinstance(scores, dict)
        assert ci is None

    def test_normal_mode_still_works(self):
        """Test that normal mode (without eval_subset) still works."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
            }
        )
        y = (X["x1"] + X["x2"] > 0).astype(int)

        scores, ci = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=3),
            eval_subset=None,
            confidence_level=0.95,
        )

        assert isinstance(scores, np.ndarray)
        assert len(scores) == 3
        assert isinstance(ci, tuple)

    def test_invalid_column_raises(self):
        """Test that invalid column name raises ValueError."""
        np.random.seed(42)
        X = pd.DataFrame({"x": np.random.randn(50)})
        y = (X["x"] > 0).astype(int)

        with pytest.raises(ValueError, match="not found"):
            cross_val_auc(LogisticRegression(), X, y, eval_subset=("nonexistent", "A"))

    def test_requires_dataframe(self):
        """Test that eval_subset requires DataFrame."""
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)

        with pytest.raises(ValueError, match="DataFrame"):
            cross_val_auc(LogisticRegression(), X, y, eval_subset=("col", "A"))


class TestSubgroupCIMethodology:
    """Regression tests for subgroup CI denominator and IC aggregation behavior."""

    def _ci_width(self, ci):
        return float(ci[1] - ci[0])

    def test_small_subgroup_ci_wider_than_full_population(self):
        """Small subgroup should generally have wider CI than full-population estimate."""
        rng = np.random.RandomState(7)
        n = 500
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        logits = x1 + 0.5 * x2 + rng.normal(scale=0.6, size=n)
        y = (logits > np.median(logits)).astype(int)

        group = np.array(["A"] * 80 + ["B"] * (n - 80), dtype=object)
        perm = rng.permutation(n)

        X = pd.DataFrame(
            {
                "x1": x1[perm],
                "x2": x2[perm],
                "group": group[perm],
            }
        )
        y = y[perm]

        cv = KFold(n_splits=5, shuffle=True, random_state=19)
        est = LogisticRegression(random_state=0, max_iter=1000)

        _, ci_full = cross_val_auc(
            est, X[["x1", "x2"]], y, cv=cv, confidence_level=0.95
        )
        _, ci_sub = cross_val_auc(
            est,
            X,
            y,
            cv=cv,
            confidence_level=0.95,
            eval_subset=("group", "A"),
        )

        assert ci_full is not None
        assert ci_sub is not None
        ci_sub_dict = cast(dict, ci_sub)
        assert "score_A" in ci_sub_dict
        assert self._ci_width(ci_sub_dict["score_A"]) > self._ci_width(ci_full)

    def test_subgroup_ci_width_reacts_to_subgroup_sample_size(self):
        """Holding signal generation fixed, larger subgroup should have narrower CI."""
        rng = np.random.RandomState(11)
        n = 700
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        logits = x1 + 0.4 * x2 + rng.normal(scale=0.7, size=n)
        y = (logits > np.median(logits)).astype(int)

        base = pd.DataFrame({"x1": x1, "x2": x2})
        cv = KFold(n_splits=5, shuffle=True, random_state=23)
        est = LogisticRegression(random_state=0, max_iter=1000)

        group_small = np.where(rng.rand(n) < 0.18, "A", "B")
        X_small = base.copy()
        X_small["group"] = group_small

        group_large = np.where(rng.rand(n) < 0.45, "A", "B")
        X_large = base.copy()
        X_large["group"] = group_large

        _, ci_small = cross_val_auc(
            est,
            X_small,
            y,
            cv=cv,
            confidence_level=0.95,
            eval_subset=("group", "A"),
        )
        _, ci_large = cross_val_auc(
            est,
            X_large,
            y,
            cv=cv,
            confidence_level=0.95,
            eval_subset=("group", "A"),
        )

        assert ci_small is not None
        assert ci_large is not None
        ci_small_dict = cast(dict, ci_small)
        ci_large_dict = cast(dict, ci_large)
        assert "score_A" in ci_small_dict
        assert "score_A" in ci_large_dict
        assert self._ci_width(ci_small_dict["score_A"]) > self._ci_width(
            ci_large_dict["score_A"]
        )


# =============================================================================
# TestCategoricalColumnExclusion: Verify column is never used by model
# =============================================================================


class TestCategoricalColumnExclusion:
    """Test that categorical column is never passed to the model.

    These tests use non-numeric categories that would cause sklearn
    to fail with "could not convert string to float" if passed to fit/predict.
    """

    def test_string_categories_excluded(self):
        """String values in categorical column would fail if passed to model."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "group": ["GROUP_" + str(i % 4) for i in range(n)],
            }
        )
        y = (X["x1"] + X["x2"] > 0).astype(int)

        # This succeeds only if categorical column is excluded
        scores, _ = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=3),
            eval_subset=("group", "GROUP_0"),
            confidence_level=0.95,
        )

        assert "GROUP_0" in scores

    def test_tuple_categories_excluded(self):
        """Tuple values would fail if passed to sklearn."""
        np.random.seed(42)
        n = 150
        categories = [("Group", 1), ("Group", 2), ("Group", 3)]

        X = pd.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "metadata": [categories[i % 3] for i in range(n)],
            }
        )
        y = (X["x1"] + X["x2"] > 0).astype(int)

        scores, _ = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=2),
            eval_subset=("metadata", categories[0]),
            confidence_level=0.95,
        )

        assert categories[0] in scores

    def test_model_uses_correct_features(self):
        """Verify model predictions depend only on non-categorical features."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {
                "useful_feature": np.random.randn(n),
                "noise": np.random.randn(n) * 0.1,
                "category": np.random.choice(["A", "B"], n),
            }
        )
        # Target depends ONLY on useful_feature
        y = (X["useful_feature"] > 0).astype(int)

        scores, _ = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=3),
            eval_subset=("category", None),
            confidence_level=0.95,
        )

        # Both categories should have high AUC (model uses useful_feature)
        for cat in ["A", "B"]:
            mean_auc = np.mean(scores[cat])
            assert mean_auc > 0.8, f"AUC too low for {cat}: {mean_auc:.3f}"


# =============================================================================
# TestIntegerCategories: Integer (ordinal) categorical values
# =============================================================================


class TestIntegerCategories:
    """Test integer (ordinal) categorical values."""

    def test_integer_single_category(self):
        """Test evaluation on single integer category."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "age_group": np.random.choice([1, 2, 3, 4, 5], n),
            }
        )
        y = (X["x1"] + X["x2"] > 0).astype(int)

        scores, ci = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=3),
            eval_subset=("age_group", 3),
            confidence_level=0.95,
        )

        # Key should be integer 3, not string '3'
        assert 3 in scores
        assert len(scores[3]) == 3

    def test_integer_all_categories(self):
        """Test evaluation on all integer categories."""
        np.random.seed(42)
        n = 400
        X = pd.DataFrame(
            {
                "x1": np.random.randn(n),
                "x2": np.random.randn(n),
                "bracket": np.random.choice([10, 20, 30, 40], n),
            }
        )
        y = (X["x1"] + X["x2"] > 0).astype(int)

        scores, ci = cross_val_auc(
            LogisticRegression(random_state=42, max_iter=1000),
            X,
            y,
            cv=KFold(n_splits=3),
            eval_subset=("bracket", None),
            confidence_level=0.95,
        )

        for bracket in [10, 20, 30, 40]:
            assert bracket in scores
            assert ci is not None
            assert f"score_{bracket}" in ci

    def test_zero_based_categories(self):
        """Test 0-based integer categories (0, 1, 2)."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {"x": np.random.randn(n), "category": np.random.choice([0, 1, 2], n)}
        )
        y = (X["x"] > 0).astype(int)

        scores, _ = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=2),
            eval_subset=("category", 0),
        )

        assert 0 in scores

    def test_integer_not_used_as_feature(self):
        """Verify integer categories are not used as model features."""
        np.random.seed(42)
        n = 300

        X = pd.DataFrame(
            {
                "useful_feature": np.random.randn(n),
                "noise": np.random.randn(n) * 0.01,
                "category": np.random.choice([0, 1], n),
            }
        )
        y = (X["useful_feature"] > 0).astype(int)

        scores, _ = cross_val_auc(
            LogisticRegression(random_state=42),
            X,
            y,
            cv=KFold(n_splits=3),
            eval_subset=("category", None),
            confidence_level=0.95,
        )

        # Both categories should have similar, high AUC
        auc_0 = np.mean(scores[0])
        auc_1 = np.mean(scores[1])

        assert auc_0 > 0.7
        assert auc_1 > 0.7
        assert abs(auc_0 - auc_1) < 0.3  # Similar performance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
