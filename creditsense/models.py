"""Estimators that must be importable wherever the bundle is loaded (pickles reference creditsense.models)."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier

SEED = 42


class WeightedGB(ClassifierMixin, BaseEstimator):
    """GradientBoosting that applies the cost weight (neg:pos ratio) to positives *internally*.

    scikit-learn's StackingClassifier does not forward sample_weight selectively, and forwarding it to every
    member would double-weight RF/XGBoost (which already use class_weight / scale_pos_weight). Wrapping GB
    makes its cost-sensitive fit part of the estimator itself, so it also holds inside the stack and CV.
    """
    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05, subsample=1.0,
                 min_samples_leaf=1, random_state=SEED):
        self.n_estimators = n_estimators; self.max_depth = max_depth; self.learning_rate = learning_rate
        self.subsample = subsample; self.min_samples_leaf = min_samples_leaf; self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        w = np.where(y == 1, (y == 0).sum() / max((y == 1).sum(), 1), 1.0)
        self.model_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth, learning_rate=self.learning_rate,
            subsample=self.subsample, min_samples_leaf=self.min_samples_leaf, random_state=self.random_state)
        self.model_.fit(X, y, sample_weight=w)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X): return self.model_.predict_proba(X)
    def predict(self, X): return self.model_.predict(X)
