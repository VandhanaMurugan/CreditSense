"""Shared helpers for Phase 2 (novelty) scripts."""
import os, json
import numpy as np
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(ROOT, "artifacts")          # Phase 1 artifacts (models, data) - git-ignored
OUT = os.path.join(ROOT, "phase2", "outputs")  # Phase 2 outputs
os.makedirs(OUT, exist_ok=True)
SEED = 42


def load_phase1():
    """Phase 1 train/test split exactly as produced by pipeline/step1 (same cleaning, IG-top-10 features)."""
    d = joblib.load(os.path.join(ART, "step1_data.joblib"))
    return d["X_train_sel"], d["X_test_sel"], d["y_train"], d["y_test"], d["scale_pos_weight"]


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


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def ece(y, p, bins=10):
    """Expected calibration error with equal-frequency bins."""
    y = np.asarray(y); p = np.asarray(p)
    order = np.argsort(p); idx = np.array_split(order, bins)
    return float(sum(len(i) / len(y) * abs(y[i].mean() - p[i].mean()) for i in idx))


def summarise(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return {"AUC": round(roc_auc_score(y, p), 4), "PR_AUC": round(average_precision_score(y, p), 4),
            "KS": round(ks_stat(y, p), 4), "Brier": round(brier_score_loss(y, p), 4), "ECE": round(ece(y, p), 4),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "F1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "TP": tp, "FP": fp, "FN": fn, "TN": tn}


def save_json(obj, name):
    with open(os.path.join(OUT, name), "w") as f: json.dump(obj, f, indent=2)
