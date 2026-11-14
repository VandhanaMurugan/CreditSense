"""Shared helpers for Phase 2 (novelty) scripts."""
import os, json
import numpy as np
import joblib, sys
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path: sys.path.insert(0, ROOT)
ART = os.path.join(ROOT, "artifacts")          # Phase 1 artifacts (models, data) - git-ignored
OUT = os.path.join(ROOT, "phase2", "outputs")  # Phase 2 outputs
os.makedirs(OUT, exist_ok=True)
SEED = 42


def load_phase1():
    """Phase 1 train/test split exactly as produced by pipeline/step1 (same cleaning, IG-top-10 features)."""
    d = joblib.load(os.path.join(ART, "step1_data.joblib"))
    return d["X_train_sel"], d["X_test_sel"], d["y_train"], d["y_test"], d["scale_pos_weight"]


from creditsense.models import WeightedGB  # noqa: E402  (kept importable for pickled bundles)


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
