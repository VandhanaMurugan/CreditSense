"""The released bundle pickles XGBoost as a platform-tied memory snapshot (built on Linux; it does not open on Windows).
When the snapshot cannot be read, the service loads the same XGBoost model from the portable export in models/.
These tests force that path and require identical outputs, so the fallback can never serve a different model."""
import os, shutil, numpy as np, pandas as pd, pytest
from creditsense import serve
from creditsense.serve import CreditSense, load_bundle, PORTABLE_XGB, _sha256
RELEASE = serve._CANDIDATES[-1]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pytestmark = pytest.mark.skipif(not os.path.exists(RELEASE), reason="needs the committed release bundle")


@pytest.fixture
def snapshot_unreadable(monkeypatch):
    """Make XGBoost refuse its pickled snapshot, as it does on Windows for a Linux-built bundle."""
    from xgboost.core import Booster, XGBoostError
    def refuse(self, state): raise XGBoostError("input stream corrupted")
    monkeypatch.setattr(Booster, "__setstate__", refuse)


def test_export_is_registered_for_the_release_bundle():
    assert _sha256(RELEASE) in PORTABLE_XGB and os.path.exists(PORTABLE_XGB[_sha256(RELEASE)])


def test_fallback_gives_identical_scores_decisions_and_reasons(snapshot_unreadable, monkeypatch):
    fallback = CreditSense(RELEASE)
    assert fallback.b["xgboost_loaded_from"] == "creditsense_xgb_v2.0.0.ubj"
    monkeypatch.undo()                                             # XGBoost reads snapshots again
    native = CreditSense(RELEASE)
    assert "xgboost_loaded_from" not in native.b
    df = pd.read_csv(os.path.join(ROOT, "data", "demo", "batch_drift.csv"))
    a, b = native.score_batch(df), fallback.score_batch(df)
    assert np.array_equal(a["pd"].values, b["pd"].values) and (a["decision"] == b["decision"]).all()
    applicant = df.iloc[0][native.b["raw_inputs"]].to_dict()
    assert native.score(applicant) == fallback.score(applicant)    # includes reason codes and the exactness check


def test_normal_loading_is_restored_after_the_fallback(snapshot_unreadable):
    from xgboost.core import Booster
    refusing = Booster.__setstate__
    load_bundle(RELEASE)
    assert Booster.__setstate__ is refusing                        # the loader put back what it found


def test_unregistered_bundle_fails_with_an_explanation(snapshot_unreadable, tmp_path):
    other = tmp_path / "other.joblib"
    shutil.copy(RELEASE, other)
    with open(other, "ab") as f: f.write(b"\n")                    # a different file, so a different hash
    with pytest.raises(RuntimeError, match="no portable export is registered"):
        load_bundle(str(other))
