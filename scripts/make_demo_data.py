"""Demo CSVs for the UI (drawn from the untouched GMSC test split; public Kaggle data, no personal identifiers).
  data/demo/sample_applicants.csv            20 applicants across risk tiers, plus one out-of-range case
  data/demo/batch_drift.csv                  2,000 applicants skewed to high utilisation (PSI alarm, calibration still fine)
  data/demo/batch_downturn_with_outcomes.csv 2,000 applicants with outcomes at double the default rate (PSI quiet, PDs too low)
"""
import os, sys, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from creditsense.serve import CreditSense
from creditsense.features import RAW_INPUTS
T = "SeriousDlqin2yrs"; out = os.path.join(ROOT, "data", "demo"); os.makedirs(out, exist_ok=True)
raw = pd.read_csv(os.path.join(ROOT, "data", "GiveMeSomeCredit_dataset.csv"), index_col=0); raw = raw[raw["age"] > 0]
_, te = train_test_split(raw, test_size=0.25, random_state=42, stratify=raw[T]); te = te.reset_index(drop=True)
rng = np.random.default_rng(7); cs = CreditSense()
p = cs.pd(te[RAW_INPUTS]); tier = np.digitize(p, cs.b["tiers"]) + 1
keep = RAW_INPUTS + ["DebtRatio", "MonthlyIncome", "NumberOfDependents"]
idx = np.concatenate([rng.choice(np.where(tier == t)[0], 4, replace=False) for t in range(1, 6)])
s = te.loc[idx, keep].copy(); s.insert(0, "applicant_id", [f"A{i+1:03d}" for i in range(len(s))])
odd = s.iloc[[0]].copy(); odd["applicant_id"] = "A021"; odd["RevolvingUtilizationOfUnsecuredLines"] = 5000.0
pd.concat([s, odd]).to_csv(os.path.join(out, "sample_applicants.csv"), index=False)
u = te["RevolvingUtilizationOfUnsecuredLines"].rank(pct=True).values ** 2
te.loc[rng.choice(len(te), 2000, replace=False, p=u / u.sum()), keep].to_csv(os.path.join(out, "batch_drift.csv"), index=False)
i1, i0 = np.where(te[T] == 1)[0], np.where(te[T] == 0)[0]; n1 = int(2 * te[T].mean() * 2000)
d = te.loc[np.r_[rng.choice(i1, n1, replace=False), rng.choice(i0, 2000 - n1, replace=False)], keep + [T]].sample(frac=1, random_state=7)
d.to_csv(os.path.join(out, "batch_downturn_with_outcomes.csv"), index=False)
print("demo data written")
