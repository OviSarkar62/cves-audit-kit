"""Clinical table builder and model. Plain: turns the participant table into model inputs, with variable roles kept separate."""
import numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
def build(S, htn0=True):
    yn = lambda c: S[c].astype(str).str.strip().str.upper().map(lambda x: 1. if x in ("YES", "Y", "1", "TRUE") else (0. if x in ("NO", "N", "0", "FALSE") else np.nan))
    num = lambda c: pd.to_numeric(S[c].astype(str).str.strip().replace({"": np.nan, "N/A": np.nan, "n/a": np.nan, "NONE": np.nan}), errors="coerce")
    X = pd.DataFrame(index=S.index); X["age"] = num("age"); X["bmi"] = num("bmi"); X["pack_years"] = num("pack_years"); X["htn"] = yn("htn_patient_medical_history"); h = num("HTN YRS PATIENT MEDICAL HISTORY"); X["htn_yrs"] = h.where(h < 1900)
    if htn0: X.loc[(X.htn == 0) & X.htn_yrs.isna(), "htn_yrs"] = 0
    X["cur_tob"] = yn("current_tobacco_use"); X["prev_tob"] = yn("previous_tobacco_use"); X["hyperlip"] = yn("HYPERLIPIDEMIA PATIENT MEDICAL HISTORY"); X["diabetes"] = yn("DM PATIENT MEDICAL HISTORY"); X["sex"] = S["gender"].astype(str).str.strip().str.upper().str[:1].map({"M": 1., "F": 0.})
    X["gait"] = pd.concat([num("GAIT - Walk 1 speed (m/s)"), num("GAIT - Walk 2 speed (m/s)")], axis=1).mean(axis=1)
    for c, n in (("ANTIPLATELETS", "antiplatelet"), ("ANTICOAGULANTS", "anticoag"), ("STATINS", "statin"), ("ANTIHYPERLIPIDEMIC", "antihyperlip")): X[n] = yn(c)
    return X, (S.group.astype(str).str.strip().str.upper() == "STROKE").astype(int).values
R = ["age", "bmi", "pack_years", "htn", "htn_yrs", "cur_tob", "prev_tob", "hyperlip", "diabetes", "sex"]; M = ["antiplatelet", "anticoag", "statin", "antihyperlip"]; P = ["gait"] + M; A = R + P
mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
def oof(Xv, y, seed):
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xv, y): p[te] = mk().fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]
    return p
rep = lambda Xv, y, n=100: float(np.mean([AUC(y, oof(Xv, y, s)) for s in range(n)]))
