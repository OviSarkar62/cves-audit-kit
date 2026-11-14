"""RQ1: clinical sets by role. Refit bootstrap, Shapley on AUC across risk/gait/medications, bootstrapped coefficients, leak checks, n=91. Aggregates only."""
import sys, numpy as np, pandas as pd, itertools, math
from clinical import *
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as AUC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
root = Path(sys.argv[1]); S0 = pd.read_csv(root / "subjects.csv"); S0.columns = [c.strip() for c in S0.columns]; S0["sid"] = S0.subject_number.astype(str).str.strip().str.lower().str[:5]
ssb = {h.stem[:5].lower() for h in (root / "data/sit-stand-balance").glob("*.hea")}
comp = S0[S0.completed_visit_status.astype(str).str.upper().str.contains("COMPLET")]; S = comp[comp.sid.isin(ssb)]; X, y = build(S); X0, _ = build(S, False)
print("n", len(y), "stroke", int(y.sum()), "| missing", {c: int(X[c].isna().sum()) for c in X if X[c].isna().any()}, "| before recoding htn_yrs", int(X0.htn_yrs.isna().sum()), "blank; by group stroke/control", int(X0.htn_yrs[y == 1].isna().sum()), int(X0.htn_yrs[y == 0].isna().sum()))
sets = {"R": R, "P": P, "A": A, "R+gait": R + ["gait"], "R+meds": R + M, "gait": ["gait"], "meds": M}
pt = {k: rep(X[v].values, y) for k, v in sets.items()}; print("point (100 reps):", {k: round(v, 3) for k, v in pt.items()})
print("draft coding:", {k: round(rep(X0[sets[k]].values, y), 3) for k in ("R", "P", "A")})
rng = np.random.default_rng(7); boot = {k: [] for k in sets}; coef = []; N = len(y); B = 2000
for b in range(B):
    i = rng.integers(0, N, N); o = np.setdiff1d(np.arange(N), i)
    if len(set(y[i])) < 2 or len(set(y[o])) < 2: continue
    for k, v in sets.items(): boot[k].append(AUC(y[o], mk().fit(X[v].values[i], y[i]).predict_proba(X[v].values[o])[:, 1]))
    if b < 1000: coef.append(mk().fit(X[A].values[i], y[i]).named_steps["logisticregression"].coef_[0])
ci = lambda a: tuple(np.round(np.percentile(a, [2.5, 97.5]), 3)); bb = {k: np.array(v) for k, v in boot.items()}
print("refit OOB CI:", {k: ci(bb[k]) for k in ("R", "P", "A", "R+gait", "R+meds")}); print("diffs P-R", ci(bb["P"] - bb["R"]), "A-R", ci(bb["A"] - bb["R"]), "A-P", ci(bb["A"] - bb["P"]))
# three-player Shapley on AUC over R, gait, meds; v(empty) = 0.5
pl = {"risk": R, "gait": ["gait"], "meds": M}; names = list(pl)
def val(S_, Xd=X, yd=y, reps=60): return .5 if not S_ else rep(Xd[[c for s in S_ for c in pl[s]]].values, yd, reps)
cache = {}
def sh(Xd=X, yd=y, reps=60):
    v = {frozenset(c): val(c, Xd, yd, reps) for r in range(4) for c in itertools.combinations(names, r)}; out = {}
    for n in names:
        o = [q for q in names if q != n]; out[n] = sum(math.factorial(len(c)) * math.factorial(2 - len(c)) / 6 * (v[frozenset(c) | {n}] - v[frozenset(c)]) for r in range(3) for c in itertools.combinations(o, r))
    return out
s = sh(); print("Shapley on AUC above 0.5:", {k: round(v, 3) for k, v in s.items()}, "sum", round(sum(s.values()), 3))
bs = []
for b in range(200):
    i = rng.integers(0, N, N); o = np.setdiff1d(np.arange(N), i)
    if len(set(y[o])) < 2 or y[i].sum() < 5: continue
    def vb(S_):
        if not S_: return .5
        c = [q for s_ in S_ for q in pl[s_]]; return AUC(y[o], mk().fit(X[c].values[i], y[i]).predict_proba(X[c].values[o])[:, 1])
    v = {frozenset(c): vb(c) for r in range(4) for c in itertools.combinations(names, r)}
    bs.append([sum(math.factorial(len(c)) * math.factorial(2 - len(c)) / 6 * (v[frozenset(c) | {n}] - v[frozenset(c)]) for r in range(3) for c in itertools.combinations([q for q in names if q != n], r)) for n in names])
print("Shapley bootstrap 95%:", {n: tuple(np.round(np.percentile(np.array(bs)[:, k], [2.5, 97.5]), 3)) for k, n in enumerate(names)})
C = np.array(coef); m0 = mk().fit(X[A].values, y); b0 = m0.named_steps["logisticregression"].coef_[0]; order = np.argsort(-np.abs(b0)); print("standardised coefficients, rank-1 share, sign stability:")
top = np.argmax(np.abs(C), 1)
for j in order: print("  %-13s beta %+.2f; 95%% %s; top-ranked %.0f%%; same sign %.0f%%" % (A[j], b0[j], np.round(np.percentile(C[:, j], [2.5, 97.5]), 2).tolist(), 100 * (top == j).mean(), 100 * (np.sign(C[:, j]) == np.sign(b0[j])).mean()))
Z = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(X[["age"]]); from sklearn.linear_model import LogisticRegression as LR
print("age: marginal beta %+.2f; adjusted %+.2f; group means stroke %.1f control %.1f" % (LR(C=1., class_weight="balanced").fit(Z, y).coef_[0, 0], b0[A.index("age")], X.age[y == 1].mean(), X.age[y == 0].mean()))
from sklearn.model_selection import LeaveOneOut
def loso(c):
    Mx = X[c].values; p_ = np.zeros(N)
    for tr, te in LeaveOneOut().split(Mx): p_[te] = mk().fit(Mx[tr], y[tr]).predict_proba(Mx[te])[:, 1]
    return round(AUC(y, p_), 3)
print("leave-one-participant-out:", {k: loso(sets[k]) for k in ("R", "P", "A", "R+gait", "R+meds")}); print("diffs R+gait-R", ci(bb["R+gait"] - bb["R"]), "R+meds-R", ci(bb["R+meds"] - bb["R"]))
tia = S["Stroke Group"].astype(str).str.strip().str.upper().eq("TIA").values; kp = ~tia; print("TIA participants", int(tia.sum()), "; excluded:", {k: round(rep(X[sets[k]].values[kp], y[kp], 50), 3) for k in ("R", "P", "A", "R+meds")})
# leaks and checks
Xv = X[A].values; Xl = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(Xv); leak = float(np.mean([AUC(y, oof(Xl, y, s)) for s in range(50)])); print("fold-wise A %.3f vs preprocessing on all data %.3f" % (rep(Xv, y, 50), leak))
perf = [np.mean([AUC(y[te], mk().fit(Xv[tr], y[tr]).predict_proba(Xv[te])[:, 1]) for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(Xv, y)]) for s in range(50)]; print("pooled vs per-fold A: %.3f vs %.3f" % (rep(Xv, y, 50), np.mean(perf)))
nul = [rep(Xv, rng.permutation(y), 2) for _ in range(200)]; print("shuffled-label A: mean %.3f p95 %.3f" % (np.mean(nul), np.percentile(nul, 95)))
o = np.mean([oof(Xv, y, s) for s in range(20)], 0); lg = np.log(np.clip(o, 1e-4, 1 - 1e-4) / (1 - np.clip(o, 1e-4, 1 - 1e-4))); print("calibration slope (out-of-fold, balanced weights, not for clinical use): %.2f" % LR(C=1e6).fit(lg[:, None], y).coef_[0, 0])
Xn, yn = build(comp); print("all completed visits n", len(yn), "stroke", int(yn.sum()), "gait missing", int(Xn.gait.isna().sum()), {k: round(rep(Xn[sets[k]].values, yn, 50), 3) for k in ("R", "P", "A")})
sr = ["age", "sex", "pack_years", "prev_tob", "diabetes", "htn", "htn_yrs", "hyperlip"]; print("strict-R (drop BMI, current tobacco) %.3f" % rep(X[sr].values, y, 50))
rt = comp.assign(i=comp.sid.isin(ssb)); print("retained gait, stroke/control:", {g: (round(float(Xn.gait[(yn == int(g == "S")) & rt.i.values].median()), 2), round(float(Xn.gait[(yn == int(g == "S")) & ~rt.i.values].median()), 2)) for g in "SC"}, "(in, out)")
