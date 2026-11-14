"""RQ4 increment, RQ5 links, gated reliability, and a withdrawn CO2 comparison. Aggregates only."""
import sys, numpy as np, pandas as pd, warnings
from pathlib import Path
from scipy import stats
from core import *
from clinical import *
warnings.filterwarnings("ignore")
root = Path(sys.argv[1]); D = pd.read_pickle(sys.argv[2]).drop(columns=["age", "sex", "sex_"], errors="ignore"); rng = np.random.default_rng(2028)
w = D[D.sd == "L"]
for col in ("un_ph", "un_ph_s", "un_ph_all"):
    q = w.pivot_table(index="sid", columns="seg", values=col)
    for a, b in (("first300", "second300"), ("first300", "t60-360")):
        z = q[[a, b]].dropna(); print("reliability", col, a, "vs", b, "n", len(z), "rho %.2f p %.3f; median abs diff %.1f" % (*stats.spearmanr(z[a], z[b]), (z[a] - z[b]).abs().median()))
S0 = pd.read_csv(root / "subjects.csv")
S0.columns = [c.strip() for c in S0.columns]; S0["sid"] = S0.subject_number.astype(str).str.strip().str.lower().str[:5]; ssb = {h.stem[:5].lower() for h in (root / "data/sit-stand-balance").glob("*.hea")}; S1 = S0[S0.completed_visit_status.astype(str).str.upper().str.contains("COMPLET") & S0.sid.isin(ssb)].set_index("sid"); X, y0 = build(S1); X["y"] = y0
def boot(a, b, y, B=1000):
    o = []; n = len(y)
    for _ in range(B):
        i = rng.integers(0, n, n); t = np.setdiff1d(np.arange(n), i)
        if len(set(y[i])) < 2 or len(set(y[t])) < 2: continue
        o.append(AUC(y[t], mk().fit(b[i], y[i]).predict_proba(b[t])[:, 1]) - AUC(y[t], mk().fit(a[i], y[i]).predict_proba(a[t])[:, 1]))
    return np.percentile(o, [2.5, 97.5]).round(3)
for seg, p in (("first300", "sit-stand-balance"), ("t60-360", "head-up-tilt")):
    x = D[(D.seg == seg) & (D.sd == "L")].set_index("sid").drop(columns=["stroke"]).join(X[R + P + ["y"]], how="inner")
    for arm in ("pr", "ro"):
        q = x.dropna(subset=[f"{arm}_ph", f"{arm}_g"]); yy = q.y.values.astype(int); base = {"A": q[A].values.astype(float), "R": q[R].values.astype(float)}; ph = q[[f"{arm}_g", f"{arm}_ph"]].values
        for k, v in base.items():
            aa, bb = rep(v, yy, 60), rep(np.c_[v, ph], yy, 60); print(p, arm, k, "n", len(q), "S", int(yy.sum()), "base %.3f plus %.3f delta %+.3f CI" % (aa, bb, bb - aa), boot(v, np.c_[v, ph], yy).tolist())
        print("  physiology alone %.3f; regime alone %.3f; A+regime %.3f; A+regime+physio %.3f" % (rep(ph, yy, 60), rep(q[["hi"]].values.astype(float), yy, 30), rep(np.c_[base["A"], q.hi], yy, 60), rep(np.c_[base["A"], q.hi, ph], yy, 60)))

print("gait mean and median by group in cohort", {k: (round(float(X.gait[X.y == k].mean()), 2), round(float(X.gait[X.y == k].median()), 2)) for k in (0, 1)})
print("\n== RQ5 three links, left channel")
for proto, fs, seg in (("head-up-tilt", 500, "t60-360"), ("sit-stand-balance", 1000, "first300")):
    rows = []
    for h in records(root, proto):
        sid = pid(h)
        if sid not in X.index: continue
        r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; P_ = r.p_signal
        q = first_seated(P_[:, n.index("fz")], fs) if proto[0] == "s" else (60 * fs, 360 * fs)
        if q is None: continue
        v = np.nan_to_num(P_[q[0]:q[1], n.index("mcal")]); a = np.nan_to_num(P_[q[0]:q[1], n.index("abp")])
        if not valid(v): continue
        rows.append(dict(sid=sid, m=np.abs(v).mean(), sd=v.std(), p95=np.percentile(v, 95), p5=np.percentile(v, 5), ma=a.mean(), sa=a.std(), cv=v.std() / max(abs(v.mean()), 1e-9), sk=stats.skew(v), ku=stats.kurtosis(v), cva=a.std() / a.mean(), ska=stats.skew(a)))
    F = pd.DataFrame(rows).set_index("sid"); F["y"] = X.y.reindex(F.index); F["hi"] = D[(D.seg == seg) & (D.sd == "L")].set_index("sid").hi.reindex(F.index); F = F.dropna(subset=["y", "hi"]); y = F.y.values.astype(int); h_ = F.hi.values.astype(int)
    naive = F[["m", "sd", "p95", "p5", "ma", "sa"]].values; free = F[["cv", "sk", "ku", "cva", "ska"]].values
    def score(M, t, reps=40): return np.mean([oof(M, t, s) for s in range(reps)], 0)
    sn, sf = score(naive, y), score(free, y)
    print(proto, "n", len(y), "regime hi", int(h_.sum()), "| Link 1 signals->regime: amplitude features %.3f; scale-free %.3f" % (rep(naive, h_, 30), rep(free, h_, 30)), "| group AUC: amplitude %.3f; scale-free %.3f; regime alone %.3f" % (rep(naive, y, 30), rep(free, y, 30), rep(h_[:, None].astype(float), y, 20)))
    for nm, sc in (("amplitude", sn),):
        print("  Link 2 score vs regime within group:", {("stroke" if g else "control"): (round(AUC(h_[y == g], sc[y == g]), 3), int(h_[y == g].sum()), int((1 - h_[y == g]).sum())) for g in (0, 1)})
        print("  Link 3 group AUC within regime:", {("high" if g else "low"): (round(AUC(y[h_ == g], sc[h_ == g]), 3), int(y[h_ == g].sum()), int((1 - y[h_ == g]).sum())) for g in (0, 1)})
print("\n== WITHDRAWN, do not report as a positive control: LF phase at highest vs lowest EtCO2 300 s windows; CO2 is confounded with time and protocol stage")
res = []
for h in records(root, "head-up-tilt"):
    r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; fs = 500; a = np.nan_to_num(r.p_signal[:, n.index("abp")]); co = np.nan_to_num(r.p_signal[:, n.index("co2")]); v = np.nan_to_num(r.p_signal[:, n.index("mcal")])
    if not valid(v): continue
    W = []
    for s in range(0, len(a) - 300 * fs, 60 * fs):
        c = co[s:s + 300 * fs]; pk, _ = sg.find_peaks(c, distance=2 * fs, prominence=5)
        if len(pk) >= 20: W.append((float(np.median(c[pk])), s))
    if len(W) < 3: continue
    W.sort(); lo, hi = W[0], W[-1]
    if hi[0] - lo[0] < 5 or abs(hi[1] - lo[1]) < 300 * fs: continue
    tf = lambda s: tfa(*beats(a[s:s + 300 * fs], v[s:s + 300 * fs], fs)); A_, B_ = tf(lo[1]), tf(hi[1]); res.append((hi[0] - lo[0], A_["ph_all"], B_["ph_all"], A_["ph"], B_["ph"], A_["g"], B_["g"]))
Q = np.array(res, float); dp = Q[:, 2] - Q[:, 1]; dg = Q[:, 4] - Q[:, 3]; gg = Q[:, 6] - Q[:, 5]; print("pairs", len(Q), "EtCO2 gap median %.1f" % np.median(Q[:, 0]), "| phase high-low CO2, all bins: median %+.1f, IQR %s, Wilcoxon p %.2g; share falling %.0f%%" % (np.median(dp), np.round(np.percentile(dp, [25, 75]), 1).tolist(), stats.wilcoxon(dp).pvalue, 100 * (dp < 0).mean()), "| CARNet gate n %d median %+.1f p %.2g" % (np.isfinite(dg).sum(), np.nanmedian(dg), stats.wilcoxon(dg[np.isfinite(dg)]).pvalue))
