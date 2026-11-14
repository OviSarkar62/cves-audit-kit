"""Second-round checks after review: one regime definition, recording order, measured regime bias, non-circular shortcut test, reliability with intervals, clock effect on the draft's own contrast, lab-table scale, marker sensitivity, mains lines, n = 91 attribution.
Usage: step5_robustness.py DATA step1.pkl step4.pkl. Aggregates only."""
import sys, re, math, itertools, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
from pathlib import Path
from scipy import stats
from scipy.stats import gaussian_kde
from statsmodels.stats.contingency_tables import Table2x2
from core import *
from clinical import *
warnings.filterwarnings("ignore")
root = Path(sys.argv[1]); S1 = pd.read_pickle(sys.argv[2]); D = pd.read_pickle(sys.argv[3]); rng = np.random.default_rng(2029)
Sb = pd.read_csv(root / "subjects.csv"); Sb.columns = [c.strip() for c in Sb.columns]; Sb["sid"] = Sb.subject_number.astype(str).str.strip().str.lower().str[:5]; Si = Sb.set_index("sid")
def anti(y, bw=.25):
    y = np.asarray(y, float); y = y[np.isfinite(y)]; g = np.linspace(y.min(), y.max(), 400); k = gaussian_kde(y, bw_method=bw)(g); pk = sg.find_peaks(k)[0]; t = sorted(pk[np.argsort(k[pk])[-2:]]); return g[t[0] + np.argmin(k[t[0]:t[1]])]
print("== one regime definition: left span, all records with a usable span")
reg = {}
for p in ("head-up-tilt", "sit-stand-balance"):
    x = S1[S1.proto == p].copy(); v = x.validL.fillna(False).astype(bool); ls = np.log10(x.spanL.clip(lower=1e-3)); th = anti(ls[v]); x["hi"] = (ls < th).astype(int); reg[p] = x.set_index("sid").hi
    T = np.array([[((x.hi == 1) & (x.stroke == 1)).sum(), ((x.hi == 0) & (x.stroke == 1)).sum()], [((x.hi == 1) & (x.stroke == 0)).sum(), ((x.hi == 0) & (x.stroke == 0)).sum()]]); tb = Table2x2(T); print(p, "all", T.tolist(), "OR %.1f (%.1f-%.1f) exact p %.2g" % (tb.oddsratio, *tb.oddsratio_confint(), stats.fisher_exact(T)[1]))
    y = x[v]; T = np.array([[((y.hi == 1) & (y.stroke == 1)).sum(), ((y.hi == 0) & (y.stroke == 1)).sum()], [((y.hi == 1) & (y.stroke == 0)).sum(), ((y.hi == 0) & (y.stroke == 0)).sum()]]); tb = Table2x2(T); print(" valid only", T.tolist(), "OR %.1f (%.1f-%.1f)" % (tb.oddsratio, *tb.oddsratio_confint()), "| invalid channels in high regime", int(x[~v].hi.sum()), "of", int((~v).sum()))
    lg = y.lagxL > anti(y.lagxL); print(" span agrees with gain %.0f%%, with lag %.0f%% (n %d); stored span ratio low/high %.2f" % (100 * ((y.gainL >= 3e4) == (y.hi == 1)).mean(), 100 * (lg == (y.hi == 1)).mean(), len(y), y[y.hi == 0].spanL.median() / y[y.hi == 1].spanL.median()))
    x["num"] = x.sid.str[1:].astype(int); from sklearn.metrics import roc_auc_score as Au
    print(" order->regime %.3f; order->group %.3f; regime->group %.3f; within low regime order->group %.3f (n %d); within high %.3f (n %d)" % (Au(x.hi, x.num), Au(x.stroke, x.num), Au(x.stroke, x.hi), Au(x[x.hi == 0].stroke, x[x.hi == 0].num), (x.hi == 0).sum(), Au(x[x.hi == 1].stroke, x[x.hi == 1].num), (x.hi == 1).sum()))
    x["z"] = (x.num - x.num.mean()) / x.num.std(); m = smf.logit("stroke ~ hi + z", x).fit(disp=0); print(" logit group ~ regime + order: regime OR %.2f (%.2f-%.2f); order OR/SD %.1f (%.1f-%.1f)" % (np.exp(m.params["hi"]), *np.exp(m.conf_int().loc["hi"]), np.exp(m.params["z"]), *np.exp(m.conf_int().loc["z"])))
print("\n== instrumental rule on the paper's regime definition (left span, valid channels): high minus low, within group")
for p in ("head-up-tilt", "sit-stand-balance"):
    x = S1[S1.proto == p].copy(); x = x[x.validL.fillna(False).astype(bool)]; x["hi"] = x.sid.map(reg[p])
    for st, y in x.groupby("stroke"):
        h = y.hi == 1; print(p, "stroke" if st else "control", "n lo/hi", int((~h).sum()), int(h.sum()), "| ECG-ABP %.0f ms | ECG-MCA %.0f ms | ABP-MCA lag %.0f ms" % (y[h].r_abp.median() - y[~h].r_abp.median(), y[h].r_mcaL.median() - y[~h].r_mcaL.median(), y[h].lagxL.median() - y[~h].lagxL.median()))
print("\n== measured regime effect on LF phase vs pure delay; implied group bias")
for p, seg in (("head-up-tilt", "t60-360"), ("sit-stand-balance", "first300")):
    x = D[(D.proto == p) & (D.seg == seg) & (D.sd == "L")].copy(); x["v"] = x.bands.map(lambda z: z[(.07, .20)]); m = smf.ols("v ~ hi + stroke", x).fit(cov_type="HC3", use_t=True); ci = m.conf_int().loc["hi"]; share = x[x.stroke == 1].hi.mean() - x[x.stroke == 0].hi.mean(); tau = x[x.hi == 1].lag.median() - x[x.hi == 0].lag.median(); print(p, "tau %.0f ms; pure delay %.1f; measured %.1f (%.1f, %.1f); share diff %.2f; implied bias pure %.1f, measured %.1f (%.1f, %.1f)" % (tau * 1e3, -360 * .135 * tau, m.params["hi"], ci[0], ci[1], share, share * -360 * .135 * tau, share * m.params["hi"], share * ci[0], share * ci[1]))
print("\n== effect of adding recording order to the phase model, CARNet gate, left")
for p, seg in (("head-up-tilt", "t60-360"), ("sit-stand-balance", "first300")):
    x = D[(D.proto == p) & (D.seg == seg) & (D.sd == "L")].copy(); x["num"] = x.sid.str[1:].astype(int); x["z"] = (x.num - x.num.mean()) / x.num.std(); x["sx"] = x.sex
    for arm in ("un", "pr", "ro"):
        q = x.dropna(subset=[f"{arm}_ph", "age", "et", "sx"]); q = q.assign(y=q[f"{arm}_ph"]); a = smf.ols("y ~ stroke + age + C(sx) + et", q).fit(cov_type="HC3", use_t=True); b = smf.ols("y ~ stroke + age + C(sx) + et + z", q).fit(cov_type="HC3", use_t=True); c = b.conf_int().loc["stroke"]; print(p, arm, "n", len(q), "base %+.1f; with order %+.1f (%.1f, %.1f)" % (a.params["stroke"], b.params["stroke"], c[0], c[1]))
print("\n== precision and robust estimates, corrected clock, CARNet gate, first300/tilt")
G = pd.read_csv(Path(sys.argv[3]).with_suffix(".grid.csv")); g = G[G.seg.isin(["first300", "t60-360"]) & (G.gate == "CARNet")]; print(g[["proto", "sd", "arm", "n", "b", "lo", "hi", "rob", "lev"]].round(1).to_string(index=False)); print("half-width median %.1f; widest %.1f" % (((g.hi - g.lo) / 2).median(), ((g.hi - g.lo) / 2).max()))
print("\n== reliability, left, unaligned: person-level")
w = D[D.sd == "L"]
def rel(a, b, col):
    q = w.pivot_table(index="sid", columns="seg", values=col)[[a, b]].dropna(); A_, B_ = q[a].values, q[b].values; n = len(q); bs = [stats.spearmanr(A_[i], B_[i])[0] for i in (rng.integers(0, n, n) for _ in range(2000))]; sh = [np.median(np.abs(A_ - rng.permutation(B_))) for _ in range(2000)]
    M = (A_ + B_) / 2; msb = 2 * M.var(ddof=1); msw = ((A_ - B_) ** 2).sum() / (2 * n); icc = (msb - msw) / (msb + msw); print(a, "vs", b, col, "n", n, "rho %.2f (%.2f, %.2f); ICC %.2f; median |diff| %.1f vs shuffled %.1f; SD diff %.1f" % (stats.spearmanr(A_, B_)[0], *np.percentile(bs, [2.5, 97.5]), icc, np.median(np.abs(A_ - B_)), np.median(sh), (A_ - B_).std()))
for col in ("un_ph", "un_ph_all"):
    rel("first300", "second300", col); rel("first300", "t60-360", col)
j = S1[S1.proto == "head-up-tilt"].set_index("sid").join(S1[S1.proto == "sit-stand-balance"].set_index("sid"), lsuffix="_t", rsuffix="_b", how="inner"); j = j[j.validL_t.fillna(False).astype(bool) & j.validL_b.fillna(False).astype(bool)]; d = (j.lagxL_t - j.lagxL_b).values; print("lag same person: median |diff| %.0f ms vs shuffled %.0f; rho %.2f, n %d" % (np.median(np.abs(d)), np.median([np.median(np.abs(j.lagxL_t.values - rng.permutation(j.lagxL_b.values))) for _ in range(2000)]), stats.spearmanr(j.lagxL_t, j.lagxL_b)[0], len(j)))
print("\n== non-circular shortcut test: regime defined from timing only")
STORED = {}
Xc, yc = build(Si.loc[[s for s in Si.index if s in set(S1.sid) and "COMPLET" in str(Si.loc[s, "completed_visit_status"]).upper()]])
for proto, fs, in (("head-up-tilt", 500), ("sit-stand-balance", 1000)):
    rows = []
    for h in records(root, proto):
        sid = pid(h); r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; P_ = r.p_signal; q = first_seated(P_[:, n.index("fz")], fs) if proto[0] == "s" else (60 * fs, 360 * fs)
        if q is None: continue
        v = np.nan_to_num(P_[q[0]:q[1], n.index("mcal")]); a = np.nan_to_num(P_[q[0]:q[1], n.index("abp")])
        if not valid(v): continue
        rows.append(dict(sid=sid, m=np.abs(v).mean(), sd=v.std(), p95=np.percentile(v, 95), p5=np.percentile(v, 5), ma=a.mean(), sa=a.std(), cv=v.std() / max(abs(v.mean()), 1e-9), sk=stats.skew(v), ku=stats.kurtosis(v), cva=a.std() / a.mean(), ska=stats.skew(a), lag=lag_xcorr(a, v, fs)))
    F = pd.DataFrame(rows).set_index("sid"); STORED[proto] = F.m; F["y"] = (Si.group.reindex(F.index).astype(str).str.strip().str.lower() == "stroke").astype(int); F["lr"] = (F.lag > anti(F.lag)).astype(int); y = F.y.values; l = F.lr.values
    amp, abp, free = F[["m", "sd", "p95", "p5"]].values, F[["ma", "sa"]].values, F[["cv", "sk", "ku"]].values
    print(proto, "n", len(F), "| lag-regime predicted by MCA amplitude %.3f; ABP amplitude only %.3f; MCA scale-free %.3f" % (rep(amp, l, 30), rep(abp, l, 30), rep(free, l, 30)), "| group AUC: MCA amplitude %.3f; lag-regime alone %.3f; scale-free %.3f" % (rep(amp, y, 30), rep(l[:, None].astype(float), y, 20), rep(free, y, 30)))
    sc = np.mean([oof(amp, y, s) for s in range(30)], 0)
    for g, nm in ((0, "low"), (1, "high")):
        a_, b_ = y[l == g], sc[l == g]; bs = []
        for _ in range(2000):
            i = rng.integers(0, len(a_), len(a_))
            if len(set(a_[i])) == 2: bs.append(AUC(a_[i], b_[i]))
        print("  within %s lag-regime group AUC %.2f (%.2f-%.2f), stroke %d control %d" % (nm, AUC(a_, b_), *np.percentile(bs, [2.5, 97.5]), a_.sum(), (1 - a_).sum()))
print("\n== lab table vs stored MCA x 100 by regime (sit-stand-balance, left)")
cols = [c for c in Sb.columns if re.search(r"SitEO", c) and re.search(r"MCA|BFV|CBF", c, re.I)]; print("columns", cols[:6])
x = S1[S1.proto == "sit-stand-balance"].set_index("sid"); ls = np.log10(x.spanL.clip(lower=1e-3)); v_ = x.validL.fillna(False).astype(bool); h_ = ls < anti(ls[v_]); lab = pd.to_numeric(Si["Mean MCAL SitEO"], errors="coerce").reindex(x.index); q = lab / (STORED["sit-stand-balance"].reindex(x.index) * 100); q = q[v_ & q.notna() & (q > 0)]
print("lab mean left MCA / stored mean x 100: low regime median %.2f IQR %s n %d; high regime median %.2f IQR %s n %d; between-regime ratio %.2f" % (q[~h_].median(), np.round(q[~h_].quantile([.25, .75]).values, 2).tolist(), (~h_[q.index]).sum(), q[h_].median(), np.round(q[h_].quantile([.25, .75]).values, 2).tolist(), h_[q.index].sum(), q[h_].median() / q[~h_].median()))
print("\n== tilt sensitivity: excluding records with a marker inside 60-360 s; mains check")
for h in []: pass
mk_in = set(); lines = {}
for proto, fs in (("head-up-tilt", 500), ("sit-stand", 500), ("sit-stand-balance", 500)):
    o = []
    for h in records(root, proto):
        r = wfdb.rdrecord(str(h.with_suffix("")), channel_names=["marker", "ecg"]); mrk = np.nan_to_num(r.p_signal[:, 0]); e = np.nan_to_num(r.p_signal[:, 1])
        if proto == "head-up-tilt":
            on = np.where(np.diff((np.abs(mrk - np.median(mrk)) > .5 * (np.abs(mrk - np.median(mrk)).max() + 1e-9)).astype(int)) == 1)[0] / 500
            if ((on >= 60) & (on <= 360)).any(): mk_in.add(pid(h))
        f, Pw = sg.welch(e[:2 ** 18] - e[:2 ** 18].mean(), fs=1., nperseg=2 ** 14); b = median_filter(Pw, size=101); o.append(float(np.max(Pw[np.abs(f - .18) < .0008]) / np.median(b[np.abs(f - .18) < .0008])))
    print(proto, "ECG line at 0.18 cycles per sample above 5x background: %d of %d" % ((np.array(o) > 5).sum(), len(o)))
x = D[(D.proto == "head-up-tilt") & (D.seg == "t60-360") & (D.sd == "L") & ~D.sid.isin(mk_in)].copy(); print("tilt records with marker inside window", len(mk_in))
for arm in ("un", "pr", "ro"):
    q = x.dropna(subset=[f"{arm}_ph", "age", "et", "sex"]); q = q.assign(y=q[f"{arm}_ph"]); m = smf.ols("y ~ stroke + age + C(sex) + et", q).fit(cov_type="HC3", use_t=True); c = m.conf_int().loc["stroke"]; print(" excluded", arm, "n", len(q), "%+.1f (%.1f, %.1f)" % (m.params["stroke"], c[0], c[1]))
print("\n== Shapley on AUC, both cohorts, 1000 resamples")
ssb = set(S1[S1.proto == "sit-stand-balance"].sid); comp = Sb[Sb.completed_visit_status.astype(str).str.upper().str.contains("COMPLET")]; pl = {"risk": R, "gait": ["gait"], "meds": M}; names = list(pl)
def shap_of(v): return [sum(math.factorial(len(c)) * math.factorial(2 - len(c)) / 6 * (v[frozenset(c) | {n}] - v[frozenset(c)]) for r in range(3) for c in itertools.combinations([q for q in names if q != n], r)) for n in names]
for nm, Sx in (("n=64", comp[comp.sid.isin(ssb)]), ("n=91", comp)):
    X, y = build(Sx); N = len(y); v = {frozenset(c): (.5 if not c else rep(X[[q for s_ in c for q in pl[s_]]].values, y, 60)) for r_ in range(4) for c in itertools.combinations(names, r_)}; pt = shap_of(v); bs = []
    for _ in range(1000):
        i = rng.integers(0, N, N); o = np.setdiff1d(np.arange(N), i)
        if len(set(y[o])) < 2 or len(set(y[i])) < 2: continue
        u = {frozenset(c): (.5 if not c else AUC(y[o], mk().fit(X[[q for s_ in c for q in pl[s_]]].values[i], y[i]).predict_proba(X[[q for s_ in c for q in pl[s_]]].values[o])[:, 1])) for r_ in range(4) for c in itertools.combinations(names, r_)}; bs.append(shap_of(u))
    b = np.array(bs); print(nm, "stroke", int(y.sum()), {n: (round(pt[k], 3), tuple(np.round(np.percentile(b[:, k], [2.5, 97.5]), 3))) for k, n in enumerate(names)})
