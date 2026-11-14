"""RQ3 grid, regime offset by frequency, bias analysis, reliability, RQ4 increment, RQ5 shortcut links. Aggregates only."""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf, statsmodels.api as sm, warnings
from pathlib import Path
from scipy import stats
from clinical import *
warnings.filterwarnings("ignore")
root = Path(sys.argv[1]); D = pd.read_pickle(sys.argv[2]); D["sex_"] = D.sex; rng = np.random.default_rng(2027)
def fit(x, y, extra=""):
    x = x.dropna(subset=[y, "age", "et", "sex_"]).copy(); x["deg"] = x[y]; nS, nC = int(x.stroke.sum()), int((1 - x.stroke).sum())
    if min(nS, nC) < 5: return dict(n=len(x), S=nS, C=nC, b=np.nan, lo=np.nan, hi=np.nan, p=np.nan, perm=np.nan, rob=np.nan, lev=np.nan)
    m = smf.ols(f"deg ~ stroke + age + C(sex_) + et{extra}", x).fit(cov_type="HC3", use_t=True); c = m.conf_int().loc["stroke"]; rob = smf.rlm(f"deg ~ stroke + age + C(sex_) + et{extra}", x, M=sm.robust.norms.HuberT()).fit().params["stroke"]
    Xm = m.model.exog.copy(); j = m.model.exog_names.index("stroke"); yv = x.deg.values; obs = np.linalg.lstsq(Xm, yv, rcond=None)[0][j]; cnt = 0; gr = x.hi.values
    for _ in range(2000):
        Xp = Xm.copy(); sv = Xm[:, j].copy()
        for g in np.unique(gr): i = np.where(gr == g)[0]; sv[i] = rng.permutation(sv[i])
        Xp[:, j] = sv; cnt += abs(np.linalg.lstsq(Xp, yv, rcond=None)[0][j]) >= abs(obs)
    return dict(n=len(x), S=nS, C=nC, b=m.params["stroke"], lo=c[0], hi=c[1], p=m.pvalues["stroke"], perm=(cnt + 1) / 2001, rob=rob, lev=float(m.get_influence().hat_matrix_diag.max()))
print("== RQ3 grid: left and right, stroke minus control, degrees"); rows = []
for (p, seg, sd), x in D.groupby(["proto", "seg", "sd"]):
    if seg == "second300": continue
    for gate, sfx in (("CARNet", "ph"), ("surrogate", "ph_s"), ("all bins", "ph_all")):
        for arm in ("un", "pr", "ro"):
            r = fit(x, f"{arm}_{sfx}"); r.update(proto=p[:4], seg=seg, sd=sd, gate=gate, arm=arm); rows.append(r)
G = pd.DataFrame(rows); pd.set_option("display.width", 200); pd.set_option("display.max_rows", 500); print(G[["proto", "seg", "sd", "gate", "arm", "n", "S", "C", "b", "lo", "hi", "p", "perm", "rob", "lev"]].round(3).to_string(index=False)); G.to_csv(Path(sys.argv[2]).with_suffix(".grid.csv"), index=False)
c = G[(G.seg != "HEADERCLOCK") & (G.arm != "un") & (G.gate != "all bins")]; print("corrected-clock aligned cells:", len(c), "| CI excludes 0:", int(((c.lo > 0) | (c.hi < 0)).sum()), "| median %.1f range %.1f to %.1f | half-width median %.1f" % (c.b.median(), c.b.min(), c.b.max(), ((c.hi - c.lo) / 2).median()))
h = G[G.seg == "HEADERCLOCK"]; print("header-clock cells, left CARNet:", h[(h.sd == "L") & (h.gate == "CARNet")][["arm", "n", "b", "lo", "hi", "p"]].round(2).to_dict("records"))
print("\n== regime by group within analysed left channels, K, coherence")
for (p, seg), x in D[D.sd == "L"].groupby(["proto", "seg"]): print(p, seg, "n", len(x), "S hi/lo", int(((x.stroke == 1) & (x.hi == 1)).sum()), int(((x.stroke == 1) & (x.hi == 0)).sum()), "C hi/lo", int(((x.stroke == 0) & (x.hi == 1)).sum()), int(((x.stroke == 0) & (x.hi == 0)).sum()), "K med", x.un_K.median(), "coh %.2f" % x.un_coh.median(), "secs med %.0f" % x.secs.median(), "lag lo/hi %.0f/%.0f" % (x[x.hi == 0].lag.median() * 1e3, x[x.hi == 1].lag.median() * 1e3), "phase un med by C-lo,C-hi,S-lo,S-hi", [round(float(g.un_ph.median()), 1) for _, g in x.groupby(["stroke", "hi"])])
print("\n== regime effect on unaligned phase by frequency (high minus low, adjusted for group), against pure delay at band centre")
for (p, seg, sd), x in D[D.seg.isin(["first300", "t60-360"])].groupby(["proto", "seg", "sd"]):
    tau = x[x.hi == 1].lag.median() - x[x.hi == 0].lag.median(); o = []
    for b in [(.03, .07), (.07, .14), (.14, .20), (.07, .20), (.20, .30), (.30, .50)]:
        q = x.assign(v=x.bands.map(lambda z: z[b])); m = smf.ols("v ~ hi + stroke", q).fit(cov_type="HC3", use_t=True); ci = m.conf_int().loc["hi"]; o.append(f"{b[0]:.2f}-{b[1]:.2f}: {m.params['hi']:+.1f} ({ci[0]:.1f},{ci[1]:.1f}) pred {-360 * sum(b) / 2 * tau:+.1f}")
    print(p, sd, "tau %.0f ms |" % (tau * 1e3), " | ".join(o))
print("\n== bias analysis: expected group difference from regime alone")
for (p, seg), x in D[(D.sd == "L") & D.seg.isin(["first300", "t60-360"])].groupby(["proto", "seg"]):
    pd_ = x[x.stroke == 1].hi.mean() - x[x.stroke == 0].hi.mean(); tau = x[x.hi == 1].lag.median() - x[x.hi == 0].lag.median(); print(p, "high-regime share difference %.2f; tau %.0f ms; pure-delay phase at LF centre %.1f deg; expected bias %.1f deg" % (pd_, tau * 1e3, -360 * .135 * tau, pd_ * -360 * .135 * tau))
print("\n== reliability, left, all bins, unaligned")
w = D[D.sd == "L"].pivot_table(index="sid", columns="seg", values="un_ph_all")
for a, b in (("first300", "second300"), ("first300", "t60-360")):
    q = w[[a, b]].dropna(); print(a, "vs", b, "n", len(q), "Spearman %.2f (p %.3f); median abs diff %.1f deg" % (*stats.spearmanr(q[a], q[b]), (q[a] - q[b]).abs().median()))
