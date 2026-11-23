"""Additional aggregate checks: cohort statuses, grid summaries, smallest detectable difference, coherence by group, environment.
Plain: the few extra numbers the final paper needs, so none is computed by hand. Usage: step6_additions.py DATA step4.pkl. Aggregates only."""
import sys, re, platform, importlib, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings, wfdb
from scipy import signal as sg
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.metrics import roc_auc_score
from core import records, pid, first_seated, beats_seg
from pathlib import Path
warnings.filterwarnings("ignore")
root = Path(sys.argv[1]); pk = Path(sys.argv[2]); G = pd.read_csv(pk.with_suffix(".grid.csv")); D = pd.read_pickle(pk)
s = pd.read_csv(root / "subjects.csv"); s.columns = [c.strip() for c in s.columns]; s["sid"] = s.subject_number.astype(str).str.strip().str.lower().str[:5]
print("== environment:", "python", platform.python_version(), *[f"{m} {importlib.import_module(m).__version__}" for m in ("numpy", "scipy", "pandas", "statsmodels", "sklearn", "wfdb")])
grp = s.group.astype(str).str.strip().str.upper()
print("\n== cohort: participant table rows", len(s), "| stroke", int((grp == "STROKE").sum()), "control", int((grp == "CONTROL").sum()))
print("visit status counts:", s.completed_visit_status.astype(str).str.strip().str.upper().value_counts().to_dict())
ids = {p: {h.stem[:5].lower() for h in (root / "data" / p).glob("*.hea")} for p in ("sit-stand-balance", "head-up-tilt")}
for nm, k in (("sit-stand-balance", ids["sit-stand-balance"]), ("head-up-tilt", ids["head-up-tilt"]), ("both", ids["sit-stand-balance"] & ids["head-up-tilt"]), ("either", ids["sit-stand-balance"] | ids["head-up-tilt"])):
    x = s[s.sid.isin(k)]; print(f"{nm}: n {len(x)}, stroke {int((x.group.str.upper().str.strip() == 'STROKE').sum())}, TIA {int(x['Stroke Group'].astype(str).str.strip().str.upper().eq('TIA').sum())}, completed {int(x.completed_visit_status.astype(str).str.upper().str.contains('COMPLET').sum())}")
from clinical import build
comp = s[s.completed_visit_status.astype(str).str.upper().str.contains("COMPLET") & s.sid.isin(ids["sit-stand-balance"])].set_index("sid"); X, yb = build(comp)
num = lambda c: pd.to_numeric(comp[c], errors="coerce")
X["nihss"] = num("NIHSS"); X["mrs"] = num("MRS"); X["sbp"] = num("SBP BASELINE"); X["dbp"] = num("DBP BASELINE"); X["hr"] = num("(Baseline Mean) HR BP BASELINE")
side = comp["Stroke Side"].astype(str).str.upper().str.strip()
print("\n== baseline table, sit-stand-balance cohort (stroke / control): median [IQR] or n (%); n with data")
for c in ("age", "bmi", "pack_years", "gait", "sbp", "dbp", "hr", "nihss", "mrs"):
    out = []; dd = 2 if c == "gait" else 1
    for g in (1, 0):
        v = X[c][yb == g].dropna(); out.append(f"{v.median():.{dd}f} [{v.quantile(.25):.{dd}f}-{v.quantile(.75):.{dd}f}] (n {len(v)})" if len(v) else "NA")
    print(f"  {c}: {out[0]} / {out[1]}")
for c in ("sex", "htn", "diabetes", "hyperlip", "cur_tob", "antiplatelet", "anticoag", "statin", "antihyperlip"):
    out = []
    for g in (1, 0):
        v = X[c][yb == g].dropna(); out.append(f"{int(v.sum())}/{len(v)} ({100 * v.mean():.0f}%)")
    print(f"  {c} (yes): {out[0]} / {out[1]}")
print("  stroke side among stroke:", side[yb == 1].map(lambda v: "bilateral" if "BILAT" in v else ("left" if v.startswith("L") else ("right" if v.startswith("R") else "not recorded or other"))).value_counts().to_dict())
print("\n== phase grid summaries (corrected clock)")
main = G[G.seg.isin(["first300", "t60-360"]) & (G.gate == "CARNet")]
for nm, g in (("12 main-gate cells, both sides, all arms", main), ("6 main-gate cells, left", main[main.sd == "L"]), ("8 main-gate aligned cells", main[main.arm.isin(["pr", "ro"])]),
              ("36 aligned cells, three segments x three gates", G[G.seg.isin(["first300", "t60-360", "longest300"]) & G.arm.isin(["pr", "ro"])])):
    print(f"{nm}: n cells {len(g)}, estimates {g.b.min():.1f} to {g.b.max():.1f}, median {g.b.median():.1f}, intervals excluding 0 {int(((g.lo > 0) | (g.hi < 0)).sum())}, median half-width {((g.hi - g.lo) / 2).median():.1f}, widest single interval {g.loc[(g.hi - g.lo).idxmax(), 'lo']:.1f} to {g.loc[(g.hi - g.lo).idxmax(), 'hi']:.1f}")
m = main.assign(mde=2.8 * (main.hi - main.lo) / (2 * 1.96))
print("smallest detectable difference (80 percent power, two-sided 5 percent; 2.8 x SE), main gate: median %.1f, range %.1f to %.1f deg" % (m.mde.median(), m.mde.min(), m.mde.max()))
for (p, sd), g in m.groupby(["proto", "sd"]): print(f"  {p} {sd}: {g.mde.min():.1f} to {g.mde.max():.1f}")
print("\n== enrolment-order proxy: participant number vs first recording date in dated headers (aggregate only)")
first = {}
for h in (root / "data").rglob("*.hea"):
    try: hd = wfdb.rdheader(str(h.with_suffix("")))
    except Exception: continue
    mm = re.match(r"s(\d{4})", h.name.lower())
    if hd.base_date and mm: k = int(mm.group(1)); first[k] = min(first.get(k, 10 ** 9), hd.base_date.toordinal())
ks = sorted(first); print("participants with a dated header", len(ks), "| Spearman %.3f" % spearmanr(ks, [first[i] for i in ks])[0])
kk = [i for i in ks if f"s{i:04d}" in set(s.sid)]; gg = [int(s.set_index("sid").group.astype(str).str.strip().str.upper()[f"s{i:04d}"] == "STROKE") for i in kk]; print("recording date predicts group: AUC %.3f (n %d)" % (roc_auc_score(gg, [first[i] for i in kk]), len(kk)))
print("\n== blood-pressure variability: ABP low-frequency power by group")
rows = []
for proto, fs in (("head-up-tilt", 500), ("sit-stand-balance", 1000)):
    for h in records(root, proto):
        r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; P = r.p_signal; q = first_seated(P[:, n.index("fz")], fs) if proto[0] == "s" else (60 * fs, 360 * fs)
        if q is None: continue
        a = np.nan_to_num(P[:, n.index("abp")]); x, _ = beats_seg(a, a, fs, *q); x = (x / x.mean() - 1) * 100; f, pw = sg.welch(x - x.mean(), fs=10, nperseg=1000, noverlap=500, window="hann"); mk_ = (f >= .07) & (f <= .2)
        rows.append(dict(proto=proto, sid=pid(h), lf=float(np.trapezoid(pw[mk_], f[mk_]))))
B = pd.DataFrame(rows); B["stroke"] = (B.sid.map(s.set_index("sid").group).astype(str).str.strip().str.upper() == "STROKE").astype(int)
for p, x in B.groupby("proto"): print(p, "median stroke %.2f vs control %.2f; Mann-Whitney p %.3f" % (x[x.stroke == 1].lf.median(), x[x.stroke == 0].lf.median(), mannwhitneyu(x[x.stroke == 1].lf, x[x.stroke == 0].lf).pvalue))
D = D.merge(B[["proto", "sid", "lf"]], on=["proto", "sid"], how="left"); D["llf"] = np.log(D.lf)
print("\n== EXPLORATORY: low-frequency coherence (0.07-0.20 Hz), stroke minus control, left, unaligned")
D["num"] = D.sid.str[1:].astype(int); S_ = s.set_index("sid")
for (p, seg), x in D[(D.sd == "L") & D.seg.isin(["first300", "second300", "t60-360"])].groupby(["proto", "seg"]):
    x = x.copy(); x["z"] = (x.num - x.num.mean()) / x.num.std(); x["age"] = pd.to_numeric(x.sid.map(S_.age), errors="coerce"); x["sx"] = x.sid.map(S_.gender).astype(str).str[0]
    q = x.dropna(subset=["un_coh", "age", "et", "llf"])
    for f in ("stroke + hi", "stroke + hi + z + age + C(sx) + et + map", "stroke + hi + z + llf"):
        r = smf.ols("un_coh ~ " + f, q).fit(cov_type="HC3", use_t=True); c = r.conf_int().loc["stroke"]; print(f"{p} {seg} | {f} | n {len(q)} | {r.params['stroke']:+.3f} ({c[0]:+.3f}, {c[1]:+.3f}) p {r.pvalues['stroke']:.3f}")
    rng = np.random.default_rng(7); obs = q[q.stroke == 1].un_coh.median() - q[q.stroke == 0].un_coh.median(); cnt = 0
    for _ in range(5000):
        y = q.stroke.values.copy()
        for h in (0, 1): i = np.where(q.hi.values == h)[0]; y[i] = rng.permutation(y[i])
        cnt += abs(np.median(q.un_coh.values[y == 1]) - np.median(q.un_coh.values[y == 0])) >= abs(obs)
    print(f"   median difference {obs:+.3f}; permutation within regime (5000) p {(cnt + 1) / 5001:.3f}")
    print("   medians by group x regime (n, coherence):", {("S" if a else "C") + ("hi" if b else "lo"): (len(y), round(y.un_coh.median(), 2)) for (a, b), y in q.groupby(["stroke", "hi"])})

print("\n== affected hemisphere (stroke, side recorded) vs controls (mean of sides), main gate, per-record aligned")
side = S_["Stroke Side"].astype(str).str.upper().str.strip()
aff = side.map(lambda v: None if "BILAT" in v else ("L" if re.match(r"L|LEFT", v) else ("R" if re.match(r"R|RIGHT", v) else None)))
for (p, seg), x in D[D.seg.isin(["first300", "t60-360"])].groupby(["proto", "seg"]):
    w = x.pivot_table(index="sid", values="pr_ph", columns="sd"); st = x.drop_duplicates("sid").set_index("sid").stroke; vals = []
    for sid, row in w.iterrows():
        if st[sid] == 1:
            a_ = aff.get(sid)
            if a_ in ("L", "R") and pd.notna(row.get(a_)): vals.append((sid, 1, row[a_]))
        elif row.notna().any(): vals.append((sid, 0, row.dropna().mean()))
    V = pd.DataFrame(vals, columns=["sid", "stroke", "ph"]); V["age"] = pd.to_numeric(V.sid.map(S_.age), errors="coerce"); r = smf.ols("ph ~ stroke + age", V.dropna()).fit(cov_type="HC3", use_t=True); c = r.conf_int().loc["stroke"]
    print(f"{p}: stroke {int((V.stroke == 1).sum())}, control {int((V.stroke == 0).sum())}; affected side minus control {r.params['stroke']:+.1f} ({c[0]:.1f}, {c[1]:.1f}) p {r.pvalues['stroke']:.2f}")
