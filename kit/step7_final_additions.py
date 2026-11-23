"""Final additions for the submission: robust (Huber) estimates at the main gate, regime-label increment with intervals,
analysed-segment gaps and physiology by group and regime, and seated-run durations. Usage: step7_final_additions.py DATA STEP4_PKL REPORTS_DIR
DATA = the CVES folder with subjects.csv; STEP4_PKL = per-participant file from step4_signals.py (private folder); REPORTS_DIR = folder holding 4_grid.txt from this run. Aggregates only."""
import sys, re, numpy as np, pandas as pd, warnings
from pathlib import Path
from scipy import stats
from core import *
from clinical import *
warnings.filterwarnings("ignore")
ROW = re.compile(r"^\s*(head|sit-)\s+(\S+)\s+([LR])\s+(CARNet|surrogate|all bins)\s+(un|pr|ro)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s*$")
def parse_grid(path):
    rows = []
    for ln in Path(path).read_text().splitlines():
        m = ROW.match(ln)
        if m:
            g = m.groups(); rows.append(dict(proto=g[0], seg=g[1], sd=g[2], gate=g[3], arm=g[4], n=int(g[5]), S=int(g[6]), C=int(g[7]), b=float(g[8]), lo=float(g[9]), hi=float(g[10]), p=float(g[11]), perm=float(g[12]), rob=float(g[13]), lev=float(g[14])))
    return pd.DataFrame(rows)
def robust_main_gate(path):
    g = parse_grid(path); m = g[(g.gate == "CARNet") & g.seg.isin(["t60-360", "first300"])]
    print("== robust (Huber) estimates, main CARNet gate, main segments (tilt 60-360 s; seated first 300 s), stroke minus control")
    print("cells %d | Huber negative in %d | Huber range %.1f to %.1f, median %.1f | least-squares range %.1f to %.1f | intervals excluding zero %d" % (len(m), int((m.rob < 0).sum()), m.rob.min(), m.rob.max(), m.rob.median(), m.b.min(), m.b.max(), int(((m.lo > 0) | (m.hi < 0)).sum())))
    for _, r in m.iterrows(): print("  %s %s %s %s n %d: least squares %+.1f (%.1f, %.1f); Huber %+.1f" % (r.proto, r.seg, r.sd, r.arm, r.n, r.b, r.lo, r.hi, r.rob))
    return m
def runs_all(fz, fs):   # copied unchanged from step4_signals.py
    k = median_filter(np.nan_to_num(np.abs(fz))[::fs // 10], size=11); low = k < .5 * k.max(); o = []; s = None
    for i, x in enumerate(np.r_[low, False]):
        if x and s is None: s = i
        if not x and s is not None:
            if i - s >= 100: o.append((s, i))
            s = None
    return o
def q3(v):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    return "NA" if len(v) == 0 else "%.1f [%.1f-%.1f]" % (np.median(v), *np.percentile(v, [25, 75]))
# ---- data sections below
if __name__ == "__main__":
    root = Path(sys.argv[1]); D = pd.read_pickle(sys.argv[2]).drop(columns=["age", "sex", "sex_"], errors="ignore"); reports = Path(sys.argv[3]); rng = np.random.default_rng(2029)
    robust_main_gate(reports / "4_grid.txt")

    print("\n== regime label added to the clinical variables (A = 15 clinical variables), paired refit bootstrap (1000), same participants as Table 7")
    S0 = pd.read_csv(root / "subjects.csv"); S0.columns = [c.strip() for c in S0.columns]; S0["sid"] = S0.subject_number.astype(str).str.strip().str.lower().str[:5]
    ssb = {h.stem[:5].lower() for h in (root / "data/sit-stand-balance").glob("*.hea")}; S1 = S0[S0.completed_visit_status.astype(str).str.upper().str.contains("COMPLET") & S0.sid.isin(ssb)].set_index("sid"); X, y0 = build(S1); X["y"] = y0
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
            q = x.dropna(subset=[f"{arm}_ph", f"{arm}_g"]); yy = q.y.values.astype(int); a = q[A].values.astype(float); b = np.c_[a, q.hi.values.astype(float)]; ph = q[[f"{arm}_g", f"{arm}_ph"]].values; c = np.c_[a, ph]
            ra, rb, rc = rep(a, yy, 60), rep(b, yy, 60), rep(c, yy, 60)
            print(p, arm, "n", len(q), "S", int(yy.sum()), "| A %.3f; A+regime %.3f, change %+.3f CI" % (ra, rb, rb - ra), boot(a, b, yy).tolist(), "| A+phase and gain %.3f, change %+.3f CI" % (rc, rc - ra), boot(a, c, yy).tolist())
    print("(the supine increment analysis is restricted to participants who are also in the seated cohort: inner join with X)")

    print("\n== analysed segments: gaps (invalid samples in the WFDB file) and physiology by group and regime, left channel")
    for proto, seg, fs in (("head-up-tilt", "t60-360", 500), ("sit-stand-balance", "first300", 1000)):
        d = D[(D.proto == proto) & (D.seg == seg) & (D.sd == "L")].set_index("sid"); rows = []
        for h in records(root, proto):
            sid = pid(h)
            if sid not in d.index: continue
            r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; Pm = r.p_signal; q = first_seated(Pm[:, n.index("fz")], fs) if proto[0] == "s" else (60 * fs, 360 * fs)
            if q is None: continue
            s0, s1 = q; raw = {c: Pm[s0:s1, n.index(c)] for c in ("abp", "mcal", "co2")}; gap = {c: float(np.isnan(v).mean()) for c, v in raw.items()}
            a = np.nan_to_num(raw["abp"]); c2 = np.nan_to_num(raw["co2"]); pk, _ = sg.find_peaks(a, distance=int(.4 * fs), prominence=np.std(a) * .5); pc, _ = sg.find_peaks(c2, distance=int(2 * fs), prominence=5)
            rows.append(dict(sid=sid, gap_abp=gap["abp"], gap_mca=gap["mcal"], gap_co2=gap["co2"], hr=60 / (np.median(np.diff(pk)) / fs) if len(pk) > 5 else np.nan, rr=60 / (np.median(np.diff(pc)) / fs) if len(pc) >= 5 else np.nan))
        F = pd.DataFrame(rows).set_index("sid").join(d[["stroke", "hi", "map", "et"]]); anygap = (F[["gap_abp", "gap_mca", "gap_co2"]] > 0).any(axis=1)
        print(proto, seg, "| records analysed %d | records with any invalid sample in the segment: ABP %d, MCA %d, CO2 %d, any %d | largest gap fraction %.4f" % (len(F), int((F.gap_abp > 0).sum()), int((F.gap_mca > 0).sum()), int((F.gap_co2 > 0).sum()), int(anygap.sum()), F[["gap_abp", "gap_mca", "gap_co2"]].max().max()))
        for (g, rg), z in F.groupby(["stroke", "hi"]): print("  %s, %s regime, n %d: mean ABP %s mmHg; heart rate %s /min; breathing rate %s /min; end-tidal CO2 %s mmHg" % ("stroke" if g else "control", "high" if rg else "low", len(z), q3(z["map"]), q3(z.hr), q3(z.rr), q3(z.et)))

    print("\n== seated-run durations, all sit-stand-balance recordings (force-plate rule of step4_signals.py)")
    L = {"first": [], "longest": [], "second": []}; nrec = 0; none = 0; full1 = []
    for h in records(root, "sit-stand-balance"):
        nrec += 1; r = wfdb.rdrecord(str(h.with_suffix(""))); o = runs_all(r.p_signal[:, r.sig_name.index("fz")], 1000)
        if not o: none += 1; continue
        dur = lambda q: (q[1] - q[0]) / 10; L["first"].append(min(300, dur(o[0]))); full1.append(dur(o[0])); L["longest"].append(min(300, dur(max(o, key=dur))))
        if len(o) > 1: L["second"].append(min(300, dur(o[1])))
    print("recordings %d, none with a seated run %d" % (nrec, none))
    for k, v in L.items(): v = np.array(v); print("  %s run, tail segment used: n %d | 300 s: %d | 200 to under 300 s: %d | under 200 s: %d" % (k, len(v), int((v >= 300).sum()), int(((v >= 200) & (v < 300)).sum()), int((v < 200).sum())))
    print("  first seated run, full duration: median %.0f s (IQR %.0f to %.0f), range %.0f to %.0f" % (np.median(full1), *np.percentile(full1, [25, 75]), min(full1), max(full1)))
