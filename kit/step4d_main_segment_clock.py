"""Main-segment version of step4d: same samples, header clock (500 Hz) vs corrected clock (1000 Hz), on the paper's main segment (last 300 s of the first seated run, as in step4_signals.py first300). Usage: step4d_main_segment_clock.py DATA. Aggregates only."""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from core import *
warnings.filterwarnings("ignore")
root = Path(sys.argv[1]); S = pd.read_csv(root / "subjects.csv"); S.columns = [c.strip() for c in S.columns]; S["sid"] = S.subject_number.astype(str).str.strip().str.lower().str[:5]; S = S.set_index("sid"); rows = []
def runs_all(fz, fs):   # copied unchanged from step4_signals.py
    k = median_filter(np.nan_to_num(np.abs(fz))[::fs // 10], size=11); low = k < .5 * k.max(); o = []; s = None
    for i, x in enumerate(np.r_[low, False]):
        if x and s is None: s = i
        if not x and s is not None:
            if i - s >= 100: o.append((s, i))
            s = None
    return o
for h in records(root, "sit-stand-balance"):
    r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; P = r.p_signal; o = runs_all(P[:, n.index("fz")], 1000)
    if not o: continue
    q = o[0]; s0, s1 = int(max(q[0], q[1] - 3000) / 10 * 1000), int(q[1] / 10 * 1000)     # main segment: last 300 s of the first seated run, in sample indices of the true 1000 Hz clock
    if (s1 - s0) / 1000 < 200: continue
    af, vf = np.nan_to_num(P[:, n.index("abp")]), np.nan_to_num(P[:, n.index("mcal")])
    if not valid(vf[s0:s1]): continue
    c = np.nan_to_num(P[s0:s1, n.index("co2")]); pk, _ = sg.find_peaks(c, distance=2000, prominence=5); et = float(np.median(c[pk])) if len(pk) >= 5 else np.nan
    row = dict(sid=h.stem[:5].lower(), et=et); hl = lag_xcorr(af[s0:s1], vf[s0:s1], 500); tl = lag_xcorr(af[s0:s1], vf[s0:s1], 1000)
    for lab, f, lg, band in (("H_LF", 500, hl, (.07, .20)), ("T_LF", 1000, tl, (.07, .20)), ("T_0.14-0.40", 1000, tl, (.14, .40))):
        A_, V_ = beats_seg(af, vf, f, s0, s1); _r = tfa(A_, V_, band=band); row[lab + "_un"] = _r["ph"]; row[lab + "_all"] = _r["ph_all"]; row[lab + "_K"] = _r["K"]; A_, V_ = beats_seg(af, shift(vf, lg, f), f, s0, s1); row[lab + "_al"] = tfa(A_, V_, band=band)["ph"]
    rows.append(row)
D = pd.DataFrame(rows).set_index("sid"); D["stroke"] = (S.group.reindex(D.index).astype(str).str.strip().str.lower() == "stroke").astype(int); D["age"] = pd.to_numeric(S.age.reindex(D.index), errors="coerce"); D["sx"] = S.gender.reindex(D.index).astype(str).str.strip().str.upper().str[0]
for c in [c for c in D if c[0] in "HT" and "_" in c and not c.endswith("_K")]:
    q = D.dropna(subset=[c, "age", "et", "sx"]).assign(y=lambda z: z[c]); m = smf.ols("y ~ stroke + age + C(sx) + et", q).fit(cov_type="HC3", use_t=True); i = m.conf_int().loc["stroke"]; print(c, "n", len(q), "(%d/%d)" % (q.stroke.sum(), (1 - q.stroke).sum()), "%+.1f (%.1f, %.1f) p %.3f; median phase %.1f" % (m.params["stroke"], i[0], i[1], m.pvalues["stroke"], q.y.median()))

print("windows per record, labelled vs correct clock: %.0f vs %.0f" % (D.H_LF_K.median(), D.T_LF_K.median()))
import statsmodels.api as sm
rng = np.random.default_rng(31)
def coef(q, c, rob=False):
    q = q.assign(y=q[c]); f = "y ~ stroke + age + C(sx) + et"
    return (smf.rlm(f, q, M=sm.robust.norms.HuberT()).fit() if rob else smf.ols(f, q).fit()).params["stroke"]
print("Paired change on the same participants, correct minus labelled clock, participant bootstrap (2000):")
for a, b, lab in (("H_LF_un", "T_LF_un", "main gate, unaligned"), ("H_LF_al", "T_LF_al", "main gate, aligned"), ("H_LF_all", "T_LF_all", "all bins, unaligned")):
    q = D.dropna(subset=[a, b, "age", "et", "sx"]); n = len(q); ob = coef(q, b) - coef(q, a); bs = []
    for _ in range(2000):
        s = q.iloc[rng.integers(0, n, n)]
        try: bs.append(coef(s, b) - coef(s, a))
        except Exception: pass
    print(" ", lab, "n", n, "labelled %+.1f; correct %+.1f; change %+.1f (%.1f, %.1f); share of resamples above zero %.3f; Huber labelled %+.1f, correct %+.1f" % (coef(q, a), coef(q, b), ob, *np.percentile(bs, [2.5, 97.5]), np.mean(np.array(bs) > 0), coef(q, a, True), coef(q, b, True)))
q = D.dropna(subset=["T_LF_un", "age", "et", "sx"]); print("correct clock, phase SD %.1f; labelled clock, phase SD %.1f; correct-clock records beyond +/-90 deg: %d" % (D.T_LF_un.std(), D.H_LF_un.std(), (D.T_LF_un.abs() > 90).sum()))
