"""RQ3 grid, frequency-resolved regime offset, bias analysis, positive control, reliability. Saves one row per channel to a path outside the repo. Aggregates only on stdout."""
import sys, numpy as np, pandas as pd, statsmodels.formula.api as smf, warnings
from scipy import stats
from scipy.stats import gaussian_kde
from core import *
warnings.filterwarnings("ignore")
root = Path(sys.argv[1]); out = sys.argv[2]; S = pd.read_csv(root / "subjects.csv"); S.columns = [c.strip() for c in S.columns]; S["sid"] = S.subject_number.astype(str).str.strip().str.lower().str[:5]; S = S.set_index("sid")
rng = np.random.default_rng(2026); BANDS = [(.03, .07), (.07, .14), (.14, .20), (.07, .20), (.20, .30), (.30, .50)]
def etco2(c, fs):
    pk, _ = sg.find_peaks(c, distance=int(2 * fs), prominence=5); return float(np.median(c[pk])) if len(pk) >= 5 else np.nan
def runs_all(fz, fs):
    k = median_filter(np.nan_to_num(np.abs(fz))[::fs // 10], size=11); low = k < .5 * k.max(); o = []; s = None
    for i, x in enumerate(np.r_[low, False]):
        if x and s is None: s = i
        if not x and s is not None:
            if i - s >= 100: o.append((s, i))
            s = None
    return o
rows = []; pre = []
for proto, fs in (("head-up-tilt", 500), ("sit-stand-balance", 1000)):
    for h in records(root, proto):
        r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; P = r.p_signal; a = np.nan_to_num(P[:, n.index("abp")]); e = np.nan_to_num(P[:, n.index("ecg")]); co2 = np.nan_to_num(P[:, n.index("co2")]); sid = pid(h)
        for ch, sd in (("mcal", "L"), ("mcar", "R")):
            v = np.nan_to_num(P[:, n.index(ch)]); pre.append(dict(proto=proto, sid=sid, sd=sd, gain=header_gain(h, ch), span=float(np.percentile(v, 99.5) - np.percentile(v, .5))))
d0 = pd.DataFrame(pre); thr = {}
for (p, sd), x in d0.groupby(["proto", "sd"]):
    y = np.log10(x.span.clip(lower=1e-3)); y = y[(y > y.median() - 1.2)].values; g = np.linspace(y.min(), y.max(), 400); k = gaussian_kde(y, bw_method=.25)(g); pk = sg.find_peaks(k)[0]; t = sorted(pk[np.argsort(k[pk])[-2:]]); thr[(p, sd)] = g[t[0] + np.argmin(k[t[0]:t[1]])]
print("regime thresholds", {k: round(v, 2) for k, v in thr.items()})
d0["hi"] = [int(np.log10(max(s, 1e-3)) < thr[(p, sd)]) for s, p, sd in zip(d0.span, d0.proto, d0.sd)]; REG = {(a, b, c): v for a, b, c, v in zip(d0.proto, d0.sid, d0.sd, d0.hi)}   # high regime = small stored span
def segs(proto, P, n, fs):
    if proto == "head-up-tilt": return {"t60-360": ((60 * fs, 360 * fs), fs)}
    o = runs_all(P[:, n.index("fz")], fs); out = {}
    if not o: return out
    tail = lambda q, L=300: (int(max(q[0], q[1] - L * 10) / 10 * fs), int(q[1] / 10 * fs))
    out["first300"] = (tail(o[0]), fs); out["longest300"] = (tail(max(o, key=lambda q: q[1] - q[0])), fs); out["HEADERCLOCK"] = (tail(o[0]), 500)
    if len(o) > 1: out["second300"] = (tail(o[1]), fs)
    return out
lagmap = []
for proto, fs in (("head-up-tilt", 500), ("sit-stand-balance", 1000)):
    for h in records(root, proto):
        r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; P = r.p_signal; sid = pid(h)
        for seg, ((s0, s1), f) in segs(proto, P, n, fs).items():
            af = np.nan_to_num(P[:, n.index("abp")]); a = af[s0:s1]; co2 = np.nan_to_num(P[s0:s1, n.index("co2")]); et = etco2(co2, f)
            for ch, sd in (("mcal", "L"), ("mcar", "R")):
                vf = np.nan_to_num(P[:, n.index(ch)]); v = vf[s0:s1]
                if not valid(v) or (s1 - s0) / f < 200: continue
                lg = lag_xcorr(a, v, f); row = dict(proto=proto, seg=seg, sid=sid, sd=sd, f=f, lag=lg, secs=(s1 - s0) / f, et=et, map=float(a.mean()), hi=REG[(proto, sid, sd)])
                row["un"] = tfa(*beats_seg(af, vf, f, s0, s1), rng=rng, nsur=200); row["pr"] = tfa(*beats_seg(af, shift(vf, lg, f), f, s0, s1), rng=rng, nsur=200)
                row["bands"] = {b: tfa(*beats_seg(af, vf, f, s0, s1), band=b)["ph_all"] for b in BANDS}; rows.append(dict(row, _raw=None))
    print(proto, "first pass done", flush=True)
D = pd.DataFrame(rows); off = D.groupby(["proto", "seg", "sd"]).apply(lambda x: x[x.hi == 1].lag.median() - x[x.hi == 0].lag.median()).to_dict(); print("regime lag offsets, ms:", {k: round(v * 1e3) for k, v in off.items()})
so = []
for proto, fs in (("head-up-tilt", 500), ("sit-stand-balance", 1000)):
    for h in records(root, proto):
        r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; P = r.p_signal; sid = pid(h)
        for seg, ((s0, s1), f) in segs(proto, P, n, fs).items():
            af = np.nan_to_num(P[:, n.index("abp")])
            for ch, sd in (("mcal", "L"), ("mcar", "R")):
                vf = np.nan_to_num(P[:, n.index(ch)])
                if not valid(vf[s0:s1]) or (s1 - s0) / f < 200: continue
                hi = REG[(proto, sid, sd)]; x = tfa(*beats_seg(af, shift(vf, off[(proto, seg, sd)] if hi else 0, f), f, s0, s1), rng=rng, nsur=200); so.append(dict(proto=proto, seg=seg, sid=sid, sd=sd, ro=x))
D = D.merge(pd.DataFrame(so), on=["proto", "seg", "sid", "sd"]); D["stroke"] = (D.sid.map(S.group).astype(str).str.strip().str.lower() == "stroke").astype(int); D["age"] = pd.to_numeric(D.sid.map(S.age), errors="coerce"); D["sex"] = D.sid.map(S.gender).astype(str).str.strip().str.upper().str[0]
for arm in ("un", "pr", "ro"):
    for k in ("ph", "g", "ph_s", "g_s", "ph_all", "K", "coh", "nb"): D[f"{arm}_{k}"] = D[arm].map(lambda x: x.get(k))
D.drop(columns=["un", "pr", "ro", "_raw"]).to_pickle(out)
