"""RQ2: clock gate, mains lines, ECG anchoring within group, regime by timing, paired delays, group link, injection by band. Aggregates only."""
import sys, re, numpy as np, pandas as pd, json
from scipy.stats import fisher_exact, gaussian_kde, spearmanr, mannwhitneyu
from statsmodels.stats.contingency_tables import Table2x2
from core import *
root = Path(sys.argv[1]); out = Path(sys.argv[2]); s = pd.read_csv(root / "subjects.csv"); s.columns = [c.strip() for c in s.columns]; s["sid"] = s.subject_number.astype(str).str.strip().str.lower().str[:5]; G = dict(zip(s.sid, (s.group.astype(str).str.strip().str.lower() == "stroke").astype(int)))
rows = []
for proto, fs in CLOCK.items():
    for h in records(root, proto):
        r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; P = r.p_signal; a = np.nan_to_num(P[:, n.index("abp")]); co2 = np.nan_to_num(P[:, n.index("co2")]); e = np.nan_to_num(P[:, n.index("ecg")])
        row = dict(proto=proto, sid=pid(h), stroke=G.get(pid(h), np.nan), hr=60 * peak_hz(a, fs, .5, 3), br=60 * peak_hz(co2, fs, .05, 1), dur=len(a) / fs)
        if proto == "sit-stand": rows.append(row); continue
        seg = first_seated(P[:, n.index("fz")], fs) if proto[0] == "s" and "fz" in n else (60 * fs, 360 * fs)
        if seg is None: rows.append(row); continue
        s0, s1 = seg; row["secs"] = (s1 - s0) / fs
        row["r_abp"] = r_locked(e[s0:s1], a[s0:s1], fs)
        for ch, sd in (("mcal", "L"), ("mcar", "R")):
            v = np.nan_to_num(P[:, n.index(ch)]); w = v[s0:s1]; row["gain" + sd] = header_gain(h, ch); row["valid" + sd] = valid(w); row["span" + sd] = float(np.percentile(v, 99.5) - np.percentile(v, .5))
            if row["valid" + sd]: row["r_mca" + sd] = r_locked(e[s0:s1], w, fs); row["lagx" + sd] = lag_xcorr(a[s0:s1], w, fs) * 1e3; row["lags" + sd] = lag_slope(a[s0:s1], w, fs) * 1e3
        rows.append(row)
d = pd.DataFrame(rows); d.to_pickle(out); print("records", d.groupby("proto").size().to_dict())
print("\n== RQ2 clock gate (share with HR 40-110 and breathing 6-30 per minute at corrected clock)")
for p, x in d.groupby("proto"): ok = x.hr.between(40, 110) & x.br.between(6, 30); print(p, "HR med %.1f br med %.1f dur med %.0f s; pass %d/%d = %.0f%%" % (x.hr.median(), x.br.median(), x.dur.median(), ok.sum(), len(x), 100 * ok.mean()))
x = d[d.proto == "sit-stand-balance"]; print("as labelled at 500 Hz, sit-stand-balance: HR med %.1f, breathing med %.1f, duration med %.0f s" % (x.hr.median() / 2, x.br.median() / 2, x.dur.median() * 2))
print("(Group link by gain below is a comparator. The paper's regime definition and OR are in 7_robustness.)")
x = d[d.proto.isin(["sit-stand-balance"])]; print("failing records: HR out", int((~x.hr.between(40, 110)).sum()), "breathing out", int((~x.br.between(6, 30)).sum()))
for lab, c in (("HR", "hr"), ("breathing", "br")):
    w = d[d.proto != "sit-stand"].pivot(index="sid", columns="proto", values=c).dropna(); q = w["head-up-tilt"] / w["sit-stand-balance"]; print("same person tilt/seated", lab, "ratio med %.2f IQR %s n=%d" % (q.median(), np.round(q.quantile([.25, .75]).values, 2), len(q)))
col = [c for c in s.columns if re.search(r"HR", c) and re.search(r"sit", c, re.I)][:6]; print("lab HR columns", col)
print("\n== regime by ECG-anchored MCA delay")
reg = {}
for p, fs in (("head-up-tilt", 500), ("sit-stand-balance", 1000)):
    x = d[d.proto == p]
    for sd in "LR":
        y = x[f"r_mca{sd}"].dropna().values; g = np.linspace(y.min(), y.max(), 500); k = gaussian_kde(y)(g); pk = sg.find_peaks(k)[0]; top = sorted(pk[np.argsort(k[pk])[-2:]]) if len(pk) > 1 else None; thr = g[top[0] + np.argmin(k[top[0]:top[1]])] if top else np.nan; reg[(p, sd)] = thr
        z = x.dropna(subset=[f"r_mca{sd}"]); hi = z[f"r_mca{sd}"] > thr; gh = z[f"gain{sd}"] >= 3e4; print(p, sd, "n", len(z), "antimode %.0f ms; modes %s; agreement with gain %.0f%%; delay hi/lo by gain %.0f/%.0f" % (thr, np.round(g[top]).tolist() if top else None, 100 * (hi == gh).mean(), z[gh][f"r_mca{sd}"].median(), z[~gh][f"r_mca{sd}"].median()))
print("\n== RQ2 instrumental: within each group, high minus low gain regime")
for p in ("head-up-tilt", "sit-stand-balance"):
    x = d[d.proto == p]
    for st, y in x.groupby("stroke"):
        y = y[y.validL.fillna(False).astype(bool)]; h = y.gainL >= 3e4; print(p, "stroke" if st else "control", "n lo/hi", int((~h).sum()), int(h.sum()), "| ECG-ABP shift %.0f ms | ECG-MCA shift %.0f ms | ABP-MCA lag shift %.0f ms" % (y[h].r_abp.median() - y[~h].r_abp.median(), y[h].r_mcaL.median() - y[~h].r_mcaL.median(), y[h].lagxL.median() - y[~h].lagxL.median()))
t = d[d.proto == "head-up-tilt"].set_index("sid"); b = d[d.proto == "sit-stand-balance"].set_index("sid"); j = t.join(b, lsuffix="_t", rsuffix="_b", how="inner"); j = j[j.validL_t.fillna(False).astype(bool) & j.validL_b.fillna(False).astype(bool)]; dd = j.lagxL_t - j.lagxL_b
print("same person tilt minus seated ABP-MCA lag: n %d, median %.0f ms, median |diff| %.0f ms, rho %.2f" % (len(dd), dd.median(), dd.abs().median(), spearmanr(j.lagxL_t, j.lagxL_b)[0]))
sw = j[(j.gainL_t >= 3e4) != (j.gainL_b >= 3e4)]; print("regime switchers between sessions n", len(sw), "lag moves in expected direction", int((np.sign(sw.lagxL_t - sw.lagxL_b) == np.where(sw.gainL_t >= 3e4, 1, -1)).sum()))
print("two estimators", {p: round(spearmanr(x.lagxL, x.lagsL, nan_policy="omit")[0], 2) for p, x in d[d.proto != "sit-stand"].groupby("proto")})
print("\n== group link, by channel gain and by timing")
for p in ("head-up-tilt", "sit-stand-balance"):
    x = d[d.proto == p]
    for sd in "LR":
        for nm, hi in (("gain", x[f"gain{sd}"] >= 3e4), ("timing", x[f"r_mca{sd}"] > reg[(p, sd)])):
            m = x[f"r_mca{sd}"].notna() if nm == "timing" else x.stroke.notna(); st = x.stroke[m] == 1; hh = hi[m]; T = np.array([[(hh & st).sum(), (~hh & st).sum()], [(hh & ~st).sum(), (~hh & ~st).sum()]]); tb = Table2x2(T); print(p, sd, nm, T.tolist(), "OR %.2f (%.2f-%.2f) p %.2g" % (tb.oddsratio, *tb.oddsratio_confint(), fisher_exact(T)[1]))
print("\n== order, amplitude")
for p in ("head-up-tilt", "sit-stand-balance"):
    x = d[d.proto == p].copy(); x["num"] = x.sid.str[1:].astype(int); from sklearn.metrics import roc_auc_score as A
    print(p, "order->regime(left gain) %.3f; order->group %.3f" % (A(x.gainL >= 3e4, x.num), A(x.stroke, x.num)), "| low/high stored span ratio (left) %.2f" % (x[(x.gainL < 3e4)].spanL.median() / x[x.gainL >= 3e4].spanL.median()))
