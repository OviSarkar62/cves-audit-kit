"""Group-median phase, gain and coherence by frequency, 0.02-0.5 Hz, for the full-curve figure CARNet recommends. Usage: curves.py DATA out.csv. Group-level aggregates only."""
import sys, numpy as np, pandas as pd
from core import *
root = Path(sys.argv[1]); S = pd.read_csv(root / "subjects.csv"); S.columns = [c.strip() for c in S.columns]; S["sid"] = S.subject_number.astype(str).str.strip().str.lower().str[:5]; G = dict(zip(S.sid, S.group.astype(str).str.strip().str.lower())); out = []
for proto, fs in (("head-up-tilt", 500), ("sit-stand-balance", 1000)):
    for h in records(root, proto):
        r = wfdb.rdrecord(str(h.with_suffix(""))); n = r.sig_name; P = r.p_signal; q = first_seated(P[:, n.index("fz")], fs) if proto[0] == "s" else (60 * fs, 360 * fs)
        if q is None: continue
        a, v = np.nan_to_num(P[:, n.index("abp")]), np.nan_to_num(P[:, n.index("mcal")])
        if not valid(v[q[0]:q[1]]) or (q[1] - q[0]) / fs < 200: continue
        x, y = beats_seg(a, v, fs, *q); x, y = (x / x.mean() - 1) * 100, (y / y.mean() - 1) * 100; f, pxy, pxx, coh = spectra(x, y); m = (f >= .02) & (f <= .5)
        out += [dict(proto=proto, group=G.get(pid(h)), f=round(float(ff), 3), phase=float(np.degrees(np.angle(c))), gain=float(abs(c / d)), coh=float(k)) for ff, c, d, k in zip(f[m], pxy[m], pxx[m], coh[m])]
D = pd.DataFrame(out); T = D.groupby(["proto", "group", "f"]).agg(phase=("phase", "median"), gain=("gain", "median"), coh=("coh", "median"), n=("phase", "size")).round(3).reset_index(); T.to_csv(sys.argv[2], index=False); print("wrote", len(T), "rows; n by protocol and group:", T.groupby(["proto", "group"]).n.max().to_dict())
