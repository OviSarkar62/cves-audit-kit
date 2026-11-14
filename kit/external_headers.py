"""Headers and clocks only, for same-laboratory datasets. Reads no group labels and compares no groups. Usage: external_headers.py DATASET_DIR. Prints per-directory aggregates."""
import sys, numpy as np, wfdb, collections
from pathlib import Path
from core import peak_hz, valid
root = Path(sys.argv[1]); by = collections.defaultdict(list)
for h in root.rglob("*.hea"): by[h.parent.relative_to(root)].append(h)
for d, H in sorted(by.items(), key=str):
    fs = collections.Counter(); hr = []; br = []; g = []; n = 0
    for h in sorted(H):
        try: hd = wfdb.rdheader(str(h.with_suffix("")))
        except Exception: continue
        fs[round(float(hd.fs), 2) if hd.fs else None] += 1; names = [str(x).lower() for x in (hd.sig_name or [])]; ch = {x: i for i, x in enumerate(names)}
        if "abp" not in ch and "bp" not in ch: continue
        if hd.fs is None or hd.sig_len is None : continue
        n += 1; want = [hd.sig_name[ch[k]] for k in ("abp", "bp", "co2", "mcal", "mcar") if k in ch]
        try: r = wfdb.rdrecord(str(h.with_suffix("")), channel_names=want)
        except Exception: continue
        m = {str(x).lower(): i for i, x in enumerate(r.sig_name)}; p = r.p_signal; a = p[:, m["abp"] if "abp" in m else m["bp"]]; hr.append(60 * peak_hz(a, r.fs, .5, 3))
        if "co2" in m: br.append(60 * peak_hz(p[:, m["co2"]], r.fs, .05, 1))
        for c in ("mcal", "mcar"):
            if c in m and valid(np.nan_to_num(p[:, m[c]])): v = np.nan_to_num(p[:, m[c]]); g.append(np.log10(max(np.percentile(v, 99.5) - np.percentile(v, .5), 1e-3)))
    print(d, "| records", len(H), "| header fs", dict(fs), "| sampled", n, "| HR med %s IQR %s" % (np.round(np.median(hr), 1) if hr else None, np.round(np.percentile(hr, [25, 75]), 1).tolist() if hr else None), "| breathing med %s" % (np.round(np.median(br), 1) if br else None), "| log10 MCA span hist", np.histogram(g, bins=np.arange(-1.5, 2.01, .25))[0].tolist() if g else None, flush=True)
