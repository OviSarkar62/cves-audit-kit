"""Reads every original WinDAQ file in the raw archive: sampling interval, channel calibration slope, raw MCA range. Plain: asks the lab's own files how fast they recorded and whether MCA scaling differs by regime.
Usage: raw_clock_census.py raw-data.tar.gz step1.pkl. Prints counts and medians only. WinDAQ layout: byte 2 = channel-info offset, byte 3 = bytes per channel, 4-5 = header bytes, 6-9 = data bytes, 28-35 = seconds per scan; each channel starts with a float slope."""
import tarfile, struct, re, collections, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
c = collections.Counter(); rows = []
for_kind = {"SC": "sit-stand-balance", "SB": "head-up-tilt"}
with tarfile.open(sys.argv[1], "r|gz") as t:
    for m in t:
        if not m.isfile() or not m.name.lower().endswith(".wdq"): continue
        b = m.name.split("/")[-1]; mt = re.search(r"(\d{4})(SA|SB|SC)", b, re.I); k = mt.group(2).upper() if mt else "other"; f = t.extractfile(m); h0 = f.read(14)
        if len(h0) < 14: c[(k, "too short")] += 1; continue
        nch = h0[0] & 31; off, chb, hdr, dbytes = h0[4], h0[5], struct.unpack("<H", h0[6:8])[0], struct.unpack("<I", h0[8:12])[0]; rest = f.read(m.size - 14 if k in for_kind and nch == 10 and "original" not in b.lower() else 40); h = h0 + rest
        if len(h) < 36: c[(k, nch, "too short")] += 1; continue
        dt = round(struct.unpack("<d", h[28:36])[0], 6) if off == 110 else "unreadable"; c[(k, nch, dt)] += 1
        if k not in for_kind or nch != 10 or "original" in b.lower() or off != 110 or chb != 36 or dbytes == 0 or len(h) < hdr + dbytes: continue
        slope = [struct.unpack("<f", h[off + i * chb:off + i * chb + 4])[0] for i in range(5)]; d = np.frombuffer(h[hdr:hdr + dbytes // (2 * nch) * 2 * nch], dtype="<i2") >> 2; d = d.reshape(-1, nch)
        rows.append(dict(sid="s" + mt.group(1), proto=for_kind[k], nch=nch, dt=dt, N=len(d), **{f"sl{i}": slope[i] for i in range(5)}, **{f"rg{i}": float(np.percentile(d[:, i], 99.5) - np.percentile(d[:, i], .5)) for i in range(5)}))
print("sampling interval census:"); [print(" ", k, v) for k, v in sorted(c.items(), key=str)]
R = pd.DataFrame(rows); print("files read with layout", R.groupby("proto").size().to_dict() if len(R) else 0)
if len(R):
    s1 = pd.read_pickle(sys.argv[2]); R = R.drop_duplicates(["sid", "proto"]).merge(s1[["sid", "proto", "gainL", "gainR", "spanL", "spanR", "lagxL", "lagxR", "stroke"]], on=["sid", "proto"])
    for p, x in R.groupby("proto"):
        print(p, "matched", len(x), "interval", x.dt.value_counts().to_dict(), "samples median", int(x.N.median()))
        for i, sd in ((3, "R"), (4, "L")):
            h = x[f"gain{sd}"] >= 3e4; print(f"  file channel {i} vs MCA {sd}: slope median lo/hi regime {x[~h][f'sl{i}'].median():.4g}/{x[h][f'sl{i}'].median():.4g}; distinct slopes {x[f'sl{i}'].round(5).nunique()}; raw counts range lo/hi {x[~h][f'rg{i}'].median():.0f}/{x[h][f'rg{i}'].median():.0f}; rank agreement with stored span {spearmanr(x[f'rg{i}']*x[f'sl{i}'].abs(), x[f'span{sd}'], nan_policy='omit')[0]:.2f}")
        print("  channel 2 (ABP) raw range lo/hi", round(x[x.gainL < 3e4].rg2.median()), round(x[x.gainL >= 3e4].rg2.median()))
