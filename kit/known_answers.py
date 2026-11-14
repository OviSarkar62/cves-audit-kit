"""Pipeline must pass these before any real result counts. Plain: feed it problems whose answers we already know."""
import numpy as np, sys
from core import *
rng = np.random.default_rng(11); ok = []
def check(name, cond, detail): ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + ": " + detail)
def pulse(fs, secs, rng):
    t = np.arange(int(secs * fs)) / fs; x = 90 + 10 * np.sin(2 * np.pi * .1 * t); beat = 0; y = np.zeros_like(t); per = 1.0; i = 0
    while i < len(t):
        n = int(per * fs); y[i:i + n] = np.exp(-np.arange(min(n, len(t) - i)) / (.15 * fs)) * 30; i += n; per = 1 + .1 * rng.standard_normal() * .5
    return x + y + rng.normal(0, .3, len(t))
for fs in (500, 1000):
    a = pulse(fs, 300, rng); v = np.r_[np.full(int(.3 * fs), a[0]), a[:-int(.3 * fs)]] * .5 + rng.normal(0, .1, len(a)); x, s = lag_xcorr(a, v, fs), lag_slope(a, v, fs)
    check(f"delay estimators recover 300 ms at {fs} Hz", abs(x - .3) <= 1 / fs and abs(s - .3) < .01, f"xcorr {x*1e3:.1f} ms, slope {s*1e3:.1f} ms")
# known phase: first-order high-pass with corner 0.05 Hz has phase atan(fc/f) lead
fc = .05; est = []; true = []
fsim = np.arange(.07, .2001, .01); true = np.degrees(np.arctan(fc / fsim)).mean()
for _ in range(60):
    w = rng.normal(0, 1, 3000 + 2000); b, c = sg.butter(1, fc, btype="high", fs=10); y = sg.lfilter(b, c, w)[2000:]; x = w[2000:]; x = x + 100; y = y + 50 + rng.normal(0, .02, len(y))
    est.append(tfa(x, y)["ph"])
check("known transfer phase within 1 degree", abs(np.nanmean(est) - true) < 1, f"true {true:.1f}, recovered {np.nanmean(est):.1f}")
# null: independent signals
for secs, label in ((300, "300 s"), (600, "600 s")):
    pb = []; pr = []; ps = []
    for _ in range(300 if secs == 300 else 150):
        x = sg.lfilter([1], [1, -.8], rng.normal(0, 1, secs * 10)) + 100; y = sg.lfilter([1], [1, -.8], rng.normal(0, 1, secs * 10)) + 100; r = tfa(x, y); pb.append(r["nb"] / r["nbins"]); pr.append(r["nb"] >= 2)
    print(f"INFO null {label}: per-bin false pass {np.mean(pb):.3f}; per-record pass (>=2 bins) {np.mean(pr):.3f}")
    check(f"null per-bin false pass near 5% ({label})", .02 <= np.mean(pb) <= .09, f"{np.mean(pb):.3f}")
# leakage guard: labels are not an input to any core function
sys.exit(0 if all(ok) else 1)
