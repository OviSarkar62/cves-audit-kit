"""Shared functions for the CVES measurement audit. Plain: turns raw pressure and brain-flow recordings into the numbers the paper reports,
using the corrected clock. Technical: CARNet-style TFA on beat-to-beat series; no record names ever leave this module's callers."""
import re
import numpy as np, wfdb
from pathlib import Path
from scipy import signal as sg
from scipy.ndimage import median_filter

CLOCK = {"sit-stand-balance": 1000, "head-up-tilt": 500, "sit-stand": 500}   # true Hz, not the header value
CRIT = {3:.51,4:.41,5:.34,6:.29,7:.25,8:.22,9:.20,10:.18,11:.17,12:.15,13:.14,14:.13,15:.12,20:.09,25:.08}  # CARNet Table 2, 95%

def crit(K):
    k = sorted(CRIT)
    if K <= k[0]: return CRIT[k[0]]
    if K >= k[-1]: return 1 - 0.05 ** (1 / (0.95 * K - 1))
    lo, hi = max(x for x in k if x <= K), min(x for x in k if x >= K)
    return CRIT[lo] if lo == hi else CRIT[lo] + (CRIT[hi] - CRIT[lo]) * (K - lo) / (hi - lo)

def records(root, proto): return sorted((Path(root) / "data" / proto).glob("*.hea"))
def pid(h): return h.stem[:5].lower()     # stays in memory; callers report aggregates only
def header_gain(h, ch):
    for ln in h.read_text(errors="ignore").splitlines()[1:]:
        t = ln.split()
        if t and t[-1] == ch: return float(re.split(r"[(/]", t[2])[0])
def valid(v): return bool(np.median(np.abs(v)) >= .05 and v.std() / max(abs(v.mean()), 1e-9) >= .10)
def peak_hz(x, fs, lo, hi, secs=256):
    x = np.nan_to_num(x); x = x - x.mean(); n = min(len(x), int(fs * secs)); f, p = sg.welch(x, fs=fs, nperseg=n, noverlap=n // 2); m = (f >= lo) & (f <= hi)
    return float(f[m][np.argmax(p[m])])
def lag_xcorr(a, v, fs):
    b, c = sg.butter(3, [.7, 10], btype="band", fs=fs); fa, fv = sg.filtfilt(b, c, a), sg.filtfilt(b, c, v)
    L = int(1.5 * fs); cc = sg.correlate(fv, fa, mode="full"); mid = len(fa) - 1
    return (np.argmax(cc[mid - L: mid + L + 1]) - L) / fs
def lag_slope(a, v, fs):
    f, pxy = sg.csd(a - a.mean(), v - v.mean(), fs=fs, nperseg=int(8 * fs), noverlap=int(4 * fs)); _, coh = sg.coherence(a, v, fs=fs, nperseg=int(8 * fs), noverlap=int(4 * fs))
    m = (f >= 1) & (f <= 5) & (coh >= .5)
    return -np.polyfit(f[m], np.unwrap(np.angle(pxy[m])), 1)[0] / (2 * np.pi) if m.sum() >= 5 else np.nan
def r_locked(ecg, y, fs, win=.6):
    """Delay from the ECG R peak to the peak of the R-locked mean of y, in ms; independent of the other channel."""
    R, _ = sg.find_peaks(ecg, distance=int(.4 * fs), prominence=np.std(ecg) * 1.5); w = int(win * fs); seg = [y[p:p + w] for p in R if p + w < len(y)]
    return np.argmax(np.mean(seg, 0)) / fs * 1e3 if len(seg) >= 30 else np.nan
def first_seated(fz, fs, secs=300):
    """Last `secs` of the first seated run: force plate below half its record maximum after 1 s median smoothing."""
    k = median_filter(np.nan_to_num(np.abs(fz))[::fs // 10], size=11); low = k < .5 * k.max(); s = None; L = secs * 10
    for i, x in enumerate(np.r_[low, False]):
        if x and s is None: s = i
        if not x and s is not None:
            if i - s >= 100: return (int(max(s, i - L) / 10 * fs), int(i / 10 * fs))
            s = None
    return None
def beats(a, v, fs):
    pk, _ = sg.find_peaks(a, distance=int(.4 * fs), prominence=np.std(a) * .5); t = pk[:-1] / fs
    ma = np.array([a[i:j].mean() for i, j in zip(pk[:-1], pk[1:])]); mv = np.array([v[i:j].mean() for i, j in zip(pk[:-1], pk[1:])]); tt = np.arange(t[0], t[-1], .1)
    return np.interp(tt, t, ma), np.interp(tt, t, mv)
def beats_seg(a, v, fs, s0, s1, pad=5):
    """Beat series on the exact segment grid. Beats are detected up to `pad` seconds beyond each edge, so 300 s gives 3000 samples and 5 windows."""
    lo, hi = max(0, s0 - pad * fs), min(len(a), s1 + pad * fs); A, V = a[lo:hi], v[lo:hi]; pk, _ = sg.find_peaks(A, distance=int(.4 * fs), prominence=np.std(A) * .5); t = (pk[:-1] + lo) / fs
    ma = np.array([A[i:j].mean() for i, j in zip(pk[:-1], pk[1:])]); mv = np.array([V[i:j].mean() for i, j in zip(pk[:-1], pk[1:])]); tt = s0 / fs + np.arange(int(round((s1 - s0) / fs * 10))) * .1
    return np.interp(tt, t, ma), np.interp(tt, t, mv)
_tri = lambda s: np.convolve(s, [.25, .5, .25], mode="same")
def spectra(a, v):
    n = 1000; kw = dict(fs=10, nperseg=n, noverlap=n // 2, window="hann", detrend=False); a, v = a - a.mean(), v - v.mean()
    f, pxy = sg.csd(a, v, **kw); _, pxx = sg.welch(a, **kw); _, pyy = sg.welch(v, **kw); pxy, pxx, pyy = _tri(pxy.real) + 1j * _tri(pxy.imag), _tri(pxx), _tri(pyy)
    return f, pxy, pxx, np.abs(pxy) ** 2 / (pxx * pyy)
def tfa(a, v, band=(.07, .20), rng=None, nsur=0):
    a, v = (a / a.mean() - 1) * 100, (v / v.mean() - 1) * 100; f, pxy, pxx, coh = spectra(a, v); m = (f >= band[0] - 1e-9) & (f <= band[1] + 1e-9); K = max(1, 1 + (len(a) - 1000) // 500)
    def est(keep):
        if keep.sum() < 2: return np.nan, np.nan
        return float(np.degrees(np.angle((pxy[m][keep] / np.abs(pxy[m][keep])).mean()))), float(np.abs(pxy[m][keep] / pxx[m][keep]).mean())
    out = dict(K=K, nbins=int(m.sum()), coh=float(coh[m].mean())); keep = coh[m] > crit(K); out["ph"], out["g"] = est(keep); out["nb"] = int(keep.sum())
    out["ph_all"] = est(np.ones(m.sum(), bool))[0]
    if nsur:
        sur = np.array([spectra(a, np.roll(v, rng.integers(200, len(v) - 200)))[3][m] for _ in range(nsur)]); keep = coh[m] > np.percentile(sur, 95, axis=0); out["ph_s"], out["g_s"] = est(keep); out["nb_s"] = int(keep.sum())
    return out
def shift(v, s, fs):
    """Positive s moves v earlier (removes a delay of s seconds)."""
    k = int(round(s * fs))
    return np.r_[v[k:], np.full(k, v[-1])] if k > 0 else (v.copy() if k == 0 else np.r_[np.full(-k, v[0]), v[:len(v) + k]])
