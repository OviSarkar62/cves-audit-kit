# helpers.py - shared by every notebook in the CVES repair series (written by notebook 00).
# Plain: runs the analysis kit step by step, compares each report with the reference run,
# parses the reports into tables, and draws the figures. It never reads group labels itself.
import os, re, ast, sys, json, shutil, hashlib, platform, subprocess, time, zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("CVES_BASE", str(Path(__file__).resolve().parents[1])))  # default: the repository root
CFG_FILE = BASE / "config.json"
NUM = re.compile(r"[-+−]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", re.I)
REPORTS = ["0_known_answers", "1_acquisition", "3_clinical", "4_log", "4_grid", "5_tests", "6_draft_clock", "7_robustness", "8_curves"]


# ---------------------------------------------------------------- paths and config
def paths():
    return SimpleNamespace(base=BASE, kit=BASE / "kit", work=BASE / "private", reports=BASE / "reports",
                           ref=BASE / "reference", figs=BASE / "figures")


def save_cfg(**kw):
    BASE.mkdir(parents=True, exist_ok=True)
    CFG_FILE.write_text(json.dumps(kw, indent=2))


def setup(need_data=True):
    """Load config, put the kit on sys.path, make sure the dataset is present on the local disk."""
    p = paths()
    if not CFG_FILE.exists():
        raise SystemExit("Run notebook 00_Setup first (config.json is missing).")
    c = json.loads(CFG_FILE.read_text())
    for d in (p.work, p.reports, p.figs, p.ref):
        d.mkdir(parents=True, exist_ok=True)
    if str(p.kit) not in sys.path:
        sys.path.insert(0, str(p.kit))
    os.chdir(p.kit)
    cfg = SimpleNamespace(**vars(p), data=Path(c["data"]), drive_data=c.get("drive_data", ""), download=c.get("download", False))
    if need_data:
        ensure_data(cfg)
    print("Kit:", cfg.kit, "\nData:", cfg.data, "\nPrivate (never share):", cfg.work, "\nReports (aggregates only):", cfg.reports)
    return cfg


def ensure_data(cfg):
    if (cfg.data / "subjects.csv").exists():
        return
    print("Dataset is not on this runtime's disk yet. Restoring it ...")
    if cfg.drive_data:
        get_from_drive(cfg.drive_data, cfg.data)
    elif cfg.download:
        download_physionet(cfg.data)
    if not (cfg.data / "subjects.csv").exists():
        raise SystemExit("No subjects.csv under %s. Re-run notebook 00_Setup, section 4." % cfg.data)


def get_from_drive(src, dst):
    src = Path(src)
    dst = Path(dst)
    if src.suffix.lower() == ".zip":
        with zipfile.ZipFile(src) as z:
            z.extractall(dst.parent / "_unzip")
        root = next(iter(sorted((dst.parent / "_unzip").rglob("subjects.csv"))), None)
        if root is None:
            raise SystemExit("No subjects.csv inside the zip.")
        shutil.copytree(root.parent, dst, dirs_exist_ok=True)
        return
    skip_ext = {".gsheet", ".gdoc", ".gslides", ".gform", ".gdraw", ".gmap", ".gjam", ".gscript", ".tar.gz"}
    n_ok, skipped = 0, []
    for f in sorted(src.rglob("*")):
        rel = f.relative_to(src)
        try:
            if f.is_dir():
                (dst / rel).mkdir(parents=True, exist_ok=True)
                continue
        except OSError:
            continue
        if f.suffix.lower() in skip_ext or f.name.endswith(".tar.gz"):
            skipped.append(str(rel))
            continue
        out = dst / rel
        if out.exists() and out.stat().st_size == f.stat().st_size:
            n_ok += 1
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(f, out)
            n_ok += 1
        except OSError as e:
            skipped.append("%s (%s)" % (rel, e))
    print("Copied/kept %d files. Skipped %d: %s" % (n_ok, len(skipped), skipped[:8]))


def download_physionet(dst):
    # Skips the 77 GB raw archive; the kit does not need it.
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-q", "-r", "-N", "-c", "-np", "-nH", "--cut-dirs=3", "-e", "robots=off", "-R", "*.tar.gz",
           "-P", str(dst), "https://physionet.org/files/cves/1.0.0/"]
    print("Downloading (this can take a while):", " ".join(cmd))
    subprocess.run(cmd, check=False)


def check_data(cfg, expect={"head-up-tilt": 68, "sit-stand": 71, "sit-stand-balance": 64}):
    ok = (cfg.data / "subjects.csv").exists()
    print("subjects.csv:", "found" if ok else "MISSING")
    for proto, n in expect.items():
        k = len(list((cfg.data / "data" / proto).glob("*.hea")))
        print("  %-18s %3d records (the reference run had %d)  %s" % (proto, k, n, "OK" if k == n else "CHECK"))
        ok &= (k == n)
    return ok


def check_private_folder(cfg):
    r = subprocess.run(["git", "-C", str(cfg.work), "rev-parse"], capture_output=True, text=True)
    if r.returncode == 0:
        raise SystemExit("Stop: the private folder is inside a git repository. Choose another folder.")
    print("Private folder is not inside a git repository: OK")


# ---------------------------------------------------------------- the kit and the reference reports
def compile_kit(kit_dir):
    bad = []
    for f in sorted(Path(kit_dir).glob("*.py")):
        r = subprocess.run([sys.executable, "-m", "py_compile", str(f)], capture_output=True, text=True)
        if r.returncode:
            bad.append((f.name, r.stderr.strip().splitlines()[-1]))
    return bad


# ---------------------------------------------------------------- running a kit step
def run_step(cfg, name, script, *args):
    """Run one kit script, stream its output, save it as reports/<name>.txt. Stops at the first failure, like run_all.sh."""
    cmd = [sys.executable, "-u", script] + [str(a) for a in args]
    print("=== %s: %s" % (name, " ".join(cmd[2:])), flush=True)
    t0, lines = time.time(), []
    p = subprocess.Popen(cmd, cwd=str(cfg.kit), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    for ln in p.stdout:
        lines.append(ln)
        print(ln, end="")
    rc = p.wait()
    txt = "".join(lines)
    (cfg.reports / (name + ".txt")).write_text(txt, encoding="utf-8")
    print("--- %s finished in %.0f s (exit code %d); saved to %s" % (name, time.time() - t0, rc, cfg.reports / (name + ".txt")))
    if rc != 0:
        raise RuntimeError("Step %s failed. Read the last lines above; do not patch individual records." % name)
    return txt


def read_report(cfg, name):
    return (cfg.reports / (name + ".txt")).read_text(encoding="utf-8")


# ---------------------------------------------------------------- comparing with the reference run
def _skel(line):
    return NUM.sub("#", line).strip()


def _tok(line):
    return [t.replace("−", "-") for t in NUM.findall(line)]


def _close(a, b):
    if "." not in a and "." not in b and "e" not in a.lower() and "e" not in b.lower():
        return float(a) == float(b)
    x, y = float(a), float(b)
    tol = 0.011 if max(abs(x), abs(y)) < 1.5 else 1.0
    return abs(x - y) <= tol


def compare(cfg, name, show_ok=False):
    """Line-by-line comparison. Rule of thumb: counts exact, AUC +/-0.01, phase and timing +/-1."""
    rf = cfg.ref / (name + ".txt")
    if not rf.exists():
        print("No reference report for", name)
        return None
    ref = [l for l in rf.read_text(encoding="utf-8").splitlines() if l.strip()]
    mine = [l for l in read_report(cfg, name).splitlines() if l.strip()]
    pool = {}
    for l in mine:
        pool.setdefault(_skel(l), []).append(l)
    rows, ok, missing = [], 0, 0
    for l in ref:
        if not NUM.search(l):
            continue
        cand = pool.get(_skel(l))
        if not cand:
            missing += 1
            rows.append(dict(status="line differs or is missing", reference=l.strip()[:150], yours=""))
            continue
        m = cand.pop(0)
        a, b = _tok(l), _tok(m)
        bad = [(x, y) for x, y in zip(a, b) if not _close(x, y)]
        if bad:
            rows.append(dict(status="numbers differ", reference=l.strip()[:150], yours=m.strip()[:150]))
        else:
            ok += 1
            if show_ok:
                rows.append(dict(status="ok", reference=l.strip()[:150], yours=m.strip()[:150]))
    df = pd.DataFrame(rows)
    n_all = ok + len(df[df.status != "ok"]) if len(df) else ok
    print("%s: %d of %d numeric reference lines match within tolerance; %d differ or are missing." % (name, ok, n_all, n_all - ok))
    if len(df):
        pd.set_option("display.max_colwidth", 160, "display.width", 250)
        try:
            from IPython.display import display
            display(df[df.status != "ok"] if not show_ok else df)
        except Exception:
            print(df)
    else:
        print("All matching. Differences of a few degrees are expected only where the analysis notes say so (library versions, implementation).")
    return df


# ---------------------------------------------------------------- parsing the reports
def parse_np(s):
    return ast.literal_eval(re.sub(r"np\.float64\(([^)]*)\)", r"\1", s))


GRID = re.compile(r"^\s*(\S+)\s+(\S+)\s+([LR])\s+(CARNet|surrogate|all bins)\s+(un|pr|ro)\s+(\d+)\s+(\d+)\s+(\d+)" + r"\s+([-+]?\d+\.\d+)" * 7 + r"\s*$")


def parse_grid(text):
    rows = []
    for ln in text.splitlines():
        m = GRID.match(ln)
        if m:
            g = m.groups()
            rows.append(dict(proto=g[0], seg=g[1], side=g[2], gate=g[3], arm=g[4], n=int(g[5]), S=int(g[6]), C=int(g[7]),
                             b=float(g[8]), lo=float(g[9]), hi=float(g[10]), p=float(g[11]), perm=float(g[12]), rob=float(g[13]), lev=float(g[14])))
    d = pd.DataFrame(rows)
    if len(d):
        d["protocol"] = d.proto.map({"head": "head-up tilt", "sit-": "sit-stand-balance"}).fillna(d.proto)
        d["clock"] = np.where(d.seg == "HEADERCLOCK", "labelled 500 Hz", "corrected")
    return d


def clinical_tables(t3, t7):
    out = {}
    m = re.search(r"^point \(100 reps\): (\{.*\})$", t3, re.M)
    a = parse_np(m.group(1)) if m else {}
    m = re.search(r"^all completed visits n (\d+) stroke (\d+) gait missing \d+ (\{.*\})", t3, re.M)
    b = parse_np(m.group(3)) if m else {}
    out["auc"] = pd.DataFrame({"n = 64": pd.Series(a), "n = 91 (all completed visits)": pd.Series(b)}).loc[lambda x: x.index.isin(["R", "P", "A", "R+gait", "R+meds", "gait", "meds"])]
    sh = {}
    for m in re.finditer(r"^n=(\d+) stroke (\d+) (\{.*\})$", t7, re.M):
        for k, (v, ci) in parse_np(m.group(3)).items():
            sh.setdefault("n = " + m.group(1), {})[k] = "%.3f (%.3f to %.3f)" % (v, ci[0], ci[1])
    out["shapley"] = pd.DataFrame(sh)
    rows = []
    for m in re.finditer(r"^\s+(\w+)\s+beta ([-+]\d+\.\d+); 95% \[([-\d.]+), ([-\d.]+)\]; top-ranked (\d+)%; same sign (\d+)%", t3, re.M):
        rows.append(dict(variable=m.group(1), beta=float(m.group(2)), lo=float(m.group(3)), hi=float(m.group(4)), top_ranked_pct=int(m.group(5)), same_sign_pct=int(m.group(6))))
    out["coef"] = pd.DataFrame(rows)
    return out


# ---------------------------------------------------------------- figures
def _plt():
    import matplotlib
    matplotlib.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    import matplotlib.pyplot as plt
    return plt


def clock_figure(step1_pkl, out):
    """Heart rate and breathing: sit-stand-balance as labelled (500 Hz), as corrected (1000 Hz), and the supine tilt session."""
    plt = _plt()
    d = pd.read_pickle(step1_pkl)
    s, t = d[d.proto == "sit-stand-balance"], d[d.proto == "head-up-tilt"]
    cols = [("Seated, as labelled\n(500 Hz)", s.hr / 2, s.br / 2), ("Seated, corrected\n(1000 Hz)", s.hr, s.br), ("Supine session\n(head-up tilt)", t.hr, t.br)]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    rng = np.random.default_rng(0)
    for k, (title, lo, hi, idx) in enumerate([("Heart rate (per minute)", 40, 110, 1), ("Breathing rate (per minute)", 6, 30, 2)]):
        a = ax[k]
        a.axhspan(lo, hi, color="#2a9d8f", alpha=.12, label="plausible range")
        for i, c in enumerate(cols):
            y = c[idx].dropna().values
            a.boxplot([y], positions=[i], widths=.5, showfliers=False)
            a.scatter(i + rng.uniform(-.18, .18, len(y)), y, s=9, color="#264653", alpha=.6)
        a.set_xticks(range(3))
        a.set_xticklabels([c[0] for c in cols])
        a.set_title(title)
        a.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return fig


def clock_gate(step1_pkl):
    d = pd.read_pickle(step1_pkl)
    x = d[d.proto == "sit-stand-balance"]
    ok = x.hr.between(40, 110) & x.br.between(6, 30)
    share = ok.mean()
    print("sit-stand-balance at 1000 Hz: %d of %d records pass the plausibility gate (%.0f%%). The kit requires at least 90%%." % (ok.sum(), len(x), 100 * share))
    return share >= .9


def grid_forest(g, out):
    plt = _plt()
    g = g.copy()
    g["label"] = g.protocol.str.replace("sit-stand-balance", "seated") + " " + g.seg + " | " + g.arm
    order = list(dict.fromkeys(g.label))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharey=True, sharex=True)
    col = {"un": "#264653", "pr": "#e76f51", "ro": "#2a9d8f"}
    for i, side in enumerate("LR"):
        for j, gate in enumerate(["CARNet", "surrogate", "all bins"]):
            a = axes[i, j]
            x = g[(g.side == side) & (g.gate == gate)]
            for _, r in x.iterrows():
                y = order.index(r.label)
                a.plot([r.lo, r.hi], [y, y], color=col[r.arm], lw=1.6, alpha=.5 if r.seg == "HEADERCLOCK" else 1, ls="--" if r.seg == "HEADERCLOCK" else "-")
                a.plot(r.b, y, "o", color=col[r.arm], ms=5)
            a.axvline(0, color="grey", lw=.8)
            a.set_title("%s channel, %s gate" % ("Left" if side == "L" else "Right", gate))
            a.set_yticks(range(len(order)))
            a.set_yticklabels(order, fontsize=7)
            a.set_xlabel("stroke minus control (degrees)")
    axes[0, 0].invert_yaxis()
    fig.suptitle("Phase grid: every clock x segment x gate x alignment x side. Dashed = labelled (wrong) clock. un = unaligned, pr = per-record, ro = regime-offset.", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return fig


def curves_figure(csv, out):
    plt = _plt()
    c = pd.read_csv(csv)
    fig, ax = plt.subplots(3, 2, figsize=(11, 9), sharex=True)
    for j, proto in enumerate(["head-up-tilt", "sit-stand-balance"]):
        for i, (v, lab) in enumerate([("phase", "Phase (degrees)"), ("gain", "Gain (%/%)"), ("coh", "Coherence")]):
            for grp, x in c[c.proto == proto].groupby("group"):
                ax[i, j].plot(x.f, x[v], label=str(grp))
            ax[i, j].axvspan(.07, .20, color="grey", alpha=.12)
            ax[i, j].set_ylabel(lab)
        ax[0, j].set_title(proto)
        ax[2, j].set_xlabel("Frequency (Hz); grey = low-frequency band 0.07-0.20 Hz")
    ax[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return fig


def coef_figure(coef, out):
    plt = _plt()
    c = coef.sort_values("beta")
    fig, a = plt.subplots(figsize=(6, 5))
    a.errorbar(c.beta, range(len(c)), xerr=[c.beta - c.lo, c.hi - c.beta], fmt="o", color="#264653", ecolor="#8ab", capsize=2)
    a.axvline(0, color="grey", lw=.8)
    a.set_yticks(range(len(c)))
    a.set_yticklabels(c.variable)
    a.set_xlabel("standardised coefficient (bootstrap 95% interval)")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    return fig


# ---------------------------------------------------------------- freezing
def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def manifest(cfg):
    import importlib.metadata as md
    pk = {}
    for n in ["numpy", "scipy", "pandas", "statsmodels", "scikit-learn", "wfdb", "matplotlib"]:
        try:
            pk[n] = md.version(n)
        except Exception:
            pk[n] = "not installed"
    m = dict(date=time.strftime("%Y-%m-%d %H:%M:%S"), python=platform.python_version(), packages=pk,
             kit={f.name: sha256(f) for f in sorted(cfg.kit.glob("*")) if f.is_file()},
             reports={f.name: sha256(f) for f in sorted(cfg.reports.glob("*.txt"))},
             figures={f.name: sha256(f) for f in sorted(cfg.figs.glob("*")) if f.is_file()})
    (cfg.reports / "MANIFEST.json").write_text(json.dumps(m, indent=2))
    return m


def package(cfg, out):
    """Zip the kit, reports and figures only. Per-participant files (pkl) and the dataset are never included."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, pat in ((cfg.kit, "*"), (cfg.reports, "*"), (cfg.figs, "*")):
            for f in sorted(folder.glob(pat)):
                if f.is_file() and f.suffix not in (".pkl", ".pyc"):
                    z.write(f, folder.name + "/" + f.name)
    return out


# ---------------------------------------------------------------- small readers
def section(text, key):
    """Lines of the report block that starts with '== <key...>' up to the next '==' line."""
    out, on = [], False
    for ln in text.splitlines():
        if ln.startswith("=="):
            on = ln[2:].strip().lower().startswith(key.lower())
            if on:
                out.append(ln)
            continue
        if on:
            out.append(ln)
    return "\n".join(out)


def grep(text, pattern):
    return "\n".join(l for l in text.splitlines() if re.search(pattern, l))


DRAFT = re.compile(r"^(\S+) n (\d+) \((\d+)/(\d+)\) ([+-]\d+\.\d+) \((-?\d+\.\d+), (-?\d+\.\d+)\) p (\S+); median phase (\S+)", re.M)


def core_exhibit(t6):
    """The core exhibit: same samples, labelled clock vs correct clock (from 6_draft_clock)."""
    name = {"H_LF": "Header clock (500 Hz, as the draft used), 0.07-0.20 Hz", "T_LF": "Correct clock (1000 Hz), true low-frequency band 0.07-0.20 Hz",
            "T_0.14-0.40": "Correct clock, the band the draft really measured (0.14-0.40 Hz)"}
    arm = {"un": "unaligned", "all": "unaligned, all bins", "al": "aligned"}
    rows = []
    for m in DRAFT.finditer(t6):
        k, a = m.group(1).rsplit("_", 1)
        rows.append(dict(analysis=name.get(k, k), gate_or_arm=arm.get(a, a), n="%s (%s/%s)" % m.group(2, 3, 4), stroke_minus_control_deg=float(m.group(5)),
                         ci=(float(m.group(6)), float(m.group(7))), p=m.group(8), median_phase_deg=float(m.group(9))))
    return pd.DataFrame(rows)


INCR = re.compile(r"^(\S+) (\w+) (A|R) n (\d+) S (\d+) base ([\d.]+) plus ([\d.]+) delta ([-+][\d.]+) CI \[([-\d.]+), ([-\d.]+)\]", re.M)


def increment_rule(t5):
    """RQ4 rule from the analysis lock: an increment counts only if the refit CI is above 0 and the gain exceeds 0.04."""
    rows = []
    for m in INCR.finditer(t5):
        d, lo, hi = float(m.group(8)), float(m.group(9)), float(m.group(10))
        rows.append(dict(protocol=m.group(1), alignment=m.group(2), set=m.group(3), n=int(m.group(4)), base=float(m.group(6)), plus=float(m.group(7)),
                         delta=d, ci_lo=lo, ci_hi=hi, counts_as_increment=bool(lo > 0 and d > 0.04)))
    return pd.DataFrame(rows)


def rq3_rule(g):
    """RQ3 rule: claim a difference only if both alignments (per-record and regime-offset) exclude 0 with the same sign under both gates."""
    g = g[g.clock == "corrected"]
    rows = []
    for (pr, seg, sd), x in g.groupby(["protocol", "seg", "side"]):
        res = {}
        for gate in ("CARNet", "all bins"):
            y = x[(x.gate == gate) & x.arm.isin(["pr", "ro"])]
            sign = [np.sign(r.b) if (r.hi < 0 or r.lo > 0) else 0 for _, r in y.iterrows()]
            res[gate] = len(sign) == 2 and sign[0] != 0 and sign[0] == sign[1]
        rows.append(dict(protocol=pr, segment=seg, side=sd, main_gate=res["CARNet"], all_bins=res["all bins"], difference_claimable=res["CARNet"] and res["all bins"]))
    return pd.DataFrame(rows)
