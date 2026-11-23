"""Compares two runs of the same code (for example the published reports and a rerun on another computer or library version). Usage: step7b_version_compare.py REPORTS_A REPORTS_B. Aggregates only."""
import sys, re, numpy as np
from pathlib import Path
from step7_final_additions import parse_grid
a, b = Path(sys.argv[1]), Path(sys.argv[2]); A, B = parse_grid(a / "4_grid.txt"), parse_grid(b / "4_grid.txt"); key = ["proto", "seg", "sd", "gate", "arm"]
M = A.merge(B, on=key, suffixes=("_a", "_b")); M["d"] = (M.b_a - M.b_b).abs()
print("== version comparison, phase grid (stroke minus control, degrees)")
print("cells compared %d of %d and %d | estimate change: median %.2f, 90th percentile %.2f, largest %.2f | n differs in %d cells" % (len(M), len(A), len(B), M.d.median(), M.d.quantile(.9), M.d.max(), int((M.n_a != M.n_b).sum())))
mg = M[(M.gate == "CARNet") & M.seg.isin(["t60-360", "first300"])]
print("main gate: estimates run A %.1f to %.1f, run B %.1f to %.1f" % (mg.b_a.min(), mg.b_a.max(), mg.b_b.min(), mg.b_b.max()))
for f in ("9_additions.txt", "6_draft_clock.txt", "6b_main_segment_clock.txt"):
    for nm, d in (("A", a), ("B", b)):
        p = d / f
        if p.exists():
            for ln in p.read_text().splitlines():
                if re.search(r"affected side minus control|main gate, unaligned|^H_LF_un|^T_LF_un", ln): print(f, nm, ln.strip()[:170])
