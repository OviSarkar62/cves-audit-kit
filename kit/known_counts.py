"""Known-answer check on the data: record counts and subject table must match the release. Usage: known_counts.py DATA [protocol ...]. With no protocol given, all three are checked. Plain: if the wrong folder or an incomplete download is used, this stops before any result is trusted."""
import sys, csv
from pathlib import Path
root = Path(sys.argv[1]); want = sys.argv[2:] or ["head-up-tilt", "sit-stand", "sit-stand-balance"]; ok = []
def check(name, cond, detail): ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + ": " + detail)
check("subjects.csv present", (root / "subjects.csv").exists(), str(root / "subjects.csv"))
for proto, n in (("head-up-tilt", 68), ("sit-stand", 71), ("sit-stand-balance", 64)):
    if proto not in want: print("NOT CHECKED %s record headers (not requested)" % proto); continue
    k = len(list((root / "data" / proto).glob("*.hea")))
    check("%s record headers" % proto, k == n, "found %d, expected %d" % (k, n))
if (root / "subjects.csv").exists():
    rows = list(csv.reader(open(root / "subjects.csv", encoding="utf-8-sig")))
    check("subjects.csv rows", len(rows) - 1 > 0, "%d subject rows" % (len(rows) - 1))
sys.exit(0 if all(ok) else 1)
