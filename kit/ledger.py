"""Lists every number in a draft with its section so nothing is missed in the rewrite. Convert first: pandoc draft.docx -t plain --wrap=none -o draft.txt. Usage: python ledger.py draft.txt ledger.csv. Stops at the reference list; masks four-digit years."""
import re, csv, sys
sec = "Front matter"; rows = []; num = re.compile(r"[−+-]?\d+(?:\.\d+)?(?:\s?×\s?10[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)?%?"); head = re.compile(r"^(?:\[Heading \d\]\s*)?(\d(?:\.\d+)?\.?\s+[A-Z].{0,70}|Abstract|References|Limitations|Conclusion)\s*$")
for i, ln in enumerate(open(sys.argv[1], encoding="utf-8"), 1):
    t0 = ln.strip(); m0 = head.match(t0)
    if m0:
        sec = m0.group(1)[:50]
        if sec.lower().startswith("references"): break
        continue
    body = re.sub(r"\[[\d,\s–-]+\]", " ", ln)
    for m in num.finditer(body):
        t = m.group(); y = bool(re.fullmatch(r"(19|20)\d\d", t)); rows.append((i, sec, "YEAR" if y else t, "[masked]" if y else body[max(0, m.start() - 45):m.end() + 25].replace("\n", " ").strip()))
w = csv.writer(open(sys.argv[2], "w", newline="", encoding="utf-8")); w.writerow(["line", "section", "value", "context", "status", "kit_report"]); [w.writerow(list(r) + ["", ""]) for r in rows]; print(len(rows), "numbers in", len({r[1] for r in rows}), "sections")
