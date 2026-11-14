#!/bin/sh
# Usage: sh run_all.sh /absolute/path/to/cves-1.0.0 /absolute/path/to/private-folder-outside-git
# Plain: runs every check in order and stops at the first failure, including a failed known-answer test.
# The working folder holds per-participant files. Keep it private and never share it. Reports contain aggregates only.
R="$1"; W="$2"; PY="${PYTHON:-python3}"
[ -f "$R/subjects.csv" ] || { echo "Stopped: no subjects.csv in '$R'. Give the absolute path to the CVES folder."; exit 1; }
[ -n "$W" ] || W="$(mktemp -d)"
mkdir -p "$W" || exit 1
if git -C "$W" rev-parse >/dev/null 2>&1; then echo "Stopped: the working folder is inside a git repository, so per-participant files could be committed."; exit 1; fi
cd "$(dirname "$0")" || exit 1
step() { echo "--- $1"; f="$W/$1.txt"; shift; "$@" > "$f" 2>&1; rc=$?; cat "$f"; [ $rc -eq 0 ] || { echo "Stopped: this step failed. See $f"; exit 1; }; }
echo "Working folder: $W"
step 0_known_answers $PY known_answers.py
step 1_acquisition $PY step1_acquisition.py "$R" "$W/step1.pkl"
step 3_clinical $PY step3_clinical.py "$R"
step 4_log $PY step4_signals.py "$R" "$W/step4.pkl"
step 4_grid $PY step4b_analysis.py "$R" "$W/step4.pkl"
step 5_tests $PY step4c_tests.py "$R" "$W/step4.pkl"
step 6_draft_clock $PY step4d_draft_clock.py "$R"
step 7_robustness $PY step5_robustness.py "$R" "$W/step1.pkl" "$W/step4.pkl"
step 8_curves $PY curves.py "$R" "$W/figS1_curves.csv"
step 9_additions $PY step6_additions.py "$R" "$W/step4.pkl"
echo "All steps passed. Optional and slow: $PY raw_clock_census.py $R/conversion/raw-data.tar.gz $W/step1.pkl"
