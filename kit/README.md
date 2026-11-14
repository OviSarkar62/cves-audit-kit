# CVES audit kit

Regenerates the audit's numbers from the public CVES v1.0.0 files. Plain: a one-command check of everything the signal and clinical sections report.

`sh run_all.sh /path/to/cves-1.0.0 /private/folder/outside/git` (Python 3.11 with numpy, scipy, pandas, statsmodels, scikit-learn, wfdb).

| Script | What it answers |
|---|---|
| `known_answers.py` | Does the pipeline recover answers we already know? |
| `step1_acquisition.py` | Clock, regime, timing, group link |
| `step3_clinical.py` | Clinical sets, intervals, Shapley, checks |
| `step4_signals.py`, `step4b_analysis.py` | Phase grid, regime by frequency, bias |
| `step4c_tests.py` | Reliability, signal increment, span-regime links; also prints a withdrawn CO₂ comparison that shouldn't be reported |
| `raw_clock_census.py` | Sampling interval and channel scale in each original lab file (slow: reads the 77 GB archive; takes the step 1 file) |
| `ledger.py` | Lists every number in a draft |

`ANALYSIS_LOCK.md` holds the fixed choices and logged deviations. All results are exploratory until regenerated from a frozen commit. Per-participant files stay in the private working folder.

`curves.py` saves group-level full curves. `step4d_draft_clock.py` shows the clock's effect on the draft's own contrast. `step5_robustness.py` holds the second-round checks: one regime definition, enrolment order, measured bias, timing-defined shortcut test, repeatability, lab-table scale, mains lines and n = 91 attribution.

`external_headers.py` checks same-laboratory datasets for clock errors, reading no group labels.
