# CVES audit kit

Regenerates the audit's numbers from the public CVES v1.0.0 files. One command checks everything the signal and clinical sections report.

`sh run_all.sh /path/to/cves-1.0.0 /private/folder/outside/git`

Python 3.13.16 with the pinned versions in `requirements.txt` (numpy, scipy, pandas, statsmodels, scikit-learn, wfdb).

The known-answer test needs no dataset: `python3 known_answers.py` runs on its own and checks that the pipeline recovers an injected delay and a known transfer-function phase.

| Script | What it answers |
|---|---|
| `known_answers.py` | Does the pipeline recover answers we already know? (no data needed) |
| `known_counts.py` | Do the participant and recording counts match the dataset tables? |
| `step1_acquisition.py` | Clock, regime, timing, group link |
| `step3_clinical.py` | Clinical sets, intervals, Shapley, checks |
| `step4_signals.py`, `step4b_analysis.py` | Phase grid, regime by frequency, bias |
| `step4c_tests.py` | Reliability, signal increment, span-regime links; also prints a withdrawn CO₂ comparison that shouldn't be reported |
| `step4d_draft_clock.py` | What the clock did to the draft's own contrast (the earlier segment rule) |
| `step4d_main_segment_clock.py` | Same comparison on the paper's main segment (last 300 s of the first seated run), paired, with window counts |
| `step5_robustness.py` | Second-round checks: one regime definition, enrolment order, measured bias, timing-defined shortcut test, repeatability, lab-table scale, mains lines and n = 91 attribution |
| `step6_additions.py` | Gait summaries, coherence models and affected-hemisphere comparison |
| `step7_final_additions.py` | Robust (Huber) estimates at the main gate, regime-label increment to AUC, signal gaps, physiology by group and regime, seated-run durations |
| `step7b_version_compare.py` | Compares two reports folders from different library versions or platforms |
| `curves.py` | Saves group-level full curves |
| `external_headers.py` | Checks same-laboratory datasets for clock errors, reading no group labels |
| `raw_clock_census.py` | Sampling interval and channel scale in each original lab file (slow: reads the 77 GB archive; takes the step 1 file; optional) |
| `ledger.py` | Lists every number in a draft |

`ANALYSIS_LOCK.md` holds the fixed choices and logged deviations. All results are exploratory until regenerated from a frozen commit. Per-participant files stay in the private working folder, outside git.
