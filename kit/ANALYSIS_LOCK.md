# Analysis lock (not a pre-specification: results have been seen)

Fixed 2026-11-02 before the final rerun. Any later change is logged below as a deviation.

| Item | Value |
|---|---|
| Sit-stand-balance clock | 1000 Hz; tilt 500 Hz |
| Regime | ECG-anchored MCA delay by channel (peak of the R-locked mean, valid channels): split at the antimode between the two modes. Agreement with channel gain (3e4) reported |
| Outcome | Left LF phase, 0.07-0.20 Hz, degrees, stroke minus control |
| Segment, sit-stand-balance | Last 300 s of first seated run (force plate below half of record max, 1 s median) |
| Segment, tilt | 60-360 s; records with a marker inside excluded as sensitivity |
| TFA | Beat-to-beat means, linear interpolation 10 Hz, Welch 100 s Hann 50%, triangular smoothing, no detrend or filter |
| Gate | CARNet Table 2 main, at least 2 bins; surrogates (200) sensitivity |
| Alignment | Per-record cross-correlation and regime-offset, both reported, no hierarchy |
| Model | age + sex + EtCO2; HC3 with t reference; within-regime permutation (5000); no adjusted model in a cell below 5 per arm |
| Grid | clock x segment x gate x alignment x side, published whole; no share-significant summary |
| RQ1 | Descriptive; A/R/P; C = 1.0; refit out-of-bag bootstrap 2000; sensitivity on completed visits n = 91 |
| RQ2 clock | At least 90% of records: HR 40-110, breathing 6-30 per min at 1000 Hz; 0.18 cycles/sample line only in this directory |
| RQ2 instrumental | Within each group: ECG-ABP moves under 20 ms; ECG-MCA moves over 80 ms; same-person cross-session offset within 40 ms |
| RQ2 group link | Fisher OR CI excludes 1 |
| RQ3 | Difference only if both alignments' 95% CIs exclude 0 with the same sign under both gates; otherwise "not detected, precision +/-X" |
| RQ4 | Per protocol: added if refit CI above 0 and gain above 0.04 |
| RQ5 | Three links all required: signals predict regime; score vs regime within group; within-regime AUC |
| Reproduction tolerance | n exact; AUC +/-0.01; phase +/-1 degree; one reconciliation, then "unresolved" |

## Deviations
1. **2026-11-02, regime definition.** ECG-anchored MCA delay was not usably bimodal (agreement with gain 48-82%, right side unimodal). Replaced by robust amplitude span (99.5th minus 0.5th percentile, log10) by channel, split at the antimode; agrees with gain in 92-96%, and ABP-MCA lag, which validates it without being used to define it, agrees in 90-100%. Decided before any phase contrast under the lock was run.
2. **2026-11-02, instrumental rule applied, not changed.** ECG-ABP shift was 39 ms in sit-stand-balance controls (limit 20 ms; 5 high-regime records), and ECG-MCA shift was 80 ms there (limit: over 80 ms). The rule fails, so "instrumental" is dropped from the claim. Wording: "regime-locked timing offset, cause unconfirmed".
3. **2026-11-02, reconciliation (the one allowed).** The independent reimplementation matched on rates, lags and regime tables, but phase contrasts differed by up to about 11 degrees, mostly because trimming beats at the segment edge gave 2990 samples, 4 windows and a 0.41 critical value instead of 3000, 5 and 0.34. Beats are now detected up to 5 s beyond the edges on an exact 300 s grid. Any remaining gap above tolerance is reported as implementation sensitivity.
4. **2026-11-02, after cold review.** Permutations run at 2000, not 5000; ORs use Woolf intervals. Regime for all reporting is left span on valid channels (flat channels fall in the small-span regime by construction). Added: recording order, clock comparison on the draft's rule, measured regime bias, timing-defined regime for the shortcut test, reliability intervals, lab-table scale, marker sensitivity, n = 91 attribution. The CO2 comparison is withdrawn as a positive control: its windows confound CO2 with time and protocol state. RQ5's amplitude links are no longer judged against the span-defined regime, which would be circular.
5. **2026-11-02, final recheck.** Instrumental rule rerun on the span definition. Added a paired bootstrap of the clock's effect, with the same participants on both clocks.
