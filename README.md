# CVES audit kit

Code, aggregate outputs and notebooks for the manuscript *Clinical variables, recording artifacts and cerebral pressure-flow phase in chronic stroke: a reanalysis of the public CVES dataset* (Sarkar and Khaled).

The kit audits sampling rate, scale and timing in the public PhysioNet CVES recordings, computes low-frequency pressure-flow phase on the corrected clock under every analysis choice, and evaluates clinical variable sets and signal increments.

## Data (not included)
The recordings are **not** redistributed here. Download CVES v1.0.0 from PhysioNet (doi:10.13026/C2DW96, Open Data Commons Attribution License) and point the kit at the folder that contains `subjects.csv`.

## Layout
- `kit/` analysis scripts. `run_all.sh` runs every step in order and stops at the first failure, including a failed known-answer test. `step6_additions.py` adds the baseline table, cohort statuses, smallest detectable difference, enrolment-order check, blood-pressure variability, exploratory coherence and affected-hemisphere checks. `step7_final_additions.py` adds the robust-estimate summary, the regime-label increment with intervals, signal gaps and physiology by group and regime, and seated-run durations; `step7b_version_compare.py` compares two runs of the grid.
- `reports/` aggregate text output of each step (counts, medians, intervals only), plus `MANIFEST.json` and `ledger.csv`.
- `notebooks/` Colab notebooks that run the kit step by step. Outputs are cleared.
- `figures/` the figures used in the manuscript (one copy of each, including Figure 4). `figures_code/` the other figures drawn by the code.

## Run
```
sh kit/run_all.sh /absolute/path/to/cves-1.0.0 /absolute/path/to/private-folder-outside-git
```
The second argument is a working folder that holds per-participant intermediate files (`*.pkl`). Keep it outside this repository; it is listed in `.gitignore`. The notebooks default to a Google Drive layout; set `CVES_BASE` to change it (see `kit/helpers.py`, `config.example.json`).

## Environment (this release was produced with)
- Python 3.13.16 (the release run); install with `pip install -r requirements.txt`
- numpy 2.4.6
- scipy 1.16.3
- pandas 2.2.3
- statsmodels 0.15.0
- scikit-learn 1.6.1
- wfdb 4.3.1
- matplotlib 3.10.0
- nbformat 5.11.1

Re-running with other library versions can move individual phase estimates (the authors measured a median shift of a few tenths of a degree and up to about 4 degrees in one cell) and the n of a few cells, without changing any conclusion in the paper. See the manuscript for the exact comparison.

## Privacy
Outputs are aggregates only. No participant-level files, record names or recording dates are in this repository. Do not commit the working folder.

## Notes on AI assistance
Claude (Anthropic) was used to draft and test analysis code and to independently reimplement analyses as a cross-check. The authors ran every analysis and checked each reported number against its output.

## References
1. Novak V. Cerebral Vasoregulation in Elderly with Stroke (CVES), version 1.0.0. PhysioNet; 2018. doi:10.13026/C2DW96. Open Data Commons Attribution License.
2. Novak V, Hu K, Desrochers L, Novak P, Caplan L, Lipsitz L, Selim M. Cerebral flow velocities during daily activities depend on blood pressure in patients with chronic ischemic infarctions. Stroke 2010;41(1):61-66. doi:10.1161/STROKEAHA.109.565556. (The original publication that the dataset page asks users to cite.)
3. Goldberger AL, Amaral LAN, Glass L, Hausdorff JM, Ivanov PCh, Mark RG, Mietus JE, Moody GB, Peng C-K, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: components of a new research resource for complex physiologic signals. Circulation 2000;101(23):e215-e220.
4. Pollard T, Moody BE, Lehman L, Gow B, Fernandes C, Xie C, Johnson A, Mark RG, Heldt T. PhysioNet as a global platform for biomedical research. Nature Health 2026. doi:10.1038/s44360-026-00096-z.
5. Claassen JAHR, Meel-van den Abeelen ASS, Simpson D, Panerai RB; CARNet. Transfer function analysis of dynamic cerebral autoregulation: a white paper from the International Cerebral Autoregulation Research Network. J Cereb Blood Flow Metab 2016;36(4):665-680. doi:10.1177/0271678X15626425.
6. Panerai RB, et al. (18 authors; CARNet). Transfer function analysis of dynamic cerebral autoregulation: a CARNet white paper 2022 update. J Cereb Blood Flow Metab 2023;43(1):3-25. doi:10.1177/0271678X221119760.
7. Welch PD. The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. IEEE Trans Audio Electroacoust 1967;15(2):70-73.

## Licence
Code: MIT (`LICENSE`). Documentation and aggregate reports: CC BY 4.0.

## Citation
Archived releases: v1.0.1 https://doi.org/10.5281/zenodo.24451017. Every version, including v1.0.2, is listed under https://doi.org/10.5281/zenodo.24451016, which always resolves to the latest release. See `CITATION.cff`. Please also cite the dataset, the original publication (reference 2) and the PhysioNet platform paper (reference 4), as requested on the dataset page.
