# Replication code: probabilistic US election forecasts (2016–2024)

Code behind the analyses of the paper *What do α and β add to the Brier score of probabilistic election forecasts? Evidence from 33 US forecasters (2016–2024) and a prospective test for 2026*, and its supplement. The 2026 prospective evaluation is pre-registered separately (https://github.com/rcprati/us-preregistration).

**Data are not included.** All inputs are third-party data without a stated licence, used locally for research; see `data/README.md` for sources and the expected layout under `data/`. Outputs are written to `outputs_local/` (git-ignored).

## Scripts

Run from the repository root with `PYTHONPATH=code`. All scripts except `us_national_shock.py`, `us_figure_national_shock.py` and the ones marked below need only numpy, pandas, scipy and matplotlib; the null-model scripts take a few minutes each.

| Script | Content | Supplement | Seed |
|---|---|---|---|
| `us_jhk_alpha_beta.py` | α, β, Brier by forecaster, cycle and office | S1, S2 | — |
| `us_null_correlated.py` | correlated-error calibration null; writes `us_null_rho.csv` | S3 | 45 |
| `us_null_loco.py` | leave-one-cycle-out version | S3 | 46 |
| `us_figure_beta_alpha_plane.py` | Figure 1, the (β, α) plane (run `us_null_correlated.py` first) | main text | 47 |
| `us_robustness.py` | rating-conversion sensitivity and cycle-level jackknife | S4, S6 | — |
| `us_ranking_persistence.py` | persistence of forecaster rankings between cycles | S5 | 48 |
| `us_house_538_versions.py` | FiveThirtyEight House Lite, Classic, Deluxe | S7 | 50 |
| `us_alpha_beta_cal_viability.py` | what α_cal and β_cal predict before the outcome | S8 | 52 |
| `us_alpha_beta_cal_evolution.py` | within-cycle evolution of (α−β)_cal; Figure 3 | S9 | 53 |
| `us_national_shock.py` | latent national shock δ | main text 5.5 | 54 |
| `us_national_shock_bootstrap.py` | bootstrap over cycles of the RMSE differences (run `us_national_shock.py` first) | main text 5.5 | 55 |
| `us_figure_national_shock.py` | Figure 2 (run `us_national_shock.py` first) | main text | 54 |

`us_null_model.py` is the shared module of the null-model scripts. Each script was checked against the original (Portuguese) version used for the paper, comparing printed numbers and output tables; the figure scripts also reproduce the original figures byte for byte. The raw-file extraction step of `us_house_538_versions.py` (`--extract`) could not be re-run because the raw FiveThirtyEight files were deleted after extraction; the analysis step uses the saved extract.

## Notes

- The 2026 pre-registration scripts (`us_expected_splitticket_2026.py`, `us_reference_2024.py`, `us_reference_2026.py`, `us_splitticket_history.py`) are in the pre-registration repository, not here.
## Licence

The code, text and outputs written by the author are released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), the same licence as the pre-registration repository; the full legal text is in `LICENSE`. The licence does **not** cover third-party data (JHK Forecasts, FiveThirtyEight, Rieke), which are not part of this repository and keep their own terms.
