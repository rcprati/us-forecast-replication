# Inputs (not included)

The scripts read these files under `data/`. They are third-party data and are **not redistributed** here.

| Path | Source | Used by |
|---|---|---|
| `jhk/jhk_output.csv` | JHK Forecasts, forecast-history database (https://projects.jhkforecasts.com/forecast-history); 33 forecasters, 2016–2024; no licence stated | all scripts |
| `538/presidential_state_toplines_2020.csv` | FiveThirtyEight 2020 presidential state toplines (Wayback Machine capture) | `us_alpha_beta_cal_evolution.py` |
| `538/forecast_results_2018.csv` | FiveThirtyEight 2018 House forecast results | `us_house_538_versions.py`, `us_alpha_beta_cal_evolution.py` |
| `538/house_2020_2022_extract.csv` | Compact extract of FiveThirtyEight's `house_district_toplines_2020.csv` and `_2022.csv` (Wayback Machine, captured 2023-04-27); the raw files were deleted after extraction | `us_house_538_versions.py`, `us_alpha_beta_cal_evolution.py` |
| `rieke/win_state.csv` | Rieke 2024 presidential model (https://github.com/markjrieke/2024-potus) | `us_alpha_beta_cal_evolution.py` |

SHA-256 values of the files used for the reported results are given in the paper's supplement (Section S11) and, for the 2026 evaluation, in the pre-registration repository.
