# Inputs (not included)

The scripts read these files under `data/`. They are third-party data and are **not redistributed** here.

| Path | Source | SHA-256 | Used by |
|---|---|---|---|
| `jhk/jhk_output.csv` | JHK Forecasts, forecast-history database (https://projects.jhkforecasts.com/forecast-history); 33 forecasters, 2016–2024; no licence stated | `0834c4fc53e9ffc2e1d7f651a6c87fa606bb02db325c41ce42ea8db25a15a67e` | all scripts |
| `538/presidential_state_toplines_2020.csv` | FiveThirtyEight 2020 presidential state toplines (Wayback Machine capture) | `5883ad9b8cffa64d47eac37ca21129034fed0ff95d165bee3466a56341ab2005` | `us_alpha_beta_cal_evolution.py` |
| `538/forecast_results_2018.csv` | FiveThirtyEight 2018 House forecast results | `f06b04061ecf9af5996c433b9b9de3be2045ef615e53bfdf794ebae587e136a8` | `us_house_538_versions.py`, `us_alpha_beta_cal_evolution.py` |
| `538/house_2020_2022_extract.csv` | Compact extract of FiveThirtyEight's `house_district_toplines_2020.csv` and `_2022.csv` (Wayback Machine, captured 2023-04-27); the raw files were deleted after extraction | `61ab2b4e84973a684e3db7231f522007bfbaa51a6c40d698458632d0b3282af1` | `us_house_538_versions.py`, `us_alpha_beta_cal_evolution.py` |
| `rieke/win_state.csv` | Rieke 2024 presidential model (https://github.com/markjrieke/2024-potus) | `1fec8e099d6fbfb623c7635fe870f4db157324b8595e0733525a6f4693f7895d` | `us_alpha_beta_cal_evolution.py` |

The SHA-256 values above are those of the files used for the reported results; the files of the 2026 evaluation are hashed in the pre-registration repository. The comparison of the latent shock with the average poll error in the presidential races (main text, Section 5.5) uses a poll-error table that is not part of this repository.
