# Inputs (not included)

The scripts read these files under `data/`. They are third-party data and are **not redistributed** here.

| Path | Source | SHA-256 | Used by |
|---|---|---|---|
| `jhk/jhk_output.csv` | JHK Forecasts, forecast-history database (https://projects.jhkforecasts.com/forecast-history); 33 forecasters, 2016–2024; no licence stated | `0834c4fc53e9ffc2e1d7f651a6c87fa606bb02db325c41ce42ea8db25a15a67e` | all scripts |
| `538/presidential_state_toplines_2020.csv` | FiveThirtyEight 2020 presidential state toplines (Wayback Machine capture) | `5883ad9b8cffa64d47eac37ca21129034fed0ff95d165bee3466a56341ab2005` | `us_alpha_beta_cal_evolution.py` |
| `538/forecast_results_2018.csv` | FiveThirtyEight 2018 House forecast results | `f06b04061ecf9af5996c433b9b9de3be2045ef615e53bfdf794ebae587e136a8` | `us_house_538_versions.py`, `us_alpha_beta_cal_evolution.py` |
| `538/house_2020_2022_extract.csv` | Compact extract of FiveThirtyEight's `house_district_toplines_2020.csv` and `_2022.csv` (Wayback Machine, captured 2023-04-27); the raw files were deleted after extraction | `61ab2b4e84973a684e3db7231f522007bfbaa51a6c40d698458632d0b3282af1` | `us_house_538_versions.py`, `us_alpha_beta_cal_evolution.py` |
| `rieke/win_state.csv` | Rieke 2024 presidential model (https://github.com/markjrieke/2024-potus) | `1fec8e099d6fbfb623c7635fe870f4db157324b8595e0733525a6f4693f7895d` | `us_alpha_beta_cal_evolution.py` |
| `validation/economist_model_output.zip` | The Economist, 2020 presidential model output (site data, capture of 2020-11-03) | `b8cce0da03ff2996b6e1d45aa884cf13d116ddb67277a022e7ef513ef8cf9e88` | `us_validate_jhk.py` |
| `forecasters/splitticket_house_2024_datawrapper.csv` | Split Ticket 2024 House forecast (chart data of the forecast page) | `5ba5cd63fdc34c54ad3544c67dd57d8e97ede3be816bc1f4a2cac1f22d43638d` | `us_validate_jhk.py` |
| `forecasters/splitticket_senate_2024_datawrapper.csv` | Split Ticket 2024 Senate forecast (chart data of the forecast page) | `db24d9b8d230526d3faace5bcd2c76066d341f17c4787dea6d2655f4c2c6f2a9` | `us_validate_jhk.py` |
| `538/raw_polls.csv` | FiveThirtyEight polls archive (`raw_polls.csv`, 1998–2022) | `9f0b185506e59117c150fb988e73b612346effe8e3a8f0d90fdafcad07f54756` | `us_poll_error.py` |
| `538/president_polls_2024_wayback.csv` | FiveThirtyEight 2024 `president_polls.csv` (Wayback Machine capture of 2024-12-02; only polls ended between 14 Oct and 4 Nov) | `d2fabd1b7450a801e55306043bf67a1d4d64426f30d3d9daf64078e49d8b24bb` | `us_polls_538_2024.py` |
| `results/election_results_presidential.csv` | Presidential results by state and candidate (2024 rows used) | `d4fed2115322ad70626e672bdf6072043174687c6cdda425fe4198027935a72a` | `us_poll_error.py` |

The SHA-256 values above are those of the files used for the reported results; the files of the 2026 evaluation are hashed in the pre-registration repository.
