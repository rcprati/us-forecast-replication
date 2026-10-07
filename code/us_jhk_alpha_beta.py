"""
Overview of alpha, beta and Brier score by forecaster, cycle and office, using the JHK Forecasts database (data/jhk/jhk_output.csv; projects.jhkforecasts.com/forecast-history; 22,391 rows, 33 forecasters, 2016-2024).
probwin = P(Republican wins); probwin_outcome = 1 if the Republican won. Here p = P(Democrat wins) = 1 - probwin and y = 1 - probwin_outcome (alpha over Democratic wins, beta over Democratic losses).
'model' forecasters = >= 30 distinct probwin values; 'categorical' = ratings mapped to probabilities by a fixed JHK table (e.g. Solid R = 0.99; Likely = 0.90; Lean = 0.75; Tossup = 0.50). Descriptive, no CIs. No random numbers (no seed).
Input: data/jhk/jhk_output.csv. Output: outputs_local/us_jhk_alpha_beta.csv (a per forecaster-cycle table; not required by the other scripts).
Run from the repository root: python code/us_jhk_alpha_beta.py
"""
import os, numpy as np, pandas as pd
os.makedirs('outputs_local', exist_ok=True)
d = pd.read_csv('data/jhk/jhk_output.csv'); d['p'] = 1 - d.probwin; d['y'] = 1 - d.probwin_outcome
kind = (d.groupby('forecast').probwin.nunique() >= 30).map({True: 'model', False: 'categorical'})
def m(g):
    pos, neg = g.y == 1, g.y == 0; a = ((1 - g.p[pos]) ** 2).mean() if pos.any() else np.nan; b = (g.p[neg] ** 2).mean() if neg.any() else np.nan
    return pd.Series(dict(n=len(g), pi=g.y.mean(), p_mean=g.p.mean(), BS=((g.y - g.p) ** 2).mean(), alpha=a, beta=b, dif=a - b, BBS=(a + b) / 2, n_pos=int(pos.sum()), n_neg=int(neg.sum())))
R = d.groupby(['election_id', 'forecast']).apply(m).reset_index(); R['kind'] = R.forecast.map(kind); R.to_csv('outputs_local/us_jhk_alpha_beta.csv', index=False)
pd.set_option('display.width', 220, 'display.max_columns', 30, 'display.max_rows', 300)
sel = R[(R.kind == 'model') | R.forecast.isin(['cook', 'sabato', 'inside'])]; sel = sel[sel.n >= 30]
for eid in ['2018_house', '2020_house', '2022_house', '2024_house', '2016_pres', '2020_pres', '2024_pres', '2018_senate', '2020_senate', '2022_senate', '2024_senate']:
    g = sel[sel.election_id == eid].sort_values('BS'); print(f"\n== {eid} (pi_D = {g.pi.iloc[0]:.3f}, n = {int(g.n.iloc[0])}) ==")
    print(g[['forecast', 'kind', 'p_mean', 'BS', 'alpha', 'beta', 'dif', 'BBS']].round(4).to_string(index=False))
