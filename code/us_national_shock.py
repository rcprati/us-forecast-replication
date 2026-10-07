"""
alpha-beta read as a LATENT NATIONAL SHOCK delta (probit-scale units; delta > 0 = the Democrat does better than forecast). Descriptive. Requires numpy, pandas, scipy.
Model: given delta, y_i ~ Bernoulli(q_i(delta)), q_i = Phi(Phi^-1(p_i) + delta). Expected (ratio of expectations): alpha(delta) = sum q(1-p)^2 / sum q; beta(delta) = sum (1-q)p^2 / sum(1-q); dif(delta) = alpha(delta) - beta(delta), increasing in delta.
Estimators per forecaster-cycle: delta_dif (the delta that reproduces the observed alpha-beta) and delta_win (the delta that reproduces the NUMBER of Democratic wins, sum q(delta) = sum y; a calibration probit intercept).
Tests: (1) consistency across forecasters in the same cycle (cycle ICC of delta, median and dispersion per cycle); (2) prediction of each forecaster's alpha-beta with ONE delta per cycle estimated from the OTHER forecasters of the cycle (leave-one-forecaster-out), against (alpha-beta)_cal (delta = 0) and the mean of the others (cycle fixed effect).
Input: data/jhk/jhk_output.csv. Writes outputs_local/us_national_shock.csv and outputs_local/us_national_shock_lofo.csv. Run from the repository root.
"""
import os, numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from scipy.optimize import brentq
import warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 220, 'display.max_columns', 30); os.makedirs('outputs_local', exist_ok=True)
d = pd.read_csv('data/jhk/jhk_output.csv'); d = d[~d.forecast.isin({'jhk_plus', 'jhk_simple', 'leantoss', 'solid_purple'})].copy(); d['p'] = (1 - d.probwin).clip(0.0005, 0.9995); d['y'] = 1 - d.probwin_outcome
def dif_model(p, dl): q = norm.cdf(norm.ppf(p) + dl); return (q * (1 - p) ** 2).sum() / q.sum() - ((1 - q) * p ** 2).sum() / (1 - q).sum()
def root(f, lo=-3.0, hi=3.0):
    try: return brentq(f, lo, hi)
    except ValueError: return np.nan
rows = []
for (eid, fc), g in d.groupby(['election_id', 'forecast']):
    if len(g) < 30: continue
    p = g.p.values; y = g.y.values; pos, neg = y == 1, y == 0; dif = ((1 - p[pos]) ** 2).mean() - (p[neg] ** 2).mean(); z = norm.ppf(p)
    rows.append(dict(eid=eid, office=g.office.iloc[0], year=int(eid[:4]), fc=fc, n=len(g), dif=dif, difc=dif_model(p, 0.0), dhat_dif=root(lambda dl: dif_model(p, dl) - dif), dhat_win=root(lambda dl: norm.cdf(z + dl).sum() - y.sum()), pbar=p.mean(), wins=y.mean(), p=p))
R = pd.DataFrame(rows); print(len(R), 'forecaster-cycle-office rows; delta_dif undefined in', int(R.dhat_dif.isna().sum()), '; delta_win undefined in', int(R.dhat_win.isna().sum()))
print('\n=== (1) delta per cycle: median [q25; q75] across forecasters ===')
S = R.groupby('eid').agg(n=('fc', 'size'), dif_med=('dif', 'median'), dhat_dif=('dhat_dif', 'median'), dhat_dif_q25=('dhat_dif', lambda s: s.quantile(.25)), dhat_dif_q75=('dhat_dif', lambda s: s.quantile(.75)), dhat_win=('dhat_win', 'median'), dhat_win_q25=('dhat_win', lambda s: s.quantile(.25)), dhat_win_q75=('dhat_win', lambda s: s.quantile(.75)))
print(S.round(3).to_string()); print('\ncorrelation between delta_dif and delta_win (forecaster-cycles):', round(R[['dhat_dif', 'dhat_win']].dropna().corr().iloc[0, 1], 3), '| Spearman', round(spearmanr(R.dhat_dif, R.dhat_win, nan_policy='omit')[0], 3))
def icc(df, col):
    df = df.dropna(subset=[col]); k = df.groupby('eid')[col]; n = k.size(); C = len(n); N = n.sum(); ms = k.mean(); gm = df[col].mean(); msb = (n * (ms - gm) ** 2).sum() / (C - 1); msw = ((df[col] - df.eid.map(ms)) ** 2).sum() / (N - C); n0 = (N - (n ** 2).sum() / N) / (C - 1); su = max((msb - msw) / n0, 0); return su / (su + msw)
print('\n=== cycle ICC (share of between-forecaster variance explained by the cycle) ===')
for cg in ('house', 'senate', 'pres', 'all'):
    s = R if cg == 'all' else R[R.office == cg]; print(f"  {cg:7s}: alpha-beta {icc(s, 'dif'):.2f} | delta_dif {icc(s, 'dhat_dif'):.2f} | delta_win {icc(s, 'dhat_win'):.2f}")
print("\n=== (2) prediction of each forecaster's alpha-beta with ONE delta per cycle from the OTHER forecasters (leave-one-forecaster-out); RMSE ===")
res = []
for eid, g in R.groupby('eid'):
    for i, r in g.iterrows():
        o = g.drop(i)
        if len(o) < 2: continue
        dl = o.dhat_win.median(); dl2 = o.dhat_dif.median()
        res.append(dict(eid=eid, office=r.office, dif=r.dif, p0=r.difc, p_win=dif_model(r.p, dl) if np.isfinite(dl) else np.nan, p_dif=dif_model(r.p, dl2) if np.isfinite(dl2) else np.nan, p_fe=o.dif.mean()))
P = pd.DataFrame(res).dropna(); P.to_csv('outputs_local/us_national_shock_lofo.csv', index=False)
for cg in ('house', 'senate', 'pres', 'all'):
    s = P if cg == 'all' else P[P.office == cg]; print(f"  {cg:7s} n={len(s):3d} | RMSE: (a-b)_cal (delta=0) {np.sqrt(((s.dif - s.p0) ** 2).mean()):.4f} | cycle delta (others' wins) {np.sqrt(((s.dif - s.p_win) ** 2).mean()):.4f} | cycle delta (others' a-b) {np.sqrt(((s.dif - s.p_dif) ** 2).mean()):.4f} | mean of others (fixed effect) {np.sqrt(((s.dif - s.p_fe) ** 2).mean()):.4f} | R2 (wins) {1 - ((s.dif - s.p_win) ** 2).sum() / ((s.dif - s.dif.mean()) ** 2).sum():.2f}")
R.drop(columns='p').to_csv('outputs_local/us_national_shock.csv', index=False)
