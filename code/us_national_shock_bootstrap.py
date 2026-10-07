"""
Bootstrap over cycle-office pairs (12 cycles) of the differences in RMSE of the leave-one-forecaster-out prediction of alpha-beta (supplement and main text, Section 5.5): the delta estimated from the other forecasters (from their win counts or from their alpha-beta) against a cycle fixed effect, and the fixed effect against delta = 0. Seed 55; B = 10,000; 95% percentile intervals.
Input: outputs_local/us_national_shock_lofo.csv, written by us_national_shock.py (run it first). Run from the repository root.
"""
import numpy as np, pandas as pd
P = pd.read_csv('outputs_local/us_national_shock_lofo.csv'); rng = np.random.default_rng(55); B = 10000
for m in ('p0', 'p_win', 'p_dif', 'p_fe'): P['e_' + m] = (P.dif - P[m]) ** 2
def rmse(df, m): return np.sqrt(df['e_' + m].mean())
def boot(sub):
    ids = sub.eid.unique(); out = []
    for _ in range(B):
        d = pd.concat([sub[sub.eid == i] for i in rng.choice(ids, len(ids))]); out.append([rmse(d, 'p_win') - rmse(d, 'p_fe'), rmse(d, 'p_dif') - rmse(d, 'p_fe'), rmse(d, 'p_fe') - rmse(d, 'p0')])
    return np.array(out)
print('Delta RMSE (negative = the first predictor is better); 95% CI by bootstrap over cycle-office pairs\n')
for cg in ('all', 'house', 'senate', 'pres'):
    s = P if cg == 'all' else P[P.office == cg]; b = boot(s); n = s.eid.nunique()
    f = lambda k: f"{np.sqrt(s['e_p_win'].mean()) - np.sqrt(s['e_p_fe'].mean()) if k == 0 else (np.sqrt(s['e_p_dif'].mean()) - np.sqrt(s['e_p_fe'].mean()) if k == 1 else np.sqrt(s['e_p_fe'].mean()) - np.sqrt(s['e_p0'].mean())):+.4f} [{np.percentile(b[:, k], 2.5):+.4f}; {np.percentile(b[:, k], 97.5):+.4f}]"
    print(f"{cg:7s} ({n} cycles) | delta(wins) - FE: {f(0)} | delta(a-b) - FE: {f(1)} | FE - (delta=0): {f(2)}")
