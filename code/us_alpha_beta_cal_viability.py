"""
Viability of alpha_cal and beta_cal as a contribution of the article (JHK, 33 forecasters, 5 cycles, M0). Seed 52. Descriptive.
(1) Within cycle: slope and R2 of alpha_obs on alpha_cal (and beta) across forecasters (calibrated => slope 1 if the forecaster is calibrated); with cycle fixed effect.
(2) Between cycles: correlation between the per-cycle mean of (alpha_cal, beta_cal, dif_cal) and of the observed (alpha, beta, dif).
(3) Out-of-sample prediction of a forecaster's alpha, beta and alpha-beta in a cycle from earlier data: (a) alpha_cal of the cycle itself (computable BEFORE the result); (b) the same forecaster's observed value in the previous cycle (persistence);
    (c) alpha_cal x calibration factor (obs/cal ratio of the same forecaster's previous cycle); (d) office mean in the previous cycle (all forecasters); (e) the global mean of alpha (constant). RMSE and error relative to the best reference.
(4) Sign: P(sign of (alpha-beta)_obs = sign of (alpha-beta)_cal), against the base rate (most common sign in the office).
Input: data/jhk/jhk_output.csv. Output: printed to stdout only (no files written). Run from the repository root: python code/us_alpha_beta_cal_viability.py
"""
import numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
rng = np.random.default_rng(52); pd.set_option('display.width', 220, 'display.max_columns', 30)
d = pd.read_csv('data/jhk/jhk_output.csv'); d = d[~d.forecast.isin({'jhk_plus', 'jhk_simple', 'leantoss', 'solid_purple'})]; d['p'] = (1 - d.probwin).clip(0.0005, 0.9995); d['y'] = 1 - d.probwin_outcome
nun = d.groupby('forecast').probwin.nunique(); models = set(nun[nun >= 30].index)
rows = []
for (eid, fc), g in d.groupby(['election_id', 'forecast']):
    if len(g) < 30: continue
    p = g.p.values; y = g.y.values; pos, neg = y == 1, y == 0; a = ((1 - p[pos]) ** 2).mean(); b = (p[neg] ** 2).mean(); ac = (p * (1 - p) ** 2).sum() / p.sum(); bc = ((1 - p) * p ** 2).sum() / (1 - p).sum()
    rows.append(dict(eid=eid, office=g.office.iloc[0], year=int(eid[:4]), fc=fc, type='model' if fc in models else 'categorical', a=a, b=b, dif=a - b, ac=ac, bc=bc, difc=ac - bc, pi=y.mean()))
R = pd.DataFrame(rows); print(len(R), 'forecaster-cycle-office rows')
print('\n=== (1) within cycle: slope (alpha_obs on alpha_cal; beta_obs on beta_cal; dif on dif_cal) and R2 with cycle fixed effect ===')
for cg in ('house', 'senate', 'pres'):
    s = R[R.office == cg]; out = {}
    for name, o, c in (('alpha', 'a', 'ac'), ('beta', 'b', 'bc'), ('alpha-beta', 'dif', 'difc')):
        x = s[c] - s.groupby('eid')[c].transform('mean'); y_ = s[o] - s.groupby('eid')[o].transform('mean'); sl = (x * y_).sum() / (x ** 2).sum(); r2 = np.corrcoef(x, y_)[0, 1] ** 2; out[name] = (round(sl, 2), round(r2, 2))
    print(f"  {cg:7s} n={len(s):2d} | (slope, within-cycle R2): {out}")
print('\n=== (2) between cycles: correlation between per-cycle means (cal vs obs) ===')
for cg in ('house', 'senate', 'pres'):
    m = R[R.office == cg].groupby('year')[['a', 'b', 'dif', 'ac', 'bc', 'difc']].mean(); print(f"  {cg:7s} ({len(m)} cycles): corr alpha = {m.a.corr(m.ac):+.2f}, beta = {m.b.corr(m.bc):+.2f}, alpha-beta = {m.dif.corr(m.difc):+.2f}")
mm = R.groupby(['office', 'year'])[['a', 'b', 'dif', 'ac', 'bc', 'difc']].mean(); print(f"  all office-cycles ({len(mm)}): corr alpha = {mm.a.corr(mm.ac):+.2f}, beta = {mm.b.corr(mm.bc):+.2f}, alpha-beta = {mm.dif.corr(mm.difc):+.2f}")
print('\n=== (3) out-of-sample prediction (RMSE; forecaster-cycles with a previous cycle of the same forecaster and office) ===')
R = R.sort_values(['office', 'fc', 'year']); R['a_prev'] = R.groupby(['office', 'fc']).a.shift(); R['b_prev'] = R.groupby(['office', 'fc']).b.shift(); R['dif_prev'] = R.groupby(['office', 'fc']).dif.shift(); R['ac_prev'] = R.groupby(['office', 'fc']).ac.shift(); R['bc_prev'] = R.groupby(['office', 'fc']).bc.shift()
years_prev = {c: sorted(R[R.office == c].year.unique()) for c in R.office.unique()}
def prev_year(c, a): l = years_prev[c]; i = l.index(a); return l[i - 1] if i > 0 else None
R['pa'] = [prev_year(c, a) for c, a in zip(R.office, R.year)]; med = R.groupby(['office', 'year'])[['a', 'b', 'dif']].mean()
R['a_office_prev'] = [med.loc[(c, pa), 'a'] if pd.notna(pa) else np.nan for c, pa in zip(R.office, R.pa)]; R['b_office_prev'] = [med.loc[(c, pa), 'b'] if pd.notna(pa) else np.nan for c, pa in zip(R.office, R.pa)]; R['dif_office_prev'] = [med.loc[(c, pa), 'dif'] if pd.notna(pa) else np.nan for c, pa in zip(R.office, R.pa)]
T = R.dropna(subset=['a_prev']).copy(); print(f"  {len(T)} forecaster-cycles with a previous cycle")
lin = {}
for name, o, c, pr, cp, cg_ in (('alpha', 'a', 'ac', 'a_prev', 'ac_prev', 'a_office_prev'), ('beta', 'b', 'bc', 'b_prev', 'bc_prev', 'b_office_prev')):
    factor = (T[pr] / T[cp]).clip(0.1, 10); preds = {f'(a) {name}_cal of the cycle itself': T[c], '(b) observed in the previous cycle (persistence)': T[pr], f'(c) {name}_cal x previous-cycle factor': T[c] * factor, '(d) office mean in the previous cycle': T[cg_], '(e) constant (global office mean)': T.groupby('office')[o].transform('mean')}
    print(f"  --- {name} (observed mean {T[o].mean():.4f}) ---")
    for k, v in preds.items(): print(f"     {k:50s} RMSE = {np.sqrt(((T[o] - v) ** 2).mean()):.4f}  | mean error {np.mean(v - T[o]):+.4f}")
fat = (T.dif_prev); preds = {'(a) (alpha-beta)_cal': T.difc, '(b) persistence': T.dif_prev, '(d) office mean in the previous cycle': T.dif_office_prev, '(e) 0': 0 * T.dif}
print(f"  --- alpha-beta (observed mean {T.dif.mean():+.4f}) ---")
for k, v in preds.items(): print(f"     {k:50s} RMSE = {np.sqrt(((T.dif - v) ** 2).mean()):.4f}")
print('\n=== (4) sign of alpha-beta: (alpha-beta)_cal against the base rate ===')
for cg in ('house', 'senate', 'pres', 'all'):
    s = R if cg == 'all' else R[R.office == cg]; hit = (np.sign(s.dif) == np.sign(s.difc)).mean(); base = max((s.dif > 0).mean(), (s.dif < 0).mean()); sp = (np.sign(s.dif) == np.sign(s.dif_prev.fillna(0))).mean() if 'dif_prev' in s else np.nan
    print(f"  {cg:7s} n={len(s):3d}: P(obs sign = cal sign) = {hit:.2f} | base rate (most common sign) = {base:.2f} | mean alpha-beta_cal {s.difc.mean():+.4f}, observed {s.dif.mean():+.4f}")
print('\n=== shift obs - cal by office and type (median) ===')
R['da'] = R.a - R.ac; R['db'] = R.b - R.bc; print(R.groupby(['office', 'type'])[['da', 'db']].median().round(4).to_string())
