"""
ROBUSTNESS of the alpha/beta overview and the ranking test (JHK database). Seed 51 (no random draws are made).
PART 1: sensitivity to the CONVERSION of categorical ratings into probabilities (JHK fixes Solid = 0.99; Likely = 0.90; Lean = 0.75; Tossup = 0.50). Mappings: M0 = JHK (as in the file); M1 conservative (Solid .98, Likely .85, Lean .70, Tilt .55);
         M2 aggressive (Solid .999, Likely .95, Lean .85, Tilt .62); M3 EMPIRICAL excluding the evaluated cycle (win frequency by rating and office in the OTHER cycles, all categorical forecasters pooled, smoothed (k+1)/(n+2)).
         For categorical forecasters, recomputes BS, the ratio BS/BS_cal (under-confidence), alpha-beta, the ranking and the persistence of the ranking (test D) under each mapping.
PART 2: CYCLE-level inference (leave-one-cycle-out jackknife, CI with t on C-1 d.f.): (a) ratio BS/BS_cal by office (median across forecasters per cycle; models and categorical separately);
         (b) cycle ICC of alpha-beta (share of the between-forecaster variance of alpha-beta explained by the cycle) by office; (c) fraction of forecasters whose alpha-beta sign matches the cycle's majority sign.
Input: data/jhk/jhk_output.csv (self-contained; does not need other scripts). Outputs: outputs_local/us_robustness_conversion.csv and outputs_local/us_robustness_cycle.csv.
Run from the repository root: python code/us_robustness.py
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, t as tdist, kendalltau
import warnings; warnings.filterwarnings('ignore')
os.makedirs('outputs_local', exist_ok=True)
rng = np.random.default_rng(51); pd.set_option('display.width', 230, 'display.max_columns', 40)
d = pd.read_csv('data/jhk/jhk_output.csv'); EXC = {'jhk_plus', 'jhk_simple', 'leantoss', 'solid_purple'}; d = d[~d.forecast.isin(EXC)].copy()
nun = d.groupby('forecast').probwin.nunique(); CAT = set(nun[nun < 30].index); d['cat'] = d.forecast.isin(CAT); d['year'] = d.election_id.str[:4].astype(int)
d['y'] = 1 - d.probwin_outcome  # y = Democrat won
BASE = {'Solid R': 0.99, 'Likely R': 0.90, 'Lean R': 0.75, 'Tilt R': 0.60, 'Tossup': 0.5}
def fixed_map(sol, lik, lea, til):
    m = {'Solid R': sol, 'Likely R': lik, 'Lean R': lea, 'Tilt R': til, 'Tossup': 0.5}; m.update({k.replace(' R', ' D'): 1 - v for k, v in list(m.items()) if k.endswith(' R')}); return m
MAPS = {'M1 conservative': fixed_map(.98, .85, .70, .55), 'M2 aggressive': fixed_map(.999, .95, .85, .62)}
def pD(df, name, year_excl=None):
    if name == 'M0 JHK': return (1 - df.probwin).clip(0.0005, 0.9995)
    if name in MAPS: m = MAPS[name]; return df.rating.map(lambda r: 1 - m.get(r, np.nan)).clip(0.0005, 0.9995)
def empirical_map(year_excl):
    c = d[d.cat & (d.year != year_excl)]; g = c.groupby(['office', 'rating']).probwin_outcome.agg(['sum', 'size']); return ((g['sum'] + 1) / (g['size'] + 2)).to_dict()   # P(R wins | rating, office)
def metrics(g, p):
    y = g.y.values; pos, neg = y == 1, y == 0; a = ((1 - p[pos]) ** 2).mean(); b = (p[neg] ** 2).mean(); bs = ((y - p) ** 2).mean(); return dict(BS=bs, BBS=(a + b) / 2, alpha=a, beta=b, dif=a - b, ratio=bs / (p * (1 - p)).mean(), n=len(g))
def table(name):
    rows = []
    for (eid, fc), g in d.groupby(['election_id', 'forecast']):
        if len(g) < 30: continue
        if g.cat.iloc[0]:
            if name == 'M3 empirical': me = empirical_map(g.year.iloc[0]); p = np.array([1 - me.get((g.office.iloc[0], r), np.nan) for r in g.rating]).clip(0.0005, 0.9995)
            else: p = pD(g, name).values
            if np.isnan(p).any(): continue
        else: p = (1 - g.probwin).clip(0.0005, 0.9995).values
        rows.append(dict(election_id=eid, forecast=fc, office=g.office.iloc[0], year=g.year.iloc[0], cat=bool(g.cat.iloc[0]), mapping=name, **metrics(g, p)))
    return pd.DataFrame(rows)
T = pd.concat([table(n) for n in ['M0 JHK', 'M1 conservative', 'M2 aggressive', 'M3 empirical']], ignore_index=True); T.to_csv('outputs_local/us_robustness_conversion.csv', index=False)
print('=== PART 1: ratio BS/BS_cal (median across forecasters) by type, office and mapping; <1 = under-confident ===')
r = T.groupby(['mapping', 'cat', 'office']).ratio.median().unstack('mapping').round(2); r.index = r.index.set_levels(['model', 'categorical'], level='cat'); print(r.to_string())
print('\n=== PART 1b: median alpha-beta of categorical forecasters by cycle-office and mapping (cycle sign) ===')
c = T[T.cat].groupby(['election_id', 'mapping']).dif.median().unstack('mapping').round(3); print(c.to_string())
print('\n=== PART 1c: best forecaster by BS in each cycle-office, by mapping ===')
for mp in T.mapping.unique():
    x = T[T.mapping == mp]; w = []
    for eid, g in x.groupby('election_id'):
        g = g.sort_values('BS'); w.append(f"{eid}:{g.forecast.iloc[0]}" )
    print(f"  {mp:15s} best by BS: {', '.join(w)}")
print('\n=== PART 1d: ranking persistence by Brier score (mean Spearman across transitions; common forecasters only), by mapping ===')
res = []
for mp in T.mapping.unique():
    x = T[T.mapping == mp]
    for cg in ('house', 'senate', 'pres'):
        s = x[x.office == cg]; years = sorted(s.year.unique()); out = {}
        for m in ('BS', 'BBS'):
            P = s.pivot(index='forecast', columns='year', values=m); rs = []
            for a0, a1 in zip(years[:-1], years[1:]):
                z = P[[a0, a1]].dropna()
                if len(z) >= 4: rs.append(spearmanr(z[a0], z[a1])[0])
            out[m] = np.mean(rs) if rs else np.nan
        res.append(dict(mapping=mp, office=cg, BS=out['BS'], BBS=out['BBS']))
print(pd.DataFrame(res).pivot(index='office', columns='mapping', values='BS').round(2).to_string()); print('(BBS):'); print(pd.DataFrame(res).pivot(index='office', columns='mapping', values='BBS').round(2).to_string())
# ---------------- PART 2 ----------------
def jack(vals):
    v = np.array(vals, float); C = len(v); th = v.mean(); loo = np.array([np.delete(v, i).mean() for i in range(C)]); se = np.sqrt((C - 1) / C * ((loo - loo.mean()) ** 2).sum()); tc = tdist.ppf(0.975, C - 1); return th, se, th - tc * se, th + tc * se
def icc(df, col='dif', g='year'):
    k = df.groupby(g)[col]; n = k.size(); C = len(n); N = n.sum(); ms = k.mean(); gm = df[col].mean(); msb = (n * (ms - gm) ** 2).sum() / (C - 1); msw = ((df[col] - df[g].map(ms)) ** 2).sum() / (N - C); n0 = (N - (n ** 2).sum() / N) / (C - 1); su = max((msb - msw) / n0, 0); return su / (su + msw)
print('\n=== PART 2a: ratio BS/BS_cal by office: mean across cycles of the median across forecasters; per-cycle CI (jackknife, t) ===')
rows = []; B0 = T[T.mapping == 'M0 JHK']; B3 = T[T.mapping == 'M3 empirical']
for name, base in (('M0 JHK', B0), ('M3 empirical', B3)):
    for kind in (False, True):
        for cg in ('house', 'senate', 'pres'):
            s = base[(base.office == cg) & (base.cat == kind)]
            if s.empty: continue
            v = s.groupby('year').ratio.median().values; th, se, lo, hi = jack(v); rows.append(dict(mapping=name, type='categorical' if kind else 'model', office=cg, cycles=len(v), mean=th, lo=lo, hi=hi, per_cycle=np.round(v, 2).tolist()))
RA = pd.DataFrame(rows); print(RA.round(3).to_string(index=False))
print('\n=== PART 2b: cycle ICC of alpha-beta (between-forecaster variance explained by the cycle), per-cycle jackknife CI ===')
rows2 = []
for cg in ('house', 'senate', 'pres'):
    s = B0[B0.office == cg]; years = sorted(s.year.unique()); th = icc(s); loo = [icc(s[s.year != a]) for a in years]; C = len(years); se = np.sqrt((C - 1) / C * ((np.array(loo) - np.mean(loo)) ** 2).sum()); tc = tdist.ppf(0.975, C - 1)
    mj = s.groupby('year').dif.mean().round(3).to_dict(); sg = np.mean([(np.sign(g.dif) == np.sign(g.dif.median())).mean() for _, g in s.groupby('year')])
    rows2.append(dict(office=cg, cycles=C, forecaster_cycles=len(s), ICC=th, lo=max(th - tc * se, 0), hi=min(th + tc * se, 1), without_one_cycle=[round(x, 2) for x in loo], mean_dif_by_cycle=mj, sign_same_as_cycle=sg))
RB = pd.DataFrame(rows2); print(RB.round(3).to_string(index=False)); RA.to_csv('outputs_local/us_robustness_cycle.csv', index=False)
