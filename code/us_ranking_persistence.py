"""
RANKING TEST D with the JHK database: does the ranking of forecasters in one cycle, by Brier score (BS) or balanced Brier score (BBS) = (alpha+beta)/2, predict the ranking in the next cycle? Seed 48.
Forecasters: >= 4 common to consecutive cycles of the same office; JHK clones (jhk_plus, jhk_simple) and extreme forecasters (leantoss, solid_purple) excluded. Offices: House (2018, 2020, 2022, 2024), Senate (2016-2024), president (2016, 2020, 2024).
Ranking metrics (lower = better position): BS, BBS, ratio BS/BS_cal (confidence: below 1 = under-confident) and |alpha - beta| (asymmetry between classes).
Statistics: Spearman between the ranking at t and at t+1 (same metric and crossed BBS_t -> BS_{t+1}); PERMUTATION test (B = 20,000) of the mean Spearman across an office's transitions; hit rate of the best forecaster; BS vs BBS agreement within the cycle (Kendall tau).
Also the 'historical' predictor: mean ranking in earlier cycles against the ranking of the current cycle. Descriptive; few forecasters and transitions.
Input: data/jhk/jhk_output.csv (self-contained; does not need other scripts). Outputs: outputs_local/us_ranking_persistence.csv and outputs_local/us_ranking_persistence_summary.csv.
Run from the repository root: python code/us_ranking_persistence.py
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, kendalltau, rankdata
import warnings; warnings.filterwarnings('ignore')
os.makedirs('outputs_local', exist_ok=True)
rng = np.random.default_rng(48); pd.set_option('display.width', 220, 'display.max_columns', 30)
d = pd.read_csv('data/jhk/jhk_output.csv'); d['p'] = (1 - d.probwin).clip(0.0005, 0.9995); d['y'] = 1 - d.probwin_outcome
EXC = {'jhk_plus', 'jhk_simple', 'leantoss', 'solid_purple'}; d = d[~d.forecast.isin(EXC)]
def met(g):
    pos, neg = g.y == 1, g.y == 0; a = ((1 - g.p[pos]) ** 2).mean(); b = (g.p[neg] ** 2).mean(); bs = ((g.y - g.p) ** 2).mean(); return pd.Series(dict(BS=bs, BBS=(a + b) / 2, ratio=bs / (g.p * (1 - g.p)).mean(), asym=abs(a - b), n=len(g)))
M = d.groupby(['election_id', 'forecast']).apply(met).reset_index(); M = M[M.n >= 30]; M['office'] = M.election_id.str.split('_').str[1]; M['year'] = M.election_id.str[:4].astype(int)
METS = ['BS', 'BBS', 'ratio', 'asym']; rows = []; perm_in = {}
for cg in ('house', 'senate', 'pres'):
    years = sorted(M[M.office == cg].year.unique()); P = {m: M[M.office == cg].pivot(index='forecast', columns='year', values=m) for m in METS}
    print(f"\n=== {cg}: cycles {years} ===")
    # BS vs BBS agreement within the cycle
    for a in years:
        x = P['BS'][a].dropna(); y = P['BBS'][a].dropna(); com = x.index.intersection(y.index); tau = kendalltau(x[com], y[com])[0]; best = (x.idxmin(), y.idxmin())
        print(f"  {a}: n={len(com)} forecasters | tau(BS, BBS)={tau:.2f} | best by BS = {best[0]}, by BBS = {best[1]}")
    for a0, a1 in zip(years[:-1], years[1:]):
        for m in METS:
            x = P[m][[a0, a1]].dropna()
            if len(x) < 4: continue
            r = spearmanr(x[a0], x[a1])[0]; top = x[a0].idxmin() == x[a1].idxmin(); top2 = x[a0].idxmin() in x[a1].nsmallest(2).index
            rows.append(dict(office=cg, from_year=a0, to_year=a1, metric=m, n=len(x), spearman=r, same_best=top, best_top2=top2))
        x = pd.concat([P['BBS'][a0], P['BS'][a1]], axis=1, keys=['a', 'b']).dropna()
        if len(x) >= 4: rows.append(dict(office=cg, from_year=a0, to_year=a1, metric='BBS->BS', n=len(x), spearman=spearmanr(x.a, x.b)[0], same_best=x.a.idxmin() == x.b.idxmin(), best_top2=x.a.idxmin() in x.b.nsmallest(2).index))
R = pd.DataFrame(rows); R.to_csv('outputs_local/us_ranking_persistence.csv', index=False)
print('\n=== Spearman between the ranking at t and at t+1, by transition ==='); print(R.pivot_table(index=['office', 'from_year', 'to_year'], columns='metric', values='spearman').round(2).to_string())
print('\n=== mean Spearman across transitions and permutation p (H0: no persistence) ===')
out = []
for cg in ('house', 'senate', 'pres'):
    for m in METS + ['BBS->BS']:
        x = R[(R.office == cg) & (R.metric == m)]
        if len(x) == 0: continue
        obs = x.spearman.mean(); years = sorted(M[M.office == cg].year.unique()); null = []
        for _ in range(20000):
            rs = []
            for (a0, a1), n in zip(zip(x.from_year, x.to_year), x.n):
                rs.append(spearmanr(np.arange(n), rng.permutation(n))[0])
            null.append(np.mean(rs))
        p = (np.array(null) >= obs).mean(); out.append(dict(office=cg, metric=m, transitions=len(x), mean=obs, p_perm=p, same_best=x.same_best.mean(), best_top2=x.best_top2.mean()))
O = pd.DataFrame(out); print(O.round(3).to_string(index=False)); O.to_csv('outputs_local/us_ranking_persistence_summary.csv', index=False)
print('\n=== historical predictor: mean ranking in earlier cycles vs ranking of the current cycle (BS) ===')
for cg in ('house', 'senate', 'pres'):
    years = sorted(M[M.office == cg].year.unique()); P = M[M.office == cg].pivot(index='forecast', columns='year', values='BS')
    for k in range(2, len(years)):
        prev = P[years[:k]].dropna(how='all'); cur = P[years[k]].dropna(); com = cur.index.intersection(prev.dropna(thresh=max(1, k - 1)).index)
        if len(com) < 4: continue
        rk_prev = prev.loc[com].rank(axis=0).mean(axis=1); print(f"  {cg} {years[k]}: n={len(com)} | Spearman(mean earlier rank, current rank) = {spearmanr(rk_prev, cur[com])[0]:+.2f}")
