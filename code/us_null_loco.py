"""
Robustness of Sec. 5.44: correlated null with rho estimated WITHOUT the tested cycle (leave-one-cycle-out; reference forecaster fte) to avoid circularity. Same functions as us_null_correlated.py (shared module us_null_model.py). Seed 46.
For each cycle-office c: estimate (rho_nat, rho_state) from the other cycles of the same office (fte) and test all forecasters of c against that null.
Input: data/jhk/jhk_output.csv. Output: outputs_local/us_null_loco.csv.
Run from the repository root:  python code/us_null_loco.py
"""
import os, numpy as np, pandas as pd
import us_null_model as M
pd.set_option('display.width', 220, 'display.max_columns', 30, 'display.max_rows', 400)
os.makedirs('outputs_local', exist_ok=True)
M.seed(46); d = M.load_data()
B = 4000; rows = []; RH = {}
for office, states in (('house', True), ('pres', False), ('senate', False)):
    cs_all = M.cycles(d, office, 'fte')
    for eid, _ in cs_all:
        cs = [c for c in cs_all if c[0] != eid]; best, _, _ = M.fit_rho(cs, states, B=300)
        RH[eid] = (best[1], best[2]); print(eid, 'rho (without the cycle itself) =', RH[eid], flush=True)
for eid, g0 in d.groupby('election_id'):
    rn, rs = RH[eid]
    for fc, g in g0.groupby('forecast'):
        if len(g) < 30: continue
        g = g.sort_values('s_id'); p = g.p.values; y = g.y.values; est = g.state.values; oa, ob, od, obs = M.estat1(p, y); Yc = M.sim(p, est, rn, rs, B); ac, bc, dc, sc = M.estat(p, Yc)
        rows.append(dict(cycle_office=eid, forecaster=fc, n=len(g), diff=od, BS=obs, rho_nat=rn, rho_state=rs, p_diff_loco=M.pbil(dc, od), p_BS_loco=M.pbil(sc, obs), p_alpha_loco=M.pbil(ac, oa), p_beta_loco=M.pbil(bc, ob)))
R = pd.DataFrame(rows); R.to_csv('outputs_local/us_null_loco.csv', index=False)
R['office'] = R.cycle_office.str.split('_').str[1]
print('\nfraction of tests with p < 0.05 (LOCO null):'); print(R.groupby('office').apply(lambda x: pd.Series(dict(tests=len(x), diff=(x.p_diff_loco < 0.05).mean(), BS=(x.p_BS_loco < 0.05).mean(), alpha=(x.p_alpha_loco < 0.05).mean(), beta=(x.p_beta_loco < 0.05).mean()))).round(3).to_string())
print('\nalpha-beta with p_loco < 0.05:'); print(R[R.p_diff_loco < 0.05][['cycle_office', 'forecaster', 'diff', 'rho_nat', 'rho_state', 'p_diff_loco']].round(3).to_string(index=False))
