"""
Sec. 5.44: calibration null with CORRELATED ERROR for alpha, beta and Brier, applied to all forecasters and cycles of the JHK database (data/jhk/jhk_output.csv). Seed 45.
Null (Gaussian copula, calibrated by construction): y_i = 1{ Phi^{-1}(p_i) + e_i > 0 }, e_i = sqrt(rho_n) z_nat + sqrt(rho_s) z_state(i) + sqrt(1 - rho_n - rho_s) eps_i, so that P(y_i = 1) = p_i (exact marginals) and dependence across units comes from the national shock (rho_n) and the state shock (rho_s, House only).
ESTIMATION by moments (reference forecaster: fte; robustness with cook, ddhq and econ): (i) between-cycle variance of the mean residual rbar = mean(y - p), E[rbar^2] under (rho_n, rho_s); (ii) between-state variance of the state mean residual (House, states with >= 3 districts). Grid over (rho_n, rho_s); President/Senate use rho_n only.
TEST: for each (cycle, office, forecaster) with >= 30 units, two-sided p-values for alpha-beta, alpha, beta and Brier against (a) the INDEPENDENT null (rho = 0) and (b) the CORRELATED null (estimated rho); design effect DEFF = Var_corr(rbar)/Var_indep(rbar). B = 4000. Descriptive.
Shared code: us_null_model.py. Input: data/jhk/jhk_output.csv.
Outputs: outputs_local/us_null_correlated.csv and outputs_local/us_null_rho.csv.
Run from the repository root:  python code/us_null_correlated.py
"""
import os, numpy as np, pandas as pd
import us_null_model as M
pd.set_option('display.width', 220, 'display.max_columns', 30, 'display.max_rows', 400)
os.makedirs('outputs_local', exist_ok=True)
M.seed(45); d = M.load_data()
RHO = {}; rows_rho = []
for office, states in (('house', True), ('pres', False), ('senate', False)):
    for fc in ('fte', 'cook', 'ddhq', 'econ'):
        cs = M.cycles(d, office, fc)
        if len(cs) < 2: continue
        best, o1, o2 = M.fit_rho(cs, states)
        RHO[(office, fc)] = (best[1], best[2]); rows_rho.append(dict(office=office, reference=fc, cycles=len(cs), S1_obs=o1, S2_obs=o2, rho_nat=best[1], rho_state=best[2], S1_exp=best[3], S2_exp=best[4]))
        print(f"{office:7s} ref={fc:5s} cycles={len(cs)} | rbar² obs {o1:.5f}" + (f", between-state variance obs {o2:.5f}" if states else '') + f" -> rho_nat = {best[1]}, rho_state = {best[2]}")
pd.DataFrame(rows_rho).to_csv('outputs_local/us_null_rho.csv', index=False)
# ---------------- TEST ----------------
B = 4000; rows = []; rhoF = {o: RHO[(o, 'fte')] for o in ('house', 'pres', 'senate')}
for eid, g0 in d.groupby('election_id'):
    office = g0.office.iloc[0]; rn, rs = rhoF[office]
    for fc, g in g0.groupby('forecast'):
        if len(g) < 30: continue
        g = g.sort_values('s_id'); p = g.p.values; y = g.y.values; est = g.state.values; oa, ob, od, obs = M.estat1(p, y)
        Yi = M.sim(p, est, 0, 0, B); Yc = M.sim(p, est, rn, rs, B); ai, bi, di, si = M.estat(p, Yi); ac, bc, dc, sc = M.estat(p, Yc)
        vi = (Yi - p).mean(1).var(); vc = (Yc - p).mean(1).var()
        rows.append(dict(cycle_office=eid, forecaster=fc, n=len(g), alpha=oa, beta=ob, diff=od, BS=obs, p_diff_ind=M.pbil(di, od), p_diff_corr=M.pbil(dc, od), p_BS_ind=M.pbil(si, obs), p_BS_corr=M.pbil(sc, obs), p_alpha_corr=M.pbil(ac, oa), p_beta_corr=M.pbil(bc, ob), diff_cal=np.median(dc), diff_5=np.percentile(dc, 5), diff_95=np.percentile(dc, 95), DEFF=vc / vi))
R = pd.DataFrame(rows); R.to_csv('outputs_local/us_null_correlated.csv', index=False)
print('\nrho used (fte):', rhoF)
print("\nDEFF (variance of the mean residual, correlated/independent), median by cycle-office:"); print(R.groupby('cycle_office').DEFF.median().round(1).to_dict())
g = R.groupby(R.cycle_office.str.split('_').str[1]).apply(lambda x: pd.Series(dict(tests=len(x), rej_diff_ind=(x.p_diff_ind < 0.05).mean(), rej_diff_corr=(x.p_diff_corr < 0.05).mean(), rej_BS_ind=(x.p_BS_ind < 0.05).mean(), rej_BS_corr=(x.p_BS_corr < 0.05).mean())))
print('\nfraction of tests with p < 0.05 (alpha-beta and Brier), independent null vs correlated null:'); print(g.round(3).to_string())
sel = R[R.forecaster.isin(['fte', 'econ', 'ddhq', 'cook', 'sabato'])].sort_values(['cycle_office', 'forecaster']); print('\n', sel[['cycle_office', 'forecaster', 'n', 'diff', 'diff_cal', 'diff_5', 'diff_95', 'p_diff_ind', 'p_diff_corr', 'BS', 'p_BS_ind', 'p_BS_corr']].round(3).to_string(index=False))
