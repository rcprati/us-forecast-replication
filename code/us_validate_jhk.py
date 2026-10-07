"""
Validation of the JHK Forecasts database against the original files of the forecasters, where those could be obtained:
 (1) FiveThirtyEight House 2018 (Lite, Classic, Deluxe) against the JHK forecaster `fte` (2018_house): mean absolute difference in P(D), correlation, agreement of outcomes;
 (2) The Economist's 2020 presidential model (state win probabilities, 51 units) against the JHK forecaster `econ` (2020_pres);
 (3) Split Ticket's 2024 House and Senate files against the JHK forecaster `split` (2024_house, 2024_senate).
Inputs: data/jhk/jhk_output.csv, data/538/forecast_results_2018.csv, data/validation/economist_model_output.zip, data/forecasters/splitticket_{house,senate}_2024_datawrapper.csv. Standard pandas; run from the repository root. Reported values (project note): (1) 0.0013 (Deluxe), 0.0136 (Classic), 0.0313 (Lite), correlation 1.0000, outcomes 434/434; (2) 0.0009, correlation 1.0; (3) maximum difference 1e-5 (House, 435/435) and 0 (Senate, 34/34).
"""
import io, zipfile, numpy as np, pandas as pd
J = pd.read_csv('data/jhk/jhk_output.csv'); J['pD'] = 1 - J.probwin; J['yD'] = 1 - J.probwin_outcome
def key(cd):
    s, n = cd.split('-'); return s + (str(int(n)) if n != 'AL' else '1')
# (1) FiveThirtyEight House 2018
f = pd.read_csv('data/538/forecast_results_2018.csv'); f = f[f.branch == 'House'].copy(); f['s_id'] = f.race.map(key)
j = J[(J.forecast == 'fte') & (J.election_id == '2018_house')].set_index('s_id')
print('(1) FiveThirtyEight House 2018 vs JHK `fte`')
for v in ('lite', 'classic', 'deluxe'):
    g = f[f.version == v].drop_duplicates('s_id').set_index('s_id').join(j[['pD', 'yD']], how='inner')
    print(f"  {v:8s} districts {len(g):4d} | mean |dP(D)| {np.abs(g.Democrat_WinProbability - g.pD).mean():.4f} | correlation {np.corrcoef(g.Democrat_WinProbability, g.pD)[0, 1]:.4f} | outcomes agree {int((g.Democrat_Won == g.yD).sum())}/{len(g)}")
# (2) The Economist 2020 president
with zipfile.ZipFile('data/validation/economist_model_output.zip') as z:
    name = [n for n in z.namelist() if n.endswith('state_averages_and_predictions_topline.csv')][0]; E = pd.read_csv(io.BytesIO(z.read(name))).set_index('state')
je = J[(J.forecast == 'econ') & (J.election_id == '2020_pres')].set_index('s_id'); g = E[['projected_win_prob']].join(je[['pD']], how='inner')
print(f"(2) The Economist 2020 president vs JHK `econ`: units {len(g)} | mean |dP(D)| {np.abs(g.projected_win_prob - g.pD).mean():.4f} | correlation {np.corrcoef(g.projected_win_prob, g.pD)[0, 1]:.4f}")
# (3) Split Ticket 2024
h = pd.read_csv('data/forecasters/splitticket_house_2024_datawrapper.csv', encoding='utf-8-sig'); h['s_id'] = h.cd.map(key); h['p'] = h.dem_prob_winning / 100
jh = J[(J.forecast == 'split') & (J.election_id == '2024_house')].set_index('s_id'); g = h.set_index('s_id')[['p']].join(jh[['pD']], how='inner')
print(f"(3) Split Ticket 2024 House vs JHK `split`: units {len(g)} of {len(h)} | max |dP(D)| {np.abs(g.p - g.pD).max():.1e}")
s = pd.read_csv('data/forecasters/splitticket_senate_2024_datawrapper.csv', encoding='utf-8-sig'); s['s_id'] = s.STUSPS.replace({'NS': 'NES'}); s['p'] = s.dem_prob_winning / 100
js = J[(J.forecast == 'split') & (J.election_id == '2024_senate')].set_index('s_id'); g = s.set_index('s_id')[['p']].join(js[['pD']], how='inner')
print(f"    Split Ticket 2024 Senate vs JHK `split`: units {len(g)} of {len(s)} | max |dP(D)| {np.abs(g.p - g.pD).max():.1e}")
