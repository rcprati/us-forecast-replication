"""
Average error of the state-level polls in the presidential races, by cycle (2000-2024): the poll error quoted in the main text, Section 5.5 (+4.27, +4.63 and +3.05 percentage points in 2016, 2020 and 2024; positive = the poll overstated the Democrat).
Error per state and cycle = two-party margin of the polls (mean of the polls ending within W = 21 days of the election, at least MINP = 2 polls) minus the actual two-party margin, in percentage points, margin = Democrat - Republican.
2000-2020: FiveThirtyEight's raw_polls.csv (Democrat vs Republican, general election, 50 states + DC). 2024: the table written by us_polls_538_2024.py, with the actual results from data/results/election_results_presidential.csv (stage 'general', 2024).
Writes outputs_local/us_poll_error_president.csv (cycle, state, e, real, n). Run from the repository root.
"""
import sys, os, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
W = int(sys.argv[1]) if len(sys.argv) > 1 else 21; MINP = int(sys.argv[2]) if len(sys.argv) > 2 else 2
os.makedirs('outputs_local', exist_ok=True)
# ---- 2000-2020
d = pd.read_csv('data/538/raw_polls.csv')
d = d[(d.type_simple == 'Pres-G') & (d.cand1_party == 'DEM') & (d.cand2_party == 'REP') & (d.time_to_election <= W) & d.location.str.fullmatch(r'[A-Z]{2}') & ~d.location.isin(['US', 'PR', 'VI', 'M1', 'M2', 'N1', 'N2', 'N3'])].copy()
d = d[d.cycle % 2 == 0]
d['e'] = 100 * ((d.cand1_pct - d.cand2_pct) / (d.cand1_pct + d.cand2_pct) - (d.cand1_actual - d.cand2_actual) / (d.cand1_actual + d.cand2_actual))
d['real'] = 100 * (d.cand1_actual - d.cand2_actual) / (d.cand1_actual + d.cand2_actual)
R = d.groupby(['cycle', 'location']).agg(e=('e', 'mean'), n=('e', 'size'), real=('real', 'first')).reset_index().rename(columns={'location': 'state'}); R = R[R.n >= MINP]
# ---- 2024
r = pd.read_csv('data/results/election_results_presidential.csv'); r = r[(r.cycle == 2024) & (r.stage == 'general') & r.state_abbrev.str.fullmatch(r'[A-Z]{2}', na=False) & ~r.state_abbrev.isin(['US', 'PR'])]
def votes(g, party): return g[g.ballot_party == party].groupby('candidate_name').votes.sum().max()
res = r.groupby('state_abbrev').apply(lambda g: pd.Series(dict(D=votes(g, 'DEM'), R=votes(g, 'REP')))).dropna(); res['m_r'] = 100 * (res.D - res.R) / (res.D + res.R)
p = pd.read_csv('outputs_local/polls_538_2024_state.csv'); p = p[(p.days <= W) & p.state.isin(res.index)].copy(); p['m_p'] = 100 * (p.D - p.R) / (p.D + p.R)
a = p.groupby('state').agg(m_p=('m_p', 'mean'), n=('m_p', 'size')).reset_index(); a = a[a.n >= MINP].merge(res[['m_r']], left_on='state', right_index=True)
a['e'] = a.m_p - a.m_r; a24 = pd.DataFrame(dict(cycle=2024, state=a.state, e=a.e, n=a.n, real=a.m_r))
U = pd.concat([R, a24], ignore_index=True); U.to_csv('outputs_local/us_poll_error_president.csv', index=False)
print(f"window = {W} days, at least {MINP} polls per state")
print(f"{'cycle':>6s}{'states':>8s}{'mean error (pp)':>18s}")
for c, g in U.groupby('cycle'): print(f"{c:6d}{len(g):8d}{g.e.mean():+18.2f}")
