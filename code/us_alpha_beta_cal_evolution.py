"""
EVOLUTION of (alpha-beta)_cal (and alpha_cal, beta_cal, Brier_cal, mean P) within a single cycle, computed from each day's probabilities (without the outcome), against the observed alpha-beta (which uses the outcome). Seed 53 (not used: descriptive).
Daily series: (A) Rieke 2024 president (51 units: 50 states + DC; 2024-05-01 to 2024-11-05, 189 days); (B) 538 polls-plus 2020 president (51 units; 2020-06-01 to 2020-11-03);
(C) 538 House in 2020 and 2022 (435 districts x Lite/Classic/Deluxe; horizons 0, 7, 14, 30, 60 and 90 days; data/538/house_2020_2022_extract.csv) and 2018 at the final forecast only.
Outcomes: JHK (forecaster fte). Statistics: (alpha-beta)_cal(t) and (alpha-beta)_obs(t), the sign of (alpha-beta)_cal(t) against the final observed sign, the 'drift' (alpha-beta)_cal(t) - (alpha-beta)_cal(final) and the mean P that the Democrat wins.
Inputs: data/jhk/jhk_output.csv, data/rieke/win_state.csv, data/538/presidential_state_toplines_2020.csv, data/538/house_2020_2022_extract.csv, data/538/forecast_results_2018.csv.
Run from the repository root: python code/us_alpha_beta_cal_evolution.py
Writes outputs_local/us_alpha_beta_cal_evolution.csv and outputs_local/figures/us_alpha_beta_cal_evolution_en.png.
"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')
os.makedirs('outputs_local/figures', exist_ok=True)
UF = {'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia': 'DC', 'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'}   # state name -> postal code
jhk = pd.read_csv('data/jhk/jhk_output.csv'); jhk = jhk[jhk.forecast == 'fte']
def outcome(eid): g = jhk[jhk.election_id == eid]; return dict(zip(g.s_id, 1 - g.probwin_outcome))
def met(p, y):
    p = np.clip(np.asarray(p, float), 0.0005, 0.9995); y = np.asarray(y, float); pos, neg = y == 1, y == 0; a = ((1 - p[pos]) ** 2).mean(); b = (p[neg] ** 2).mean(); ac = (p * (1 - p) ** 2).sum() / p.sum(); bc = ((1 - p) * p ** 2).sum() / (1 - p).sum()
    return dict(pbar=p.mean(), alpha_cal=ac, beta_cal=bc, diff_cal=ac - bc, BS_cal=(p * (1 - p)).mean(), diff_obs=a - b, BS_obs=((y - p) ** 2).mean(), n=len(p))
rows = []
# (A) Rieke 2024
W = pd.read_csv('data/rieke/win_state.csv'); W = W[W.state.isin(UF)].copy(); W['uf'] = W.state.map(UF); W['run_date'] = pd.to_datetime(W.run_date); y24 = outcome('2024_pres'); end = W.run_date.max()
for dt, g in W.groupby('run_date'):
    g = g[g.uf.isin(y24)]; rows.append(dict(series='Rieke 2024 (pres)', days=(end - dt).days, **met(g.p_win.values, [y24[u] for u in g.uf])))
# (B) 538 polls-plus 2020 president
P = pd.read_csv('data/538/presidential_state_toplines_2020.csv', low_memory=False); P['d'] = pd.to_datetime(P.modeldate, format='%m/%d/%Y'); P = P[P.state.isin(UF)].copy(); P['uf'] = P.state.map(UF); y20 = outcome('2020_pres'); end = P.d.max()
for dt, g in P.groupby('d'):
    g = g[g.uf.isin(y20)]; rows.append(dict(series='538 2020 (pres)', days=(end - dt).days, **met(g.winstate_chal.values, [y20[u] for u in g.uf])))
# (C) 538 House
C = pd.read_csv('data/538/house_2020_2022_extract.csv').rename(columns={'ciclo': 'cycle', 'horizonte': 'horizon', 'versao': 'version'})   # accepts the older extract with Portuguese column names
for (cy, v, h), g in C.groupby(['cycle', 'version', 'horizon']): rows.append(dict(series=f'538 House {cy} ({v})', days=h, **met(g.p.values, g.y.values)))
f18 = pd.read_csv('data/538/forecast_results_2018.csv'); f18 = f18[f18.branch == 'House']
for v, g in f18.groupby('version'): rows.append(dict(series=f'538 House 2018 ({v})', days=0, **met(g.Democrat_WinProbability.values, g.Democrat_Won.astype(float).values)))
E_ = pd.DataFrame(rows).sort_values(['series', 'days'], ascending=[True, False]); E_.to_csv('outputs_local/us_alpha_beta_cal_evolution.csv', index=False)
pd.set_option('display.width', 220, 'display.max_columns', 30)
print('=== daily series: values at 150, 120, 90, 60, 30, 14, 7 and 0 days ===')
for s in ('Rieke 2024 (pres)', '538 2020 (pres)'):
    g = E_[E_.series == s].set_index('days'); sel = [h for h in (150, 120, 90, 60, 30, 14, 7, 0) if h in g.index]; print(f'\n{s}: n = {int(g.n.iloc[0])} units'); print(g.loc[sel, ['pbar', 'alpha_cal', 'beta_cal', 'diff_cal', 'diff_obs', 'BS_cal', 'BS_obs']].round(3).to_string())
    fin = g.loc[0]; print(f"  final observed sign: {np.sign(fin.diff_obs):+.0f} | sign of (α−β)_cal by horizon:", {h: int(np.sign(g.loc[h, 'diff_cal'])) for h in sel}, '| drift (α−β)_cal(t) − (α−β)_cal(0):', {h: round(g.loc[h, 'diff_cal'] - fin.diff_cal, 3) for h in sel})
print('\n=== 538 House: (α−β)_cal and observed by horizon ===')
Ch = E_[E_.series.str.startswith('538 House')]; print(Ch.pivot_table(index='series', columns='days', values='diff_cal').round(3).to_string()); print(); print(Ch.pivot_table(index='series', columns='days', values='diff_obs').round(3).to_string()); print(); print(Ch.pivot_table(index='series', columns='days', values='pbar').round(3).to_string())
# sign agreement
print('\n=== sign of (α−β)_cal(t) equal to the final observed sign ===')
for s, g in E_.groupby('series'):
    if g.days.nunique() < 3: continue
    fin = np.sign(g[g.days == 0].diff_obs.iloc[0]); r = g[g.days.isin([90, 60, 30, 14, 7, 0])]; print(f"  {s:30s}: final sign {fin:+.0f} | agreement by horizon {dict(zip(r.days, (np.sign(r.diff_cal) == fin).astype(int)))}")
# --------- figure ---------
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'; AZ, LA = '#2a78d6', '#eb6834'
def figure():
    T = dict(tit='Evolution of (α−β) expected under calibration and observed within a cycle', sub='Expected: computed from that day’s probabilities, without the outcome. Observed: uses the outcome. Days before the election.', cal='(α−β)_cal', obs='(α−β) observed', x='days before the election', y='α−β', ab='Biden withdraws (Jul 21)', ov='Democrats overstated (β > α)', un='Democrats understated (α > β)', r='Rieke 2024, president', f='538 2020, president (polls-plus)', h='538 House 2020 and 2022 (Deluxe)', hx='days before the election')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    fig, axs = plt.subplots(1, 3, figsize=(16.5, 5.6), dpi=200, facecolor=SURF)
    for ax, s, tt in ((axs[0], 'Rieke 2024 (pres)', T['r']), (axs[1], '538 2020 (pres)', T['f'])):
        g = E_[E_.series == s].sort_values('days'); ax.set_facecolor(SURF); ax.plot(g.days, g.diff_cal, color=AZ, lw=2, label=T['cal']); ax.plot(g.days, g.diff_obs, color=LA, lw=2, ls=(0, (4, 2)), label=T['obs']); ax.axhline(0, color=INK2, lw=0.8)
        ax.invert_xaxis(); ax.grid(color=GRID, lw=0.6); ax.set_title(tt, loc='left', fontweight='bold', fontsize=11.5, color=INK); ax.set_xlabel(T['x']); ax.set_ylabel(T['y'])
        if s.startswith('Rieke'): ax.axvline((pd.Timestamp('2024-11-05') - pd.Timestamp('2024-07-21')).days, color=INK2, lw=0.9, ls=':'); ax.text((pd.Timestamp('2024-11-05') - pd.Timestamp('2024-07-21')).days - 2, ax.get_ylim()[1] * 0.92, T['ab'], fontsize=8.5, color=INK2, ha='right')
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax = axs[2]; ax.set_facecolor(SURF)
    for cy, col, mk in ((2020, '#1baf7a', 's'), (2022, '#eda100', '^')):
        g = E_[E_.series == f'538 House {cy} (deluxe)'].sort_values('days'); ax.plot(g.days, g.diff_cal, color=col, lw=2, marker=mk, ms=6, label=f'{cy} {T["cal"]}'); ax.plot(g.days, g.diff_obs, color=col, lw=1.6, ls=(0, (4, 2)), marker=mk, ms=6, mfc=SURF, label=f'{cy} {T["obs"]}')
    ax.axhline(0, color=INK2, lw=0.8); ax.invert_xaxis(); ax.grid(color=GRID, lw=0.6); ax.set_title(T['h'], loc='left', fontweight='bold', fontsize=11.5, color=INK); ax.set_xlabel(T['hx']); ax.set_ylabel(T['y'])
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    axs[0].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='upper right'); axs[1].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='lower left'); axs[2].legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc='center left', bbox_to_anchor=(0.02, 0.42))
    fig.suptitle(T['tit'], x=0.01, ha='left', fontsize=13.5, fontweight='bold', color=INK, y=0.99); fig.text(0.01, 0.935, T['sub'], fontsize=9.3, color=INK2, ha='left'); fig.tight_layout(rect=(0, 0, 1, 0.92)); return fig
figure().savefig('outputs_local/figures/us_alpha_beta_cal_evolution_en.png', facecolor=SURF); print('\nfigure saved')
