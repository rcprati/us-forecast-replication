"""
Figure: the (beta, alpha) plane of US probabilistic forecasters (JHK database, 2016-2024; Sec. 5.43-5.44). Seed 47.
Each filled marker = a forecaster in a cycle (colour and shape = cycle); the small grey point linked to it by a thin arrow = (beta_cal, alpha_cal), the value expected under calibration computed from the forecaster's own probabilities.
The 90% ellipse (cycle colour) = distribution of (beta, alpha) of 538 under the calibrated null with correlated error (Gaussian copula; rho from outputs_local/us_null_rho.csv).
Diagonal alpha = beta: no asymmetry between classes; above = costly Democratic wins (Democrats understated); below = costly Democratic losses (Democrats overstated). Thin dashed lines: constant balanced Brier (alpha+beta)/2.
Shared code: us_null_model.py. Inputs: data/jhk/jhk_output.csv; outputs_local/us_null_rho.csv (written by us_null_correlated.py, run it first).
Outputs: outputs_local/figures/us_beta_alpha_plane_en.png and .svg.
Run from the repository root:  python code/us_figure_beta_alpha_plane.py
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
from scipy.stats import chi2
import us_null_model as M
os.makedirs('outputs_local/figures', exist_ok=True)
M.seed(47); d = M.load_data()
rho = pd.read_csv('outputs_local/us_null_rho.csv'); rho = rho[rho.reference == 'fte'].set_index('office')
def ab(p, y):
    pos, neg = y == 1, y == 0; return ((1 - p[pos]) ** 2).mean(), (p[neg] ** 2).mean()
def cal(p): return (p * (1 - p) ** 2).sum() / p.sum(), ((1 - p) * p ** 2).sum() / (1 - p).sum()
def simab(p, est, rn, rs, B=1500):
    a, b, _, _ = M.estat(p, M.sim(p, est, rn, rs, B)); return b, a
rows = []
for (eid, fc), g in d.groupby(['election_id', 'forecast']):
    if len(g) < 30: continue
    p = g.p.values; y = g.y.values; a, b = ab(p, y); ac, bc = cal(p); rows.append(dict(eid=eid, year=int(eid[:4]), office=eid.split('_')[1], fc=fc, alpha=a, beta=b, alpha_cal=ac, beta_cal=bc))
R = pd.DataFrame(rows)
# --- style (colour tokens: reference palette, light mode) ---
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
COR = {2016: '#2a78d6', 2018: '#eb6834', 2020: '#1baf7a', 2022: '#eda100', 2024: '#e87ba4'}; MK = {2016: 'P', 2018: 'o', 2020: 's', 2022: '^', 2024: 'D'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
fig, axs = plt.subplots(1, 3, figsize=(16.5, 6.6), dpi=200, facecolor=SURF); LIM = {'house': 0.075, 'pres': 0.135, 'senate': 0.16}; TIT = {'house': 'House (434–435 districts)', 'pres': 'President (56 units)', 'senate': 'Senate (34–35 races)'}
for ax, cg in zip(axs, ['house', 'pres', 'senate']):
    ax.set_facecolor(SURF); L = LIM[cg]; ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect('equal'); ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    ax.plot([0, L], [0, L], color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    for c in np.arange(0.02, 2 * L, 0.02): ax.plot([0, 2 * c], [2 * c, 0], color=GRID, lw=0.7, ls=(0, (2, 3)), zorder=0)
    g = R[R.office == cg]
    # ellipse of the correlated null (538) per cycle
    for eid, gg in d[(d.office == cg) & (d.forecast == 'fte')].groupby('election_id'):
        gg = gg.sort_values('s_id'); b, a = simab(gg.p.values, gg.state.values, rho.loc[cg, 'rho_nat'], rho.loc[cg, 'rho_state']); C = np.cov(b, a); w, V = np.linalg.eigh(C); ang = np.degrees(np.arctan2(V[1, 1], V[0, 1])); k = np.sqrt(chi2.ppf(0.9, 2))
        ax.add_patch(Ellipse((b.mean(), a.mean()), 2 * k * np.sqrt(w[1]), 2 * k * np.sqrt(w[0]), angle=ang, fill=False, ec=COR[int(eid[:4])], lw=1.3, ls=(0, (1, 1.6)), zorder=2))
    for r in g.itertuples():
        if r.fc == 'fte': ax.annotate('', xy=(r.beta, r.alpha), xytext=(r.beta_cal, r.alpha_cal), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=0.9, shrinkA=2, shrinkB=4, mutation_scale=8), zorder=4); ax.scatter([r.beta_cal], [r.alpha_cal], s=16, color='#8a8984', zorder=4, linewidths=0)
        ax.scatter([r.beta], [r.alpha], s=38 if r.fc != 'fte' else 80, marker=MK[r.year], color=COR[r.year], edgecolor=SURF, linewidths=1.0 if r.fc != 'fte' else 1.8, zorder=6 if r.fc == 'fte' else 5)
    for r in g[g.fc == 'fte'].itertuples(): ax.annotate(f"538 {r.year}", (r.beta, r.alpha), xytext=(5, 5), textcoords='offset points', fontsize=8.5, color=INK, zorder=7, bbox=dict(boxstyle='round,pad=0.12', fc=SURF, ec='none', alpha=0.85))
    ax.set_title(TIT[cg], loc='left', fontsize=11.5, color=INK, fontweight='bold', pad=8); ax.set_xlabel('β: cost on Democratic losses  E[p² | D loses]'); ax.set_ylabel('α: cost on Democratic wins  E[(1−p)² | D wins]' if cg == 'house' else '')
    bb = dict(boxstyle='round,pad=0.2', fc=SURF, ec='none', alpha=0.92); ax.text(0.02 * L, 0.975 * L, 'Democrats understated (α > β)', fontsize=8.5, color=INK2, va='top', zorder=9, bbox=bb); ax.text(0.98 * L, 0.025 * L, 'Democrats overstated (β > α)', fontsize=8.5, color=INK2, ha='right', va='bottom', zorder=9, bbox=bb)
leg = [Line2D([], [], marker=MK[y], color=COR[y], ls='', markersize=8, markeredgecolor=SURF, label=str(y)) for y in COR] + [Line2D([], [], marker='o', color='#8a8984', ls='', markersize=5, label='538: expected under calibration (arrow to observed)'), Line2D([], [], color=INK2, lw=1.3, ls=(0, (1, 1.6)), label='538: 90% region under calibration with a national shock'), Line2D([], [], color=INK2, lw=1, ls=(0, (4, 3)), label='α = β'), Line2D([], [], color=GRID, lw=1, ls=(0, (2, 3)), label='constant balanced Brier')]
fig.legend(handles=leg, loc='lower center', ncol=5, frameon=False, fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.5, -0.005))
fig.suptitle('The (β, α) plane: the side of the error belongs to the cycle, and most forecasters err less than expected under calibration', x=0.01, ha='left', fontsize=13.5, fontweight='bold', color=INK, y=0.995)
fig.text(0.01, 0.962,  'Each marker is a forecaster in a cycle (33 forecasters, 2016–2024, JHK Forecasts database).\nArrows (538): from expected under calibration to observed; outline: 90% region of the 538 forecast under calibration with a national shock (ρ estimated on the same data).', fontsize=9.3, color=INK2, ha='left', va='top', linespacing=1.4)
fig.tight_layout(rect=(0, 0.11, 1, 0.91)); fig.savefig('outputs_local/figures/us_beta_alpha_plane_en.png', facecolor=SURF); fig.savefig('outputs_local/figures/us_beta_alpha_plane_en.svg', facecolor=SURF)
print('ok', len(R), 'points'); print(R.groupby('office').size().to_dict())
