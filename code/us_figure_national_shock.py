"""
Figure: latent national shock delta-hat (probit scale; > 0 = the Democrat does better than forecast) by cycle and office, from outputs_local/us_national_shock.csv (written by code/us_national_shock.py). One dot = one forecaster (delta-hat from the number of wins); large marker + bar = median and quartiles.
Input: outputs_local/us_national_shock.csv (run code/us_national_shock.py first). Output: outputs_local/figures/us_national_shock_en.png. Seed 54 (jitter).
Run from the repository root: python code/us_figure_national_shock.py
"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
os.makedirs('outputs_local/figures', exist_ok=True)
R = pd.read_csv('outputs_local/us_national_shock.csv'); SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'; COR = {'house': '#2a78d6', 'senate': '#eb6834', 'pres': '#1baf7a'}
def fig():
    T = dict(tit='The latent national shock of each cycle, as estimated by each forecaster', sub='δ̂ (probit scale; > 0 = the Democrat did better than forecast). Each dot is a forecaster (δ̂ from the number of wins); large marker and bar = median and quartiles.', y='δ̂ (latent standard deviations)', c={'house': 'House', 'senate': 'Senate', 'pres': 'President'}, up='Democrats understated', dn='Democrats overstated')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2, 'text.color': INK})
    f, axs = plt.subplots(1, 3, figsize=(15, 5.2), dpi=200, facecolor=SURF, sharey=True); rng = np.random.default_rng(54)
    for ax, cg in zip(axs, ['house', 'pres', 'senate']):
        ax.set_facecolor(SURF); s = R[R.office == cg]; years = sorted(s.year.unique())
        for i, a in enumerate(years):
            v = s[s.year == a].dhat_win.dropna().values; ax.scatter(i + rng.uniform(-0.14, 0.14, len(v)), v, s=22, color=COR[cg], alpha=0.40, edgecolors='none', zorder=3)
            q1, md, q3 = np.percentile(v, [25, 50, 75]); ax.plot([i, i], [q1, q3], color=COR[cg], lw=3, solid_capstyle='round', zorder=4); ax.scatter([i], [md], s=80, color=COR[cg], edgecolor=SURF, linewidths=1.5, zorder=5); ax.text(i + 0.18, md, f'{md:+.2f}', fontsize=8.5, color=INK, va='center', zorder=6)
        ax.axhline(0, color=INK2, lw=0.9, zorder=2); ax.set_xticks(range(len(years))); ax.set_xticklabels(years); ax.set_xlim(-0.6, len(years) - 0.4); ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.set_title(T['c'][cg] + f' (n = {int(s.groupby("year").size().min())}–{int(s.groupby("year").size().max())} forecasters)', loc='left', fontsize=11.5, fontweight='bold', color=INK)
        for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    axs[0].set_ylabel(T['y']); axs[0].text(-0.55, 0.97, T['up'], fontsize=8.5, color=INK2, va='top'); axs[0].text(-0.55, -0.97, T['dn'], fontsize=8.5, color=INK2, va='bottom'); axs[0].set_ylim(-1.1, 1.1)
    f.suptitle(T['tit'], x=0.01, ha='left', fontsize=13.5, fontweight='bold', color=INK, y=0.995); f.text(0.01, 0.925, T['sub'], fontsize=9.3, color=INK2, ha='left'); f.tight_layout(rect=(0, 0, 1, 0.91)); return f
fig().savefig('outputs_local/figures/us_national_shock_en.png', facecolor=SURF); print('ok')
