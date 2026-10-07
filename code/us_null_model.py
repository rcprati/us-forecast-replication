"""
Shared simulation/estimation code for the US correlated-error calibration null (sections 5.43-5.44).
Imported by us_null_correlated.py, us_null_loco.py and us_beta_alpha_plane_figure (us_figure_beta_alpha_plane.py). Not run directly.
Null (Gaussian copula, calibrated by construction): y_i = 1{ Phi^{-1}(p_i) + e_i > 0 }, e_i = sqrt(rho_n) z_nat + sqrt(rho_s) z_state(i) + sqrt(1 - rho_n - rho_s) eps_i,
so that P(y_i = 1) = p_i (exact marginals) and dependence across units comes from the national shock (rho_n) and the state shock (rho_s).
Input: data/jhk/jhk_output.csv (JHK Forecasts database; third-party data, not distributed with this repository).
The random generator is module-level: call seed(n) before simulating (seeds 45, 46, 47 in the scripts).
"""
import numpy as np, pandas as pd
from scipy.stats import norm
import warnings; warnings.filterwarnings('ignore')
DATA = 'data/jhk/jhk_output.csv'
rng = np.random.default_rng(0)
def seed(s):
    global rng; rng = np.random.default_rng(s)
def load_data(path=DATA):
    d = pd.read_csv(path); d['p'] = (1 - d.probwin).clip(0.0005, 0.9995); d['y'] = 1 - d.probwin_outcome
    d['state'] = d.s_id.str.extract(r'^([A-Z]{2})')[0]; return d
def sim(p, est, rn, rs, B):
    n = len(p); eu = {e: i for i, e in enumerate(pd.unique(est))}; gi = np.array([eu[e] for e in est]); zn = rng.normal(size=(B, 1)); zs = rng.normal(size=(B, len(eu)))[:, gi] if rs > 0 else 0; eps = rng.normal(size=(B, n))
    e = np.sqrt(rn) * zn + (np.sqrt(rs) * zs if rs > 0 else 0) + np.sqrt(max(1 - rn - rs, 1e-9)) * eps; return (norm.ppf(p)[None, :] + e > 0).astype(np.float64)
def estat(p, Y):
    S = Y.sum(1); a = ((1 - p) ** 2 * Y).sum(1) / np.maximum(S, 1); b = (p ** 2 * (1 - Y)).sum(1) / np.maximum((1 - Y).sum(1), 1); bs = ((Y - p) ** 2).mean(1); return a, b, a - b, bs
def estat1(p, y):
    a, b, dif, bs = estat(p, y[None, :].astype(float)); return a[0], b[0], dif[0], bs[0]
def pbil(nul, obs): return 2 * min((nul <= obs).mean(), (nul >= obs).mean())
# ---------------- rho estimation (method of moments) ----------------
GN = [0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]; GS = [0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3]
def cycles(d, office, fc):
    return [(eid, g.sort_values('s_id')) for eid, g in d[(d.office == office) & (d.forecast == fc)].groupby('election_id') if len(g) >= 30]
def obs_stats(cs, states=True):
    s1 = np.mean([(g.y - g.p).mean() ** 2 for _, g in cs]); s2 = None
    if states:
        v = []
        for _, g in cs:
            m = (g.y - g.p).groupby(g.state).agg(['mean', 'size']); m = m[m['size'] >= 3]; v.append(np.average((m['mean'] - (g.y - g.p).mean()) ** 2, weights=m['size'])) if len(m) > 3 else None
        s2 = np.mean([x for x in v if x is not None])
    return s1, s2
def exp_stats(cs, rn, rs, states, B=400):
    s1 = []; s2 = []
    for _, g in cs:
        p = g.p.values; Y = sim(p, g.state.values, rn, rs, B); R = Y - p; s1.append((R.mean(1) ** 2).mean())
        if states:
            e = g.state.values; ue = pd.unique(e); w = []
            for u in ue:
                idx = np.where(e == u)[0]
                if len(idx) >= 3: w.append((R[:, idx].mean(1) - R.mean(1)) ** 2 * len(idx))
            if w: s2.append(np.sum(w, axis=0).mean() / sum(len(np.where(e == u)[0]) for u in ue if (e == u).sum() >= 3))
    return np.mean(s1), (np.mean(s2) if s2 else None)
def fit_rho(cs, states, B=400):
    """Grid search over (rho_n, rho_s); returns (loss, rho_n, rho_s, S1_exp, S2_exp), plus observed (S1, S2)."""
    o1, o2 = obs_stats(cs, states); best = None
    for rn in GN:
        for rs in (GS if states else [0]):
            if rn + rs >= 0.95: continue
            e1, e2 = exp_stats(cs, rn, rs, states, B=B); loss = ((o1 - e1) / e1) ** 2 + (((o2 - e2) / e2) ** 2 if states and e2 else 0)
            if best is None or loss < best[0]: best = (loss, rn, rs, e1, e2)
    return best, o1, o2
