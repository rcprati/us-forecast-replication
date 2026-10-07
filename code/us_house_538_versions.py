"""
FiveThirtyEight's three model versions (Lite = polls only; Classic = + fundamentals; Deluxe = + expert ratings) for the US House in 2018, 2020 and 2022 (435 districts), and their evolution over the campaign in 2020 and 2022. Seed 50.
Inputs: data/538/house_2020_2022_extract.csv (compact extract of the 538 district toplines; see extract_raw below), data/538/forecast_results_2018.csv and the JHK outcomes (data/jhk/jhk_output.csv, forecaster fte).
Step 1, extract_raw() (OPTIONAL, UNTESTED here: the raw files were deleted from the original project after the extraction, so this function could not be re-run): reads the raw 538 files data/538/raw/house_district_toplines_2020.csv and data/538/raw/house_district_toplines_2022.csv (Wayback Machine snapshots of 2023-04-27, 46 MB and 75 MB) and writes data/538/house_2020_2022_extract.csv (final forecast and horizons 0, 7, 14, 30, 60, 90 days, by district and version). Run it only if the raw files are present: python code/us_house_538_versions.py --extract
Step 2, analysis() (default): python code/us_house_538_versions.py (from the repository root). Writes outputs_local/us_house_538_versions.csv.
Statistics: alpha, beta, Brier, balanced Brier by version and cycle; paired differences between versions with a bootstrap of STATES (B = 4000); Brier and alpha-beta by horizon.
"""
import os, sys, numpy as np, pandas as pd
import warnings; warnings.filterwarnings('ignore')
pd.set_option('display.width', 220, 'display.max_columns', 30)
EXTRACT = 'data/538/house_2020_2022_extract.csv'
def extract_raw(files=('data/538/raw/house_district_toplines_2020.csv', 'data/538/raw/house_district_toplines_2022.csv')):
    cols = ['cycle', 'district', 'forecastdate', 'expression', 'winner_Dparty', 'mean_netpartymargin', 'p10_netpartymargin', 'p90_netpartymargin']
    H = {}
    for f in files:
        x = pd.read_csv(f, usecols=cols, on_bad_lines='skip', low_memory=False); x['dt'] = pd.to_datetime(x.forecastdate, format='%m/%d/%y', errors='coerce'); x = x.dropna(subset=['dt', 'winner_Dparty']); cyc = int(x.cycle.iloc[0]); end = x.dt.max(); x['version'] = x.expression.str.strip('_'); x['p'] = x.winner_Dparty.astype(float)
        for h in (0, 7, 14, 30, 60, 90):
            d0 = end - pd.Timedelta(days=h); s = x[x.dt == d0]
            if s.empty: s = x[x.dt <= d0].sort_values('dt').groupby(['district', 'version']).tail(1)
            H[(cyc, h)] = s.assign(cycle=cyc, horizon=h)[['cycle', 'horizon', 'district', 'version', 'p', 'mean_netpartymargin', 'p10_netpartymargin', 'p90_netpartymargin', 'dt']]
    C = pd.concat(H.values()); C['s_id'] = C.district.str.replace('-', '', regex=False)
    out = pd.read_csv('data/jhk/jhk_output.csv'); out = out[(out.forecast == 'fte') & out.election_id.isin(['2020_house', '2022_house'])].assign(cycle=lambda d: d.election_id.str[:4].astype(int), y=lambda d: 1 - d.probwin_outcome)[['cycle', 's_id', 'y']]
    C = C.merge(out, on=['cycle', 's_id'], how='left'); print('without outcome:', int(C.y.isna().sum()), 'of', len(C))
    os.makedirs(os.path.dirname(EXTRACT), exist_ok=True); C.drop(columns=['dt']).to_csv(EXTRACT, index=False)
def analysis():
    rng = np.random.default_rng(50); os.makedirs('outputs_local', exist_ok=True)
    C = pd.read_csv(EXTRACT).rename(columns={'ciclo': 'cycle', 'horizonte': 'horizon', 'versao': 'version'})   # accepts the older extract with Portuguese column names
    print('without outcome:', int(C.y.isna().sum()), 'of', len(C))
    f18 = pd.read_csv('data/538/forecast_results_2018.csv'); f18 = f18[f18.branch == 'House'].assign(cycle=2018, s_id=lambda d: d.race.str.replace('-', '', regex=False), version=lambda d: d.version, p=lambda d: d.Democrat_WinProbability, y=lambda d: d.Democrat_Won.astype(float), horizon=0)[['cycle', 'horizon', 's_id', 'version', 'p', 'y']]
    C['p'] = C.p.astype(float); allc = pd.concat([C[['cycle', 'horizon', 's_id', 'version', 'p', 'y']], f18]); allc['state'] = allc.s_id.str[:2]
    def ab(g):
        pos, neg = g.y == 1, g.y == 0; a = ((1 - g.p[pos]) ** 2).mean(); b = (g.p[neg] ** 2).mean(); return pd.Series(dict(n=len(g), pi=g.y.mean(), BS=((g.y - g.p) ** 2).mean(), alpha=a, beta=b, diff=a - b, BBS=(a + b) / 2, BS_cal=(g.p * (1 - g.p)).mean()))
    fin = allc[allc.horizon == 0].dropna(subset=['y']); T = fin.groupby(['cycle', 'version']).apply(ab).reset_index(); T['ratio'] = T.BS / T.BS_cal; T.to_csv('outputs_local/us_house_538_versions.csv', index=False)
    print('=== final forecast by version (P(D)); pi = share of districts won by the Democrat ===')
    print(T.round(4).to_string(index=False))
    print('\n=== paired Brier differences (bootstrap by state, B = 4000); negative = the first is better ===')
    for cyc in (2018, 2020, 2022):
        W = fin[fin.cycle == cyc].pivot_table(index=['s_id', 'state'], columns='version', values='p').reset_index(); y = fin[(fin.cycle == cyc) & (fin.version == 'classic')].set_index('s_id').y.reindex(W.s_id).values; ests = W.state.values; ue = pd.unique(ests); ix = {e: np.where(ests == e)[0] for e in ue}
        for a_, b_ in (('deluxe', 'classic'), ('classic', 'lite'), ('deluxe', 'lite')):
            d_ = (y - W[a_].values) ** 2 - (y - W[b_].values) ** 2; bt = []
            for _ in range(4000):
                sel = np.concatenate([ix[e] for e in rng.choice(ue, len(ue))]); bt.append(d_[sel].mean())
            print(f"  {cyc} {a_:7s} − {b_:7s}: {d_.mean():+.4f} [{np.percentile(bt, 2.5):+.4f}; {np.percentile(bt, 97.5):+.4f}] (n={len(d_)}, {len(ue)} states)")
    print('\n=== evolution over the campaign (Brier and alpha-beta by horizon, days before the election) ===')
    Hh = allc[allc.cycle.isin([2020, 2022])].dropna(subset=['y']).groupby(['cycle', 'version', 'horizon']).apply(ab).reset_index()
    print(Hh.pivot_table(index=['cycle', 'version'], columns='horizon', values='BS').round(4).to_string()); print(); print(Hh.pivot_table(index=['cycle', 'version'], columns='horizon', values='diff').round(3).to_string())
if __name__ == '__main__':
    if '--extract' in sys.argv: extract_raw()
    else: analysis()
