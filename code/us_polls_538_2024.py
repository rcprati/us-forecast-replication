"""
Builds the state-level 2024 presidential poll table from FiveThirtyEight's president_polls.csv (Wayback Machine capture of 2024-12-02; only polls ended between 14 Oct and 4 Nov 2024).
Keeps STATE rows only (excludes the Maine and Nebraska congressional-district polls), the general-election, non-hypothetical Harris vs Trump question, one row per poll (population priority lv > rv > v > a).
days = days between the end of fieldwork and 2024-11-05. Input: data/538/president_polls_2024_wayback.csv. Output: outputs_local/polls_538_2024_state.csv. Run from the repository root.
"""
import os, pandas as pd
os.makedirs('outputs_local', exist_ok=True)
p = pd.read_csv('data/538/president_polls_2024_wayback.csv', low_memory=False)
p = p[(p.hypothetical == False) & (p.stage == 'general') & p.candidate_name.isin(['Kamala Harris', 'Donald Trump'])]
ST = {'Alaska': 'AK', 'Arizona': 'AZ', 'California': 'CA', 'Colorado': 'CO', 'Florida': 'FL', 'Georgia': 'GA', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'Wisconsin': 'WI', 'Wyoming': 'WY'}
p = p[p.state.isin(ST)].copy(); p['state_abbr'] = p.state.map(ST); p['rk'] = p.population.map({'lv': 0, 'rv': 1, 'v': 2, 'a': 3}).fillna(9)
w = p.pivot_table(index=['poll_id', 'state_abbr', 'pollster', 'end_date', 'population', 'rk', 'sample_size'], columns='candidate_name', values='pct').reset_index().dropna(subset=['Kamala Harris', 'Donald Trump'])
w = w.sort_values('rk').drop_duplicates('poll_id'); w['end'] = pd.to_datetime(w.end_date, format='%m/%d/%y'); w['days'] = (pd.Timestamp('2024-11-05') - w.end).dt.days
out = pd.DataFrame(dict(state=w.state_abbr, source=w.pollster, end=w.end.dt.strftime('%Y-%m-%d'), D=w['Kamala Harris'], R=w['Donald Trump'], days=w.days, universe=w.population.str.upper().replace({'A': 'RV', 'V': 'RV'}), n=w.sample_size, year=2024))
out.to_csv('outputs_local/polls_538_2024_state.csv', index=False); print(f"{len(out)} state polls; {out.state.nunique()} states; days {out.days.min()}-{out.days.max()}")
