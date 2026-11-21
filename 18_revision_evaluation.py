"""Evaluation of the backtest: accuracy by horizon, prediction intervals, two-way
cluster-robust tests and bootstrap, nested combination, 2021 break, robustness and
country results. Output: output/revision/*.csv, results_revision.md
"""
import os
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
from scipy import stats

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output", "revision")
m06 = SourceFileLoader("m06", os.path.join(BASE, "06_dm_test.py")).load_module()
TARGET = 'ict_specialists_pct'

LOCAL = ['Naive', 'ARIMA', 'ETS', 'Prophet']
PANEL = ['XGBoost', 'Random Forest', 'Decision Tree', 'FE OLS', 'Pooled OLS']
MAIN = ['Naive', 'ARIMA', 'ETS', 'Prophet', 'Decision Tree', 'Random Forest', 'XGBoost',
        'Ensemble', 'FE OLS', 'Pooled OLS']
STAT_GROUP, ML_GROUP = ['ARIMA', 'ETS'], ['XGBoost', 'Random Forest', 'Decision Tree', 'Prophet']
B = 9999
RNG = np.random.default_rng(2026)
MD = []      # markdown report lines


def md(s=''):
    MD.append(s)
    print(s)


def table_md(df, floatfmt=3):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append('' if np.isnan(v) else (f'{v:.{floatfmt}f}' if abs(v) < 1000 else f'{v:.0f}'))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


# ------------------------------------------------------------------ loading
def load():
    bt = pd.read_csv(os.path.join(OUT, "backtest.csv"))
    p = os.path.join(OUT, "backtest_aic.csv")
    if os.path.exists(p):
        bt = pd.concat([bt, pd.read_csv(p)], ignore_index=True)
    bt = bt[bt.origin <= 2023]                         # 2024 = final forecasts (no actuals)
    loc = bt[bt.spec == 'local']
    sets = {}
    for spec in ['exante', 'exante_cty', 'conditional']:
        pan = bt[(bt.spec == spec) & bt.model.isin(PANEL)]
        w = pd.concat([loc, pan]).pivot_table(index=['origin', 'geo', 'year', 'h'],
                                             columns='model', values='forecast').reset_index()
        act = bt.groupby(['geo', 'year'])['actual'].first()
        w['actual'] = [act.get((g, y), np.nan) for g, y in zip(w.geo, w.year)]
        w['Ensemble'] = (w['ARIMA'] + w['XGBoost']) / 2
        sets[spec] = w
    native = loc[['model', 'origin', 'geo', 'year', 'h', 'lo80', 'hi80', 'lo95', 'hi95']]
    return bt, sets, native


def metrics(a, f):
    e = np.asarray(a) - np.asarray(f)
    return {'MAE': np.mean(np.abs(e)), 'RMSE': np.sqrt(np.mean(e ** 2)),
            'MAPE': np.mean(np.abs(e) / np.asarray(a)) * 100, 'Bias': np.mean(e), 'N': len(e)}


# -------------------------------------------------------------------- tests
def twoway(df, d):
    """Mean loss differential: two-way cluster-robust t test and pigeonhole bootstrap."""
    d = np.asarray(d, float)
    n, dbar = len(d), d.mean()
    u = d - dbar
    geo, yr = df['geo'].values, df['year'].values
    gc, gt = len(np.unique(geo)), len(np.unique(yr))
    sc = pd.Series(u).groupby(geo).sum().values
    st = pd.Series(u).groupby(yr).sum().values
    vc = (sc ** 2).sum() / n ** 2 * gc / (gc - 1)
    vt = (st ** 2).sum() / n ** 2 * gt / (gt - 1)
    v0 = (u ** 2).sum() / n ** 2 * n / (n - 1)
    v = vc + vt - v0
    if v <= 0:
        v = max(vc, vt)
    t = dbar / np.sqrt(v)
    p = 2 * stats.t.sf(abs(t), df=min(gc, gt) - 1)
    # pigeonhole bootstrap: resample countries and years independently
    M = pd.DataFrame({'g': geo, 'y': yr, 'd': d}).pivot_table(index='g', columns='y', values='d')
    mask = (~M.isna()).values.astype(float)
    Mz = np.nan_to_num(M.values)
    wc = RNG.multinomial(M.shape[0], np.ones(M.shape[0]) / M.shape[0], size=B).astype(float)
    wt = RNG.multinomial(M.shape[1], np.ones(M.shape[1]) / M.shape[1], size=B).astype(float)
    num = np.einsum('bc,ct,bt->b', wc, Mz, wt)
    den = np.einsum('bc,ct,bt->b', wc, mask, wt)
    mb = num / den
    pb = float(np.mean(np.abs(mb - dbar) >= abs(dbar)))
    return dbar, t, p, pb


def loss(e, a, kind):
    e = np.asarray(e, float)
    if kind == 'squared':
        return e ** 2
    if kind == 'absolute':
        return np.abs(e)
    return np.abs(e) / np.asarray(a, float)


def compare(w, test, bench, kind='squared', pooled=True):
    """Positive statistic => `test` is more accurate than `bench`."""
    d = loss(w.actual - w[bench], w.actual, kind) - loss(w.actual - w[test], w.actual, kind)
    dbar, t, p, pb = twoway(w, d)
    out = {'test': test, 'bench': bench, 'loss': kind, 'mean_d': dbar,
           't_2way': t, 'p_2way': p, 'p_boot': pb, 'N': len(w)}
    if pooled:
        e_b, e_t = (w.actual - w[bench]).values, (w.actual - w[test]).values
        if kind == 'relative':
            e_b, e_t = e_b / w.actual.values, e_t / w.actual.values
        dm, pp = m06.diebold_mariano(e_b, e_t, h=1, loss='squared' if kind == 'squared' else 'absolute')
        out.update({'DM_pooled': dm, 'p_pooled': pp})
    return out


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return adj


# --------------------------------------------------------- empirical intervals
def empirical_intervals(w, models):
    """Sequential empirical PIs from past log errors at the same horizon."""
    rows = []
    for m in models:
        le = np.log(w.actual / w[m])
        base = w.assign(le=le)
        for (h, T), g in base.groupby(['h', 'origin']):
            cal = base[(base.h == h) & (base.origin + h <= T)]['le'].dropna()
            if len(cal) < 27:
                continue
            q = np.quantile(cal, [0.10, 0.90, 0.025, 0.975])
            for _, r in g.iterrows():
                rows.append({'model': m, 'h': h, 'origin': T, 'geo': r.geo, 'year': r.year,
                             'actual': r.actual, 'forecast': r[m],
                             'lo80': r[m] * np.exp(q[0]), 'hi80': r[m] * np.exp(q[1]),
                             'lo95': r[m] * np.exp(q[2]), 'hi95': r[m] * np.exp(q[3]),
                             'n_cal': len(cal)})
    return pd.DataFrame(rows)


def interval_stats(df, a=0.2, lo='lo80', hi='hi80'):
    y, l, u = df.actual.values, df[lo].values, df[hi].values
    cov = np.mean((y >= l) & (y <= u)) * 100
    width = np.mean(u - l)
    score = np.mean((u - l) + 2 / a * (l - y) * (y < l) + 2 / a * (y - u) * (y > u))
    return cov, width, score


# ----------------------------------------------------------------- nested
def nested_ensemble(w):
    """Choose ensemble components and weights at each origin from past errors only."""
    rows, picks = [], []
    for (h, T), g in w.groupby(['h', 'origin']):
        past = w[(w.h == h) & (w.origin + h <= T)]
        if past.empty:
            continue
        mse = {m: np.mean((past.actual - past[m]) ** 2) for m in STAT_GROUP + ML_GROUP}
        s = min(STAT_GROUP, key=mse.get)
        ml = min(ML_GROUP, key=mse.get)
        ws = (1 / mse[s]) / (1 / mse[s] + 1 / mse[ml])
        picks.append({'h': h, 'origin': T, 'stat': s, 'ml': ml, 'w_stat_invmse': ws})
        for _, r in g.iterrows():
            rows.append({'h': h, 'origin': T, 'geo': r.geo, 'year': r.year,
                         'Nested equal': (r[s] + r[ml]) / 2,
                         'Nested inv-MSE': ws * r[s] + (1 - ws) * r[ml]})
    return pd.DataFrame(rows), pd.DataFrame(picks)


def main():
    bt, sets, native = load()
    w = sets['exante'].dropna(subset=MAIN + ['actual']).copy()
    md('# Revision results (generated by 18_revision_evaluation.py)')
    md()
    md(f"Main specification: panel models with exogenous predictors lagged by one year, "
       f"fold-wise median imputation; common subsample of all models.")
    md()

    # 1. accuracy by horizon
    acc = []
    for h in range(1, 6):
        g = w[w.h == h]
        for m in MAIN:
            r = metrics(g.actual, g[m])
            acc.append({'h': h, 'model': m, **r})
    acc = pd.DataFrame(acc)
    acc.to_csv(os.path.join(OUT, "accuracy_by_horizon.csv"), index=False)
    md('## 1. Accuracy by horizon (common subsample)')
    for h in range(1, 6):
        g = w[w.h == h]
        md(f"\n### h = {h}: N = {len(g)}, target years {g.year.min()}-{g.year.max()}, "
           f"origins {g.origin.min()}-{g.origin.max()}")
        md(table_md(acc[acc.h == h].drop(columns='h')))
    # common target years 2017-2024 for all horizons
    md('\n### Common target years 2017-2024 (all horizons)')
    acc2 = []
    for h in range(1, 6):
        g = w[(w.h == h) & (w.year >= 2017)]
        for m in MAIN:
            acc2.append({'h': h, 'model': m, **metrics(g.actual, g[m])})
    acc2 = pd.DataFrame(acc2)
    acc2.to_csv(os.path.join(OUT, "accuracy_by_horizon_2017_2024.csv"), index=False)
    md(table_md(acc2.pivot(index='model', columns='h', values='MAE').reset_index()))

    # 2. intervals
    md('\n## 2. Prediction intervals')
    emp = empirical_intervals(w, MAIN)
    emp.to_csv(os.path.join(OUT, "intervals_empirical_backtest.csv"), index=False)
    pis = []
    for (m, h), g in emp.groupby(['model', 'h']):
        c80, w80, s80 = interval_stats(g, 0.2, 'lo80', 'hi80')
        c95, w95, s95 = interval_stats(g, 0.05, 'lo95', 'hi95')
        pis.append({'model': m, 'h': h, 'type': 'empirical', 'cov80': c80, 'width80': w80,
                    'score80': s80, 'cov95': c95, 'width95': w95, 'score95': s95, 'N': len(g)})
        nat = native[native.model == m].merge(g[['origin', 'geo', 'year', 'h', 'actual']],
                                              on=['origin', 'geo', 'year', 'h'])
        if len(nat) and nat.lo80.notna().all():
            c80, w80, s80 = interval_stats(nat, 0.2, 'lo80', 'hi80')
            c95, w95, s95 = interval_stats(nat, 0.05, 'lo95', 'hi95')
            pis.append({'model': m, 'h': h, 'type': 'native', 'cov80': c80, 'width80': w80,
                        'score80': s80, 'cov95': c95, 'width95': w95, 'score95': s95,
                        'N': len(nat)})
    pis = pd.DataFrame(pis)
    pis.to_csv(os.path.join(OUT, "interval_coverage.csv"), index=False)
    md('Empirical intervals are evaluated only where at least one earlier origin '
       'provides calibration errors at the same horizon.')
    md(table_md(pis.sort_values(['type', 'model', 'h'])))
    # native intervals on all rows (no calibration constraint)
    natall = native.merge(w[['origin', 'geo', 'year', 'h', 'actual']], on=['origin', 'geo', 'year', 'h'])
    natall = natall.dropna(subset=['lo80'])
    rows = []
    for (m, h), g in natall.groupby(['model', 'h']):
        c80, w80, _ = interval_stats(g, 0.2, 'lo80', 'hi80')
        c95, w95, _ = interval_stats(g, 0.05, 'lo95', 'hi95')
        rows.append({'model': m, 'h': h, 'cov80': c80, 'width80': w80, 'cov95': c95,
                     'width95': w95, 'N': len(g)})
    md('\nNative intervals, all backtest rows:')
    md(table_md(pd.DataFrame(rows)))

    # 3. tests
    md('\n## 3. Tests of equal accuracy, h = 1')
    g1 = w[w.h == 1].copy()
    pairs = [('XGBoost', 'ARIMA'), ('XGBoost', 'ETS'), ('XGBoost', 'Prophet'),
             ('XGBoost', 'Random Forest'), ('XGBoost', 'Decision Tree'),
             ('Random Forest', 'Decision Tree'), ('Random Forest', 'ARIMA'),
             ('ARIMA', 'ETS'), ('ARIMA', 'Prophet'), ('ARIMA', 'Decision Tree'),
             ('Ensemble', 'XGBoost'), ('Ensemble', 'ARIMA'),
             ('ARIMA', 'Naive'), ('XGBoost', 'Naive'), ('Ensemble', 'Naive'),
             ('XGBoost', 'FE OLS'), ('XGBoost', 'Pooled OLS'), ('FE OLS', 'ARIMA'),
             ('Pooled OLS', 'ARIMA')]
    tests = []
    for kind in ['squared', 'absolute', 'relative']:
        for te, be in pairs:
            tests.append(compare(g1, te, be, kind))
    tests = pd.DataFrame(tests)
    # Holm: over the 16 comparisons reported in the article (without the naive benchmark)
    # and, for reference, over all 19 comparisons
    for kind in ['squared', 'absolute', 'relative']:
        k = tests.loss == kind
        kn = k & (tests.test != 'Naive') & (tests.bench != 'Naive')
        tests.loc[kn, 'p_2way_holm'] = holm(tests.loc[kn, 'p_2way'].values)
        tests.loc[k, 'p_2way_holm_all19'] = holm(tests.loc[k, 'p_2way'].values)
    tests.to_csv(os.path.join(OUT, "tests_h1.csv"), index=False)
    for kind in ['squared', 'absolute', 'relative']:
        md(f'\n### loss = {kind}')
        md(table_md(tests[tests.loss == kind].drop(columns='loss')))
    # tests by horizon for key pairs (two-way cluster and bootstrap only)
    md('\n### Key pairs by horizon (squared and relative loss)')
    th = []
    for h in range(1, 6):
        gh = w[w.h == h]
        for te, be in [('XGBoost', 'ARIMA'), ('Random Forest', 'ARIMA'), ('Ensemble', 'ARIMA'),
                       ('Ensemble', 'XGBoost'), ('ARIMA', 'Naive'), ('XGBoost', 'Naive'),
                       ('Ensemble', 'Naive'), ('FE OLS', 'ARIMA'), ('ETS', 'ARIMA')]:
            for kind in ['squared', 'relative']:
                r = compare(gh, te, be, kind, pooled=False)
                r['h'] = h
                th.append(r)
    th = pd.DataFrame(th)
    th.to_csv(os.path.join(OUT, "tests_by_horizon.csv"), index=False)
    md(table_md(th[['h', 'test', 'bench', 'loss', 'mean_d', 't_2way', 'p_2way', 'p_boot', 'N']]))

    # 4. nested ensemble
    md('\n## 4. Nested validation of the combination')
    ne, picks = nested_ensemble(w)
    picks.to_csv(os.path.join(OUT, "nested_picks.csv"), index=False)
    wn = w.merge(ne, on=['h', 'origin', 'geo', 'year'])
    nres, ntest = [], []
    for h in range(1, 6):
        g = wn[wn.h == h]
        for m in ['Nested equal', 'Nested inv-MSE', 'Ensemble', 'ARIMA', 'XGBoost', 'Naive']:
            nres.append({'h': h, 'model': m, **metrics(g.actual, g[m])})
        for te in ['Nested equal', 'Nested inv-MSE']:
            for be in ['ARIMA', 'XGBoost', 'Naive', 'Ensemble']:
                for kind in ['squared', 'absolute', 'relative']:
                    r = compare(g, te, be, kind, pooled=False)
                    r['h'] = h
                    ntest.append(r)
    nres, ntest = pd.DataFrame(nres), pd.DataFrame(ntest)
    nres.to_csv(os.path.join(OUT, "nested_accuracy.csv"), index=False)
    ntest.to_csv(os.path.join(OUT, "nested_tests.csv"), index=False)
    md(table_md(nres))
    md(table_md(ntest[['h', 'test', 'bench', 'loss', 'mean_d', 't_2way', 'p_2way', 'p_boot', 'N']]))
    md('\nComponents chosen (h = 1): ' +
       str(picks[picks.h == 1][['origin', 'stat', 'ml', 'w_stat_invmse']].round(3).to_dict('records')))

    # 5. 2021 break
    md('\n## 5. EU-LFS break in 2021')
    panel = pd.read_csv(os.path.join(BASE, "data", "panel_dataset.csv"))
    piv = panel.pivot_table(index='geo', columns='year', values=TARGET)
    dlt = piv.diff(axis=1)
    md(f"Mean annual change of the share (unweighted, pp): 2013-2020 = "
       f"{dlt.loc[:, 2013:2020].mean().mean():.3f}; 2021 = {dlt[2021].mean():.3f}; "
       f"2022-2024 = {dlt.loc[:, 2022:2024].mean().mean():.3f}; 2020 = {dlt[2020].mean():.3f}; "
       f"countries with a fall in 2021: {(dlt[2021] < 0).sum()}; "
       f"SD of changes in 2021 = {dlt[2021].std():.3f} vs mean SD 2013-2020 = "
       f"{dlt.loc[:, 2013:2020].std().mean():.3f}")
    byyear = []
    for y, g in g1.groupby('year'):
        r = {'year': y}
        for m in MAIN:
            r[m] = np.mean(np.abs(g.actual - g[m]))
        byyear.append(r)
    byyear = pd.DataFrame(byyear)
    byyear.to_csv(os.path.join(OUT, "mae_by_target_year_h1.csv"), index=False)
    md('\nMAE by target year, h = 1:')
    md(table_md(byyear))
    rows = []
    for label, sub in [('all years', g1), ('without 2021', g1[g1.year != 2021]),
                       ('without 2021-2022', g1[~g1.year.isin([2021, 2022])]),
                       ('2013-2020', g1[g1.year <= 2020]), ('2021-2024', g1[g1.year >= 2021])]:
        r = {'sample': label, 'N': len(sub)}
        for m in MAIN:
            r[m] = np.mean(np.abs(sub.actual - sub[m]))
        rows.append(r)
    sub21 = pd.DataFrame(rows)
    sub21.to_csv(os.path.join(OUT, "break2021_mae.csv"), index=False)
    md('\nMAE by subsample, h = 1:')
    md(table_md(sub21))
    t21 = []
    g21 = g1[g1.year != 2021]
    for te, be in [('XGBoost', 'ARIMA'), ('Random Forest', 'ARIMA'), ('Ensemble', 'ARIMA'),
                   ('Ensemble', 'XGBoost'), ('ARIMA', 'Naive'), ('Ensemble', 'Naive'),
                   ('XGBoost', 'Naive'), ('ARIMA', 'ETS'), ('XGBoost', 'Prophet')]:
        for kind in ['squared', 'relative']:
            t21.append(compare(g21, te, be, kind))
    t21 = pd.DataFrame(t21)
    t21.to_csv(os.path.join(OUT, "break2021_tests.csv"), index=False)
    md('\nTests without target year 2021 (h = 1):')
    md(table_md(t21))
    # ARIMA with break dummies
    wb = w.dropna(subset=['ARIMA breaks'])          # column comes from the local-model pivot
    rows = []
    for h in range(1, 6):
        g = wb[wb.h == h]
        for sel, lab in [(g, 'all origins'), (g[g.origin >= 2021], 'origins 2021+')]:
            if len(sel) == 0:
                continue
            r = {'h': h, 'origins': lab, 'N': len(sel),
                 'MAE ARIMA': np.mean(np.abs(sel.actual - sel.ARIMA)),
                 'MAE ARIMA breaks': np.mean(np.abs(sel.actual - sel['ARIMA breaks']))}
            if lab == 'all origins':
                c = compare(sel, 'ARIMA breaks', 'ARIMA', 'squared', pooled=False)
                r.update({'p_2way': c['p_2way'], 'p_boot': c['p_boot']})
            rows.append(r)
    md('\nARIMA(1,1,1) with step dummies (2011, 2021) vs ARIMA(1,1,1):')
    md(table_md(pd.DataFrame(rows)))

    # 6. robustness h = 1
    md('\n## 6. Robustness, h = 1')
    rob = []
    ref = g1.set_index(['origin', 'geo', 'year'])
    for spec in ['conditional', 'exante_cty']:
        ws = sets[spec]
        ws = ws[ws.h == 1].dropna(subset=['XGBoost', 'Random Forest', 'ARIMA', 'actual'])
        for m in ['XGBoost', 'Random Forest', 'Decision Tree', 'FE OLS', 'Pooled OLS', 'Ensemble']:
            x = ws.set_index(['origin', 'geo', 'year'])[[m]].rename(columns={m: 'alt'})
            j = ref.join(x, how='inner').reset_index()
            r = {'spec': spec, 'model': m, **metrics(j.actual, j.alt)}
            for be in ['ARIMA', 'XGBoost', 'Naive']:
                c = compare(j, 'alt', be, 'squared', pooled=True)
                r[f'p2way_vs_{be}'] = c['p_2way']
                r[f'pboot_vs_{be}'] = c['p_boot']
                r[f'ppool_vs_{be}'] = c['p_pooled']
                c = compare(j, 'alt', be, 'relative', pooled=False)
                r[f'p2way_rel_vs_{be}'] = c['p_2way']
            rob.append(r)
    for m in ['ARIMA AIC', 'ARIMA breaks']:
        x = bt[(bt.model == m) & (bt.h == 1)].set_index(['origin', 'geo', 'year'])[['forecast']]
        if x.empty:
            continue
        j = ref.join(x.rename(columns={'forecast': 'alt'}), how='inner').reset_index()
        r = {'spec': 'local', 'model': m, **metrics(j.actual, j.alt)}
        for be in ['ARIMA', 'XGBoost', 'Naive']:
            c = compare(j, 'alt', be, 'squared', pooled=True)
            r[f'p2way_vs_{be}'] = c['p_2way']
            r[f'pboot_vs_{be}'] = c['p_boot']
            r[f'ppool_vs_{be}'] = c['p_pooled']
            c = compare(j, 'alt', be, 'relative', pooled=False)
            r[f'p2way_rel_vs_{be}'] = c['p_2way']
        rob.append(r)
    rob = pd.DataFrame(rob)
    rob.to_csv(os.path.join(OUT, "robustness_h1.csv"), index=False)
    md(table_md(rob))
    # imputation effect: full-sample vs fold-wise medians
    leg = bt[(bt.spec == 'legacy')][['origin', 'geo', 'year', 'forecast']].rename(columns={'forecast': 'legacy'})
    cnd = sets['conditional']
    cnd = cnd[cnd.h == 1][['origin', 'geo', 'year', 'actual', 'XGBoost']]
    j = cnd.merge(leg, on=['origin', 'geo', 'year'])
    md(f"\nImputation effect (XGBoost, current-year predictors): full-sample medians "
       f"MAE = {np.mean(np.abs(j.actual - j.legacy)):.4f}, MAPE = "
       f"{np.mean(np.abs(j.actual - j.legacy) / j.actual) * 100:.3f}; fold-wise medians MAE = "
       f"{np.mean(np.abs(j.actual - j.XGBoost)):.4f}, MAPE = "
       f"{np.mean(np.abs(j.actual - j.XGBoost) / j.actual) * 100:.3f}; N = {len(j)}; "
       f"max |diff| of forecasts = {np.max(np.abs(j.legacy - j.XGBoost)):.4f}")

    # 7. countries
    md('\n## 7. Country-level results, h = 1')
    pc = g1.groupby('geo').apply(lambda g: pd.Series({m: np.mean(np.abs(g.actual - g[m])) for m in MAIN}))
    better = {m: int((pc[m] < pc['ARIMA']).sum()) for m in MAIN if m != 'ARIMA'}
    md('Countries where the model has lower MAE than ARIMA: ' + str(better))
    better_n = {m: int((pc[m] < pc['Naive']).sum()) for m in MAIN if m != 'Naive'}
    md('Countries where the model has lower MAE than Naive: ' + str(better_n))
    ape = g1.groupby('geo').apply(lambda g: pd.Series({m: np.mean(np.abs(g.actual - g[m]) / g.actual) * 100
                                                       for m in MAIN}))
    six = ['ARIMA', 'ETS', 'Prophet', 'Decision Tree', 'Random Forest', 'XGBoost']
    best = ape[six].idxmin(axis=1)
    md('Best of the six models by MAPE: ' + str(best.value_counts().to_dict()))
    best7 = ape[six + ['Naive']].idxmin(axis=1)
    md('Best incl. Naive by MAPE: ' + str(best7.value_counts().to_dict()))
    lvl = g1.groupby('geo').actual.mean().sort_values()
    terc = pd.qcut(lvl, 3, labels=['low', 'mid', 'high'])
    rows = []
    for t in ['low', 'mid', 'high']:
        geos = terc[terc == t].index
        g = g1[g1.geo.isin(geos)]
        r = {'tercile': t, 'range': f"{lvl[geos].min():.1f}-{lvl[geos].max():.1f}", 'N': len(g)}
        for m in ['Naive', 'ARIMA', 'XGBoost', 'Random Forest', 'FE OLS', 'Ensemble']:
            r[m] = np.mean(np.abs(g.actual - g[m]) / g.actual) * 100
        rows.append(r)
    md('\nMAPE by tercile of the average level (2013-2024):')
    md(table_md(pd.DataFrame(rows)))
    top6 = ['SE', 'LU', 'FI', 'EE', 'NL', 'IE']
    g6 = g1[g1.geo.isin(top6)]
    md('\nBias for six leading countries (actual - forecast): ' +
       str({m: round(float(np.mean(g6.actual - g6[m])), 3) for m in MAIN}))
    md('Bias all countries h=1: ' + str({m: round(float(np.mean(g1.actual - g1[m])), 3) for m in MAIN}))
    md(f"Correlation of ARIMA and XGBoost errors (h = 1): "
       f"{np.corrcoef(g1.actual - g1.ARIMA, g1.actual - g1.XGBoost)[0, 1]:.3f}")
    pc.round(3).to_csv(os.path.join(OUT, "mae_by_country_h1.csv"))
    ape.round(2).to_csv(os.path.join(OUT, "mape_by_country_h1.csv"))
    for c in ['SE', 'LU', 'EE', 'GR', 'ES', 'IT', 'CY', 'PT']:
        md(f"  {c}: MAE XGB {pc.loc[c, 'XGBoost']:.3f}, MAPE XGB {ape.loc[c, 'XGBoost']:.2f}, "
           f"MAE ARIMA {pc.loc[c, 'ARIMA']:.3f}, MAE Ens {pc.loc[c, 'Ensemble']:.3f}")
    md(f"MAPE XGB median {ape['XGBoost'].median():.2f}, min {ape['XGBoost'].min():.2f} "
       f"({ape['XGBoost'].idxmin()}), max {ape['XGBoost'].max():.2f} ({ape['XGBoost'].idxmax()})")

    # extrapolation of tree models
    md('\n## 8. Extrapolation of tree models (h = 1)')
    ymax = {T: panel[(panel.year <= T)][TARGET].max() for T in range(2012, 2024)}
    g1['train_max'] = g1.origin.map(ymax)
    newmax = g1[g1.actual > g1.train_max]
    md(f"Test observations above the training maximum: {len(newmax)} "
       f"({sorted(newmax.geo.unique())})")
    for m in ['Decision Tree', 'Random Forest', 'XGBoost', 'ARIMA', 'Naive']:
        above = int((g1[m] > g1.train_max).sum())
        md(f"  {m}: forecasts above the training maximum = {above}; mean forecast for new-maximum "
           f"cases = {newmax[m].mean():.3f} vs actual {newmax.actual.mean():.3f} and training max "
           f"{newmax.train_max.mean():.3f}")
    # all horizons
    w['train_max'] = w.origin.map(ymax)
    for m in ['Random Forest', 'XGBoost', 'Decision Tree']:
        md(f"  all horizons, {m}: forecasts above training max = {int((w[m] > w.train_max).sum())} "
           f"of {len(w)}; max excess = {float((w[m] - w.train_max).max()):.3f}")

    # 9. alternative interval constructions
    md('\n## 9. Alternative empirical intervals: uncentred (used), centred on the median, symmetric')
    alt = []
    for m in ['Ensemble', 'ARIMA', 'XGBoost', 'FE OLS']:
        base = w.assign(lerr=np.log(w.actual / w[m]))
        for kind in ['uncentred', 'centred', 'symmetric']:
            for h in range(1, 6):
                Y, L8, U8, L9, U9 = [], [], [], [], []
                for T, g in base[base.h == h].groupby('origin'):
                    cal = base[(base.h == h) & (base.origin + h <= T)].lerr.dropna().values
                    if len(cal) < 27:
                        continue
                    if kind == 'uncentred':
                        q = np.quantile(cal, [0.10, 0.90, 0.025, 0.975])
                    elif kind == 'centred':
                        q = np.quantile(cal - np.median(cal), [0.10, 0.90, 0.025, 0.975])
                    else:
                        a8, a9 = np.quantile(np.abs(cal), [0.80, 0.95])
                        q = [-a8, a8, -a9, a9]
                    f = g[m].values
                    Y += list(g.actual)
                    L8 += list(f * np.exp(q[0])); U8 += list(f * np.exp(q[1]))
                    L9 += list(f * np.exp(q[2])); U9 += list(f * np.exp(q[3]))
                d = pd.DataFrame({'actual': Y, 'lo80': L8, 'hi80': U8, 'lo95': L9, 'hi95': U9})
                c80, w80, s80 = interval_stats(d, 0.2, 'lo80', 'hi80')
                c95, w95, s95 = interval_stats(d, 0.05, 'lo95', 'hi95')
                alt.append({'model': m, 'type': kind, 'h': h, 'cov80': c80, 'score80': s80,
                            'cov95': c95, 'score95': s95, 'N': len(d)})
    alt = pd.DataFrame(alt)
    alt.to_csv(os.path.join(OUT, "interval_alternatives.csv"), index=False)
    for col in ['cov80', 'score80', 'cov95', 'score95']:
        md(f'\n{col}:')
        md(table_md(alt.pivot_table(index=['model', 'type'], columns='h', values=col).reset_index()))

    with open(os.path.join(OUT, "results_revision.md"), "w", encoding="utf-8") as fh:
        fh.write('\n'.join(MD) + '\n')


if __name__ == "__main__":
    main()
