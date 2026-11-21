"""Additional robustness checks: pooled and fixed-effects OLS on the same features,
XGBoost grid, ex ante combination, ARIMA with a 2011 break dummy, AIC-selected ARIMA,
Holm correction, CAGR 2011-2024.

Output: output/robustness_extra.csv, robustness_cv_extra.csv, cagr_2011_2024.csv,
dm_holm.csv
"""
import os
import warnings
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
t09 = SourceFileLoader("t09", os.path.join(BASE, "09_tree_models.py")).load_module()
m06 = t09.m06
TARGET = t09.TARGET


def metr(actual, pred):
    e = np.asarray(actual) - np.asarray(pred)
    return float(np.mean(np.abs(e))), float(np.sqrt(np.mean(e ** 2))), \
        float(np.mean(np.abs(e) / np.asarray(actual)) * 100)


def dm_vs(ref, cv, name_ref='REF'):
    """DM (squared) of cv['predicted'] against ref['predicted'] on common rows."""
    w = ref[['geo', 'year', 'actual', 'predicted']].merge(
        cv[['geo', 'year', 'predicted']], on=['geo', 'year'], suffixes=('_ref', ''))
    w['err_REF'] = w.actual - w.predicted_ref
    w['err_M'] = w.actual - w.predicted
    dm, p = m06.diebold_mariano(w.err_REF, w.err_M, h=1, loss='squared')
    t, pp, _, _ = m06.panel_dm(w, 'REF', 'M', 'squared')
    better = int((w.assign(a=w.err_REF.abs(), b=w.err_M.abs()).groupby('geo')[['a', 'b']]
                  .mean().pipe(lambda d: (d.b < d.a).sum())))
    return dm, p, t, pp, better, len(w)


class FEOLS:
    """OLS with country dummies (geo_code -> one-hot)."""

    def __init__(self):
        self.lr = LinearRegression()

    def _x(self, X):
        X = X.copy()
        d = pd.get_dummies(X.pop('geo_code').astype(int), prefix='c').astype(float)
        d = d.reindex(columns=[f'c_{i}' for i in range(27)], fill_value=0.0)
        return pd.concat([X.reset_index(drop=True), d.reset_index(drop=True)], axis=1)

    def fit(self, X, y):
        self.lr.fit(self._x(X), y)
        return self

    def predict(self, X):
        return self.lr.predict(self._x(X))


class PooledOLS:
    def __init__(self):
        self.lr = LinearRegression()

    def fit(self, X, y):
        self.lr.fit(X.drop(columns='geo_code'), y)
        return self

    def predict(self, X):
        return self.lr.predict(X.drop(columns='geo_code'))


def local_cv(df, fit_predict, name):
    rows = []
    for geo in sorted(df.geo.unique()):
        ts = df[df.geo == geo][['year', TARGET]].dropna().sort_values('year').set_index('year')[TARGET]
        years = ts.index.tolist()
        for k in range(8, len(years) - 1):
            train = ts.iloc[:k + 1]
            try:
                pred = fit_predict(train, years[k + 1])
            except Exception:
                continue
            rows.append({'geo': geo, 'year': years[k + 1], 'actual': ts.iloc[k + 1],
                         'predicted': pred, 'model': name})
    return pd.DataFrame(rows)


def arima_step(train, next_year):
    step = (train.index >= 2011).astype(float)
    if step.min() == step.max():          # no break inside the window
        return float(ARIMA(train.values, order=(1, 1, 1)).fit().forecast(1)[0])
    f = ARIMA(train.values, order=(1, 1, 1), exog=step).fit()
    return float(f.forecast(1, exog=np.array([[1.0 if next_year >= 2011 else 0.0]]))[0])


def arima_auto(train, next_year):
    best, best_aic = None, np.inf
    for p in range(3):
        for q in range(3):
            for trend in ['n', 't']:
                try:
                    f = ARIMA(train.values, order=(p, 1, q), trend=trend).fit()
                except Exception:
                    continue
                if f.aic < best_aic:
                    best, best_aic = f, f.aic
    return float(best.forecast(1)[0])


def main():
    df = pd.read_csv(os.path.join(BASE, "data", "panel_dataset.csv"))
    df_ts = df[df[TARGET].notna()].copy()
    cv_all = pd.concat([pd.read_csv(os.path.join(OUT, "cv_results.csv")),
                        pd.read_csv(os.path.join(OUT, "cv_results_trees.csv"))])
    arima = cv_all[cv_all.model == 'ARIMA']
    xgbm = cv_all[cv_all.model == 'XGBoost']
    rows, extra_cv = [], []

    def add(name, cv, note=''):
        mae, rmse, mape = metr(cv.actual, cv.predicted)
        r = {'check': name, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape, 'n': len(cv), 'note': note}
        for refname, ref in [('ARIMA', arima), ('XGBoost', xgbm)]:
            dm, p, tt, pp, better, n = dm_vs(ref, cv)
            r.update({f'DM_vs_{refname}': dm, f'p_pool_vs_{refname}': p,
                      f't_panel_vs_{refname}': tt, f'p_panel_vs_{refname}': pp,
                      f'countries_better_than_{refname}': better})
        rows.append(r)
        extra_cv.append(cv.assign(model=name))
        print(f"{name:34s} MAE={mae:.3f} RMSE={rmse:.3f} MAPE={mape:.2f} n={len(cv)}")

    # 1. panel linear models
    cv, _ = t09.run_panel_model(df_ts, PooledOLS, 'Pooled OLS', forecast=False)
    add('Pooled OLS (same features)', cv)
    cv, _ = t09.run_panel_model(df_ts, FEOLS, 'FE OLS', forecast=False)
    add('FE OLS (same features + country FE)', cv)

    # 2. XGBoost grid
    grid = []
    for depth in [3, 4, 6]:
        for lr, n in [(0.05, 400), (0.1, 200)]:
            for sub in [0.8, 1.0]:
                f = lambda: xgb.XGBRegressor(n_estimators=n, max_depth=depth, learning_rate=lr,
                                             subsample=sub, colsample_bytree=0.8, reg_alpha=0.1,
                                             reg_lambda=1.0, random_state=42, verbosity=0)
                cvg, _ = t09.run_panel_model(df_ts, f, 'XGB grid', forecast=False)
                grid.append({'max_depth': depth, 'learning_rate': lr, 'n_estimators': n,
                             'subsample': sub, 'MAE': metr(cvg.actual, cvg.predicted)[0]})
    grid = pd.DataFrame(grid)
    grid.to_csv(os.path.join(OUT, "xgb_sensitivity.csv"), index=False)
    print("XGB grid MAE range:", round(grid.MAE.min(), 4), "-", round(grid.MAE.max(), 4))

    # 3. ex ante combination
    df_lag = df_ts.sort_values(['geo', 'year']).copy()
    for c in t09.EXTRA_FEATURES:
        df_lag[c] = df_lag.groupby('geo')[c].shift(1)
    cv_xl, _ = t09.run_panel_model(df_lag, t09.xgb_factory, 'XGB ex ante', forecast=False)
    ens = arima[['geo', 'year', 'actual', 'predicted']].merge(
        cv_xl[['geo', 'year', 'predicted']], on=['geo', 'year'], suffixes=('_a', '_x'))
    ens['predicted'] = (ens.predicted_a + ens.predicted_x) / 2
    add('Ensemble ex ante (ARIMA + XGB lagged exog)', ens[['geo', 'year', 'actual', 'predicted']])

    # 4. ARIMA with a 2011 break dummy
    add('ARIMA(1,1,1) + 2011 break dummy', local_cv(df_ts, arima_step, 'ARIMA step'))

    # 5. ARIMA auto (AIC)
    add('ARIMA auto-AIC (p,q<=2, drift or not)', local_cv(df_ts, arima_auto, 'ARIMA auto'))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "robustness_extra.csv"), index=False)
    pd.concat(extra_cv).to_csv(os.path.join(OUT, "robustness_cv_extra.csv"), index=False)
    pd.set_option('display.width', 250)
    print(res.round(4).to_string(index=False))

    # 6. Holm correction
    dm = pd.read_csv(os.path.join(OUT, "dm_test_all.csv"))
    dm = dm[dm.loss == 'squared'].copy()
    for col in ['p_pooled', 'p_panel']:
        order = dm[col].sort_values().index
        m = len(order)
        adj, run = {}, 0.0
        for k, idx in enumerate(order):
            run = max(run, min(1.0, (m - k) * dm.loc[idx, col]))
            adj[idx] = run
        dm[col + '_holm'] = pd.Series(adj)
    dm.to_csv(os.path.join(OUT, "dm_holm.csv"), index=False)
    print(dm[['benchmark', 'model', 'p_pooled', 'p_pooled_holm', 'p_panel', 'p_panel_holm']].to_string(index=False))

    # 7. CAGR after the break
    piv = df.pivot_table(index='geo', columns='year', values=TARGET)
    cagr = pd.DataFrame({
        'cagr_2010_2024': ((piv[2024] / piv[2010]) ** (1 / 14) - 1) * 100,
        'cagr_2011_2024': ((piv[2024] / piv[2011]) ** (1 / 13) - 1) * 100,
        'level_2010': piv[2010], 'level_2011': piv[2011], 'level_2024': piv[2024],
        'drop_2011': piv[2011] - piv[2010]}).sort_values('cagr_2011_2024', ascending=False)
    cagr.to_csv(os.path.join(OUT, "cagr_2011_2024.csv"))
    print(cagr.round(2).to_string())
    mean = piv.mean()
    print('EU mean 2010/2011/2024:', round(mean[2010], 3), round(mean[2011], 3), round(mean[2024], 3),
          'CAGR 2011-2024:', round(((mean[2024] / mean[2011]) ** (1 / 13) - 1) * 100, 2),
          'countries dropping in 2011:', int((cagr.drop_2011 < 0).sum()))


if __name__ == "__main__":
    main()
