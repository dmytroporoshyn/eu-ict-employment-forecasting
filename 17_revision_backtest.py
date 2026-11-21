"""Multi-horizon rolling-origin backtest (origins 2012-2023, h = 1-5) and final forecasts
(origin 2024). Panel specifications: exante (main; predictors at t-1, fold-wise
medians), exante_cty (within-country imputation), conditional (predictors of the
target year), legacy (reproduces output/cv_results.csv).

Output: output/revision/backtest.csv. Option --aic adds AIC-selected ARIMA.
"""
import logging
import os
import sys
import time
import warnings
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings('ignore')
logging.getLogger('cmdstanpy').disabled = True
logging.getLogger('prophet').setLevel(logging.ERROR)

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output", "revision")
os.makedirs(OUT, exist_ok=True)
m04 = SourceFileLoader("m04", os.path.join(BASE, "04_models.py")).load_module()

TARGET = 'ict_specialists_pct'
EXOG = ['hightech_emp_pct', 'gdp_pc_eur', 'rd_pct_gdp', 'internet_use_pct',
        'services_employment_pct', 'broadband_per100', 'gdp_pc_usd',
        'ict_service_exports_pct']
LAGF = [f'{TARGET}_lag1', f'{TARGET}_lag2', f'{TARGET}_lag3',
        f'{TARGET}_roll2', f'{TARGET}_roll3', f'{TARGET}_diff1']
FEATURES = LAGF + EXOG + ['trend', 'geo_code']
H = 5
ORIGINS = list(range(2012, 2025))       # 2024 = final forecast origin
FIRST_YEAR = 2004
LAST_DATA_YEAR = 2024
RS = 42
Z80, Z95 = stats.norm.ppf(0.90), stats.norm.ppf(0.975)


# ----------------------------------------------------------------- models
def xgb_factory():
    return xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                            reg_lambda=1.0, random_state=RS, verbosity=0)


class FEOLS:
    """OLS with country dummies (geo_code -> one-hot), as in 12_robustness.py."""

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


PANEL_MODELS = {
    'XGBoost': xgb_factory,
    'Random Forest': lambda: RandomForestRegressor(
        n_estimators=200, max_features=1 / 3, min_samples_leaf=1,
        random_state=RS, n_jobs=-1),
    'Decision Tree': lambda: DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=5, random_state=RS),
    'FE OLS': FEOLS,
    'Pooled OLS': PooledOLS,
}


# ------------------------------------------------------------ panel engine
def exog_frame(panel, spec, T):
    """Exogenous features up to T+H as seen at origin T (exante: x_{t-1}, held at x_T)."""
    years = range(FIRST_YEAR, T + H + 1)
    idx = pd.MultiIndex.from_product([sorted(panel.geo.unique()), years], names=['geo', 'year'])
    x = panel.set_index(['geo', 'year'])[EXOG].reindex(idx)
    if spec.startswith('exante'):
        src = x.copy()
        fut = src.index.get_level_values('year') > T
        last = x.xs(T, level='year')                 # last observed values (year T)
        src.loc[fut, :] = last.reindex(src.index[fut].get_level_values('geo')).values
        feat = src.groupby(level='geo').shift(1)     # feature of year t = x_{t-1}
    else:
        feat = x
    return feat


def build_rows(ydf, feat):
    """Lag features (via the original create_xgb_features) + exogenous features."""
    f = m04.create_xgb_features(ydf)            # sorts by geo, year; adds trend, geo_code
    f['trend'] = f['year'] - FIRST_YEAR
    f = f.drop(columns=[c for c in EXOG if c in f.columns])
    f = f.merge(feat.reset_index(), on=['geo', 'year'], how='left')
    f = f.sort_values(['geo', 'year']).reset_index(drop=True)
    return f


def impute(train, test, how, med):
    """Fill gaps with training-fold medians (within-country first for exante_cty)."""
    train, test = train.copy(), test.copy()
    if how == 'country':
        # within-country fill from rows visible at the origin
        both = pd.concat([train.assign(_t=0), test.assign(_t=1)]).sort_values(['geo', 'year'])
        both[EXOG] = both.groupby('geo')[EXOG].transform(lambda s: s.ffill().bfill())
        train = both[both._t == 0].drop(columns='_t').sort_values(['geo', 'year'])
        test = both[both._t == 1].drop(columns='_t').sort_values(['geo', 'year'])
    train[EXOG] = train[EXOG].fillna(med)
    test[EXOG] = test[EXOG].fillna(med)
    return train, test


def panel_origin(panel, T, spec, models):
    """Fit panel models on rows <= T and forecast T+1..T+H recursively."""
    ydf = panel[(panel.year <= T) & panel[TARGET].notna()][['geo', 'country', 'year', TARGET]].copy()
    feat = exog_frame(panel, spec, T)
    rows = build_rows(ydf, feat)
    train = rows.dropna(subset=LAGF + [TARGET]).copy()
    train = train[train.year >= 2007]
    full_med = None
    if spec == 'legacy':
        # original code: medians over the full model sample (all years, x_t)
        allrows = build_rows(panel[panel[TARGET].notna()][['geo', 'country', 'year', TARGET]],
                             exog_frame(panel, 'legacy', LAST_DATA_YEAR))
        allrows = allrows.dropna(subset=LAGF + [TARGET])
        full_med = allrows[EXOG].median()
    how = 'country' if spec == 'exante_cty' else 'median'
    med = full_med if full_med is not None else train[EXOG].median()   # fold-wise medians
    out = []
    actual = panel.set_index(['geo', 'year'])[TARGET]
    names = sorted(panel.geo.unique())
    cmap = panel.groupby('geo').country.first()
    for name in models:
        hist = ydf.copy()
        model = None
        for h in range(1, H + 1):
            t = T + h
            step = pd.DataFrame({'geo': names, 'year': t})
            step['country'] = step.geo.map(cmap)
            step[TARGET] = np.nan
            r = build_rows(pd.concat([hist, step], ignore_index=True), feat)
            test = r[r.year == t].copy()
            trn_i, test_i = impute(train, test, how, med)
            if model is None:
                model = PANEL_MODELS[name]()
                model.fit(trn_i[FEATURES].astype(float), trn_i[TARGET].astype(float))
            pred = model.predict(test_i[FEATURES].astype(float))
            step[TARGET] = pred
            hist = pd.concat([hist, step], ignore_index=True)
            for g, p in zip(test_i.geo, pred):
                a = actual.get((g, t), np.nan) if t <= LAST_DATA_YEAR else np.nan
                out.append({'spec': spec, 'model': name, 'origin': T, 'geo': g, 'year': t,
                            'h': h, 'actual': a, 'forecast': float(p)})
            if spec in ('legacy',) and h == 1:
                break
    return out


# ------------------------------------------------------------ local engine
def local_origin(panel, T, with_aic=False):
    from prophet import Prophet
    out = []
    for g in sorted(panel.geo.unique()):
        s = panel[(panel.geo == g) & panel[TARGET].notna()].set_index('year')[TARGET]
        tr = s[s.index <= T]
        if len(tr) < 9:
            continue
        years = [T + h for h in range(1, H + 1)]
        act = [s.get(y, np.nan) for y in years]

        def emit(model, fc, lo80=None, hi80=None, lo95=None, hi95=None):
            for i, y in enumerate(years):
                out.append({'spec': 'local', 'model': model, 'origin': T, 'geo': g, 'year': y,
                            'h': i + 1, 'actual': act[i], 'forecast': float(fc[i]),
                            'lo80': np.nan if lo80 is None else float(lo80[i]),
                            'hi80': np.nan if hi80 is None else float(hi80[i]),
                            'lo95': np.nan if lo95 is None else float(lo95[i]),
                            'hi95': np.nan if hi95 is None else float(hi95[i])})

        # naive random walk (no drift)
        last = tr.iloc[-1]
        sd = np.sqrt(np.mean(np.diff(tr.values) ** 2))
        hs = np.arange(1, H + 1)
        fc = np.repeat(last, H)
        emit('Naive', fc, fc - Z80 * sd * np.sqrt(hs), fc + Z80 * sd * np.sqrt(hs),
             fc - Z95 * sd * np.sqrt(hs), fc + Z95 * sd * np.sqrt(hs))

        # ARIMA(1,1,1) without constant (same call as 04_models.py)
        try:
            f = ARIMA(tr, order=(1, 1, 1)).fit()
            gf = f.get_forecast(H)
            c80, c95 = gf.conf_int(alpha=0.2).values, gf.conf_int(alpha=0.05).values
            emit('ARIMA', gf.predicted_mean.values, c80[:, 0], c80[:, 1], c95[:, 0], c95[:, 1])
        except Exception as e:
            print('ARIMA fail', g, T, e)

        # ARIMA(1,1,1) with step dummies for the 2011 and 2021 breaks
        try:
            ex_cols = [b for b in (2011, 2021) if tr.index.min() < b <= tr.index.max()]
            if ex_cols:
                ex = np.column_stack([(tr.index >= b).astype(float) for b in ex_cols])
                f = ARIMA(tr.values, order=(1, 1, 1), exog=ex).fit()
                exf = np.column_stack([np.ones(H) for _ in ex_cols])
                gf = f.get_forecast(H, exog=exf)
                fc_ = gf.predicted_mean
            else:
                fc_ = ARIMA(tr.values, order=(1, 1, 1)).fit().forecast(H)
            emit('ARIMA breaks', np.asarray(fc_))
        except Exception as e:
            print('ARIMA breaks fail', g, T, e)

        # ETS: additive damped trend; intervals by simulation
        try:
            f = ExponentialSmoothing(tr, trend='add', seasonal=None, damped_trend=True).fit(optimized=True)
            fc = f.forecast(H).values
            sim = f.simulate(H, repetitions=2000, error='add', anchor='end',
                             random_state=RS).values
            q = np.percentile(sim, [10, 90, 2.5, 97.5], axis=1)
            emit('ETS', fc, q[0], q[1], q[2], q[3])
        except Exception as e:
            print('ETS fail', g, T, e)

        # Prophet (as in 04_models.py), intervals from predictive samples
        try:
            dfp = pd.DataFrame({'ds': pd.to_datetime(tr.index.astype(str), format='%Y'),
                                'y': tr.values})
            m = Prophet(yearly_seasonality=False, weekly_seasonality=False,
                        daily_seasonality=False, growth='linear',
                        changepoint_prior_scale=0.05)
            m.fit(dfp)
            fut = m.make_future_dataframe(periods=H, freq='YS')
            fcp = m.predict(fut)['yhat'].values[-H:]
            np.random.seed(RS)
            smp = m.predictive_samples(fut)['yhat'][-H:, :]
            q = np.percentile(smp, [10, 90, 2.5, 97.5], axis=1)
            emit('Prophet', fcp, q[0], q[1], q[2], q[3])
        except Exception as e:
            print('Prophet fail', g, T, e)

        if with_aic:
            best, best_aic = None, np.inf
            for p in range(3):
                for qq in range(3):
                    for trend in ['n', 't']:
                        try:
                            ff = ARIMA(tr.values, order=(p, 1, qq), trend=trend).fit()
                        except Exception:
                            continue
                        if ff.aic < best_aic:
                            best, best_aic = ff, ff.aic
            emit('ARIMA AIC', best.forecast(H))
    return out


def main():
    with_aic = '--aic' in sys.argv
    panel = pd.read_csv(os.path.join(BASE, "data", "panel_dataset.csv"))
    rows = []
    t0 = time.time()

    # 0. replication check of the original XGBoost cross-validation
    leg = []
    for T in range(2012, 2024):
        leg += panel_origin(panel, T, 'legacy', ['XGBoost'])
    leg = pd.DataFrame(leg)
    ref = pd.read_csv(os.path.join(BASE, "output", "cv_results.csv"))
    ref = ref[ref.model == 'XGBoost']
    chk = leg.merge(ref, on=['geo', 'year'], suffixes=('', '_ref'))
    diff = float((chk.forecast - chk.predicted).abs().max())
    msg = (f"Legacy replication: {len(chk)} of {len(ref)} rows matched, "
           f"max |diff| vs output/cv_results.csv = {diff:.2e}")
    print(msg, flush=True)
    with open(os.path.join(OUT, "replication_check.txt"), "w", encoding="utf-8") as fh:
        fh.write(msg + "\n")
    assert len(chk) == len(ref) and diff < 1e-6, "engine does not reproduce the original XGBoost"
    rows += leg.to_dict('records')

    # 1. panel models
    for spec in ['exante', 'exante_cty', 'conditional']:
        for T in ORIGINS:
            if spec != 'exante' and T == 2024:
                continue
            rows += panel_origin(panel, T, spec, list(PANEL_MODELS))
        print(f"panel {spec} done, {time.time() - t0:.0f} s", flush=True)

    # 2. local models
    for T in ORIGINS:
        rows += local_origin(panel, T, with_aic)
        print(f"local origin {T} done, {time.time() - t0:.0f} s", flush=True)

    res = pd.DataFrame(rows)
    fn = "backtest_aic.csv" if with_aic else "backtest.csv"
    if with_aic:
        res = res[res.model == 'ARIMA AIC']
    res.to_csv(os.path.join(OUT, fn), index=False)
    print(f"Saved output/revision/{fn}: {len(res)} rows, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
