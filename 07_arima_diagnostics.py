"""Ljung-Box diagnostics for per-country ARIMA(1,1,1) residuals.

Fits the same final ARIMA(1,1,1) models as 04_models.py (full series
per country) and tests residuals for autocorrelation at lag 5.
"""
import os
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
TARGET = 'ict_specialists_pct'

df = pd.read_csv(os.path.join(DATA_DIR, "panel_dataset.csv"))

rows = []
for geo in sorted(df['geo'].unique()):
    ts = df[df['geo'] == geo][['year', TARGET]].dropna().sort_values('year')
    ts = ts.set_index('year')[TARGET]
    if len(ts) < 10:
        continue
    fitted = ARIMA(ts, order=(1, 1, 1)).fit()
    # skip the first residual (diffuse initialization after differencing)
    resid = fitted.resid.iloc[1:]
    lb = acorr_ljungbox(resid, lags=[5], model_df=2)
    rows.append({
        'geo': geo,
        'n_obs': len(ts),
        'lb_stat_lag5': round(float(lb['lb_stat'].iloc[0]), 3),
        'lb_pvalue_lag5': round(float(lb['lb_pvalue'].iloc[0]), 4),
        'aic': round(float(fitted.aic), 2),
    })

res = pd.DataFrame(rows)
res.to_csv(os.path.join(OUTPUT_DIR, "arima_diagnostics.csv"), index=False)

n_total = len(res)
n_ok = int((res['lb_pvalue_lag5'] > 0.05).sum())
print(f"Countries fitted: {n_total}")
print(f"No residual autocorrelation at lag 5 (p > 0.05): {n_ok} of {n_total}")
print(res.to_string(index=False))
