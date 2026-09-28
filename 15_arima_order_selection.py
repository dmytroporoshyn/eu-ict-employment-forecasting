"""AIC-based check of the fixed ARIMA(1,1,1) order (Section 2.3.1 of the article).

For every country the full target series is fitted with ARIMA(p,1,q),
p, q = 0..2, without a constant (statsmodels default for d = 1, the same
setting as in 04_models.py). Reports the AIC-optimal order, the AIC gap
of ARIMA(1,1,1) to the optimum and whether the two are practically
equivalent (gap < 2 AIC units).

Output: output/arima_order_selection.csv
"""
import os
import warnings

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
TARGET = 'ict_specialists_pct'

df = pd.read_csv(os.path.join(BASE, "data", "panel_dataset.csv"))
rows = []
for geo in sorted(df.geo.unique()):
    ts = df[df.geo == geo][['year', TARGET]].dropna().sort_values('year').set_index('year')[TARGET]
    aic = {}
    for p in range(3):
        for q in range(3):
            try:
                aic[(p, q)] = ARIMA(ts.values, order=(p, 1, q)).fit().aic
            except Exception:
                pass
    best = min(aic, key=aic.get)
    gap = aic[(1, 1)] - aic[best]
    rows.append({'geo': geo, 'best_order': f'({best[0]},1,{best[1]})', 'aic_best': round(aic[best], 2),
                 'aic_111': round(aic[(1, 1)], 2), 'gap_111': round(gap, 2), 'equivalent': gap < 2})

res = pd.DataFrame(rows)
res.to_csv(os.path.join(BASE, "output", "arima_order_selection.csv"), index=False)
print(res.to_string(index=False))
print(f"\n(0,1,0) optimal for {(res.best_order == '(0,1,0)').sum()} of {len(res)} countries")
print(f"mean AIC gap of (1,1,1): {res.gap_111.mean():.2f}; practically equivalent (gap < 2): {res.equivalent.sum()}")
