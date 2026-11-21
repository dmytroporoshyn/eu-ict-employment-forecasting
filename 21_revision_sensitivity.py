"""Hyper-parameter sensitivity of the tree models (main specification, h = 1).
Output: output/revision/sensitivity_h1.csv
"""
import os
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

BASE = os.path.dirname(os.path.abspath(__file__))
r17 = SourceFileLoader("r17", os.path.join(BASE, "17_revision_backtest.py")).load_module()


def mae_h1(panel, name, factory):
    r17.PANEL_MODELS[name] = factory
    rows = []
    for T in range(2012, 2024):
        rows += [r for r in r17.panel_origin(panel, T, 'exante', [name]) if r['h'] == 1]
    d = pd.DataFrame(rows)
    return float(np.mean(np.abs(d.actual - d.forecast))), len(d)


def main():
    panel = pd.read_csv(os.path.join(BASE, "data", "panel_dataset.csv"))
    r17.H = 1                                   # only one-year-ahead forecasts are needed
    out = []
    for depth in [3, 4, 5, 6, 8, None]:
        for leaf in [1, 5, 10]:
            m, n = mae_h1(panel, 'grid', lambda: DecisionTreeRegressor(
                max_depth=depth, min_samples_leaf=leaf, random_state=42))
            out.append({'model': 'Decision Tree', 'max_depth': depth, 'min_samples_leaf': leaf, 'MAE': m, 'n': n})
    for mf in [1 / 3, 0.5, 1.0]:
        for leaf in [1, 5]:
            for depth in [None, 6]:
                m, n = mae_h1(panel, 'grid', lambda: RandomForestRegressor(
                    n_estimators=200, max_features=mf, min_samples_leaf=leaf, max_depth=depth,
                    random_state=42, n_jobs=-1))
                out.append({'model': 'Random Forest', 'max_depth': depth, 'min_samples_leaf': leaf,
                            'max_features': round(mf, 3), 'MAE': m, 'n': n})
    for depth in [3, 4, 6]:
        for lr, ne in [(0.05, 400), (0.1, 200)]:
            for sub in [0.8, 1.0]:
                m, n = mae_h1(panel, 'grid', lambda: xgb.XGBRegressor(
                    n_estimators=ne, max_depth=depth, learning_rate=lr, subsample=sub,
                    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0))
                out.append({'model': 'XGBoost', 'max_depth': depth, 'learning_rate': lr,
                            'n_estimators': ne, 'subsample': sub, 'MAE': m, 'n': n})
    res = pd.DataFrame(out)
    res.to_csv(os.path.join(BASE, "output", "revision", "sensitivity_h1.csv"), index=False)
    print(res.groupby('model').MAE.agg(['count', 'min', 'max']).round(4).to_string())


if __name__ == "__main__":
    main()
