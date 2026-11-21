"""Decision tree and random forest with the same panel design as XGBoost in 04_models.py
(features, imputation, CV, recursive forecast); only the algorithm differs. Also:
hyper-parameter sensitivity and an ex ante check with predictors lagged by one year.

Output: output/cv_results_trees.csv, forecasts_trees.csv, tree_sensitivity.csv,
robustness_lagged_exog.csv
"""
import os
import warnings
from importlib.machinery import SourceFileLoader

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE, "output")
m04 = SourceFileLoader("m04", os.path.join(BASE, "04_models.py")).load_module()
m06 = SourceFileLoader("m06", os.path.join(BASE, "06_dm_test.py")).load_module()

TARGET = m04.TARGET
FORECAST_HORIZON = m04.FORECAST_HORIZON
RANDOM_STATE = m04.RANDOM_STATE
EXTRA_FEATURES = [
    'hightech_emp_pct', 'gdp_pc_eur', 'rd_pct_gdp', 'internet_use_pct',
    'services_employment_pct', 'broadband_per100', 'gdp_pc_usd',
    'ict_service_exports_pct'
]

# Tree: depth 4 as in XGBoost, >= 5 obs per leaf. Forest: 200 full-depth trees,
# 1/3 of the features per split (Breiman's default for regression).
MODELS = {
    'Decision Tree': lambda: DecisionTreeRegressor(
        max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE),
    'Random Forest': lambda: RandomForestRegressor(
        n_estimators=200, max_features=1 / 3, min_samples_leaf=1,
        random_state=RANDOM_STATE, n_jobs=-1),
}


def xgb_factory():
    return xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, verbosity=0)


def prepare(df):
    extra = [f for f in EXTRA_FEATURES if f in df.columns]
    df_feat = m04.create_xgb_features(df)
    lag_features = [c for c in df_feat.columns if 'lag' in c or 'roll' in c or 'diff' in c]
    feature_cols = lag_features + extra + ['trend', 'geo_code']
    df_model = df_feat.dropna(subset=lag_features + [TARGET]).copy()
    for col in extra:
        df_model[col] = df_model[col].fillna(df_model[col].median())
    return df_feat, df_model, feature_cols, extra


def run_panel_model(df, factory, name, forecast=True):
    """Same logic as run_xgboost() in 04_models.py, for any regressor."""
    df_feat, df_model, feature_cols, extra = prepare(df)

    unique_years = sorted(df_model['year'].unique())
    min_train_years = 6
    rows = []
    for test_year_idx in range(min_train_years, len(unique_years)):
        test_year = unique_years[test_year_idx]
        train = df_model[df_model['year'].isin(unique_years[:test_year_idx])]
        test = df_model[df_model['year'] == test_year]
        if len(train) < 20 or len(test) == 0:
            continue
        model = factory()
        model.fit(train[feature_cols], train[TARGET])
        preds = model.predict(test[feature_cols])
        for (_, r), p in zip(test.iterrows(), preds):
            rows.append({'geo': r['geo'], 'year': int(r['year']),
                         'actual': r[TARGET], 'predicted': float(p), 'model': name})
    cv = pd.DataFrame(rows)
    if not forecast:
        return cv, None

    final = factory()
    final.fit(df_model[feature_cols], df_model[TARGET])
    fc_rows = []
    for geo in sorted(df['geo'].unique()):
        cd = df_feat[df_feat['geo'] == geo].sort_values('year').copy()
        for _ in range(FORECAST_HORIZON):
            next_year = int(cd['year'].max()) + 1
            new = cd.iloc[-1].copy()
            new['year'] = next_year
            new['trend'] = next_year - df['year'].min()
            new[f'{TARGET}_lag1'] = cd[TARGET].iloc[-1]
            new[f'{TARGET}_lag2'] = cd[TARGET].iloc[-2]
            new[f'{TARGET}_lag3'] = cd[TARGET].iloc[-3]
            new[f'{TARGET}_roll2'] = cd[TARGET].iloc[-2:].mean()
            new[f'{TARGET}_roll3'] = cd[TARGET].iloc[-3:].mean()
            new[f'{TARGET}_diff1'] = cd[TARGET].iloc[-1] - cd[TARGET].iloc[-2]
            X = pd.DataFrame([new[feature_cols]]).astype(float)
            for col in extra:
                X[col] = X[col].fillna(df_model[col].median())
            pred = float(final.predict(X)[0])
            fc_rows.append({'geo': geo, 'year': next_year, 'forecast': pred, 'model': name})
            new[TARGET] = pred
            cd = pd.concat([cd, pd.DataFrame([new])], ignore_index=True)
    return cv, pd.DataFrame(fc_rows)


def mae(cv):
    return float(np.mean(np.abs(cv['actual'] - cv['predicted'])))


def main():
    df = pd.read_csv(os.path.join(BASE, "data", "panel_dataset.csv"))
    df_ts = df[df[TARGET].notna()].copy()

    # --- 0. check: reproduce XGBoost of 04_models.py ---
    cv_x, _ = run_panel_model(df_ts, xgb_factory, 'XGBoost', forecast=False)
    ref = pd.read_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"))
    ref = ref[ref.model == 'XGBoost']
    chk = cv_x.merge(ref, on=['geo', 'year'], suffixes=('', '_ref'))
    max_diff = float((chk['predicted'] - chk['predicted_ref']).abs().max())
    print(f"Reproduction check XGBoost: {len(chk)} of {len(ref)} rows, "
          f"max |diff| = {max_diff:.2e}")
    assert len(chk) == len(ref) and max_diff < 1e-4, "XGBoost not reproduced"

    # --- 1. main specifications ---
    all_cv, all_fc = [], []
    for name, factory in MODELS.items():
        cv, fc = run_panel_model(df_ts, factory, name)
        print(f"{name:14s} MAE={mae(cv):.3f}  n={len(cv)}")
        all_cv.append(cv)
        all_fc.append(fc)
    pd.concat(all_cv).to_csv(os.path.join(OUTPUT_DIR, "cv_results_trees.csv"), index=False)
    pd.concat(all_fc).to_csv(os.path.join(OUTPUT_DIR, "forecasts_trees.csv"), index=False)

    # --- 2. hyper-parameter sensitivity (CV only) ---
    grid = []
    for depth in [3, 4, 5, 6, 8, None]:
        for leaf in [1, 5, 10]:
            cv, _ = run_panel_model(df_ts, lambda: DecisionTreeRegressor(
                max_depth=depth, min_samples_leaf=leaf, random_state=RANDOM_STATE),
                'Decision Tree', forecast=False)
            grid.append({'model': 'Decision Tree', 'max_depth': depth,
                         'min_samples_leaf': leaf, 'max_features': None, 'MAE': mae(cv)})
    for mf in [1 / 3, 0.5, 1.0]:
        for leaf in [1, 5]:
            for depth in [None, 6]:
                cv, _ = run_panel_model(df_ts, lambda: RandomForestRegressor(
                    n_estimators=200, max_features=mf, min_samples_leaf=leaf,
                    max_depth=depth, random_state=RANDOM_STATE, n_jobs=-1),
                    'Random Forest', forecast=False)
                grid.append({'model': 'Random Forest', 'max_depth': depth,
                             'min_samples_leaf': leaf, 'max_features': round(mf, 3),
                             'MAE': mae(cv)})
    sens = pd.DataFrame(grid)
    sens.to_csv(os.path.join(OUTPUT_DIR, "tree_sensitivity.csv"), index=False)
    print(sens.groupby('model')['MAE'].describe())

    # --- 3. ex ante check: exogenous predictors lagged by one year ---
    df_lag = df_ts.sort_values(['geo', 'year']).copy()
    for c in EXTRA_FEATURES:
        df_lag[c] = df_lag.groupby('geo')[c].shift(1)
    arima = pd.read_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"))
    arima = arima[arima.model == 'ARIMA'][['geo', 'year', 'actual', 'predicted']]
    rob = []
    for name, factory in [('XGBoost', xgb_factory)] + list(MODELS.items()):
        cv, _ = run_panel_model(df_lag, factory, name, forecast=False)
        w = arima.merge(cv[['geo', 'year', 'predicted']], on=['geo', 'year'],
                        suffixes=('_arima', ''))
        w['err_ARIMA'] = w['actual'] - w['predicted_arima']
        w['err_M'] = w['actual'] - w['predicted']
        dm, p_pool = m06.diebold_mariano(w['err_ARIMA'], w['err_M'], h=1, loss='squared')
        t_p, p_panel, _, _ = m06.panel_dm(w, 'ARIMA', 'M', 'squared')
        rob.append({'model': name, 'MAE': mae(cv),
                    'MAPE': float(np.mean(np.abs(cv.actual - cv.predicted) / cv.actual) * 100),
                    'DM_vs_ARIMA_pooled': dm, 'p_pooled': p_pool,
                    't_vs_ARIMA_panel': t_p, 'p_panel': p_panel, 'n_obs': len(cv)})
    rob = pd.DataFrame(rob)
    rob.to_csv(os.path.join(OUTPUT_DIR, "robustness_lagged_exog.csv"), index=False)
    print(rob.round(4).to_string(index=False))
    print("Saved: cv_results_trees.csv, forecasts_trees.csv, tree_sensitivity.csv, "
          "robustness_lagged_exog.csv")


if __name__ == "__main__":
    main()
