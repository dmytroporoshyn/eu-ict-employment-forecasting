"""
Моделювання та прогнозування зайнятості у сфері інформаційних послуг.

Методи:
  1. ARIMA (per country)
  2. ETS / Holt-Winters (per country)
  3. XGBoost (panel with lag features)
  4. Prophet (per country)

Validation: expanding window cross-validation
Metrics: MAE, RMSE, MAPE
Forecast horizon: 2025-2029
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb
from prophet import Prophet

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FORECAST_HORIZON = 5  # 2025-2029
TARGET = 'ict_specialists_pct'
RANDOM_STATE = 42

# Representative countries for per-country models
REP_COUNTRIES = ['DE', 'FR', 'SE', 'EE', 'PL', 'RO', 'IE', 'ES', 'IT']


# ============================================================
# HELPERS
# ============================================================

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate(y_true, y_pred):
    """Calculate MAE, RMSE, MAPE."""
    return {
        'MAE': mean_absolute_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAPE': mape(y_true, y_pred),
    }


def expanding_window_splits(years, min_train=8, test_size=1):
    """Generate expanding window CV splits for time series."""
    unique_years = sorted(years.unique())
    splits = []
    for i in range(min_train, len(unique_years) - test_size + 1):
        train_years = unique_years[:i]
        test_years = unique_years[i:i + test_size]
        splits.append((train_years, test_years))
    return splits


# ============================================================
# 1. ARIMA (per country)
# ============================================================

def run_arima(df, countries=None):
    """Run ARIMA for each country with expanding window CV."""
    print("\n" + "=" * 60)
    print("MODEL 1: ARIMA")
    print("=" * 60)

    if countries is None:
        countries = sorted(df['geo'].unique())

    all_cv_results = []
    all_forecasts = []

    for geo in countries:
        ts = df[df['geo'] == geo][['year', TARGET]].dropna().sort_values('year')
        ts = ts.set_index('year')[TARGET]

        if len(ts) < 10:
            continue

        # Expanding window CV
        years = ts.index.tolist()
        cv_preds = []

        for train_end_idx in range(8, len(years) - 1):
            train = ts.iloc[:train_end_idx + 1]
            actual = ts.iloc[train_end_idx + 1]

            try:
                model = ARIMA(train, order=(1, 1, 1))
                fitted = model.fit()
                pred = fitted.forecast(steps=1).iloc[0]
                cv_preds.append({
                    'geo': geo,
                    'year': years[train_end_idx + 1],
                    'actual': actual,
                    'predicted': pred,
                    'model': 'ARIMA'
                })
            except Exception:
                continue

        if cv_preds:
            cv_df = pd.DataFrame(cv_preds)
            all_cv_results.extend(cv_preds)

        # Final model on all data + forecast
        try:
            model = ARIMA(ts, order=(1, 1, 1))
            fitted = model.fit()
            forecast = fitted.forecast(steps=FORECAST_HORIZON)
            for i, val in enumerate(forecast):
                all_forecasts.append({
                    'geo': geo,
                    'year': ts.index[-1] + i + 1,
                    'forecast': val,
                    'model': 'ARIMA'
                })
        except Exception:
            pass

    cv_results = pd.DataFrame(all_cv_results)
    forecasts = pd.DataFrame(all_forecasts)

    if not cv_results.empty:
        metrics = evaluate(cv_results['actual'], cv_results['predicted'])
        print(f"  CV Results: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, MAPE={metrics['MAPE']:.2f}%")
        print(f"  CV observations: {len(cv_results)}")

    return cv_results, forecasts


# ============================================================
# 2. ETS / Holt-Winters (per country)
# ============================================================

def run_ets(df, countries=None):
    """Run ETS (Holt's linear trend) for each country."""
    print("\n" + "=" * 60)
    print("MODEL 2: ETS (Holt-Winters)")
    print("=" * 60)

    if countries is None:
        countries = sorted(df['geo'].unique())

    all_cv_results = []
    all_forecasts = []

    for geo in countries:
        ts = df[df['geo'] == geo][['year', TARGET]].dropna().sort_values('year')
        ts = ts.set_index('year')[TARGET]

        if len(ts) < 10:
            continue

        years = ts.index.tolist()
        cv_preds = []

        for train_end_idx in range(8, len(years) - 1):
            train = ts.iloc[:train_end_idx + 1]
            actual = ts.iloc[train_end_idx + 1]

            try:
                # Holt's linear trend (no seasonality for annual data)
                model = ExponentialSmoothing(
                    train, trend='add', seasonal=None, damped_trend=True
                )
                fitted = model.fit(optimized=True)
                pred = fitted.forecast(steps=1).iloc[0]
                cv_preds.append({
                    'geo': geo,
                    'year': years[train_end_idx + 1],
                    'actual': actual,
                    'predicted': pred,
                    'model': 'ETS'
                })
            except Exception:
                continue

        if cv_preds:
            all_cv_results.extend(cv_preds)

        # Final model + forecast
        try:
            model = ExponentialSmoothing(
                ts, trend='add', seasonal=None, damped_trend=True
            )
            fitted = model.fit(optimized=True)
            forecast = fitted.forecast(steps=FORECAST_HORIZON)
            for i, val in enumerate(forecast):
                all_forecasts.append({
                    'geo': geo,
                    'year': ts.index[-1] + i + 1,
                    'forecast': val,
                    'model': 'ETS'
                })
        except Exception:
            pass

    cv_results = pd.DataFrame(all_cv_results)
    forecasts = pd.DataFrame(all_forecasts)

    if not cv_results.empty:
        metrics = evaluate(cv_results['actual'], cv_results['predicted'])
        print(f"  CV Results: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, MAPE={metrics['MAPE']:.2f}%")
        print(f"  CV observations: {len(cv_results)}")

    return cv_results, forecasts


# ============================================================
# 3. XGBoost (panel model with lag features)
# ============================================================

def create_xgb_features(df, target_col=TARGET, lags=[1, 2, 3]):
    """Create lag features and rolling statistics for XGBoost."""
    result = df.copy()

    # Sort
    result = result.sort_values(['geo', 'year'])

    # Lag features of target
    for lag in lags:
        result[f'{target_col}_lag{lag}'] = result.groupby('geo')[target_col].shift(lag)

    # Rolling mean (2 and 3 year)
    result[f'{target_col}_roll2'] = result.groupby('geo')[target_col].transform(
        lambda x: x.shift(1).rolling(2, min_periods=1).mean()
    )
    result[f'{target_col}_roll3'] = result.groupby('geo')[target_col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )

    # Year-over-year change
    result[f'{target_col}_diff1'] = result.groupby('geo')[target_col].diff(1)
    result[f'{target_col}_diff1'] = result.groupby('geo')[f'{target_col}_diff1'].shift(1)

    # Trend (years since start)
    result['trend'] = result['year'] - result['year'].min()

    # Country encoding (label)
    result['geo_code'] = pd.Categorical(result['geo']).codes

    return result


def run_xgboost(df):
    """Run XGBoost panel model with expanding window CV."""
    print("\n" + "=" * 60)
    print("MODEL 3: XGBoost (Panel)")
    print("=" * 60)

    # Feature columns from the panel
    extra_features = [
        'hightech_emp_pct', 'gdp_pc_eur', 'rd_pct_gdp', 'internet_use_pct',
        'services_employment_pct', 'broadband_per100', 'gdp_pc_usd',
        'ict_service_exports_pct'
    ]
    extra_features = [f for f in extra_features if f in df.columns]

    # Create features
    df_feat = create_xgb_features(df)

    lag_features = [c for c in df_feat.columns if 'lag' in c or 'roll' in c or 'diff' in c]
    feature_cols = lag_features + extra_features + ['trend', 'geo_code']
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    # Drop rows with NaN in lag features (first 3 years per country)
    df_model = df_feat.dropna(subset=lag_features + [TARGET]).copy()

    # Fill NaN in extra features with median
    for col in extra_features:
        if col in df_model.columns:
            df_model[col] = df_model[col].fillna(df_model[col].median())

    print(f"  Features: {feature_cols}")
    print(f"  Dataset: {len(df_model)} rows")

    # Expanding window CV
    unique_years = sorted(df_model['year'].unique())
    min_train_years = 6

    all_cv_results = []

    for test_year_idx in range(min_train_years, len(unique_years)):
        test_year = unique_years[test_year_idx]
        train_years = unique_years[:test_year_idx]

        train = df_model[df_model['year'].isin(train_years)]
        test = df_model[df_model['year'] == test_year]

        if len(train) < 20 or len(test) == 0:
            continue

        X_train = train[feature_cols]
        y_train = train[TARGET]
        X_test = test[feature_cols]
        y_test = test[TARGET]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=0
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        for _, row in test.iterrows():
            idx = test.index.get_loc(row.name)
            all_cv_results.append({
                'geo': row['geo'],
                'year': int(row['year']),
                'actual': row[TARGET],
                'predicted': preds[idx] if isinstance(idx, int) else preds[0],
                'model': 'XGBoost'
            })

    cv_results = pd.DataFrame(all_cv_results)

    if not cv_results.empty:
        metrics = evaluate(cv_results['actual'], cv_results['predicted'])
        print(f"  CV Results: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, MAPE={metrics['MAPE']:.2f}%")
        print(f"  CV observations: {len(cv_results)}")

    # Final model on all data
    X_all = df_model[feature_cols]
    y_all = df_model[TARGET]
    final_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=RANDOM_STATE, verbosity=0
    )
    final_model.fit(X_all, y_all)

    # Feature importance
    importance = pd.Series(
        final_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\n  Feature importance (top 10):")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:35s} {imp:.4f}")

    # Forecast: iterative prediction for 2025-2029
    all_forecasts = []
    countries = sorted(df['geo'].unique())

    for geo in countries:
        country_data = df_feat[df_feat['geo'] == geo].sort_values('year').copy()

        for step in range(FORECAST_HORIZON):
            next_year = int(country_data['year'].max()) + 1
            last_row = country_data.iloc[-1].copy()

            new_row = last_row.copy()
            new_row['year'] = next_year
            new_row['trend'] = next_year - df['year'].min()

            # Update lags from latest known values
            new_row[f'{TARGET}_lag1'] = country_data[TARGET].iloc[-1]
            new_row[f'{TARGET}_lag2'] = country_data[TARGET].iloc[-2] if len(country_data) >= 2 else np.nan
            new_row[f'{TARGET}_lag3'] = country_data[TARGET].iloc[-3] if len(country_data) >= 3 else np.nan
            new_row[f'{TARGET}_roll2'] = country_data[TARGET].iloc[-2:].mean()
            new_row[f'{TARGET}_roll3'] = country_data[TARGET].iloc[-3:].mean()
            new_row[f'{TARGET}_diff1'] = country_data[TARGET].iloc[-1] - country_data[TARGET].iloc[-2] if len(country_data) >= 2 else 0

            # Predict
            X_pred = pd.DataFrame([new_row[feature_cols]])
            for col in extra_features:
                if col in X_pred.columns:
                    X_pred[col] = X_pred[col].fillna(df_model[col].median())

            pred = final_model.predict(X_pred)[0]

            all_forecasts.append({
                'geo': geo,
                'year': next_year,
                'forecast': pred,
                'model': 'XGBoost'
            })

            # Add predicted value to history for next iteration
            new_row[TARGET] = pred
            country_data = pd.concat([country_data, pd.DataFrame([new_row])], ignore_index=True)

    forecasts = pd.DataFrame(all_forecasts)

    return cv_results, forecasts, final_model, feature_cols


# ============================================================
# 4. Prophet (per country)
# ============================================================

def run_prophet(df, countries=None):
    """Run Prophet for each country with expanding window CV."""
    print("\n" + "=" * 60)
    print("MODEL 4: Prophet")
    print("=" * 60)

    if countries is None:
        countries = sorted(df['geo'].unique())

    all_cv_results = []
    all_forecasts = []

    for geo in countries:
        ts = df[df['geo'] == geo][['year', TARGET]].dropna().sort_values('year')

        if len(ts) < 10:
            continue

        # Prophet requires 'ds' and 'y'
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(ts['year'], format='%Y'),
            'y': ts[TARGET].values
        })

        years = ts['year'].tolist()

        # Expanding window CV
        for train_end_idx in range(8, len(years) - 1):
            train = prophet_df.iloc[:train_end_idx + 1]
            actual = prophet_df.iloc[train_end_idx + 1]['y']

            try:
                model = Prophet(
                    yearly_seasonality=False,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    growth='linear',
                    changepoint_prior_scale=0.05,
                )
                model.fit(train)
                future = model.make_future_dataframe(periods=1, freq='YS')
                forecast = model.predict(future)
                pred = forecast.iloc[-1]['yhat']

                all_cv_results.append({
                    'geo': geo,
                    'year': years[train_end_idx + 1],
                    'actual': actual,
                    'predicted': pred,
                    'model': 'Prophet'
                })
            except Exception:
                continue

        # Final model + forecast
        try:
            model = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=False,
                daily_seasonality=False,
                growth='linear',
                changepoint_prior_scale=0.05,
            )
            model.fit(prophet_df)
            future = model.make_future_dataframe(periods=FORECAST_HORIZON, freq='YS')
            forecast = model.predict(future)

            for i in range(FORECAST_HORIZON):
                row = forecast.iloc[len(prophet_df) + i]
                all_forecasts.append({
                    'geo': geo,
                    'year': years[-1] + i + 1,
                    'forecast': row['yhat'],
                    'forecast_lower': row['yhat_lower'],
                    'forecast_upper': row['yhat_upper'],
                    'model': 'Prophet'
                })
        except Exception:
            pass

    cv_results = pd.DataFrame(all_cv_results)
    forecasts = pd.DataFrame(all_forecasts)

    if not cv_results.empty:
        metrics = evaluate(cv_results['actual'], cv_results['predicted'])
        print(f"  CV Results: MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}, MAPE={metrics['MAPE']:.2f}%")
        print(f"  CV observations: {len(cv_results)}")

    return cv_results, forecasts


# ============================================================
# COMPARISON & VISUALIZATION
# ============================================================

def compare_models(all_cv, all_forecasts, df):
    """Compare all models and generate visualizations."""
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # Overall metrics by model
    summary = []
    for model_name in ['ARIMA', 'ETS', 'XGBoost', 'Prophet']:
        cv = all_cv[all_cv['model'] == model_name]
        if cv.empty:
            continue
        metrics = evaluate(cv['actual'], cv['predicted'])
        metrics['model'] = model_name
        metrics['n_obs'] = len(cv)
        summary.append(metrics)

    summary_df = pd.DataFrame(summary).set_index('model')
    print("\n  Overall CV Metrics:")
    print(summary_df.to_string())

    # Per-country metrics
    country_metrics = []
    for model_name in ['ARIMA', 'ETS', 'XGBoost', 'Prophet']:
        cv = all_cv[all_cv['model'] == model_name]
        for geo in cv['geo'].unique():
            cv_geo = cv[cv['geo'] == geo]
            if len(cv_geo) < 2:
                continue
            m = evaluate(cv_geo['actual'], cv_geo['predicted'])
            m['model'] = model_name
            m['geo'] = geo
            country_metrics.append(m)

    country_metrics_df = pd.DataFrame(country_metrics)

    # ---- PLOT 1: Model comparison bar chart ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric in zip(axes, ['MAE', 'RMSE', 'MAPE']):
        vals = summary_df[metric]
        colors = ['#4e79a7', '#59a14f', '#e15759', '#f28e2b'][:len(vals)]
        ax.bar(vals.index, vals.values, color=colors)
        ax.set_title(metric)
        ax.set_ylabel(metric + (' (%)' if metric == 'MAPE' else ''))
        for i, v in enumerate(vals.values):
            ax.text(i, v + v * 0.02, f'{v:.3f}', ha='center', fontsize=9)

    plt.suptitle('Cross-Validation Metrics: Model Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig8_model_comparison.png"), bbox_inches='tight')
    plt.close()
    print("  Saved: fig8_model_comparison.png")

    # ---- PLOT 2: Forecast comparison for representative countries ----
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ax, geo in zip(axes.flat, REP_COUNTRIES):
        country_name = df[df['geo'] == geo]['country'].iloc[0]
        hist = df[df['geo'] == geo][['year', TARGET]].dropna().sort_values('year')

        ax.plot(hist['year'], hist[TARGET], 'ko-', markersize=4, label='Historical', linewidth=1.5)

        colors_map = {'ARIMA': '#4e79a7', 'ETS': '#59a14f', 'XGBoost': '#e15759', 'Prophet': '#f28e2b'}
        for model_name, color in colors_map.items():
            fc = all_forecasts[(all_forecasts['model'] == model_name) & (all_forecasts['geo'] == geo)]
            if not fc.empty:
                # Connect last historical point to first forecast
                fc_plot = pd.concat([
                    pd.DataFrame({'year': [hist['year'].iloc[-1]], 'forecast': [hist[TARGET].iloc[-1]]}),
                    fc[['year', 'forecast']]
                ])
                ax.plot(fc_plot['year'], fc_plot['forecast'], '--', color=color,
                        linewidth=1.5, label=model_name, marker='s', markersize=3)

        ax.set_title(f'{country_name} ({geo})')
        ax.set_xlabel('Year')
        ax.set_ylabel('ICT Specialists (%)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle('ICT Specialists Forecast by Country (2025-2029)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig9_forecasts_by_country.png"), bbox_inches='tight')
    plt.close()
    print("  Saved: fig9_forecasts_by_country.png")

    # ---- PLOT 3: CV actual vs predicted scatter ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, model_name in zip(axes.flat, ['ARIMA', 'ETS', 'XGBoost', 'Prophet']):
        cv = all_cv[all_cv['model'] == model_name]
        if cv.empty:
            continue
        ax.scatter(cv['actual'], cv['predicted'], alpha=0.4, s=15, color='steelblue')
        lims = [
            min(cv['actual'].min(), cv['predicted'].min()) - 0.5,
            max(cv['actual'].max(), cv['predicted'].max()) + 0.5
        ]
        ax.plot(lims, lims, 'r--', linewidth=1, label='Perfect prediction')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        m = evaluate(cv['actual'], cv['predicted'])
        ax.set_title(f"{model_name} (MAE={m['MAE']:.3f}, MAPE={m['MAPE']:.1f}%)")
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Cross-Validation: Actual vs Predicted', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig10_actual_vs_predicted.png"), bbox_inches='tight')
    plt.close()
    print("  Saved: fig10_actual_vs_predicted.png")

    # ---- PLOT 4: Per-country MAPE heatmap ----
    if not country_metrics_df.empty:
        pivot = country_metrics_df.pivot_table(index='geo', columns='model', values='MAPE')
        pivot = pivot.sort_values(pivot.columns[0], ascending=True)

        fig, ax = plt.subplots(figsize=(10, 12))
        import seaborn as sns
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax)
        ax.set_title('MAPE (%) by Country and Model')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fig11_mape_by_country.png"), bbox_inches='tight')
        plt.close()
        print("  Saved: fig11_mape_by_country.png")

    # ---- EU27 Average Forecast ----
    print("\n  EU27 Average Forecast (2025-2029):")
    for model_name in ['ARIMA', 'ETS', 'XGBoost', 'Prophet']:
        fc = all_forecasts[all_forecasts['model'] == model_name]
        if fc.empty:
            continue
        avg = fc.groupby('year')['forecast'].mean()
        vals_str = ', '.join([f"{y}: {v:.2f}%" for y, v in avg.items()])
        print(f"    {model_name:10s}: {vals_str}")

    return summary_df, country_metrics_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Load data
    df = pd.read_csv(os.path.join(DATA_DIR, "panel_dataset.csv"))
    print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # Use full range (2004-2024) for time series models
    df_ts = df[df[TARGET].notna()].copy()

    # Run all models
    cv_arima, fc_arima = run_arima(df_ts)
    cv_ets, fc_ets = run_ets(df_ts)
    cv_xgb, fc_xgb, xgb_model, xgb_features = run_xgboost(df_ts)
    cv_prophet, fc_prophet = run_prophet(df_ts)

    # Combine results
    all_cv = pd.concat([cv_arima, cv_ets, cv_xgb, cv_prophet], ignore_index=True)
    all_forecasts = pd.concat([fc_arima, fc_ets, fc_xgb, fc_prophet], ignore_index=True)

    # Save raw results
    all_cv.to_csv(os.path.join(OUTPUT_DIR, "cv_results.csv"), index=False)
    all_forecasts.to_csv(os.path.join(OUTPUT_DIR, "forecasts_2025_2029.csv"), index=False)
    print("\n  Saved: cv_results.csv, forecasts_2025_2029.csv")

    # Compare & visualize
    summary_df, country_metrics_df = compare_models(all_cv, all_forecasts, df_ts)

    # Save summary
    summary_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_summary.csv"))

    # Save XGBoost model for SHAP analysis
    xgb_model.save_model(os.path.join(OUTPUT_DIR, "xgb_model.json"))

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
