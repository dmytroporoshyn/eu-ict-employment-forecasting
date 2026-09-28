"""
SHAP-аналіз для XGBoost та сценарне прогнозування (2025-2029).

1. SHAP feature importance + beeswarm plot
2. Сценарії: оптимістичний / базовий / песимістичний
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import shap

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

TARGET = 'ict_specialists_pct'

# ============================================================
# LOAD DATA & MODEL
# ============================================================

df = pd.read_csv(os.path.join(DATA_DIR, "panel_dataset.csv"))
forecasts = pd.read_csv(os.path.join(OUTPUT_DIR, "forecasts_2025_2029.csv"))

# Recreate features (same as 04_models.py)
def create_xgb_features(df, target_col=TARGET, lags=[1, 2, 3]):
    result = df.copy()
    result = result.sort_values(['geo', 'year'])
    for lag in lags:
        result[f'{target_col}_lag{lag}'] = result.groupby('geo')[target_col].shift(lag)
    result[f'{target_col}_roll2'] = result.groupby('geo')[target_col].transform(
        lambda x: x.shift(1).rolling(2, min_periods=1).mean()
    )
    result[f'{target_col}_roll3'] = result.groupby('geo')[target_col].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    )
    result[f'{target_col}_diff1'] = result.groupby('geo')[target_col].diff(1)
    result[f'{target_col}_diff1'] = result.groupby('geo')[f'{target_col}_diff1'].shift(1)
    result['trend'] = result['year'] - result['year'].min()
    result['geo_code'] = pd.Categorical(result['geo']).codes
    return result

extra_features = [
    'hightech_emp_pct', 'gdp_pc_eur', 'rd_pct_gdp', 'internet_use_pct',
    'services_employment_pct', 'broadband_per100', 'gdp_pc_usd',
    'ict_service_exports_pct'
]
extra_features = [f for f in extra_features if f in df.columns]

df_feat = create_xgb_features(df[df[TARGET].notna()].copy())

lag_features = [c for c in df_feat.columns if 'lag' in c or 'roll' in c or 'diff' in c]
feature_cols = lag_features + extra_features + ['trend', 'geo_code']
feature_cols = [c for c in feature_cols if c in df_feat.columns]

df_model = df_feat.dropna(subset=lag_features + [TARGET]).copy()
for col in extra_features:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna(df_model[col].median())

X = df_model[feature_cols]
y = df_model[TARGET]

# Retrain XGBoost
model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbosity=0
)
model.fit(X, y)

# ============================================================
# 1. SHAP ANALYSIS
# ============================================================

print("=" * 60)
print("SHAP ANALYSIS")
print("=" * 60)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Readable feature names for plots
feature_labels = {
    'ict_specialists_pct_lag1': 'ICT share (t-1)',
    'ict_specialists_pct_lag2': 'ICT share (t-2)',
    'ict_specialists_pct_lag3': 'ICT share (t-3)',
    'ict_specialists_pct_roll2': 'ICT share (2yr avg)',
    'ict_specialists_pct_roll3': 'ICT share (3yr avg)',
    'ict_specialists_pct_diff1': 'ICT share change (t-1)',
    'hightech_emp_pct': 'High-tech employment (%)',
    'gdp_pc_eur': 'GDP per capita (EUR)',
    'rd_pct_gdp': 'R&D expenditure (% GDP)',
    'internet_use_pct': 'Internet use (%)',
    'services_employment_pct': 'Services employment (%)',
    'broadband_per100': 'Broadband (per 100)',
    'gdp_pc_usd': 'GDP per capita (USD)',
    'ict_service_exports_pct': 'ICT service exports (%)',
    'trend': 'Time trend',
    'geo_code': 'Country code',
}

# SHAP bar plot (mean |SHAP|)
fig, ax = plt.subplots(figsize=(10, 8))
mean_shap = pd.Series(
    np.abs(shap_values.values).mean(axis=0),
    index=feature_cols
).sort_values(ascending=True)

labels = [feature_labels.get(f, f) for f in mean_shap.index]
colors = ['#e15759' if 'lag' in f or 'roll' in f or 'diff' in f else '#4e79a7'
          for f in mean_shap.index]

ax.barh(labels, mean_shap.values, color=colors)
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('XGBoost Feature Importance (SHAP)')
ax.grid(True, axis='x', alpha=0.3)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#e15759', label='Autoregressive (lag/rolling)'),
    Patch(facecolor='#4e79a7', label='Exogenous (economic indicators)'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig12_shap_importance.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig12_shap_importance.png")

# SHAP beeswarm plot
fig, ax = plt.subplots(figsize=(12, 8))
shap_values_renamed = shap.Explanation(
    values=shap_values.values,
    base_values=shap_values.base_values,
    data=shap_values.data,
    feature_names=[feature_labels.get(f, f) for f in feature_cols]
)
shap.plots.beeswarm(shap_values_renamed, show=False, max_display=16)
plt.title('SHAP Beeswarm: Feature Impact on ICT Specialists Prediction')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig13_shap_beeswarm.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig13_shap_beeswarm.png")

# Print SHAP summary
print("\n  Mean |SHAP| values:")
for feat in mean_shap.sort_values(ascending=False).index:
    label = feature_labels.get(feat, feat)
    print(f"    {label:35s} {mean_shap[feat]:.4f}")

# Exogenous share
lag_sum = mean_shap[[f for f in mean_shap.index if 'lag' in f or 'roll' in f or 'diff' in f]].sum()
exo_sum = mean_shap[[f for f in mean_shap.index if f not in
                      [c for c in mean_shap.index if 'lag' in c or 'roll' in c or 'diff' in c]]].sum()
total = lag_sum + exo_sum
print(f"\n  Autoregressive features: {lag_sum / total * 100:.1f}% of total SHAP")
print(f"  Exogenous features:     {exo_sum / total * 100:.1f}% of total SHAP")


# ============================================================
# 2. SCENARIO FORECASTING
# ============================================================

print("\n" + "=" * 60)
print("SCENARIO FORECASTING (2025-2029)")
print("=" * 60)

# Base forecast = average of ARIMA and XGBoost (two best models)
base_fc = forecasts[forecasts['model'].isin(['ARIMA', 'XGBoost'])].copy()
base_avg = base_fc.groupby(['geo', 'year'])['forecast'].mean().reset_index()
base_avg.columns = ['geo', 'year', 'base']

# Get country names
country_names = df[['geo', 'country']].drop_duplicates().set_index('geo')['country']

# Scenarios as COHERENT perturbations of a single base trajectory.
# All three scenarios share the same model-based path (base = ARIMA+XGBoost
# average) and differ only by how much of the base-implied change from the
# last actual (2024) materialises: pessimistic = 50%, base = 100%,
# optimistic = 150%. This guarantees pessimistic <= base <= optimistic for
# every country and year by construction (the previous version mixed a model
# base with a separate linear extrapolation, which broke monotonicity).

scenarios = base_avg.copy()

scenarios_list = []
for geo in sorted(scenarios['geo'].unique()):
    geo_base = scenarios[scenarios['geo'] == geo].sort_values('year')
    last_actual = df[(df['geo'] == geo) & (df[TARGET].notna())][TARGET].iloc[-1]

    for _, row in geo_base.iterrows():
        year = int(row['year'])

        # Base = model average (ARIMA + XGBoost)
        base_val = row['base']

        # Cumulative change implied by the base forecast relative to 2024
        base_delta = base_val - last_actual

        # Optimistic / pessimistic scale that same change by 1.5x / 0.5x;
        # min/max keeps the ordering correct even when the base declines
        cand_lo = last_actual + 0.5 * base_delta
        cand_hi = last_actual + 1.5 * base_delta
        pessimistic = min(cand_lo, cand_hi)
        optimistic = max(cand_lo, cand_hi)

        scenarios_list.append({
            'geo': geo,
            'country': country_names.get(geo, geo),
            'year': year,
            'pessimistic': round(pessimistic, 2),
            'base': round(base_val, 2),
            'optimistic': round(optimistic, 2),
        })

scenarios_df = pd.DataFrame(scenarios_list)
scenarios_df.to_csv(os.path.join(OUTPUT_DIR, "scenarios_2025_2029.csv"), index=False)
print("  Saved: scenarios_2025_2029.csv")

# Print EU27 average scenarios
print("\n  EU27 Average Scenarios:")
eu_scenarios = scenarios_df.groupby('year')[['pessimistic', 'base', 'optimistic']].mean()
for year, row in eu_scenarios.iterrows():
    print(f"    {year}: Pessimistic={row['pessimistic']:.2f}%  "
          f"Base={row['base']:.2f}%  Optimistic={row['optimistic']:.2f}%")

# ---- PLOT: Scenario fan chart (EU27 average) ----
fig, ax = plt.subplots(figsize=(12, 7))

# Historical EU27 average
hist_avg = df.groupby('year')[TARGET].mean().dropna()
ax.plot(hist_avg.index, hist_avg.values, 'ko-', markersize=4, linewidth=2, label='Historical (EU27 avg)')

# Scenarios
years = eu_scenarios.index
ax.fill_between(years, eu_scenarios['pessimistic'], eu_scenarios['optimistic'],
                alpha=0.2, color='steelblue', label='Pessimistic-Optimistic range')
ax.plot(years, eu_scenarios['base'], 's-', color='#e15759', linewidth=2,
        markersize=6, label='Base scenario')
ax.plot(years, eu_scenarios['optimistic'], '^--', color='#59a14f', linewidth=1.5,
        markersize=5, label='Optimistic')
ax.plot(years, eu_scenarios['pessimistic'], 'v--', color='#f28e2b', linewidth=1.5,
        markersize=5, label='Pessimistic')

# Connect historical to forecast
ax.plot([hist_avg.index[-1], years[0]],
        [hist_avg.values[-1], eu_scenarios['base'].iloc[0]],
        'k--', alpha=0.3)

ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('ICT Specialists (% of employment)', fontsize=12)
ax.set_title('ICT Specialists Share: EU27 Average Forecast Scenarios (2025-2029)', fontsize=14)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig14_scenarios_eu27.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig14_scenarios_eu27.png")

# ---- PLOT: Scenarios for representative countries ----
REP_COUNTRIES = ['DE', 'SE', 'EE', 'PL', 'RO', 'IE']

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax, geo in zip(axes.flat, REP_COUNTRIES):
    cname = country_names.get(geo, geo)
    hist = df[(df['geo'] == geo) & (df[TARGET].notna())][['year', TARGET]].sort_values('year')
    sc = scenarios_df[scenarios_df['geo'] == geo].sort_values('year')

    ax.plot(hist['year'], hist[TARGET], 'ko-', markersize=4, linewidth=1.5, label='Historical')

    ax.fill_between(sc['year'], sc['pessimistic'], sc['optimistic'],
                    alpha=0.2, color='steelblue')
    ax.plot(sc['year'], sc['base'], 's-', color='#e15759', linewidth=2, markersize=5, label='Base')
    ax.plot(sc['year'], sc['optimistic'], '^--', color='#59a14f', linewidth=1, markersize=4, label='Optimistic')
    ax.plot(sc['year'], sc['pessimistic'], 'v--', color='#f28e2b', linewidth=1, markersize=4, label='Pessimistic')

    ax.set_title(f'{cname}', fontsize=12)
    ax.set_xlabel('Year')
    ax.set_ylabel('ICT Specialists (%)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Forecast Scenarios by Country (2025-2029)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig15_scenarios_countries.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig15_scenarios_countries.png")

# ---- TABLE: 2029 forecast by country ----
print("\n  Country Forecasts for 2029:")
fc_2029 = scenarios_df[scenarios_df['year'] == 2029].sort_values('base', ascending=False)
print(f"  {'Country':15s} {'Pessim':>8s} {'Base':>8s} {'Optim':>8s}")
print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
for _, row in fc_2029.iterrows():
    print(f"  {row['country']:15s} {row['pessimistic']:8.2f} {row['base']:8.2f} {row['optimistic']:8.2f}")

print("\n" + "=" * 60)
print("ALL DONE!")
print("=" * 60)
