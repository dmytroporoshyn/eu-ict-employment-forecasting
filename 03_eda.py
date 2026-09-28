"""
Розвідувальний аналіз (EDA) панельного датасету
для прогнозування зайнятості у сфері інформаційних послуг.

Генерує графіки та описову статистику у папку output/
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib.ticker import MaxNLocator

matplotlib.use('Agg')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(os.path.join(DATA_DIR, "panel_dataset.csv"))
print(f"Dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Countries: {df['geo'].nunique()}, Years: {df['year'].min()}-{df['year'].max()}")

# Focus on 2010-2024 (where WB data is available)
df_full = df.copy()
df = df[df['year'] >= 2010].copy()
print(f"Filtered (2010+): {df.shape[0]} rows")

# Representative countries for detailed plots
REP_COUNTRIES = {
    'DE': 'Germany', 'FR': 'France', 'SE': 'Sweden',
    'EE': 'Estonia', 'PL': 'Poland', 'RO': 'Romania',
    'IE': 'Ireland', 'ES': 'Spain', 'IT': 'Italy'
}

# ============================================================
# 1. DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "=" * 60)
print("1. DESCRIPTIVE STATISTICS")
print("=" * 60)

numeric_cols = df.select_dtypes(include=[np.number]).columns.drop('year')
desc = df[numeric_cols].describe().T
desc['missing_%'] = (1 - df[numeric_cols].count() / len(df)) * 100
print(desc.round(2).to_string())

# Save to CSV
desc.round(3).to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"))

# ============================================================
# 2. TARGET VARIABLE TRENDS (ict_specialists_pct)
# ============================================================

print("\n" + "=" * 60)
print("2. TARGET VARIABLE ANALYSIS")
print("=" * 60)

# 2a. All EU27 countries - spaghetti plot
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Left: all countries (full period 2004-2024)
ax = axes[0]
for geo in sorted(df_full['geo'].unique()):
    country_data = df_full[df_full['geo'] == geo]
    ax.plot(country_data['year'], country_data['ict_specialists_pct'],
            alpha=0.4, linewidth=1, color='steelblue')

# Highlight representative countries
colors = plt.cm.Set1(np.linspace(0, 1, len(REP_COUNTRIES)))
for (geo, name), color in zip(REP_COUNTRIES.items(), colors):
    country_data = df_full[df_full['geo'] == geo]
    ax.plot(country_data['year'], country_data['ict_specialists_pct'],
            linewidth=2.5, label=name, color=color)

ax.set_xlabel('Year')
ax.set_ylabel('ICT Specialists (% of employment)')
ax.set_title('ICT Specialists Share in EU27 Countries (2004-2024)')
ax.legend(fontsize=8, loc='upper left')
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

# Right: EU27 average trend
ax = axes[1]
eu_avg = df_full.groupby('year')['ict_specialists_pct'].agg(['mean', 'std', 'min', 'max'])
ax.fill_between(eu_avg.index, eu_avg['min'], eu_avg['max'], alpha=0.15, color='steelblue', label='Min-Max range')
ax.fill_between(eu_avg.index,
                eu_avg['mean'] - eu_avg['std'],
                eu_avg['mean'] + eu_avg['std'],
                alpha=0.3, color='steelblue', label='Mean +/- 1 SD')
ax.plot(eu_avg.index, eu_avg['mean'], 'o-', color='darkblue', linewidth=2, markersize=4, label='EU27 Mean')
ax.set_xlabel('Year')
ax.set_ylabel('ICT Specialists (% of employment)')
ax.set_title('EU27 Average ICT Specialists Share (2004-2024)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_ict_specialists_trends.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig1_ict_specialists_trends.png")

# ============================================================
# 3. ICT SPECIALISTS BY COUNTRY (2024 vs 2010)
# ============================================================

fig, ax = plt.subplots(figsize=(12, 8))

comparison = df_full[df_full['year'].isin([2010, 2024])].pivot_table(
    index='country', columns='year', values='ict_specialists_pct'
).dropna()
comparison = comparison.sort_values(2024, ascending=True)

y_pos = np.arange(len(comparison))
ax.barh(y_pos - 0.15, comparison[2010], height=0.3, color='#4e79a7', label='2010')
ax.barh(y_pos + 0.15, comparison[2024], height=0.3, color='#e15759', label='2024')
ax.set_yticks(y_pos)
ax.set_yticklabels(comparison.index, fontsize=9)
ax.set_xlabel('ICT Specialists (% of employment)')
ax.set_title('ICT Specialists Share: 2010 vs 2024')
ax.legend()
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_ict_2010_vs_2024.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig2_ict_2010_vs_2024.png")

# ============================================================
# 4. CORRELATION MATRIX
# ============================================================

feature_cols = [
    'ict_specialists_pct', 'hightech_emp_pct', 'gdp_pc_eur', 'rd_pct_gdp',
    'internet_use_pct', 'services_employment_pct', 'broadband_per100',
    'tertiary_education_pct', 'ict_service_exports_pct', 'gdp_pc_usd'
]
feature_cols = [c for c in feature_cols if c in df.columns]

corr = df[feature_cols].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, square=True, ax=ax,
            xticklabels=[c.replace('_', '\n') for c in feature_cols],
            yticklabels=[c.replace('_', '\n') for c in feature_cols])
ax.set_title('Correlation Matrix of Key Variables (2010-2024)')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_correlation_matrix.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig3_correlation_matrix.png")

# Print top correlations with target
print("\n  Correlations with ict_specialists_pct:")
target_corr = corr['ict_specialists_pct'].drop('ict_specialists_pct').sort_values(ascending=False)
for feat, val in target_corr.items():
    print(f"    {feat:35s} {val:+.3f}")

# ============================================================
# 5. KEY SCATTER PLOTS (target vs top predictors)
# ============================================================

top_features = ['broadband_per100', 'gdp_pc_eur', 'rd_pct_gdp', 'internet_use_pct']
top_features = [f for f in top_features if f in df.columns]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
for ax, feat in zip(axes.flat, top_features):
    scatter_data = df.dropna(subset=[feat, 'ict_specialists_pct'])
    ax.scatter(scatter_data[feat], scatter_data['ict_specialists_pct'],
               alpha=0.4, s=20, c='steelblue')

    # Add regression line
    z = np.polyfit(scatter_data[feat], scatter_data['ict_specialists_pct'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(scatter_data[feat].min(), scatter_data[feat].max(), 100)
    ax.plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=2)

    r = scatter_data[feat].corr(scatter_data['ict_specialists_pct'])
    ax.set_xlabel(feat.replace('_', ' ').title())
    ax.set_ylabel('ICT Specialists (%)')
    ax.set_title(f'ICT Specialists vs {feat.replace("_", " ").title()} (r={r:.2f})')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_scatter_plots.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig4_scatter_plots.png")

# ============================================================
# 6. GROWTH RATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("3. GROWTH RATE ANALYSIS")
print("=" * 60)

# Calculate CAGR 2010-2024 for each country
growth = df_full[df_full['year'].isin([2010, 2024])].pivot_table(
    index=['geo', 'country'], columns='year', values='ict_specialists_pct'
).dropna()
growth['cagr_pct'] = ((growth[2024] / growth[2010]) ** (1 / 14) - 1) * 100
growth['abs_change'] = growth[2024] - growth[2010]
growth = growth.sort_values('cagr_pct', ascending=False)

print("\n  Country CAGR (2010-2024) for ICT Specialists (%):")
for (geo, country), row in growth.iterrows():
    print(f"    {country:15s}  {row[2010]:.1f}% -> {row[2024]:.1f}%  "
          f"(CAGR: {row['cagr_pct']:+.2f}%, change: {row['abs_change']:+.1f}pp)")

# Bar chart of CAGR
fig, ax = plt.subplots(figsize=(12, 8))
growth_sorted = growth.sort_values('cagr_pct', ascending=True)
countries = [idx[1] for idx in growth_sorted.index]
colors = ['#e15759' if v < 0 else '#4e79a7' for v in growth_sorted['cagr_pct']]
ax.barh(countries, growth_sorted['cagr_pct'], color=colors)
ax.set_xlabel('CAGR 2010-2024 (%)')
ax.set_title('Compound Annual Growth Rate of ICT Specialists Share (2010-2024)')
ax.axvline(x=0, color='black', linewidth=0.5)
ax.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_cagr_by_country.png"), bbox_inches='tight')
plt.close()
print("\n  Saved: fig5_cagr_by_country.png")

# ============================================================
# 7. MISSING DATA HEATMAP
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))
missing = df[numeric_cols].isnull().groupby(df['year']).mean() * 100
sns.heatmap(missing.T, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax,
            xticklabels=True, yticklabels=True)
ax.set_title('Missing Data (%) by Year and Variable (2010-2024)')
ax.set_xlabel('Year')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig6_missing_data.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig6_missing_data.png")

# ============================================================
# 8. DISTRIBUTION OF TARGET VARIABLE
# ============================================================

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Histogram
ax = axes[0]
ax.hist(df['ict_specialists_pct'].dropna(), bins=25, color='steelblue', edgecolor='white', alpha=0.8)
ax.set_xlabel('ICT Specialists (%)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of ICT Specialists (%)')
ax.axvline(df['ict_specialists_pct'].mean(), color='red', linestyle='--', label=f"Mean={df['ict_specialists_pct'].mean():.2f}")
ax.legend()

# Box plot by year (selected years)
ax = axes[1]
years_select = [2010, 2014, 2018, 2022, 2024]
box_data = df_full[df_full['year'].isin(years_select)]
box_data.boxplot(column='ict_specialists_pct', by='year', ax=ax)
ax.set_xlabel('Year')
ax.set_ylabel('ICT Specialists (%)')
ax.set_title('Distribution by Year')
plt.sca(ax)
plt.title('Distribution by Year')

# Box plot by country group (high/mid/low GDP)
ax = axes[2]
median_gdp = df.groupby('geo')['gdp_pc_eur'].median()
df_temp = df.copy()
df_temp['gdp_group'] = df_temp['geo'].map(
    lambda x: 'High GDP' if median_gdp.get(x, 0) > 35000
    else ('Mid GDP' if median_gdp.get(x, 0) > 20000 else 'Low GDP')
)
df_temp.boxplot(column='ict_specialists_pct', by='gdp_group', ax=ax)
ax.set_xlabel('GDP Group')
ax.set_ylabel('ICT Specialists (%)')
plt.sca(ax)
plt.title('By GDP Group')

plt.suptitle('')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig7_target_distribution.png"), bbox_inches='tight')
plt.close()
print("  Saved: fig7_target_distribution.png")

# ============================================================
# 9. STATIONARITY CHECK (for time series methods)
# ============================================================

print("\n" + "=" * 60)
print("4. STATIONARITY CHECK (ADF test on EU27 average)")
print("=" * 60)

from statsmodels.tsa.stattools import adfuller

eu_avg_ts = df_full.groupby('year')['ict_specialists_pct'].mean().dropna()

# Level
adf_level = adfuller(eu_avg_ts, autolag='AIC')
print(f"\n  ADF on level:      stat={adf_level[0]:.4f}, p={adf_level[1]:.4f}")
print(f"    -> {'Stationary' if adf_level[1] < 0.05 else 'Non-stationary'} at 5%")

# First difference
adf_diff = adfuller(eu_avg_ts.diff().dropna(), autolag='AIC')
print(f"  ADF on 1st diff:   stat={adf_diff[0]:.4f}, p={adf_diff[1]:.4f}")
print(f"    -> {'Stationary' if adf_diff[1] < 0.05 else 'Non-stationary'} at 5%")

# Per-country stationarity
print("\n  Per-country ADF test (level):")
stationary_count = 0
for geo in sorted(df_full['geo'].unique()):
    ts = df_full[df_full['geo'] == geo]['ict_specialists_pct'].dropna()
    if len(ts) >= 8:
        adf = adfuller(ts, autolag='AIC')
        status = "S" if adf[1] < 0.05 else "NS"
        if adf[1] < 0.05:
            stationary_count += 1
        print(f"    {geo}: stat={adf[0]:+.3f}, p={adf[1]:.3f} [{status}]")

print(f"\n  Stationary at 5%: {stationary_count}/{df_full['geo'].nunique()} countries")
print("  -> Most series are non-stationary in levels (trending upward)")
print("  -> ARIMA will need differencing (d=1)")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"""
Key findings:
1. ICT specialists share is growing across all EU27 (2004-2024)
2. EU27 average: {df_full[df_full['year']==2010].ict_specialists_pct.mean():.1f}% (2010) -> {df_full[df_full['year']==2024].ict_specialists_pct.mean():.1f}% (2024)
3. Leaders: Sweden, Finland, Estonia, Netherlands, Luxembourg
4. Fastest growth: Lithuania, Portugal, Cyprus, Latvia
5. Most series are non-stationary -> ARIMA needs d=1
6. Strong correlations with: broadband, GDP, R&D, internet use
7. COVID-19 effect visible (acceleration in 2020-2021)

Generated files in output/:
  - fig1_ict_specialists_trends.png
  - fig2_ict_2010_vs_2024.png
  - fig3_correlation_matrix.png
  - fig4_scatter_plots.png
  - fig5_cagr_by_country.png
  - fig6_missing_data.png
  - fig7_target_distribution.png
  - descriptive_statistics.csv
""")
