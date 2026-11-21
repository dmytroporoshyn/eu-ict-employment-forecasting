"""Figures of the revised manuscript (300 dpi) -> output/revision/print/rfig*.png"""
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output", "revision")
PRINT = os.path.join(OUT, "print")
os.makedirs(PRINT, exist_ok=True)
TARGET = 'ict_specialists_pct'
W, DPI, MIN_PT = 6.5, 300, 8
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9,
    'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'grid.linewidth': 0.4, 'savefig.dpi': DPI})
COLORS = {'Naive': '#7f7f7f', 'ARIMA': '#4e79a7', 'ETS': '#59a14f', 'Prophet': '#f28e2b',
          'Decision Tree': '#9c755f', 'Random Forest': '#b07aa1', 'XGBoost': '#e15759',
          'Ensemble': '#1b1b1b', 'FE OLS': '#17becf', 'Pooled OLS': '#bcbd22'}
SHORT = {'Naive': 'Naive', 'ARIMA': 'ARIMA', 'ETS': 'ETS', 'Prophet': 'Prophet',
         'Decision Tree': 'DT', 'Random Forest': 'RF', 'XGBoost': 'XGBoost',
         'Ensemble': 'ARIMA+XGB', 'FE OLS': 'FE reg.', 'Pooled OLS': 'Pooled reg.'}
df = pd.read_csv(os.path.join(BASE, "data", "panel_dataset.csv"))
names = df[['geo', 'country']].drop_duplicates().set_index('geo')['country']
LABELS = {
    'ict_specialists_pct_lag1': 'ICT share (t-1)', 'ict_specialists_pct_lag2': 'ICT share (t-2)',
    'ict_specialists_pct_lag3': 'ICT share (t-3)', 'ict_specialists_pct_roll2': 'ICT share (2-yr avg)',
    'ict_specialists_pct_roll3': 'ICT share (3-yr avg)', 'ict_specialists_pct_diff1': 'ICT share change (t-1)',
    'hightech_emp_pct': 'High-tech employment (t-1)', 'gdp_pc_eur': 'GDP per capita, EUR (t-1)',
    'rd_pct_gdp': 'R&D expenditure (t-1)', 'internet_use_pct': 'Internet use (t-1)',
    'services_employment_pct': 'Services employment (t-1)', 'broadband_per100': 'Broadband (t-1)',
    'gdp_pc_usd': 'GDP per capita, USD (t-1)', 'ict_service_exports_pct': 'ICT service exports (t-1)',
    'trend': 'Time trend', 'geo_code': 'Country code'}


def save(fig, name):
    small = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-6]
    assert not small, f"{name}: text below {MIN_PT} pt: {small[:5]}"
    path = os.path.join(PRINT, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print("saved", name)


# ------------------------------------------------------------------ Fig. 1
DESIGN = [
    ('Stage 1. Panel dataset',
     'Eurostat, World Bank: 27 EU countries × 2004–2024; target: ICT specialists,\n'
     '% of total employment (by occupation, ISCO-08); 8 exogenous predictors'),
    ('Stage 2a. Local models',
     'a separate model for each country,\nusing only the series\' own history\n\n'
     'statistical: ARIMA(1,1,1), ETS\nML: Prophet'),
    ('Stage 2b. Panel (global) models',
     'one model for all 27 countries;\nlags of the target, predictors at t−1\n\n'
     'ML: decision tree → random forest → XGBoost\ncontrol: linear regressions (pooled, FE)'),
    ('Stage 3. Rolling-origin evaluation (origins 2012–2023)',
     'forecasts h = 1–5 years ahead, same information set for all models,\n'
     'gaps imputed within each training fold; MAE, RMSE, MAPE, bias, interval coverage;\n'
     'two-way cluster-robust tests and bootstrap; nested validation of the combination'),
    ('Stage 4. Interpretation and combination',
     'SHAP decomposition of XGBoost; equal-weight ARIMA + XGBoost combination;\n'
     'robustness: conditional predictors, imputation, ARIMA order, 2011 and 2021 breaks'),
    ('Stage 5. Forecasts for 2025–2029',
     'country forecasts with empirical 80% and 95% prediction intervals;\n'
     'employment-weighted EU aggregate under three employment paths vs the 2030 target'),
]


def fig_design():
    s1, s2a, s2b, s3, s4, s5 = DESIGN
    ink, edge = '#1f2a36', '#4e6a85'
    fig, ax = plt.subplots(figsize=(W, 5.9))
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(0.13, 1.005)
    ax.axis('off')

    def box(x, y, w, h, title, body, fill='#eef3f8'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                    linewidth=0.9, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h - 0.018, title, ha='center', va='top', fontsize=9,
                fontweight='bold', color=ink)
        ax.text(x + w / 2, y + h - 0.054, body, ha='center', va='top', fontsize=8,
                color=ink, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=edge, lw=1.0, mutation_scale=10,
                                    shrinkA=1, shrinkB=1))

    G = 0.03
    y, h = 1.0, 0.125
    box(0, y - h, 1, h, *s1)
    arrow(0.5, y - h, 0.5, y - h - G)
    y -= h + G
    h, half = 0.215, 0.49
    box(0, y - h, half, h, *s2a, fill='#e6eef7')
    box(1 - half, y - h, half, h, *s2b, fill='#f7ebe9')
    arrow(half / 2, y - h, 0.42, y - h - G)
    arrow(1 - half / 2, y - h, 0.58, y - h - G)
    y -= h + G
    h = 0.15
    box(0, y - h, 1, h, *s3)
    arrow(0.5, y - h, 0.5, y - h - G)
    y -= h + G
    h = 0.125
    box(0, y - h, 1, h, *s4)
    arrow(0.5, y - h, 0.5, y - h - G)
    y -= h + G
    h = 0.125
    box(0, y - h, 1, h, *s5)
    save(fig, "rfig01_research_design.png")


# ------------------------------------------------------------------ Fig. 5
def fig_accuracy():
    acc = pd.read_csv(os.path.join(OUT, "accuracy_by_horizon.csv"))
    a = acc[acc.h == 1].set_index('model')
    order = ['ARIMA', 'ETS', 'Prophet', 'Decision Tree', 'Random Forest', 'XGBoost', 'Ensemble']
    a = a.loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.6))
    for ax, met, unit in zip(axes, ['MAE', 'RMSE', 'MAPE'], ['p.p.', 'p.p.', '%']):
        ax.bar(range(len(order)), a[met].values, color=[COLORS[m] for m in order], width=0.7)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([SHORT[m] for m in order], rotation=50, ha='right')
        ax.set_title(f'{met}, {unit}')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, a[met].max() * 1.12)
    fig.tight_layout()
    save(fig, "rfig05_accuracy_h1.png")


# ------------------------------------------------------------------ Fig. 6
def fig_horizons():
    acc = pd.read_csv(os.path.join(OUT, "accuracy_by_horizon.csv"))
    cov = pd.read_csv(os.path.join(OUT, "interval_coverage.csv"))
    show = ['ARIMA', 'ETS', 'Random Forest', 'XGBoost', 'Ensemble', 'FE OLS']
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.6))
    for m in show:
        a = acc[acc.model == m].sort_values('h')
        ls = '-' if m != 'Ensemble' else '-'
        axes[0].plot(a.h, a.MAE, marker='o', markersize=2.5, color=COLORS[m], linestyle=ls, label=SHORT[m])
        axes[1].plot(a.h, a.Bias, marker='o', markersize=2.5, color=COLORS[m], linestyle=ls)
    for m in ['ARIMA', 'XGBoost', 'Ensemble']:
        c = cov[(cov.model == m) & (cov.type == 'empirical')].sort_values('h')
        axes[2].plot(c.h, c.cov80, marker='o', markersize=2.5, color=COLORS[m])
    c = cov[(cov.model == 'ARIMA') & (cov.type == 'native')].sort_values('h')
    axes[2].plot(c.h, c.cov80, marker='s', markersize=2.5, color=COLORS['ARIMA'], linestyle='--')
    axes[2].axhline(80, color='#444', linewidth=0.7, linestyle=':')
    axes[0].set_title('(a) MAE, p.p.', loc='left')
    axes[1].set_title('(b) Bias (actual − forecast), p.p.', loc='left')
    axes[2].set_title('(c) Coverage of 80% PI, %', loc='left')
    axes[1].axhline(0, color='#444', linewidth=0.6)
    for ax in axes:
        ax.set_xlabel('Horizon h, years')
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.3)
    axes[2].set_ylim(30, 100)
    h1 = [Line2D([], [], color=COLORS[m], marker='o', markersize=3, label=SHORT[m]) for m in show]
    h2 = [Line2D([], [], color=COLORS['ARIMA'], marker='s', markersize=3, linestyle='--',
                 label='ARIMA, model-based PI')]
    fig.legend(handles=h1 + h2, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.17))
    fig.tight_layout()
    save(fig, "rfig06_horizons.png")


# ------------------------------------------------------------------ Fig. 7
def fig_scatter():
    bt = pd.read_csv(os.path.join(OUT, "backtest.csv"))
    bt = bt[(bt.h == 1) & (bt.origin <= 2023) & bt.spec.isin(['local', 'exante'])]
    models = ['ARIMA', 'ETS', 'Prophet', 'Decision Tree', 'Random Forest', 'XGBoost', 'FE OLS']
    w = bt.pivot_table(index=['origin', 'geo', 'year'], columns='model', values='forecast').reset_index()
    act = bt.groupby(['geo', 'year'])['actual'].first()
    w['actual'] = [act[(g, y)] for g, y in zip(w.geo, w.year)]
    w['Ensemble'] = (w.ARIMA + w.XGBoost) / 2
    w = w.dropna(subset=models)
    show = ['ARIMA', 'ETS', 'Prophet', 'Decision Tree', 'Random Forest', 'XGBoost', 'FE OLS', 'Ensemble']
    fig, axes = plt.subplots(2, 4, figsize=(W, 3.7), sharex=True, sharey=True)
    lo, hi = 0.8, 9.3
    for ax, m in zip(axes.flat, show):
        ax.scatter(w.actual, w[m], s=3, alpha=0.5, color=COLORS[m], linewidths=0)
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.7)
        ax.set_title(f'{SHORT[m]}\nMAE = {np.mean(np.abs(w.actual - w[m])):.3f}')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('Actual, %')
    for ax in axes[:, 0]:
        ax.set_ylabel('Forecast, %')
    fig.tight_layout()
    save(fig, "rfig07_actual_vs_pred.png")


# ------------------------------------------------------------------ Fig. 8-9
def fig_shap():
    z = np.load(os.path.join(OUT, "shap_values.npz"), allow_pickle=True)
    feats = list(z['features'])
    vals = z['values']
    mean = pd.Series(np.abs(vals).mean(axis=0), index=feats).sort_values()
    ar = [f for f in feats if 'lag' in f or 'roll' in f or 'diff' in f]
    fig, ax = plt.subplots(figsize=(W - 0.4, 3.9))
    ax.barh([LABELS[f] for f in mean.index], mean.values,
            color=['#e15759' if f in ar else '#4e79a7' for f in mean.index])
    ax.set_xlabel('Mean |SHAP value|, p.p.')
    ax.grid(True, axis='x', alpha=0.3)
    ax.legend(handles=[Patch(color='#e15759', label='Autoregressive features'),
                       Patch(color='#4e79a7', label='Exogenous and structural features')],
              loc='lower right')
    save(fig, "rfig08_shap_importance.png")
    ex = shap.Explanation(values=vals, base_values=z['base'], data=z['data'],
                          feature_names=[LABELS[f] for f in feats])
    shap.plots.beeswarm(ex, show=False, max_display=16, plot_size=(W, 4.4))
    fig = plt.gcf()
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip():
            t.set_fontsize(max(t.get_fontsize() * 0.62, MIN_PT))
    save(fig, "rfig09_shap_beeswarm.png")


# ------------------------------------------------------------------ Fig. 10
def fig_forecasts():
    fin = pd.read_csv(os.path.join(OUT, "forecasts_final.csv"))
    iv = pd.read_csv(os.path.join(OUT, "intervals_final.csv"))
    models = ['ARIMA', 'ETS', 'Prophet', 'Decision Tree', 'Random Forest', 'XGBoost',
              'FE OLS', 'Ensemble']
    fig, axes = plt.subplots(3, 3, figsize=(W, 6.6))
    for ax, geo in zip(axes.flat, ['DE', 'FR', 'SE', 'EE', 'PL', 'RO', 'IE', 'ES', 'IT']):
        h = df[(df.geo == geo) & (df.year >= 2011)][['year', TARGET]].dropna()
        e = iv[(iv.model == 'Ensemble') & (iv.geo == geo)].sort_values('year')
        ax.fill_between(e.year, e.lo80, e.hi80, color='#999999', alpha=0.25, linewidth=0)
        ax.plot(h.year, h[TARGET], 'ko-', markersize=2, linewidth=1)
        f = fin[fin.geo == geo].sort_values('year')
        for m in models:
            ax.plot([h.year.iloc[-1]] + f.year.tolist(), [h[TARGET].iloc[-1]] + f[m].tolist(),
                    '-' if m == 'Ensemble' else '--', color=COLORS[m],
                    linewidth=1.5 if m == 'Ensemble' else 0.9)
        ax.set_title(names[geo])
        ax.set_xticks([2012, 2018, 2024, 2029])
        ax.grid(True, alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('ICT specialists, %')
    handles = [Line2D([], [], color='k', marker='o', markersize=2, label='Actual')] + \
              [Line2D([], [], color=COLORS[m], linestyle='-' if m == 'Ensemble' else '--',
                      label=SHORT[m]) for m in models] + \
              [Patch(color='#999999', alpha=0.25, label='80% PI (ARIMA+XGB)')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    save(fig, "rfig10_forecasts.png")


# ------------------------------------------------------------------ Fig. 11
def fig_eu():
    hist = pd.read_csv(os.path.join(OUT, "eu_aggregate_history.csv"), index_col=0)
    agg = pd.read_csv(os.path.join(OUT, "eu_aggregate.csv"))
    hist = hist[hist.index >= 2011]
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9))
    e1 = agg[(agg.path == 'E1 constant')]
    e3 = agg[(agg.path == 'E3 employment trend')]
    ens = e1[e1.model == 'Ensemble'].sort_values('year')
    ax = axes[0]
    ax.fill_between(ens.year, ens.S_weighted_lo95, ens.S_weighted_hi95, color='#999999', alpha=0.18, linewidth=0)
    ax.fill_between(ens.year, ens.S_weighted_lo80, ens.S_weighted_hi80, color='#999999', alpha=0.35, linewidth=0)
    ax.plot(hist.index, hist.S_weighted, 'ko-', markersize=2.5, linewidth=1.1)
    x0, y0 = hist.index[-1], hist.S_weighted.iloc[-1]
    for m in ['ARIMA', 'XGBoost', 'FE OLS', 'Ensemble']:
        s = e1[e1.model == m].sort_values('year')
        ax.plot([x0] + s.year.tolist(), [y0] + s.S_weighted.tolist(), '-' if m == 'Ensemble' else '--',
                color=COLORS[m], linewidth=1.5 if m == 'Ensemble' else 1.0)
    ax.set_title('(a) Employment-weighted share, %', loc='left')
    ax.set_xticks([2011, 2015, 2020, 2024, 2029])
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ens3 = e3[e3.model == 'Ensemble'].sort_values('year')
    ax.fill_between(ens.year, ens.N_thousand_lo95 / 1000, ens.N_thousand_hi95 / 1000, color='#999999',
                    alpha=0.18, linewidth=0)
    ax.fill_between(ens.year, ens.N_thousand_lo80 / 1000, ens.N_thousand_hi80 / 1000, color='#999999',
                    alpha=0.35, linewidth=0)
    ax.plot(hist.index, hist.N_thousand / 1000, 'ko-', markersize=2.5, linewidth=1.1)
    x0, y0 = hist.index[-1], hist.N_thousand.iloc[-1] / 1000
    for m in ['ARIMA', 'XGBoost', 'FE OLS', 'Ensemble']:
        s = e1[e1.model == m].sort_values('year')
        ax.plot([x0] + s.year.tolist(), [y0] + (s.N_thousand / 1000).tolist(),
                '-' if m == 'Ensemble' else '--', color=COLORS[m], linewidth=1.5 if m == 'Ensemble' else 1.0)
    ax.plot([x0] + ens3.year.tolist(), [y0] + (ens3.N_thousand / 1000).tolist(), ':', color='k', linewidth=1.2)
    ax.axhline(20, color='#c0392b', linewidth=1.0)
    ax.text(2011.3, 19.4, 'Digital Decade target: 20 million by 2030', color='#c0392b', fontsize=8, va='top')
    ax.set_ylim(4, 21)
    ax.set_title('(b) ICT specialists, million', loc='left')
    ax.set_xticks([2011, 2015, 2020, 2024, 2029])
    ax.grid(True, alpha=0.3)
    handles = [Line2D([], [], color='k', marker='o', markersize=3, label='Actual'),
               Line2D([], [], color=COLORS['Ensemble'], label='ARIMA+XGB (E1)'),
               Line2D([], [], color='k', linestyle=':', label='ARIMA+XGB (E3)'),
               Line2D([], [], color=COLORS['ARIMA'], linestyle='--', label='ARIMA'),
               Line2D([], [], color=COLORS['XGBoost'], linestyle='--', label='XGBoost'),
               Line2D([], [], color=COLORS['FE OLS'], linestyle='--', label='FE regression'),
               Patch(color='#999999', alpha=0.35, label='80% PI'),
               Patch(color='#999999', alpha=0.18, label='95% PI')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    save(fig, "rfig11_eu_aggregate.png")


# ------------------------------------------------------------------ Fig. 12
def fig_country_pi():
    iv = pd.read_csv(os.path.join(OUT, "intervals_final.csv"))
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.3))
    for ax, geo in zip(axes.flat, ['DE', 'SE', 'EE', 'PL', 'RO', 'IE']):
        h = df[(df.geo == geo) & (df.year >= 2011)].set_index('year')[TARGET].dropna()
        e = iv[(iv.model == 'Ensemble') & (iv.geo == geo)].sort_values('year')
        ax.fill_between(e.year, e.lo95, e.hi95, color='#999999', alpha=0.18, linewidth=0)
        ax.fill_between(e.year, e.lo80, e.hi80, color='#999999', alpha=0.35, linewidth=0)
        ax.plot(h.index, h.values, 'ko-', markersize=2, linewidth=1)
        ax.plot([h.index[-1]] + e.year.tolist(), [h.values[-1]] + e.forecast.tolist(), 's-',
                color=COLORS['Ensemble'], markersize=2.5, linewidth=1.3)
        ax.set_title(names[geo])
        ax.set_xticks([2012, 2018, 2024, 2029])
        ax.grid(True, alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('ICT specialists, %')
    handles = [Line2D([], [], color='k', marker='o', markersize=3, label='Actual'),
               Line2D([], [], color='k', marker='s', markersize=3, label='ARIMA+XGB forecast'),
               Patch(color='#999999', alpha=0.35, label='80% PI'),
               Patch(color='#999999', alpha=0.18, label='95% PI')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "rfig12_country_intervals.png")


if __name__ == "__main__":
    fig_design()
    fig_accuracy()
    fig_horizons()
    fig_scatter()
    fig_shap()
    fig_forecasts()
    fig_eu()
    fig_country_pi()
