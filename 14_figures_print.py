"""
Друкарські версії всіх рисунків статті для подання в JOPI.

Вимоги журналу: >= 300 dpi (або >= 1000 px), шрифт на рисунку не менше 8 pt.
Тому кожен рисунок будується одразу у фізичному розмірі сторінки
(ширина 6,5 дюйма = 16,5 см) з базовим шрифтом 8-9 pt і зберігається з 300 dpi.
Заголовки прибрано (їх замінюють підписи у тексті), легенди спільні.

Вхід: data/panel_dataset.csv, output/*.csv, output/xgb_model.json
Вихід: output/print/fig01 ... fig11 (*.png, 300 dpi)
"""
import os
import warnings
from importlib.machinery import SourceFileLoader

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

matplotlib.use('Agg')
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
PRINT = os.path.join(OUT, "print")
os.makedirs(PRINT, exist_ok=True)
TARGET = 'ict_specialists_pct'
W = 6.5            # ширина текстового поля, дюйми
DPI = 300
MIN_PT = 8

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.titlesize': 9,
    'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2,
    'grid.linewidth': 0.4, 'savefig.dpi': DPI,
})

MODELS = ['ARIMA', 'ETS', 'XGBoost', 'Random Forest', 'Decision Tree', 'Prophet']
COLORS = {'ARIMA': '#4e79a7', 'ETS': '#59a14f', 'XGBoost': '#e15759',
          'Random Forest': '#b07aa1', 'Decision Tree': '#9c755f', 'Prophet': '#f28e2b'}
REP9 = ['DE', 'FR', 'SE', 'EE', 'PL', 'RO', 'IE', 'ES', 'IT']

df = pd.read_csv(os.path.join(BASE, "data", "panel_dataset.csv"))
names = df[['geo', 'country']].drop_duplicates().set_index('geo')['country']


def save(fig, name):
    """Зберігає рисунок і перевіряє, що жоден текст не менший за 8 pt."""
    small = [t.get_text() for t in fig.findobj(matplotlib.text.Text)
             if t.get_text().strip() and t.get_fontsize() < MIN_PT - 1e-6]
    assert not small, f"{name}: text below {MIN_PT} pt: {small[:5]}"
    path = os.path.join(PRINT, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    from PIL import Image
    w, h = Image.open(path).size
    print(f"  {name}: {w}x{h}px")


# ---------------------------------------------------------------- 1. схема
DESIGN_TEXT = {
    'uk': [
        ('Етап 1. Формування панельного датасету',
         'Eurostat, World Bank: 27 країн ЄС × 2004–2024 рр.\n'
         'цільова змінна — частка ІКТ-спеціалістів у зайнятості (%), 8 екзогенних предикторів'),
        ('Етап 2а. Локальні моделі',
         'окрема модель для кожної країни,\nлише власна історія ряду\n\n'
         'статистичні: ARIMA(1,1,1), ETS\nML: Prophet'),
        ('Етап 2б. Панельні моделі',
         'одна модель для всіх 27 країн, 16 ознак:\nлаги, ковзні середні, екзогенні змінні\n\n'
         'ML: дерево рішень → випадковий ліс\n→ XGBoost'),
        ('Етап 3. Оцінювання точності',
         'крос-валідація з розширюваним вікном: 12 точок прогнозу, тестові роки 2013–2024,\n'
         'прогноз на 1 рік уперед, 323 спільні прогнози; метрики MAE, RMSE, MAPE;\n'
         "тест Дібольда-Маріано (об'єднаний і панельний); перевірки робастності"),
        ('Етап 4. Інтерпретація та комбінування',
         'SHAP-декомпозиція моделі XGBoost: внесок авторегресійних та екзогенних ознак\n'
         'ансамбль ARIMA + XGBoost з рівними вагами'),
        ('Етап 5. Прогноз і сценарії на 2025–2029 рр.',
         'ітеративний прогноз усіма моделями; три сценарії (50%, 100% і 150% приросту\n'
         'базової траєкторії ARIMA + XGBoost) для кожної країни та агрегату ЄС-27'),
    ],
    'en': [
        ('Stage 1. Building the panel dataset',
         'Eurostat, World Bank: 27 EU countries × 2004–2024\n'
         'target variable: ICT specialists, % of total employment; 8 exogenous predictors'),
        ('Stage 2a. Local models',
         'a separate model for each country,\nusing only the series\' own history\n\n'
         'statistical: ARIMA(1,1,1), ETS\nML: Prophet'),
        ('Stage 2b. Panel models',
         'one model for all 27 countries, 16 features:\nlags, rolling means, exogenous variables\n\n'
         'ML: decision tree → random forest\n→ XGBoost'),
        ('Stage 3. Accuracy evaluation',
         'expanding-window cross-validation: 12 forecast origins, test years 2013–2024,\n'
         'one-year-ahead forecasts, 323 common forecasts; metrics: MAE, RMSE, MAPE;\n'
         'Diebold–Mariano test (pooled and panel); robustness checks'),
        ('Stage 4. Interpretation and combination',
         'SHAP decomposition of XGBoost: contributions of autoregressive and exogenous\n'
         'features; equal-weight ARIMA + XGBoost ensemble'),
        ('Stage 5. Forecasts and scenarios for 2025–2029',
         'recursive forecasts from all models; three scenarios (50%, 100% and 150%\n'
         'of baseline ARIMA + XGBoost growth) for each country and the EU-27 aggregate'),
    ],
}


def fig_design(lang='uk'):
    s1, s2a, s2b, s3, s4, s5 = DESIGN_TEXT[lang]
    ink, edge = '#1f2a36', '#4e6a85'
    fig, ax = plt.subplots(figsize=(W, 5.6))
    ax.set_xlim(-0.012, 1.012)
    ax.set_ylim(0.035, 1.005)
    ax.axis('off')

    def box(x, y, w, h, title, body, fill='#eef3f8'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.012",
                                    linewidth=0.9, edgecolor=edge, facecolor=fill))
        ax.text(x + w / 2, y + h - 0.02, title, ha='center', va='top', fontsize=9,
                fontweight='bold', color=ink)
        ax.text(x + w / 2, y + h - 0.058, body, ha='center', va='top', fontsize=8,
                color=ink, linespacing=1.35)

    def arrow(x0, y0, x1, y1):
        ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle='-|>', color=edge, lw=1.0, mutation_scale=10,
                                    shrinkA=1, shrinkB=1))

    X, Wd, G = 0.0, 1.0, 0.035
    y, h = 1.0, 0.125
    box(X, y - h, Wd, h, *s1)
    arrow(0.5, y - h, 0.5, y - h - G)
    y -= h + G
    h, half = 0.235, 0.49
    box(X, y - h, half, h, *s2a, fill='#e6eef7')
    box(1 - half, y - h, half, h, *s2b, fill='#f7ebe9')
    arrow(half / 2, y - h, 0.42, y - h - G)
    arrow(1 - half / 2, y - h, 0.58, y - h - G)
    y -= h + G
    h = 0.155
    box(X, y - h, Wd, h, *s3)
    arrow(0.5, y - h, 0.5, y - h - G)
    y -= h + G
    h = 0.13
    box(X, y - h, Wd, h, *s4)
    arrow(0.5, y - h, 0.5, y - h - G)
    y -= h + G
    h = 0.13
    box(X, y - h, Wd, h, *s5)
    save(fig, "fig01_research_design.png" if lang == 'uk' else f"fig01_research_design_{lang}.png")


# ---------------------------------------------------------------- 2. динаміка
def fig_trends():
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.1))
    ax = axes[0]
    for geo in sorted(df.geo.unique()):
        d = df[df.geo == geo]
        ax.plot(d.year, d[TARGET], color='#9fb6cd', linewidth=0.6, alpha=0.8)
    cmap = plt.cm.tab10(np.linspace(0, 1, 10))
    handles = []
    for geo, c in zip(REP9, cmap):
        d = df[df.geo == geo]
        ln, = ax.plot(d.year, d[TARGET], color=c, linewidth=1.2, label=names[geo])
        handles.append(ln)
    ax.set_xlabel('Year')
    ax.set_ylabel('ICT specialists (% of employment)')
    ax.set_title('(a) EU-27 countries', loc='left')
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    g = df.groupby('year')[TARGET].agg(['mean', 'std', 'min', 'max'])
    ax.fill_between(g.index, g['min'], g['max'], alpha=0.15, color='steelblue', label='Min–max')
    ax.fill_between(g.index, g['mean'] - g['std'], g['mean'] + g['std'], alpha=0.3,
                    color='steelblue', label='Mean ± 1 SD')
    ax.plot(g.index, g['mean'], 'o-', color='darkblue', markersize=2.5, label='Mean')
    ax.set_title('(b) Unweighted EU-27 mean', loc='left')
    ax.set_xlabel('Year')
    ax.grid(True, alpha=0.3)
    for a in axes:
        a.axvline(2011, color='#555', linestyle=':', linewidth=0.8)
    axes[1].text(2011.3, g['max'].max() * 0.97, 'series break\n(ISCO-08, 2011)', fontsize=8,
                 va='top', color='#333')
    axes[1].legend(loc='upper left', bbox_to_anchor=(0.0, 0.8), frameon=False)
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    fig.tight_layout()
    save(fig, "fig02_trends.png")


# ---------------------------------------------------------------- 3. 2011 vs 2024
def fig_2011_2024():
    c = df[df.year.isin([2011, 2024])].pivot_table(index='country', columns='year',
                                                    values=TARGET).dropna()
    c = c.sort_values(2024)
    fig, ax = plt.subplots(figsize=(W, 4.9))
    y = np.arange(len(c))
    ax.barh(y - 0.19, c[2011], height=0.38, color='#4e79a7', label='2011')
    ax.barh(y + 0.19, c[2024], height=0.38, color='#e15759', label='2024')
    ax.set_yticks(y)
    ax.set_yticklabels(c.index)
    ax.set_ylim(-0.7, len(c) - 0.3)
    ax.set_xlabel('ICT specialists (% of employment)')
    ax.legend(loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    save(fig, "fig03_2011_vs_2024.png")


# ---------------------------------------------------------------- 4. кореляції
def fig_corr():
    cols = ['ict_specialists_pct', 'hightech_emp_pct', 'gdp_pc_eur', 'rd_pct_gdp',
            'internet_use_pct', 'services_employment_pct', 'broadband_per100',
            'tertiary_education_pct', 'ict_service_exports_pct', 'gdp_pc_usd']
    lab = ['ICT share', 'High-tech emp.', 'GDP pc (EUR)', 'R&D (% GDP)', 'Internet use',
           'Services emp.', 'Broadband', 'Tertiary enrol.', 'ICT serv. exports', 'GDP pc (USD)']
    d = df[df.year >= 2010]
    corr = d[cols].corr()
    fig, ax = plt.subplots(figsize=(W, 4.7))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    m = np.ma.array(corr.values, mask=mask)
    im = ax.imshow(m, cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(len(cols)):
        for j in range(i + 1):
            v = corr.values[i, j]
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(v) > 0.6 else 'black')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(lab, rotation=45, ha='right')
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(lab)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    save(fig, "fig04_correlation.png")


# ---------------------------------------------------------------- 5. метрики
def fig_metrics():
    s = pd.read_csv(os.path.join(OUT, "model_comparison_all.csv")).set_index('model').loc[MODELS]
    short = ['ARIMA', 'ETS', 'XGBoost', 'RF', 'DT', 'Prophet']
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5))
    for ax, met, unit in zip(axes, ['MAE', 'RMSE', 'MAPE'], ['p.p.', 'p.p.', '%']):
        ax.bar(range(6), s[met].values, color=[COLORS[m] for m in MODELS], width=0.7)
        ax.set_xticks(range(6))
        ax.set_xticklabels(short, rotation=45, ha='right')
        ax.set_title(f'{met}, {unit}')
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim(0, s[met].max() * 1.12)
    fig.tight_layout()
    save(fig, "fig05_model_comparison.png")


# ---------------------------------------------------------------- 6. факт/прогноз
def fig_scatter():
    cv = pd.concat([pd.read_csv(os.path.join(OUT, "cv_results.csv")),
                    pd.read_csv(os.path.join(OUT, "cv_results_trees.csv"))])
    fig, axes = plt.subplots(2, 3, figsize=(W, 4.6), sharex=True, sharey=True)
    lo, hi = 0.8, 9.3
    for ax, m in zip(axes.flat, MODELS):
        c = cv[cv.model == m]
        ax.scatter(c.actual, c.predicted, s=4, alpha=0.5, color=COLORS[m], linewidths=0)
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.7)
        mae = np.mean(np.abs(c.actual - c.predicted))
        ax.set_title(f'{m} (MAE = {mae:.3f})')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.grid(True, alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel('Actual, %')
    for ax in axes[:, 0]:
        ax.set_ylabel('Predicted, %')
    fig.tight_layout()
    save(fig, "fig06_actual_vs_predicted.png")


# ---------------------------------------------------------------- 7-8. SHAP
LABELS = {
    'ict_specialists_pct_lag1': 'ICT share (t-1)', 'ict_specialists_pct_lag2': 'ICT share (t-2)',
    'ict_specialists_pct_lag3': 'ICT share (t-3)', 'ict_specialists_pct_roll2': 'ICT share (2-yr avg)',
    'ict_specialists_pct_roll3': 'ICT share (3-yr avg)', 'ict_specialists_pct_diff1': 'ICT share change (t-1)',
    'hightech_emp_pct': 'High-tech employment (%)', 'gdp_pc_eur': 'GDP per capita (EUR)',
    'rd_pct_gdp': 'R&D expenditure (% GDP)', 'internet_use_pct': 'Internet use (%)',
    'services_employment_pct': 'Services employment (%)', 'broadband_per100': 'Broadband (per 100)',
    'gdp_pc_usd': 'GDP per capita (USD)', 'ict_service_exports_pct': 'ICT service exports (%)',
    'trend': 'Time trend', 'geo_code': 'Country code'}


def fig_shap():
    t09 = SourceFileLoader("t09", os.path.join(BASE, "09_tree_models.py")).load_module()
    _, df_model, feats, _ = t09.prepare(df[df[TARGET].notna()].copy())
    model = xgb.XGBRegressor()
    model.load_model(os.path.join(OUT, "xgb_model.json"))
    X = df_model[feats]
    sv = shap.TreeExplainer(model)(X)
    mean = pd.Series(np.abs(sv.values).mean(axis=0), index=feats).sort_values()
    ar = [f for f in feats if 'lag' in f or 'roll' in f or 'diff' in f]
    share = mean[ar].sum() / mean.sum() * 100
    print(f"  SHAP check: lag1 = {mean['ict_specialists_pct_lag1']:.3f}, AR share = {share:.1f}%")

    fig, ax = plt.subplots(figsize=(W - 0.4, 3.9))
    col = ['#e15759' if f in ar else '#4e79a7' for f in mean.index]
    ax.barh([LABELS[f] for f in mean.index], mean.values, color=col)
    ax.set_xlabel('Mean |SHAP value|, p.p.')
    ax.grid(True, axis='x', alpha=0.3)
    ax.legend(handles=[Patch(color='#e15759', label='Autoregressive features'),
                       Patch(color='#4e79a7', label='Exogenous and structural features')],
              loc='lower right')
    save(fig, "fig07_shap_importance.png")

    ex = shap.Explanation(values=sv.values, base_values=sv.base_values, data=sv.data,
                          feature_names=[LABELS[f] for f in feats])
    shap.plots.beeswarm(ex, show=False, max_display=16, plot_size=(W, 4.4))
    fig = plt.gcf()
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_text().strip():
            t.set_fontsize(max(t.get_fontsize() * 0.62, MIN_PT))
    save(fig, "fig08_shap_beeswarm.png")


# ---------------------------------------------------------------- 9. прогнози
def fig_forecasts():
    fc = pd.concat([pd.read_csv(os.path.join(OUT, "forecasts_2025_2029.csv")),
                    pd.read_csv(os.path.join(OUT, "forecasts_trees.csv"))])
    fig, axes = plt.subplots(3, 3, figsize=(W, 6.6))
    for ax, geo in zip(axes.flat, REP9):
        h = df[df.geo == geo][['year', TARGET]].dropna().sort_values('year')
        h = h[h.year >= 2011]
        ax.plot(h.year, h[TARGET], 'ko-', markersize=2, linewidth=1)
        for m in MODELS:
            f = fc[(fc.model == m) & (fc.geo == geo)]
            ax.plot([h.year.iloc[-1]] + f.year.tolist(), [h[TARGET].iloc[-1]] + f.forecast.tolist(),
                    '--', color=COLORS[m], linewidth=1)
        ax.set_title(names[geo])
        ax.set_xticks([2012, 2018, 2024, 2029])
        ax.grid(True, alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel('ICT specialists, %')
    handles = [Line2D([], [], color='k', marker='o', markersize=2, label='Actual')] + \
              [Line2D([], [], color=COLORS[m], linestyle='--', label=m) for m in MODELS]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, "fig09_forecasts_by_country.png")


# ---------------------------------------------------------------- 10-11. сценарії
def fig_scenarios():
    sc = pd.read_csv(os.path.join(OUT, "scenarios_2025_2029.csv"))
    eu = sc.groupby('year')[['pessimistic', 'base', 'optimistic']].mean()
    hist = df.groupby('year')[TARGET].mean()
    hist = hist[hist.index >= 2011]

    def draw(ax, h, s):
        ax.plot(h.index, h.values, 'ko-', markersize=2.5, linewidth=1.1)
        ax.fill_between(s.index, s['pessimistic'], s['optimistic'], alpha=0.2, color='steelblue')
        x0, y0 = h.index[-1], h.values[-1]
        ax.plot([x0] + list(s.index), [y0] + list(s['base']), 's-', color='#e15759', markersize=2.5)
        ax.plot([x0] + list(s.index), [y0] + list(s['optimistic']), '^--', color='#59a14f',
                markersize=2.5, linewidth=0.9)
        ax.plot([x0] + list(s.index), [y0] + list(s['pessimistic']), 'v--', color='#f28e2b',
                markersize=2.5, linewidth=0.9)
        ax.grid(True, alpha=0.3)

    handles = [Line2D([], [], color='k', marker='o', markersize=3, label='Actual'),
               Line2D([], [], color='#e15759', marker='s', markersize=3, label='Base'),
               Line2D([], [], color='#59a14f', marker='^', linestyle='--', markersize=3, label='Optimistic'),
               Line2D([], [], color='#f28e2b', marker='v', linestyle='--', markersize=3, label='Pessimistic'),
               Patch(color='steelblue', alpha=0.2, label='Scenario range')]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    draw(ax, hist, eu)
    ax.set_xticks([2011, 2014, 2017, 2020, 2024, 2029])
    ax.set_xlabel('Year')
    ax.set_ylabel('ICT specialists, %')
    ax.legend(handles=handles, loc='upper left')
    save(fig, "fig10_scenarios_eu27.png")

    fig, axes = plt.subplots(2, 3, figsize=(W, 4.3))
    for ax, geo in zip(axes.flat, ['DE', 'SE', 'EE', 'PL', 'RO', 'IE']):
        h = df[(df.geo == geo) & (df.year >= 2011)].set_index('year')[TARGET].dropna()
        s = sc[sc.geo == geo].set_index('year')
        draw(ax, h, s)
        ax.set_title(names[geo])
        ax.set_xticks([2012, 2018, 2024, 2029])
    for ax in axes[:, 0]:
        ax.set_ylabel('ICT specialists, %')
    fig.legend(handles=handles, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    save(fig, "fig11_scenarios_countries.png")


if __name__ == "__main__":
    fig_design()
    fig_design('en')
    fig_trends()
    fig_2011_2024()
    fig_corr()
    fig_metrics()
    fig_scatter()
    fig_shap()
    fig_forecasts()
    fig_scenarios()
    print("Saved to output/print/")
