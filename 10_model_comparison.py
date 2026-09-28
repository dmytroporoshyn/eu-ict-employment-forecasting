"""
Зведене порівняння шести моделей (ARIMA, ETS, XGBoost, Random Forest,
Decision Tree, Prophet) та ансамблю ARIMA+XGBoost.

Вхід:  output/cv_results.csv (04_models.py), output/cv_results_trees.csv,
       output/forecasts_2025_2029.csv, output/forecasts_trees.csv (09_tree_models.py)
Вихід: output/model_comparison_all.csv   -- MAE/RMSE/MAPE кожної моделі
       output/dm_test_all.csv            -- тест Дібольда-Маріано (pooled + panel)
       output/best_model_by_country.csv  -- найточніша модель за MAPE по країнах
       output/forecasts_eu27_all.csv     -- середній прогноз ЄС-27 по моделях
       output/fig8_model_comparison.png, fig9_forecasts_by_country.png,
       output/fig10_actual_vs_predicted.png (перебудовано для шести моделей)

Реалізація DM-тесту та знакова конвенція взяті з 06_dm_test.py:
DM > 0  =>  перша модель пари (benchmark) гірша, друга -- точніша.
"""
import os
from importlib.machinery import SourceFileLoader

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use('Agg')

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output")
TARGET = 'ict_specialists_pct'
_dm = SourceFileLoader("dm_mod", os.path.join(BASE, "06_dm_test.py")).load_module()

MODELS = ['ARIMA', 'ETS', 'XGBoost', 'Random Forest', 'Decision Tree', 'Prophet']
CLASS = {'ARIMA': 'statistical', 'ETS': 'statistical', 'XGBoost': 'ML',
         'Random Forest': 'ML', 'Decision Tree': 'ML', 'Prophet': 'ML'}
COLORS = {'ARIMA': '#4e79a7', 'ETS': '#59a14f', 'XGBoost': '#e15759',
          'Random Forest': '#b07aa1', 'Decision Tree': '#9c755f', 'Prophet': '#f28e2b'}
REP_COUNTRIES = ['DE', 'FR', 'SE', 'EE', 'PL', 'RO', 'IE', 'ES', 'IT']


def metrics(actual, pred):
    err = np.asarray(actual, float) - np.asarray(pred, float)
    return {'MAE': np.mean(np.abs(err)), 'RMSE': np.sqrt(np.mean(err ** 2)),
            'MAPE': np.mean(np.abs(err / np.asarray(actual, float))) * 100}


def main():
    cv = pd.concat([pd.read_csv(os.path.join(OUT, "cv_results.csv")),
                    pd.read_csv(os.path.join(OUT, "cv_results_trees.csv"))],
                   ignore_index=True)

    # --- 1. метрики кожної моделі на власній вибірці (як у табл. 1 статті) ---
    rows = []
    for m in MODELS:
        c = cv[cv.model == m]
        r = metrics(c.actual, c.predicted)
        r.update(model=m, cls=CLASS[m], n_obs=len(c))
        rows.append(r)

    # --- спільна підвибірка (усі шість моделей) ---
    wide = cv.pivot_table(index=['geo', 'year'], columns='model',
                          values='predicted').reset_index()
    act = cv.groupby(['geo', 'year'], as_index=False)['actual'].first()
    wide = wide.merge(act, on=['geo', 'year']).dropna(subset=MODELS)
    wide['Ensemble'] = (wide['ARIMA'] + wide['XGBoost']) / 2
    r = metrics(wide.actual, wide.Ensemble)
    r.update(model='Ensemble (ARIMA+XGBoost)', cls='combination', n_obs=len(wide))
    rows.append(r)
    for m in MODELS:  # для довідки -- метрики на спільній підвибірці
        rr = metrics(wide.actual, wide[m])
        rows.append({'model': m + ' [common]', 'cls': CLASS[m], 'n_obs': len(wide), **rr})
    summary = pd.DataFrame(rows)[['model', 'cls', 'MAE', 'RMSE', 'MAPE', 'n_obs']]
    summary.to_csv(os.path.join(OUT, "model_comparison_all.csv"), index=False)
    pd.set_option('display.width', 160)
    print(summary.round(3).to_string(index=False))
    print(f"\nCommon subsample: {len(wide)} obs, {wide.geo.nunique()} countries")

    # --- 2. тест Дібольда-Маріано ---
    names = MODELS + ['Ensemble']
    for m in names:
        wide[f'err_{m}'] = wide['actual'] - wide[m]
    pairs = [('ARIMA', 'XGBoost'), ('ETS', 'XGBoost'), ('Prophet', 'XGBoost'),
             ('Decision Tree', 'XGBoost'), ('Random Forest', 'XGBoost'),
             ('Decision Tree', 'Random Forest'),
             ('ETS', 'ARIMA'), ('Prophet', 'ARIMA'),
             ('Decision Tree', 'ARIMA'), ('ARIMA', 'Random Forest'),
             ('XGBoost', 'Ensemble'), ('ARIMA', 'Ensemble')]
    dm_rows = []
    for bench, test in pairs:
        for loss in ['squared', 'absolute']:
            dm, p_pool = _dm.diebold_mariano(wide[f'err_{bench}'], wide[f'err_{test}'],
                                             h=1, loss=loss)
            t, p_panel, mean_d, n_c = _dm.panel_dm(wide, bench, test, loss)
            # у скількох країнах друга модель пари точніша (за MAE)
            per_c = wide.assign(eb=wide[f'err_{bench}'].abs(),
                                et=wide[f'err_{test}'].abs()).groupby('geo')[['eb', 'et']].mean()
            dm_rows.append({'benchmark': bench, 'model': test, 'loss': loss,
                            'DM_pooled': round(dm, 3), 'p_pooled': round(p_pool, 4),
                            't_panel': round(t, 3), 'p_panel': round(p_panel, 4),
                            'countries_model_better': int((per_c.et < per_c.eb).sum()),
                            'n_obs': len(wide)})
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(os.path.join(OUT, "dm_test_all.csv"), index=False)
    print("\n" + dm_df.to_string(index=False))

    # --- 3. найточніша модель по країнах (MAPE, як у статті) ---
    cv['ape'] = (cv.actual - cv.predicted).abs() / cv.actual * 100
    mape_c = cv.groupby(['geo', 'model']).ape.mean().unstack()[MODELS]
    best = mape_c.idxmin(axis=1).rename('best_model').to_frame()
    best['best_class'] = best.best_model.map(CLASS)
    best['best_stat'] = mape_c[['ARIMA', 'ETS']].min(axis=1)
    best['best_ml'] = mape_c[['XGBoost', 'Random Forest', 'Decision Tree', 'Prophet']].min(axis=1)
    best.join(mape_c.round(2)).to_csv(os.path.join(OUT, "best_model_by_country.csv"))
    print("\nBest model by country (MAPE):", best.best_model.value_counts().to_dict())
    print("Best class by country:", best.best_class.value_counts().to_dict())
    # лише чотири початкові моделі -- контроль збігу зі статтею (16/7/3/1)
    four = mape_c[['ARIMA', 'ETS', 'XGBoost', 'Prophet']].idxmin(axis=1).value_counts()
    print("Four original models:", four.to_dict())

    # --- 4. прогнози ЄС-27 ---
    fc = pd.concat([pd.read_csv(os.path.join(OUT, "forecasts_2025_2029.csv")),
                    pd.read_csv(os.path.join(OUT, "forecasts_trees.csv"))], ignore_index=True)
    eu = fc.groupby(['year', 'model']).forecast.mean().unstack()[MODELS]
    eu.to_csv(os.path.join(OUT, "forecasts_eu27_all.csv"))
    print("\nEU27 mean forecast:\n", eu.round(2).to_string())

    # --- 5. рисунки ---
    base = summary[summary.model.isin(MODELS)].set_index('model').loc[MODELS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    short = ['ARIMA', 'ETS', 'XGBoost', 'Random\nForest', 'Decision\nTree', 'Prophet']
    for ax, met in zip(axes, ['MAE', 'RMSE', 'MAPE']):
        vals = base[met].values
        ax.bar(range(len(vals)), vals, color=[COLORS[m] for m in MODELS])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(short, fontsize=9)
        ax.set_title(met)
        ax.set_ylabel(met + (' (%)' if met == 'MAPE' else ' (p.p.)'))
        for i, v in enumerate(vals):
            ax.text(i, v * 1.02, f'{v:.2f}' if met == 'MAPE' else f'{v:.3f}',
                    ha='center', fontsize=8.5)
        ax.set_ylim(0, vals.max() * 1.15)
    plt.suptitle('Cross-Validation Metrics: Model Comparison (2013-2024)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig8_model_comparison.png"), bbox_inches='tight', dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(17, 11.5))
    for ax, m in zip(axes.flat, MODELS):
        c = cv[cv.model == m]
        ax.scatter(c.actual, c.predicted, alpha=0.4, s=14, color=COLORS[m])
        lims = [min(c.actual.min(), c.predicted.min()) - 0.5,
                max(c.actual.max(), c.predicted.max()) + 0.5]
        ax.plot(lims, lims, 'k--', linewidth=1, label='Perfect prediction')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        r = metrics(c.actual, c.predicted)
        ax.set_title(f"{m} (MAE={r['MAE']:.3f}, MAPE={r['MAPE']:.1f}%)")
        ax.set_xlabel('Actual (%)')
        ax.set_ylabel('Predicted (%)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.suptitle('Cross-Validation: Actual vs Predicted', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig10_actual_vs_predicted.png"), bbox_inches='tight', dpi=150)
    plt.close()

    df = pd.read_csv(os.path.join(BASE, "data", "panel_dataset.csv"))
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ax, geo in zip(axes.flat, REP_COUNTRIES):
        hist = df[df.geo == geo][['year', TARGET]].dropna().sort_values('year')
        ax.plot(hist.year, hist[TARGET], 'ko-', markersize=4, label='Historical', linewidth=1.5)
        for m in MODELS:
            f = fc[(fc.model == m) & (fc.geo == geo)]
            fp = pd.concat([pd.DataFrame({'year': [hist.year.iloc[-1]],
                                          'forecast': [hist[TARGET].iloc[-1]]}),
                            f[['year', 'forecast']]])
            ax.plot(fp.year, fp.forecast, '--', color=COLORS[m], linewidth=1.4,
                    label=m, marker='s', markersize=3)
        ax.set_title(f"{df[df.geo == geo].country.iloc[0]} ({geo})")
        ax.set_xlabel('Year')
        ax.set_ylabel('ICT Specialists (%)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.suptitle('ICT Specialists Forecast by Country (2025-2029)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig9_forecasts_by_country.png"), bbox_inches='tight', dpi=150)
    plt.close()
    print("\nSaved figures fig8, fig9, fig10 and CSV outputs")


if __name__ == "__main__":
    main()
