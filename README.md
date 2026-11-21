# Forecasting ICT employment in the EU: statistical methods vs machine learning

Replication code and data for the article

> *Forecasting the share of ICT specialists in employment in EU countries: a comparison of statistical methods and machine learning algorithms* (manuscript under revision at the Journal of Operations Intelligence; submitted under the title *Forecasting employment in information services in EU countries: …*).

The study forecasts the share of ICT specialists in total employment of the 27 EU countries for 2025–2029 and compares six methods on a common test period (2013–2024): ARIMA and ETS (statistical, estimated per country), and decision tree, random forest, XGBoost (panel models estimated on all 27 countries) and Prophet (per country). Forecast accuracy is compared with Diebold–Mariano tests (pooled and panel versions), and a set of robustness checks separates the contribution of the panel design from that of the learning algorithm.

*Code and data to reproduce the results of the article on forecasting the share of ICT specialists in employment in EU countries.*

## Repository structure

```
data/            raw downloads from Eurostat and World Bank + the assembled panel (panel_dataset.csv)
output/          all numerical results (CSV) and the saved final XGBoost model (xgb_model.json)
output/print/    the eleven figures of the article at print size (300 dpi)
01_ ... 15_*.py  analysis scripts (numbers follow the order in which the analysis was developed)
```

## Requirements

Python 3.13 (tested with 3.13.1 on Windows 11). Install the dependencies with

```
pip install -r requirements.txt
```

Prophet needs a working CmdStan backend; `pip install prophet` installs it automatically on most systems.

## How to reproduce

Run the scripts from the repository root in this order (on Windows use `py` instead of `python`):

| Step | Script | What it does | Main outputs | Used in the article |
|------|--------|--------------|--------------|---------------------|
| 1 | `01_download_data.py` | downloads Eurostat and World Bank series via API | `data/*.csv` | Section 2.2 |
| 2 | `02_prepare_data.py` | assembles the EU-27 panel 2004–2024 | `data/panel_dataset.csv` | Section 2.2 |
| 3 | `03_eda.py` | descriptive statistics, correlations, ADF unit-root tests | `output/descriptive_statistics.csv` | Sections 2.3.1, 3.1 |
| 4 | `04_models.py` | expanding-window CV and 2025–2029 forecasts for ARIMA, ETS, XGBoost, Prophet | `output/cv_results.csv`, `output/forecasts_2025_2029.csv`, `output/xgb_model.json` | Tables 2, 5 |
| 5 | `05_shap_and_scenarios.py` | SHAP decomposition of XGBoost, scenarios | `output/scenarios_2025_2029.csv` | Section 3.6, Table 6 |
| 6 | `06_dm_test.py` | Diebold–Mariano tests for the four original models | `output/dm_test_results.csv` | Table 3 |
| 7 | `07_arima_diagnostics.py` | Ljung–Box residual diagnostics of ARIMA(1,1,1) | `output/arima_diagnostics.csv` | Section 2.3.1 |
| 8 | `15_arima_order_selection.py` | AIC check of the fixed ARIMA(1,1,1) order | `output/arima_order_selection.csv` | Section 2.3.1 |
| 9 | `09_tree_models.py` | decision tree and random forest with the XGBoost panel design; hyper-parameter sensitivity; ex ante check | `output/cv_results_trees.csv`, `output/forecasts_trees.csv`, `output/tree_sensitivity.csv`, `output/robustness_lagged_exog.csv` | Tables 2, 4, 5 |
| 10 | `10_model_comparison.py` | metrics and DM tests for all six models and the ensemble; best model by country | `output/model_comparison_all.csv`, `output/dm_test_all.csv`, `output/best_model_by_country.csv`, `output/forecasts_eu27_all.csv` | Tables 2, 3, 5 |
| 11 | `12_robustness.py` | panel linear regressions, XGBoost grid, ex ante ensemble, ARIMA with 2011 break dummy, AIC-selected ARIMA, Holm correction, CAGR 2011–2024 | `output/robustness_extra.csv`, `output/xgb_sensitivity.csv`, `output/dm_holm.csv`, `output/cagr_2011_2024.csv` | Table 4, Table 7, Section 3.1 |
| 12 | `14_figures_print.py` | all figures of the article at print size | `output/print/fig01–fig11` | Figures 1–11 |

### Revised manuscript (round 1 of peer review)

Scripts 16–21 reproduce all results of the revised manuscript (outputs in `output/revision/`). They implement the reviewers' requests: imputation within each training fold, the same information set for all models (exogenous predictors lagged by one year), multi-horizon rolling-origin backtesting (h = 1–5), prediction intervals, two-way (country × year) cluster-robust tests and bootstrap, nested validation of the combination, checks of the 2011 and 2021 breaks and the employment-weighted EU aggregate.

| Step | Script | What it does | Main outputs |
|------|--------|--------------|--------------|
| R1 | `16_download_employment.py` | downloads Eurostat `lfsa_egan` and EUROPOP2023 `proj_23np` | `data/eurostat_lfsa_egan.csv`, `data/eurostat_proj_23np.csv` |
| R2 | `17_revision_backtest.py` | multi-horizon backtest of all models (origins 2012–2023) and final forecasts (origin 2024); checks that the engine reproduces `output/cv_results.csv` exactly; `--aic` adds AIC-selected ARIMA (≈7 min) | `output/revision/backtest.csv`, `backtest_aic.csv` |
| R3 | `18_revision_evaluation.py` | accuracy by horizon, interval coverage, two-way tests, nested combination, 2021 break, robustness | `output/revision/results_revision.md` and CSV tables |
| R4 | `19_revision_forecast.py` | 2025–2029 forecasts with empirical intervals, SHAP, employment-weighted EU aggregate | `output/revision/forecast_summary.md`, `intervals_final.csv`, `eu_aggregate.csv` |
| R5 | `20_revision_figures.py` | figures of the revised manuscript | `output/revision/print/rfig*.png` |
| R6 | `21_revision_sensitivity.py` | hyper-parameter sensitivity in the revised specification | `output/revision/sensitivity_h1.csv` |

`08_naive_benchmark.py` compares all models with a naive random-walk forecast (`output/naive_benchmark.csv`, `output/naive_dm_tests.csv`). This benchmark was part of an earlier version of the manuscript and is not reported in the final article; it is kept for transparency. `11_research_design_figure.py` and `13_fig_2011_vs_2024.py` produce earlier (screen) versions of Figures 1 and 3 and are kept for completeness. `12_robustness.py` takes about ten minutes because the AIC-selected ARIMA is re-estimated at every cross-validation origin.

`09_tree_models.py` first re-runs the XGBoost cross-validation and asserts that it reproduces `output/cv_results.csv` exactly, which guarantees that all panel models share the same features and folds.

Results are deterministic (fixed random seeds). Random forest predictions may differ from the stored values in the 15th decimal place because of parallel summation.

## Data sources and licences

* Eurostat, datasets `isoc_sks_itspt`, `htec_emp_nat2`, `isoc_ci_ifp_iu`, `nama_10_pc`, `rd_e_gerdtot` (https://ec.europa.eu/eurostat/data/database). Reuse is authorised provided the source is acknowledged (Commission Decision 2011/833/EU).
* World Bank, World Development Indicators: `SL.SRV.EMPL.ZS`, `IT.NET.BBND.P2`, `NY.GDP.PCAP.CD`, `BX.GSR.CCIS.ZS`, `SE.TER.ENRR` (https://databank.worldbank.org/source/world-development-indicators), CC BY 4.0.

The panel data were downloaded on 12 April 2026 (file dates; values match the Eurostat release before its revision of 17 April 2026); `lfsa_egan` and `proj_23np` on 21 November 2026. Note that the Eurostat ICT-specialist series has a methodological break in 2011 (transition of the EU Labour Force Survey to ISCO-08); see Section 2.2 of the article.

## Licence

Code: MIT licence (see `LICENSE`). Derived data and results in `data/panel_dataset.csv` and `output/`: CC BY 4.0, with attribution to the original sources above.

## Citation

If you use this code, please cite the article (see `CITATION.cff`).
