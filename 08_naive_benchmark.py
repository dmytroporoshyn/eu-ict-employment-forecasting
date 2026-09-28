"""
Наївний бенчмарк (випадкове блукання) як опорна модель порівняння.

Наївний прогноз: y_hat(t) = y(t-1). Це найжорсткіший базовий орієнтир
для персистентних макроекономічних рядів: його похибка тотожно дорівнює
фактичній річній зміні показника.

Рахує на тій самій спільній підвибірці, що й 06_dm_test.py:
  1) метрики MAE/RMSE/MAPE для наївної моделі, чотирьох основних моделей
     та ансамблю (ARIMA + XGBoost) / 2;
  2) тест Дібольда-Маріано кожної моделі ПРОТИ наївної, у двох
     специфікаціях (pooled з HLN-корекцією та панельній).

Знакова конвенція успадкована з 06_dm_test.py:
  DM > 0  =>  benchmark (наївна) гірша, тобто модель точніша.
"""
import os

import numpy as np
import pandas as pd

from importlib.machinery import SourceFileLoader

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "panel_dataset.csv")
CV = os.path.join(BASE, "output", "cv_results.csv")
OUT_DIR = os.path.join(BASE, "output")
TARGET = "ict_specialists_pct"

# перевикористовуємо реалізацію тестів з 06_dm_test.py, щоб знаки й
# HLN-корекція були тотожні тим, що вже наведені у статті
_dm = SourceFileLoader("dm_mod", os.path.join(BASE, "06_dm_test.py")).load_module()
diebold_mariano = _dm.diebold_mariano
panel_dm = _dm.panel_dm


def metrics(actual, pred):
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    err = actual - pred
    mask = actual != 0
    return {
        "MAE": round(float(np.mean(np.abs(err))), 3),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "MAPE": round(float(np.mean(np.abs(err[mask] / actual[mask])) * 100), 2),
    }


def main():
    cv = pd.read_csv(CV)
    panel = pd.read_csv(DATA)[["geo", "year", TARGET]].rename(columns={TARGET: "y"})

    # значення попереднього року -> наївний прогноз
    prev = panel.copy()
    prev["year"] = prev["year"] + 1
    prev = prev.rename(columns={"y": "naive_pred"})

    models = [m for m in ["ARIMA", "ETS", "XGBoost", "Prophet"] if m in set(cv["model"])]

    wide_pred = cv.pivot_table(index=["geo", "year"], columns="model",
                               values="predicted").reset_index()
    actual = cv.groupby(["geo", "year"], as_index=False)["actual"].first()

    wide = wide_pred.merge(actual, on=["geo", "year"]).merge(prev, on=["geo", "year"])
    wide = wide.dropna(subset=models + ["naive_pred", "actual"])  # спільна підвибірка

    # ансамбль двох найточніших моделей
    wide["Ensemble"] = (wide["ARIMA"] + wide["XGBoost"]) / 2
    wide["Naive"] = wide["naive_pred"]

    print(f"Спільних спостережень: {len(wide)} по {wide['geo'].nunique()} країнах")

    # --- 1. метрики ---
    rows = []
    for name in ["Naive"] + models + ["Ensemble"]:
        m = metrics(wide["actual"], wide[name])
        m["model"] = name
        # у скількох країнах модель точніша за наївну (за MAE)
        per_c = wide.assign(
            e_m=(wide["actual"] - wide[name]).abs(),
            e_n=(wide["actual"] - wide["Naive"]).abs(),
        ).groupby("geo")[["e_m", "e_n"]].mean()
        m["countries_better_than_naive"] = int((per_c["e_m"] < per_c["e_n"]).sum())
        rows.append(m)
    met_df = pd.DataFrame(rows)[
        ["model", "MAE", "RMSE", "MAPE", "countries_better_than_naive"]
    ]
    met_df.to_csv(os.path.join(OUT_DIR, "naive_benchmark.csv"), index=False)
    print("\n" + met_df.to_string(index=False))

    # --- 2. DM-тести проти наївної моделі ---
    for name in ["Naive"] + models + ["Ensemble"]:
        wide[f"err_{name}"] = wide["actual"] - wide[name]

    dm_rows = []
    for test in models + ["Ensemble"]:
        for loss in ["squared", "absolute"]:
            dm, p_pool = diebold_mariano(wide["err_Naive"], wide[f"err_{test}"],
                                         h=1, loss=loss)
            t_panel, p_panel, mean_d, n_c = panel_dm(wide, "Naive", test, loss)
            dm_rows.append({
                "benchmark": "Naive",
                "model": test,
                "loss": loss,
                "DM_pooled": round(dm, 3),
                "p_pooled": round(p_pool, 4),
                "t_panel": round(t_panel, 3),
                "p_panel": round(p_panel, 4),
                "n_countries": n_c,
            })
    dm_df = pd.DataFrame(dm_rows)
    dm_df.to_csv(os.path.join(OUT_DIR, "naive_dm_tests.csv"), index=False)
    print("\n" + dm_df[dm_df.loss == "squared"].to_string(index=False))
    print("\nЗбережено: output/naive_benchmark.csv, output/naive_dm_tests.csv")


if __name__ == "__main__":
    main()
