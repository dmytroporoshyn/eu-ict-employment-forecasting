# -*- coding: utf-8 -*-
"""Diebold-Mariano tests for the original models.

(A) pooled DM with the Harvey-Leybourne-Newbold (HLN) correction;
(B) panel version: mean loss differential per country, t-test across countries.

Input: output/cv_results.csv. Output: output/dm_test_results.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

CV = "output/cv_results.csv"
OUT = "output/dm_test_results.csv"


def diebold_mariano(e1, e2, h=1, loss="squared"):
    """DM test for one error series with HLN correction. Returns (DM, p).
    DM > 0 means model 2 is more accurate than model 1 (benchmark)."""
    e1 = np.asarray(e1, float)
    e2 = np.asarray(e2, float)
    if loss == "squared":
        d = e1 ** 2 - e2 ** 2
    elif loss == "absolute":
        d = np.abs(e1) - np.abs(e2)
    else:
        raise ValueError(loss)

    n = len(d)
    d_bar = d.mean()
    # autocovariances up to lag h-1 (variance only for h = 1)
    gamma0 = np.sum((d - d_bar) ** 2) / n
    gammas = []
    for k in range(1, h):
        cov = np.sum((d[k:] - d_bar) * (d[:-k] - d_bar)) / n
        gammas.append(cov)
    var_d = (gamma0 + 2 * np.sum(gammas)) / n
    if var_d <= 0:
        return np.nan, np.nan
    dm = d_bar / np.sqrt(var_d)
    # Harvey-Leybourne-Newbold correction
    hln = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * hln
    p = 2 * stats.t.cdf(-abs(dm_hln), df=n - 1)
    return dm_hln, p


def panel_dm(df_wide, m_bench, m_test, loss="squared"):
    """Panel DM: country-mean loss differentials, one-sample t-test across countries."""
    e_b = df_wide[f"err_{m_bench}"]
    e_t = df_wide[f"err_{m_test}"]
    if loss == "squared":
        d = e_b ** 2 - e_t ** 2
    else:
        d = e_b.abs() - e_t.abs()
    per_country = d.groupby(df_wide["geo"]).mean()
    t_stat, p = stats.ttest_1samp(per_country, 0.0)
    return t_stat, p, per_country.mean(), per_country.shape[0]


def main():
    cv = pd.read_csv(CV)
    cv["err"] = cv["actual"] - cv["predicted"]

    # wide format: one row per (geo, year), error columns by model
    wide = cv.pivot_table(index=["geo", "year"], columns="model",
                          values="err").reset_index()
    models = [c for c in ["ARIMA", "ETS", "XGBoost", "Prophet"] if c in wide.columns]
    wide = wide.dropna(subset=models)  # common subsample
    wide = wide.rename(columns={m: f"err_{m}" for m in models})
    n_common = len(wide)
    print(f"Спільних спостережень для порівняння: {n_common}")

    pairs = [
        ("ARIMA", "XGBoost"),
        ("ETS", "XGBoost"),
        ("Prophet", "XGBoost"),
        ("ETS", "ARIMA"),
        ("Prophet", "ARIMA"),
    ]

    rows = []
    for bench, test in pairs:
        for loss in ["squared", "absolute"]:
            # (A) pooled
            dm, p_pool = diebold_mariano(wide[f"err_{bench}"],
                                         wide[f"err_{test}"], h=1, loss=loss)
            # (B) panel (by country)
            t_panel, p_panel, mean_d, n_c = panel_dm(wide, bench, test, loss)
            rows.append({
                "benchmark": bench,
                "model": test,
                "loss": loss,
                "DM_pooled": round(dm, 3),
                "p_pooled": round(p_pool, 4),
                "t_panel": round(t_panel, 3),
                "p_panel": round(p_panel, 4),
                "mean_loss_diff": round(mean_d, 5),
                "n_countries": n_c,
                "n_obs": n_common,
            })

    res = pd.DataFrame(rows)
    res.to_csv(OUT, index=False, encoding="utf-8")
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    print("\n=== Diebold-Mariano test results ===")
    print(res.to_string(index=False))
    print(f"\nЗбережено: {OUT}")

    print("\n--- Інтерпретація (squared loss, panel DM) ---")
    for _, r in res[res.loss == "squared"].iterrows():
        sig = ("значуща, p<0.01" if r.p_panel < 0.01 else
               "значуща, p<0.05" if r.p_panel < 0.05 else
               "значуща, p<0.10" if r.p_panel < 0.10 else
               "НЕзначуща")
        better = r.model if r.mean_loss_diff > 0 else r.benchmark
        print(f"  {r.model} vs {r.benchmark}: точніша {better}; "
              f"t={r.t_panel}, p={r.p_panel} ({sig})")


if __name__ == "__main__":
    main()
