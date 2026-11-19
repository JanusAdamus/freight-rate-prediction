"""Formal hypothesis tests behind every modelling decision. Writes results/stats_tests.json.

Usage: python -m src.stats_tests
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from src import pipeline as P
from src import train as T

warnings.filterwarnings("ignore", category=FutureWarning)
ROOT = T.ROOT
FINAL = T.VARIANTS[T.FINAL_VARIANT]
NW_LAG = 7  # daily loss differences are autocorrelated within a week
N_CITY_DRAWS = 20
DIRECT, INVERTED = [1, 2, 3, 6, 9], [4, 5, 7, 10]


def diebold_mariano(err_a: np.ndarray, err_b: np.ndarray, days: np.ndarray) -> dict:
    """H0: equal expected absolute error. Loss differential averaged per day (unit of dependence),
    Newey-West variance. Positive mean -> model b is better."""
    d = pd.Series(err_a - err_b).groupby(days).mean().to_numpy()
    n, centered = len(d), d - d.mean()
    var = centered @ centered / n
    for k in range(1, NW_LAG + 1):
        var += 2 * (1 - k / (NW_LAG + 1)) * (centered[k:] @ centered[:-k] / n)
    z = d.mean() / np.sqrt(var / n)
    wilcoxon_p = stats.wilcoxon(d).pvalue
    return {"mean_daily_MAE_diff": round(float(d.mean()), 2), "DM_z": round(float(z), 2),
            "DM_p": float(2 * stats.norm.sf(abs(z))), "wilcoxon_p": float(wilcoxon_p), "n_days": n}


def clean_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[(frame["posted_rate"] / frame["distance"]).between(P.RPM_MIN, P.RPM_MAX)]


def compare(train: pd.DataFrame, test: pd.DataFrame, feats_a: list[str] | None, feats_b: list[str]) -> dict:
    """a=None means the lane-median baseline."""
    test = clean_rows(test)
    actual = test["posted_rate"].to_numpy()
    pred_a = (T.lane_baseline(P.drop_rate_outliers(train), test) if feats_a is None
              else T.predict(T.fit(train, feats_a), test, feats_a))
    pred_b = T.predict(T.fit(train, feats_b), test, feats_b)
    out = diebold_mariano(np.abs(actual - pred_a), np.abs(actual - pred_b), test["date"].to_numpy())
    out.update({"MAE_a": T.metrics(actual, pred_a)["MAE"], "MAE_b": T.metrics(actual, pred_b)["MAE"]})
    return out


def model_comparisons(t: pd.DataFrame) -> dict:
    m = t["date"].dt.month
    with_mi, with_qs = P.GEO_FEATURES, FINAL + ["quote_signal"]
    sep_oct, aug = (t[m < 9], t[m >= 9]), (t[m < 8], t[m == 8])
    return {
        "final_vs_baseline_SepOct": compare(*sep_oct, None, FINAL),
        "final_vs_baseline_Aug": compare(*aug, None, FINAL),
        "final_vs_plus_market_index_SepOct": compare(*sep_oct, with_mi, FINAL),
        "final_vs_plus_market_index_Aug": compare(*aug, with_mi, FINAL),
        "final_vs_plus_quote_signal_Aug": compare(*aug, with_qs, FINAL),
    }


def selection_independent_of_holdout(t: pd.DataFrame) -> dict:
    """Was the final configuration also better on folds that end before September?"""
    m = t["date"].dt.month
    prev = T.make_model  # previous config: absolute loss, 63 leaves
    out = {}
    for name, (tr, te) in {"train<7 -> Jul-Aug": (t[m < 7], t[m.isin([7, 8])]),
                           "train<8 -> Aug": (t[m < 8], t[m == 8])}.items():
        te = clean_rows(te)
        actual = te["posted_rate"].to_numpy()
        c = P.drop_rate_outliers(tr)
        old = prev(FINAL).set_params(loss="absolute_error", max_leaf_nodes=63).fit(P.build_features(c)[FINAL], P.target(c))
        p_old = P.to_rate(old.predict(P.build_features(te)[FINAL]), te)
        p_new = T.predict(T.fit(tr, FINAL), te, FINAL)
        out[name] = diebold_mariano(np.abs(actual - p_old), np.abs(actual - p_new), te["date"].to_numpy())
    return out


def repeated_unseen_cities(t: pd.DataFrame) -> dict:
    m = t["date"].dt.month
    rows = []
    for seed in range(N_CITY_DRAWS):
        cities = np.random.default_rng(seed).choice(sorted(set(t["pickup"])), T.N_HOLDOUT_CITIES, replace=False)
        touch = t["pickup"].isin(cities) | t["delivery"].isin(cities)
        tr, te = t[(m < 9) & ~touch], clean_rows(t[(m >= 9) & touch])
        actual = te["posted_rate"].to_numpy()
        rows.append({"final": T.metrics(actual, T.predict(T.fit(tr, FINAL), te, FINAL))["MAE"],
                     "baseline": T.metrics(actual, T.lane_baseline(P.drop_rate_outliers(tr), te))["MAE"]})
    r = pd.DataFrame(rows)
    wins = int((r["final"] < r["baseline"]).sum())
    return {"draws": N_CITY_DRAWS, "final_MAE_mean": round(r["final"].mean(), 2),
            "final_MAE_min_max": [r["final"].min(), r["final"].max()],
            "baseline_MAE_mean": round(r["baseline"].mean(), 2), "final_wins": wins,
            "sign_test_p": float(stats.binomtest(wins, N_CITY_DRAWS, 0.5, alternative="greater").pvalue)}


def lane_residuals(t: pd.DataFrame) -> pd.DataFrame:
    c = clean_rows(t).copy()
    c["rpm"] = c["posted_rate"] / c["distance"]
    c["res"] = np.log(c["rpm"] / c.groupby(["pickup", "delivery", "equipment"])["rpm"].transform("median"))
    return c


def quote_signal_tests(t: pd.DataFrame, v: pd.DataFrame) -> dict:
    """Per day, H0: corr(log quote_signal, log predicted $/mile) = 0. Label-free, so it runs on validation too."""
    model = T.fit(t, FINAL)
    out = {}
    val_p = []
    for label, frame in (("train", t), ("validation", v)):
        lp = np.log(T.predict(model, frame, FINAL) / frame["distance"].to_numpy())
        g = pd.DataFrame({"date": frame["date"].to_numpy(), "lq": np.log(frame["quote_signal"].to_numpy()), "lp": lp})
        for month, gm in g.groupby(g["date"].dt.month):
            tests = [stats.pearsonr(x["lq"], x["lp"]) for _, x in gm.groupby("date")]
            r = np.array([d.statistic for d in tests])
            p = np.array([d.pvalue for d in tests])
            if label == "validation":
                val_p.extend(p)
            out[f"{label}_month_{month}"] = {"mean_daily_r": round(float(r.mean()), 3),
                                             "share_days_p<0.05": round(float((p < 0.05).mean()), 3)}
    sig = int((np.array(val_p) < 0.05).sum())
    out["validation_sig_days"] = f"{sig}/{len(val_p)}"
    out["validation_sig_days_vs_5pct_binomial_p"] = float(stats.binomtest(sig, len(val_p), 0.05).pvalue)
    aug = t.loc[t["date"].dt.month == 8, "quote_signal"]
    sep = t.loc[t["date"].dt.month == 9, "quote_signal"]
    out["KS_Aug_vs_validation_p"] = float(stats.ks_2samp(aug, v["quote_signal"]).pvalue)
    out["KS_Sep_vs_validation_p"] = float(stats.ks_2samp(sep, v["quote_signal"]).pvalue)
    return out


def outlier_tests(t: pd.DataFrame) -> dict:
    t = t.assign(rpm=t["posted_rate"] / t["distance"])
    lane = clean_rows(t).assign(rpm=lambda f: f["posted_rate"] / f["distance"]).groupby(
        ["pickup", "delivery", "equipment"])["rpm"].median().rename("lane")
    o = t.join(lane, on=["pickup", "delivery", "equipment"])
    o = o[o["date"].dt.month.isin(DIRECT) & o["lane"].notna()]
    bad = ~o["rpm"].between(P.RPM_MIN, P.RPM_MAX)
    qs_ratio = o["quote_signal"] / o["lane"]
    rate_ratio = o["rpm"] / o["lane"]
    return {
        "n_outliers_direct_months": int(bad.sum()),
        "quote/lane median outliers vs normal": [round(qs_ratio[bad].median(), 3), round(qs_ratio[~bad].median(), 3)],
        "quote/lane MannWhitney_p": float(stats.mannwhitneyu(qs_ratio[bad], qs_ratio[~bad]).pvalue),
        "posted/lane median outliers vs normal": [round(rate_ratio[bad].median(), 3), round(rate_ratio[~bad].median(), 3)],
        "posted/lane |log| MannWhitney_p": float(stats.mannwhitneyu(np.abs(np.log(rate_ratio[bad])), np.abs(np.log(rate_ratio[~bad]))).pvalue),
    }


def negative_weight_test(t: pd.DataFrame) -> dict:
    """res ~ a + b*log|w| + c*is_negative ; H0: c = 0 (negative sign carries no price information)."""
    c = lane_residuals(t).dropna(subset=["weight"])
    X = np.c_[np.ones(len(c)), np.log(c["weight"].abs()), (c["weight"] < 0).astype(float)]
    y = c["res"].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    se = np.sqrt(np.diag(resid @ resid / (len(y) - X.shape[1]) * np.linalg.inv(X.T @ X)))
    tstat = beta / se
    return {"coef_log_abs_weight": round(float(beta[1]), 4), "coef_log_abs_weight_p": float(2 * stats.t.sf(abs(tstat[1]), len(y) - 3)),
            "coef_is_negative": round(float(beta[2]), 4), "coef_is_negative_p": float(2 * stats.t.sf(abs(tstat[2]), len(y) - 3)),
            "n_negative": int((c["weight"] < 0).sum())}


def calendar_tests(t: pd.DataFrame) -> dict:
    c = lane_residuals(t)
    daily = c.groupby("date")["res"].mean()
    by_month = [g.to_numpy() for _, g in daily.groupby(daily.index.month)]
    by_dow = [g.to_numpy() for _, g in c.groupby(c["date"].dt.dayofweek)["res"]]
    holidays = ["2025-01-01", "2025-01-20", "2025-02-17", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01", "2025-10-13"]
    jumps = []
    for h in pd.to_datetime(holidays):
        near = (np.abs((daily.index - h).days) <= 7) & (daily.index != h)
        jumps.append(daily.get(h, np.nan) - daily[near].mean())
    jumps = np.array(jumps)
    month_means = daily.groupby(daily.index.month).mean()
    return {
        "season_KruskalWallis_p": float(stats.kruskal(*by_month).pvalue),
        "season_range_pct": round(float((month_means.max() - month_means.min()) * 100), 2),
        "dow_KruskalWallis_p": float(stats.kruskal(*by_dow).pvalue),
        "dow_range_pct": round(float(np.ptp(c.groupby(c["date"].dt.dayofweek)["res"].mean().to_numpy()) * 100), 2),
        "holiday_jump_mean_pct": round(float(jumps.mean() * 100), 3),
        "holiday_jump_ttest_p": float(stats.ttest_1samp(jumps, 0).pvalue),
    }


def adversarial_test(t: pd.DataFrame, v: pd.DataFrame, n_perm: int = 20) -> dict:
    feats = [f for f in FINAL if f != "day_of_week"]
    a = pd.concat([P.build_features(t[t["date"].dt.month >= 9])[feats].assign(y=0),
                   P.build_features(v)[feats].assign(y=1)], ignore_index=True)
    clf = HistGradientBoostingClassifier(max_iter=150, random_state=0)
    auc = roc_auc_score(a["y"], cross_val_predict(clf, a[feats], a["y"], cv=5, method="predict_proba")[:, 1])
    rng = np.random.default_rng(0)
    null = [roc_auc_score(yp, cross_val_predict(clf, a[feats], yp, cv=5, method="predict_proba")[:, 1])
            for yp in (rng.permutation(a["y"].to_numpy()) for _ in range(n_perm))]
    coords = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "delta_lat", "delta_lon"]
    other = [f for f in feats if f not in coords]
    auc_other = roc_auc_score(a["y"], cross_val_predict(clf, a[other], a["y"], cv=5, method="predict_proba")[:, 1])
    return {"AUC": round(float(auc), 3), "null_AUC_max": round(float(max(null)), 3),
            "permutation_p_upper_bound": 1 / (n_perm + 1), "AUC_without_coordinates": round(float(auc_other), 3)}


def conformal_and_stability(t: pd.DataFrame, v: pd.DataFrame) -> dict:
    m = t["date"].dt.month
    model = T.fit(t[m < 9], FINAL)
    cal, te = t[m == 9], t[m == 10]
    res_cal = np.abs(np.log(cal["posted_rate"] / T.predict(model, cal, FINAL)))
    res_te = np.abs(np.log(te["posted_rate"] / T.predict(model, te, FINAL)))
    out = {}
    for level in (0.8, 0.9):
        q = np.quantile(res_cal, level)
        hits = int((res_te <= q).sum())
        out[f"conformal_{int(level * 100)}"] = {
            "half_width_pct": round(float((np.exp(q) - 1) * 100), 2),
            "Oct_coverage": round(hits / len(te), 4),
            "binomial_p_coverage_below_nominal": float(stats.binomtest(hits, len(te), level, alternative="less").pvalue)}
    preds = []
    for seed in range(5):
        c = P.drop_rate_outliers(t)
        mdl = T.make_model(FINAL).set_params(random_state=seed).fit(P.build_features(c)[FINAL], P.target(c))
        preds.append(P.to_rate(mdl.predict(P.build_features(v)[FINAL]), v))
    spread = np.std(preds, axis=0) / np.mean(preds, axis=0)
    out["seed_spread_pct_median"] = round(float(np.median(spread) * 100), 3)
    out["seed_spread_pct_p99"] = round(float(np.quantile(spread, 0.99) * 100), 3)
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    adjusted, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, p * (len(items) - i)))
        adjusted[k] = running
    return adjusted


def main() -> None:
    t, v = P.load("train_test.csv"), P.load("validation.csv")
    results = {
        "model_comparisons": model_comparisons(t),
        "selection_independent_of_SepOct": selection_independent_of_holdout(t),
        "repeated_unseen_cities": repeated_unseen_cities(t),
        "quote_signal": quote_signal_tests(t, v),
        "outliers": outlier_tests(t),
        "negative_weight": negative_weight_test(t),
        "calendar": calendar_tests(t),
        "adversarial": adversarial_test(t, v),
        "conformal_and_stability": conformal_and_stability(t, v),
    }
    primary = {f"DM {k}": x["DM_p"] for k, x in results["model_comparisons"].items()}
    primary["unseen cities sign test"] = results["repeated_unseen_cities"]["sign_test_p"]
    results["holm_adjusted_primary"] = holm(primary)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "stats_tests.json").write_text(json.dumps(results, indent=2, default=float))
    print(json.dumps(results, indent=2, default=float))


if __name__ == "__main__":
    main()
