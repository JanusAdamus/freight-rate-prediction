"""Evidence behind the feature decisions: quote_signal regimes and market_index level drift.

Usage: python -m src.diagnostics
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import pipeline as P
from src import train as T


def quote_signal_regimes(data: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    """Per month: share of loads where quote_signal is within 2% of the true $/mile, and the
    per-day correlation of log(quote_signal) with a no-quote model's log($/mile) prediction.
    The correlation needs no labels, so it also classifies the validation months."""
    features = T.VARIANTS[T.FINAL_VARIANT]
    model = T.fit(data, features)
    rows = []
    for frame in (data, validation):
        pred_rpm = T.predict(model, frame, features) / frame["distance"].to_numpy()
        g = pd.DataFrame({"date": frame["date"], "lq": np.log(frame["quote_signal"]), "lp": np.log(pred_rpm)})
        if "posted_rate" in frame:
            g["close"] = (np.abs(np.log(frame["posted_rate"] / frame["distance"] / frame["quote_signal"])) < 0.02)
        daily = g.groupby("date").apply(lambda x: x["lq"].corr(x["lp"]))
        for month, corr in daily.groupby(daily.index.to_period("M")):
            in_month = g["date"].dt.to_period("M") == month
            rows.append({
                "month": str(month),
                "share_qs_within_2pct": round(g.loc[in_month, "close"].mean(), 3) if "close" in g else np.nan,
                "daily_corr_min": round(corr.min(), 2), "daily_corr_max": round(corr.max(), 2),
                "regime": "direct" if corr.min() > 0.5 else "inverted" if corr.max() < -0.5 else "noise",
            })
    return pd.DataFrame(rows)


def market_index_by_month(data: pd.DataFrame, validation: pd.DataFrame) -> pd.DataFrame:
    rpm = data["posted_rate"] / data["distance"]
    clean = data[rpm.between(P.RPM_MIN, P.RPM_MAX)]
    train = clean.groupby(clean["date"].dt.to_period("M")).agg(
        market_index=("market_index", "mean"),
        median_rpm=("posted_rate", lambda s: (s / clean.loc[s.index, "distance"]).median()))
    val = validation.groupby(validation["date"].dt.to_period("M")).agg(market_index=("market_index", "mean"))
    return pd.concat([train, val]).round(3)


def main() -> None:
    data, validation = P.load("train_test.csv"), P.load("validation.csv")
    pd.set_option("display.width", 160)
    print("quote_signal regime by month (train Jan-Oct, validation Nov-Dec):")
    print(quote_signal_regimes(data, validation).to_string(index=False))
    print("\nmarket_index level vs price level by month:")
    print(market_index_by_month(data, validation).to_string())


if __name__ == "__main__":
    main()
