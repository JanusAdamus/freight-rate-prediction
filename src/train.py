"""Validate on time + unseen-city holdouts, then fit on all clean data and write predictions.

Usage: python -m src.train
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import pipeline as P

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT_START = pd.Timestamp("2025-09-01")  # train Jan-Aug, evaluate Sep-Oct: mimics predicting Nov-Dec
N_HOLDOUT_CITIES = 8  # same count as cities that only appear in validation.csv
SEED = 0
_NO_MI = [f for f in P.GEO_FEATURES if f != "market_index"]
VARIANTS = {
    "final (no market_index, no quote_signal)": _NO_MI,
    "+ market_index": P.GEO_FEATURES,
    "+ weekly market_index": _NO_MI + ["market_index_weekly"],
    "+ quote_signal": _NO_MI + ["quote_signal"],
}
# quote_signal leaks the label in some months but is pure noise in Aug and Nov-Dec (src/diagnostics.py).
FINAL_VARIANT = "final (no market_index, no quote_signal)"


MONOTONE = {"weight_lb": 1, "distance": -1, "log_distance": -1}


def make_model(features: list[str]) -> HistGradientBoostingRegressor:
    # squared loss + 31 leaves won every rolling-origin and unseen-city fold (small but consistent margin)
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=1500, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, categorical_features=["equipment_code"],
        # physical priors: heavier -> pricier; longer -> cheaper per mile (only constrains present features)
        monotonic_cst={k: v for k, v in MONOTONE.items() if k in features},
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=SEED,
    )


def lane_baseline(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Median $/mile per lane+equipment, falling back to equipment median for unseen lanes."""
    rpm = train.assign(rpm=train["posted_rate"] / train["distance"])
    lane = rpm.groupby(["pickup", "delivery", "equipment"])["rpm"].median().rename("lane")
    equip = rpm.groupby("equipment")["rpm"].median().rename("equip")
    merged = test.join(lane, on=["pickup", "delivery", "equipment"]).join(equip, on="equipment")
    return merged["lane"].fillna(merged["equip"]).to_numpy() * test["distance"].to_numpy()


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "MAE": round(float(np.mean(np.abs(actual - predicted))), 2),
        "MAPE_%": round(float(np.mean(np.abs(actual - predicted) / actual) * 100), 2),
    }


def fit(frame: pd.DataFrame, features: list[str]) -> HistGradientBoostingRegressor:
    clean = P.drop_rate_outliers(frame)
    model = make_model(features)
    model.fit(P.build_features(clean)[features], P.target(clean))
    return model


def predict(model, frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    return P.to_rate(model.predict(P.build_features(frame)[features]), frame)


def holdouts(data: pd.DataFrame) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    past, future = data[data["date"] < HOLDOUT_START], data[data["date"] >= HOLDOUT_START]
    cities = np.random.default_rng(SEED).choice(sorted(set(data["pickup"])), N_HOLDOUT_CITIES, replace=False)
    touches = lambda f: f["pickup"].isin(cities) | f["delivery"].isin(cities)  # noqa: E731
    august = data["date"].dt.month == 8
    return {
        "time (Sep-Oct)": (past, future),
        "time + unseen cities": (past[~touches(past)], future[touches(future)]),
        # August is the only training month whose quote_signal regime matches Nov-Dec (pure noise).
        "August (Nov-Dec-like quote_signal)": (data[data["date"] < "2025-08-01"], data[august]),
    }


def evaluate(data: pd.DataFrame) -> dict:
    report = {}
    for name, (train, test) in holdouts(data).items():
        actual = test["posted_rate"].to_numpy()
        clean_mask = (test["posted_rate"] / test["distance"]).between(P.RPM_MIN, P.RPM_MAX).to_numpy()
        rows = {"lane-median baseline": lane_baseline(P.drop_rate_outliers(train), test)}
        for variant, features in VARIANTS.items():
            rows[variant] = predict(fit(train, features), test, features)
        report[name] = {
            "n_train": len(train), "n_test": len(test),
            "all rows": {k: metrics(actual, v) for k, v in rows.items()},
            "clean rows": {k: metrics(actual[clean_mask], v[clean_mask]) for k, v in rows.items()},
        }
    return report


def main() -> None:
    data = P.load("train_test.csv")
    validation = P.load("validation.csv")
    report = evaluate(data)
    print(json.dumps(report, indent=2))

    features = VARIANTS[FINAL_VARIANT]
    model = fit(data, features)
    out = pd.read_csv(ROOT / "data" / "validation_predictions_template.csv")[["load_id"]]
    rates = pd.Series(predict(model, validation, features), index=validation["load_id"])
    out["predicted_rate"] = out["load_id"].map(rates).round(2)
    if out["predicted_rate"].isna().any():
        raise ValueError("template has load_ids missing from validation.csv")
    out.to_csv(ROOT / "validation_predictions.csv", index=False)

    december = pd.read_csv(ROOT / "data" / "december_chart_inputs.csv")
    dec_frame = P.december_frame(december, P.city_coordinates(data), validation,
                                 quote_signal=float(data["quote_signal"].median()))
    december["predicted_rate"] = predict(model, dec_frame, features).round(2)
    december.to_csv(ROOT / "data" / "december_chart_inputs.csv", index=False)

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote validation_predictions.csv ({len(out)} rows) and December predictions")


if __name__ == "__main__":
    main()
