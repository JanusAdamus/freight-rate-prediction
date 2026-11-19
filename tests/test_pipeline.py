import numpy as np
import pandas as pd
import pytest

from src import pipeline as P


def _frame(**overrides) -> pd.DataFrame:
    row = {
        "pickup": "A", "delivery": "B", "pickup_lat": 38.0, "pickup_lon": -85.0,
        "delivery_lat": 41.0, "delivery_lon": -85.0, "distance": 250.0, "equipment": "Dry Van",
        "weight": -30000.0, "date": pd.Timestamp("2025-03-05"), "market_index": np.nan,
        "quote_signal": 2.0, "posted_rate": 550.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_negative_weight_becomes_positive():
    assert P.build_features(_frame())["weight_lb"].iloc[0] == 30000.0


def test_missing_market_index_is_filled():
    frame = pd.concat([_frame(), _frame(market_index=1.2)], ignore_index=True)
    assert P.fill_market_index(frame).tolist() == [1.2, 1.2]


def test_rate_outliers_dropped():
    frame = pd.concat([_frame(), _frame(posted_rate=5000.0), _frame(posted_rate=50.0)], ignore_index=True)
    assert len(P.drop_rate_outliers(frame)) == 1


def test_haversine_one_degree_latitude_is_about_69_miles():
    assert P.haversine_mi(0, 0, 1, 0) == pytest.approx(69.1, abs=0.1)


def test_target_round_trip():
    frame = _frame()
    assert P.to_rate(P.target(frame).to_numpy(), frame)[0] == pytest.approx(550.0)


def test_unknown_equipment_rejected():
    with pytest.raises(ValueError):
        P.build_features(_frame(equipment="Tanker"))


def test_december_frame_fills_coords_and_signals():
    coords = pd.DataFrame({"lat": [38.0, 41.0], "lon": [-85.0, -85.0]}, index=["A", "B"])
    validation = _frame(market_index=0.9)
    december = pd.DataFrame([{"pickup": "A", "delivery": "B", "distance": 250, "equipment": "Dry Van",
                              "weight": 32000, "date": "2025-03-05", "predicted_rate": np.nan}])
    out = P.december_frame(december, coords, validation, quote_signal=2.05)
    assert out[["pickup_lat", "delivery_lat", "market_index", "quote_signal"]].iloc[0].tolist() == [38.0, 41.0, 0.9, 2.05]
