"""Loading, cleaning and feature engineering for the freight rate model."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EARTH_RADIUS_MI = 3958.8
# Rate-per-mile outside this band is ~3.6x or ~0.3x the lane norm: entry errors, not market moves.
RPM_MIN, RPM_MAX = 1.2, 4.0
EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]
GEO_FEATURES = [
    "distance", "log_distance", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "delta_lat", "delta_lon", "straight_line_mi", "detour_ratio",
    "equipment_code", "weight_lb", "market_index", "day_of_week",
]


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, parse_dates=["date"])


def drop_rate_outliers(frame: pd.DataFrame) -> pd.DataFrame:
    rpm = frame["posted_rate"] / frame["distance"]
    return frame[rpm.between(RPM_MIN, RPM_MAX)].reset_index(drop=True)


def fill_market_index(frame: pd.DataFrame) -> pd.Series:
    """Missing market_index -> mean of the same day (known at prediction time), then global median."""
    daily = frame.groupby("date")["market_index"].transform("mean")
    return frame["market_index"].fillna(daily).fillna(frame["market_index"].median())


def market_index_weekly(frame: pd.DataFrame) -> pd.Series:
    """Daily market_index minus its centered 7-day mean: keeps the weekly wave (which tracks price)
    and drops the level (which drifts out of range between train and validation)."""
    daily = frame.assign(mi=fill_market_index(frame)).groupby("date")["mi"].mean()
    wave = daily - daily.rolling(7, center=True, min_periods=4).mean()
    return frame["date"].map(wave)


def haversine_mi(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_MI * np.arcsin(np.sqrt(a))


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Model inputs. City names are deliberately absent: 8 validation cities never appear in training."""
    unknown = set(frame["equipment"]) - set(EQUIPMENT)
    if unknown:
        raise ValueError(f"unknown equipment types: {sorted(unknown)}")
    straight = haversine_mi(frame["pickup_lat"], frame["pickup_lon"], frame["delivery_lat"], frame["delivery_lon"])
    return pd.DataFrame({
        "distance": frame["distance"],
        "log_distance": np.log(frame["distance"]),
        "pickup_lat": frame["pickup_lat"],
        "pickup_lon": frame["pickup_lon"],
        "delivery_lat": frame["delivery_lat"],
        "delivery_lon": frame["delivery_lon"],
        "delta_lat": frame["delivery_lat"] - frame["pickup_lat"],
        "delta_lon": frame["delivery_lon"] - frame["pickup_lon"],
        "straight_line_mi": straight,
        "detour_ratio": frame["distance"] / np.maximum(straight, 1.0),
        "equipment_code": frame["equipment"].map({e: i for i, e in enumerate(EQUIPMENT)}),
        # Negative weights carry normal rates: sign flip in entry. NaN stays NaN (model handles it).
        "weight_lb": frame["weight"].abs(),
        "market_index": fill_market_index(frame),
        "market_index_weekly": market_index_weekly(frame),
        "day_of_week": frame["date"].dt.dayofweek,
        "quote_signal": frame.get("quote_signal", pd.Series(np.nan, index=frame.index)),
    }, index=frame.index)


def target(frame: pd.DataFrame) -> pd.Series:
    """log rate-per-mile: removes the distance scale so the model learns lane/market effects."""
    return np.log(frame["posted_rate"] / frame["distance"])


def to_rate(log_rpm: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
    return np.exp(log_rpm) * frame["distance"].to_numpy()


def city_coordinates(frame: pd.DataFrame) -> pd.DataFrame:
    pick = frame[["pickup", "pickup_lat", "pickup_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    drop = frame[["delivery", "delivery_lat", "delivery_lon"]].set_axis(["city", "lat", "lon"], axis=1)
    return pd.concat([pick, drop]).drop_duplicates("city").set_index("city")


def december_frame(december: pd.DataFrame, coords: pd.DataFrame, validation: pd.DataFrame,
                   quote_signal: float) -> pd.DataFrame:
    """December chart rows lack coords, market_index and quote_signal. Coords come from known
    cities, market_index from the same days in validation.csv (unused by the final model, kept so
    every variant can run), and quote_signal is a per-load quote, so a neutral training median."""
    frame = december.copy()
    frame["quote_signal"] = quote_signal
    frame["date"] = pd.to_datetime(frame["date"])
    for side in ("pickup", "delivery"):
        missing = set(frame[side]) - set(coords.index)
        if missing:
            raise ValueError(f"no coordinates for {side} cities: {sorted(missing)}")
        frame[f"{side}_lat"] = frame[side].map(coords["lat"])
        frame[f"{side}_lon"] = frame[side].map(coords["lon"])
    daily_index = validation.groupby("date")["market_index"].mean()
    frame["market_index"] = frame["date"].map(daily_index)
    if frame["market_index"].isna().any():
        raise ValueError("validation.csv has no market_index for some December dates")
    return frame
