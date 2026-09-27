#!/usr/bin/env python3
"""Observable features and inference for the all-outcome entry model."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from build_conditional_path_library import FEATURE_COLUMNS
from conditional_wick_assets import SUPPORTED_ASSETS
from prospective_entry_outcomes import horizon_slug, threshold_slug


MODEL_SCHEMA_VERSION = "prospective-entry-model-v1.0.0"
DISPLAY_HORIZONS_MINUTES = (1_440, 10_080, 43_200)
DISPLAY_THRESHOLDS_PCT = (2.0, 5.0, 10.0)
OUTCOME_CLASSES = ("target_first", "adverse_first", "ambiguous_intrabar", "neither")
OUTCOME_TO_CODE = {label: index for index, label in enumerate(OUTCOME_CLASSES)}
STATE_FEATURES = (
    "entry_age_minutes",
    "entry_distance_from_target_pct",
    "peak_distance_from_target_pct",
    "drawdown_from_peak_pct",
    "departure_to_entry_bars",
)
PRE_SIGNAL_WINDOWS_MINUTES = (60, 240, 480)
RECENT_WINDOWS_MINUTES = (15, 60, 240, 1_280)


def feature_columns() -> tuple[str, ...]:
    columns: list[str] = [*FEATURE_COLUMNS, *STATE_FEATURES]
    columns.extend(
        (
            "log_entry_age_minutes",
            "log_entry_distance_pct",
            "log_peak_distance_pct",
            "log_departure_to_entry_bars",
            "entry_bar_range_pct",
            "entry_bar_body_pct",
            "signal_to_entry_aligned_return_pct",
        )
    )
    for prefix, windows in (
        ("pre", PRE_SIGNAL_WINDOWS_MINUTES),
        ("recent", RECENT_WINDOWS_MINUTES),
    ):
        for minutes in windows:
            columns.extend(
                (
                    f"{prefix}_{minutes}m_aligned_return_pct",
                    f"{prefix}_{minutes}m_realized_vol_pct",
                    f"{prefix}_{minutes}m_mean_range_pct",
                    f"{prefix}_{minutes}m_volume_ratio",
                )
            )
    columns.extend(f"asset_{asset}" for asset in SUPPORTED_ASSETS)
    columns.append("direction_upper_wick")
    return tuple(columns)


FEATURE_COLUMNS_V1 = feature_columns()


def default_artifact_root(project_root: Path) -> Path:
    override = os.environ.get("CANDLE_PROJECTION_MODEL_DIR")
    if override:
        return Path(override)
    if local_app_data := os.environ.get("LOCALAPPDATA"):
        return Path(local_app_data) / "candle_projection_algo" / "prospective_entry_models"
    return project_root / "data" / "prospective_entry_models"


def _safe_index(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    result = np.full(len(indices), np.nan, dtype=float)
    valid = (indices >= 0) & (indices < len(values))
    result[valid] = values[indices[valid]]
    return result


def _rolling_at(
    values: pd.Series, indices: np.ndarray, window: int, operation: str
) -> np.ndarray:
    rolling = values.rolling(window=max(2, window), min_periods=2)
    computed = rolling.std(ddof=0) if operation == "std" else rolling.mean()
    return _safe_index(computed.to_numpy(dtype=float), indices)


def _window_features(
    frame: pd.DataFrame,
    end_indices: np.ndarray,
    direction_sign: np.ndarray,
    window_bars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    window_bars = max(1, int(window_bars))
    close = frame["close"].to_numpy(dtype=float)
    volume = frame["volume"].astype(float)
    past_indices = end_indices - window_bars
    current_close = _safe_index(close, end_indices)
    past_close = _safe_index(close, past_indices)
    aligned_return = direction_sign * (current_close / past_close - 1.0) * 100.0
    one_bar_return = frame["close"].astype(float).pct_change().mul(100.0)
    volatility = _rolling_at(one_bar_return, end_indices, window_bars, "std")
    range_pct = (
        (frame["high"].astype(float) - frame["low"].astype(float))
        / frame["close"].astype(float).replace(0.0, np.nan)
        * 100.0
    )
    mean_range = _rolling_at(range_pct, end_indices, window_bars, "mean")
    volume_mean = volume.rolling(window=max(2, window_bars), min_periods=2).mean()
    recent_volume = _safe_index(volume_mean.to_numpy(dtype=float), end_indices)
    prior_volume = _safe_index(volume_mean.to_numpy(dtype=float), past_indices)
    return (
        aligned_return,
        volatility,
        mean_range,
        recent_volume / np.maximum(prior_volume, 1e-12),
    )


def features_from_observations(
    frame: pd.DataFrame, observations: pd.DataFrame
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS_V1, index=observations.index)
    work = observations.copy()
    interval_minutes = int(str(work["timeframe"].iloc[0]).removesuffix("m"))
    entry_indices = work["entry_index"].to_numpy(dtype=np.int64)
    signal_indices = work["signal_index"].to_numpy(dtype=np.int64)
    direction_sign = work["direction_sign"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    open_values = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    output = pd.DataFrame(index=work.index)
    for column in (*FEATURE_COLUMNS, *STATE_FEATURES):
        output[column] = pd.to_numeric(work[column], errors="coerce")
    output["log_entry_age_minutes"] = np.log1p(np.maximum(output["entry_age_minutes"], 0.0))
    output["log_entry_distance_pct"] = np.log1p(
        np.maximum(output["entry_distance_from_target_pct"], 0.0)
    )
    output["log_peak_distance_pct"] = np.log1p(
        np.maximum(output["peak_distance_from_target_pct"], 0.0)
    )
    output["log_departure_to_entry_bars"] = np.log1p(
        np.maximum(output["departure_to_entry_bars"], 0.0)
    )
    entry_close = _safe_index(close, entry_indices)
    output["entry_bar_range_pct"] = (
        (_safe_index(high, entry_indices) - _safe_index(low, entry_indices))
        / np.maximum(entry_close, 1e-12)
        * 100.0
    )
    output["entry_bar_body_pct"] = (
        np.abs(entry_close - _safe_index(open_values, entry_indices))
        / np.maximum(entry_close, 1e-12)
        * 100.0
    )
    output["signal_to_entry_aligned_return_pct"] = (
        direction_sign
        * (entry_close / _safe_index(close, signal_indices) - 1.0)
        * 100.0
    )
    for prefix, end_indices, windows in (
        ("pre", signal_indices - 1, PRE_SIGNAL_WINDOWS_MINUTES),
        ("recent", entry_indices, RECENT_WINDOWS_MINUTES),
    ):
        for minutes in windows:
            values = _window_features(
                frame,
                end_indices,
                direction_sign,
                max(1, minutes // interval_minutes),
            )
            for suffix, data in zip(
                ("aligned_return_pct", "realized_vol_pct", "mean_range_pct", "volume_ratio"),
                values,
                strict=True,
            ):
                output[f"{prefix}_{minutes}m_{suffix}"] = data
    assets = work["asset"].astype(str)
    for asset in SUPPORTED_ASSETS:
        output[f"asset_{asset}"] = assets.eq(asset).astype(float)
    output["direction_upper_wick"] = work["direction"].astype(str).eq("upper_wick").astype(float)
    return (
        output.loc[:, FEATURE_COLUMNS_V1]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )


def live_observation(
    frame: pd.DataFrame,
    signal: Mapping[str, Any],
    asset: str,
    timeframe: str,
    current_index: int,
    departure_index: int,
    current_move_pct: float,
    peak_move_pct: float,
    drawdown_from_peak_pct: float,
) -> tuple[pd.Series, pd.DataFrame]:
    signal_index = int(signal["bar_index"])
    interval_minutes = int(timeframe.removesuffix("m"))
    pre_signal_bars = max(1, (8 * 60) // interval_minutes)
    recent_bars = max(1, (256 * 5) // interval_minutes)
    row: dict[str, Any] = {
        "observation_id": "live",
        "asset": asset,
        "timeframe": timeframe,
        "direction": str(signal["direction"]),
        "direction_sign": int(signal["direction_sign"]),
        "signal_index": signal_index,
        "entry_index": current_index,
        "entry_close_time_ms": int(frame["close_time"].iat[current_index]),
        "entry_age_minutes": (current_index - signal_index) * interval_minutes,
        "entry_age_bars": current_index - signal_index,
        "entry_distance_from_target_pct": float(current_move_pct),
        "peak_distance_from_target_pct": float(peak_move_pct),
        "drawdown_from_peak_pct": float(drawdown_from_peak_pct),
        "departure_to_entry_bars": current_index - int(departure_index),
        # These observable-only coordinates let the frozen 5m architecture
        # construct the same pre-signal/recent/episode sequence fingerprint
        # used during chronological research.
        "wick_target": float(signal["wick_target"]),
        "pre_signal_start_index": max(0, signal_index - pre_signal_bars),
        "pre_signal_end_index": signal_index - 1,
        "recent_start_index": max(0, current_index - recent_bars + 1),
        "recent_end_index": current_index,
        "signal_to_entry_start_index": signal_index,
        "signal_to_entry_end_index": current_index,
    }
    for column in FEATURE_COLUMNS:
        row[column] = float(signal[column])
    observations = pd.DataFrame([row])
    return observations.iloc[0], features_from_observations(frame, observations)


def _classifier_probabilities(model: Any, matrix: np.ndarray) -> list[dict[int, float]]:
    raw = model.predict_proba(matrix)
    if not isinstance(raw, list):
        raw = [raw]
    return [
        {
            int(label): float(probability)
            for label, probability in zip(classes, values[0], strict=True)
        }
        for classes, values in zip(model.classes_, raw, strict=True)
    ]


def predict_bundle(bundle: Mapping[str, Any], features: pd.DataFrame) -> dict[str, Any]:
    matrix = features.loc[:, tuple(bundle["feature_columns"])].to_numpy(dtype=np.float32)
    encoded = dict(
        zip(
            bundle["classifier_output_names"],
            _classifier_probabilities(bundle["classifier"], matrix),
            strict=True,
        )
    )
    fill: dict[str, float] = {}
    outcomes: dict[str, dict[str, float]] = {}
    for horizon in bundle["horizons_minutes"]:
        slug = horizon_slug(int(horizon))
        fill[slug] = encoded[f"fill_{slug}"].get(1, 0.0)
        for threshold in bundle["thresholds_pct"]:
            key = threshold_slug(float(threshold))
            values = encoded[f"outcome_{slug}_vs_{key}pct"]
            outcomes[f"{slug}_vs_{key}pct"] = {
                label: values.get(code, 0.0) for label, code in OUTCOME_TO_CODE.items()
            }
    running_fill = 0.0
    for horizon in bundle["horizons_minutes"]:
        slug = horizon_slug(int(horizon))
        running_fill = max(running_fill, fill[slug])
        fill[slug] = min(1.0, running_fill)
    risk: dict[str, dict[str, float]] = {}
    time: dict[str, dict[str, float]] = {}
    for index, horizon in enumerate(bundle["horizons_minutes"]):
        slug = horizon_slug(int(horizon))
        point = max(0.0, float(bundle["risk_models"][index].predict(matrix)[0]))
        adjustment = bundle.get("risk_upper_adjustments", {}).get(slug, {})
        risk[slug] = {
            "p50": point,
            "p80": point + max(0.0, float(adjustment.get("p80", 0.0))),
            "p90": point + max(0.0, float(adjustment.get("p90", 0.0))),
        }
        point_log = float(bundle["time_models"][index].predict(matrix)[0])
        residuals = bundle.get("time_log_residual_quantiles", {}).get(
            slug, {"p10": 0.0, "p50": 0.0, "p90": 0.0}
        )
        raw_time = sorted(
            max(0.0, math.expm1(point_log + float(residuals[key])))
            for key in ("p10", "p50", "p90")
        )
        time[slug] = {
            key: min(float(horizon), value)
            for key, value in zip(("p10", "p50", "p90"), raw_time, strict=True)
        }
    running_risk = {"p50": 0.0, "p80": 0.0, "p90": 0.0}
    for horizon in bundle["horizons_minutes"]:
        slug = horizon_slug(int(horizon))
        for key in running_risk:
            running_risk[key] = max(running_risk[key], risk[slug][key])
            risk[slug][key] = running_risk[key]
    return {
        "fill_probability": fill,
        "additional_adverse_pct": risk,
        "remaining_time_minutes_if_filled_within_horizon": time,
        "competing_outcomes": outcomes,
    }


def save_bundle(bundle: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(dict(bundle), temporary, compress=3)
    temporary.replace(path)


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported prospective-entry artifact schema: {bundle.get('schema_version')}")
    return bundle
