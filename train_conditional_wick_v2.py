"""Train and calibrate a leakage-safe V2 conditional wick-risk prototype.

V2 is a risk layer for the existing V1 real-historical-trajectory selector.  It
does not alter V1, estimate an unconditional fill probability, or claim a
trading edge.  Its population is deliberately restricted to 5-minute ETHUSDT
and BTCUSDT strict-wick episodes that eventually made the library's clean fill.

For each observable post-signal snapshot, the prototype builds only state that
would have been known when that candle closed, then estimates conditional
quantiles for remaining bars to fill and maximum away-from-wick movement through
the eventual fill.  The latter includes the snapshot candle, matching V1's
existing ``future_peak_move_pct`` replay definition.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


QUANTILES = (0.10, 0.50, 0.90)
STATIC_FEATURE_COLUMNS = (
    "body_pct_of_range",
    "dominant_wick_pct_of_range",
    "opposite_wick_pct_of_range",
    "range_pct_of_close",
    "range_vs_prior_20_median",
    "volume_vs_prior_20_mean",
    "aligned_prior_1h_return_pct",
)
SNAPSHOT_FEATURE_COLUMNS = (
    "elapsed_log_bars",
    "current_move_pct",
    "peak_move_pct",
    "drawdown_from_peak_pct",
    "current_bar_range_pct",
    "current_bar_body_pct",
    "recent_bar_range_mean_3_pct",
    "recent_abs_close_change_mean_3_pct",
    "log_volume_ratio_to_signal",
    *STATIC_FEATURE_COLUMNS,
)
CATEGORICAL_FEATURE_COLUMNS = ("asset_is_btc", "direction_is_lower")
DEFAULT_SNAPSHOT_OFFSETS = "1,3,6,12,24,60,120,240,480,960"
OBSERVABLE_STATE_INPUT_COLUMNS = (
    "elapsed_bars",
    "current_move_pct",
    "peak_move_pct",
    "drawdown_from_peak_pct",
    "current_bar_range_pct",
    "current_bar_body_pct",
    "recent_bar_range_mean_3_pct",
    "recent_abs_close_change_mean_3_pct",
    "volume_ratio_to_signal",
    *STATIC_FEATURE_COLUMNS,
)


def utc_iso(timestamp_ms: int | float) -> str:
    return datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=UTC).isoformat().replace("+00:00", "Z")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp") as handle:
        frame.to_csv(handle, index=False)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_pickle(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent, suffix=".tmp") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def parse_offsets(value: str) -> list[int]:
    offsets = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not offsets or any(offset <= 0 for offset in offsets):
        raise ValueError("--snapshot-offsets must contain positive integers")
    return offsets


def observable_state_schema() -> dict[str, Any]:
    """Small contract for a later display/client that has already observed a 5m snapshot."""
    descriptions = {
        "elapsed_bars": "completed 5m candles after the signal candle",
        "current_move_pct": "direction-normalized close move away from the wick target",
        "peak_move_pct": "maximum direction-normalized high/low move through the snapshot candle",
        "drawdown_from_peak_pct": "max(0, peak_move_pct - current_move_pct)",
        "current_bar_range_pct": "absolute normalized high-low range of the snapshot candle",
        "current_bar_body_pct": "absolute normalized close-open body of the snapshot candle",
        "recent_bar_range_mean_3_pct": "mean normalized range across the latest up-to-three observed candles",
        "recent_abs_close_change_mean_3_pct": "mean absolute normalized close change across the latest up-to-three observed candles",
        "volume_ratio_to_signal": "non-negative snapshot-candle volume divided by signal-candle volume",
    }
    return {
        "schema_version": "2.0.0-prototype",
        "input_shape": "a JSON object or {\"state\": object}",
        "required_categorical_fields": {
            "asset": ["ETHUSDT", "BTCUSDT"],
            "direction": ["upper_wick", "lower_wick"],
        },
        "required_numeric_fields": [
            {"name": name, "description": descriptions.get(name, "observable signal-candle feature from episodes.csv")}
            for name in OBSERVABLE_STATE_INPUT_COLUMNS
        ],
        "derived_inside_v2": {
            "elapsed_log_bars": "log1p(elapsed_bars)",
            "log_volume_ratio_to_signal": "log1p(volume_ratio_to_signal)",
            "asset_is_btc": "asset == BTCUSDT",
            "direction_is_lower": "direction == lower_wick",
        },
        "output": {
            "remaining_bars": ["p10", "p50", "p90"],
            "future_max_away_pct": ["p10", "p50", "p90"],
        },
        "population_warning": "Only clean eventual-fill episodes; never interpret this as unconditional fill probability.",
    }


def evenly_spaced(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    """Deterministically cap a chronologically sorted frame without random rows."""
    if maximum < 1:
        raise ValueError("row cap must be positive")
    if len(frame) <= maximum:
        return frame.copy()
    positions = np.linspace(0, len(frame) - 1, num=maximum, dtype=int)
    return frame.iloc[np.unique(positions)].copy()


def load_events(library_dir: Path) -> pd.DataFrame:
    events_path = library_dir / "episodes.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"Missing episode library: {events_path}")
    events = pd.read_csv(events_path)
    required = {
        "episode_id",
        "path_file",
        "asset",
        "timeframe",
        "direction",
        "direction_sign",
        "signal_open_time_ms",
        "fill_open_time_utc",
        "interval_minutes",
        "signal_to_fill_bars",
        "signal_volume",
        *STATIC_FEATURE_COLUMNS,
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise RuntimeError(f"episodes.csv is missing required V2 columns: {', '.join(missing)}")
    events = events.loc[
        events["timeframe"].eq("5m") & events["asset"].isin(("ETHUSDT", "BTCUSDT"))
    ].copy()
    if events.empty:
        raise RuntimeError("No 5m ETHUSDT/BTCUSDT completed clean-fill episodes are available")
    events["signal_open_time_ms"] = pd.to_numeric(events["signal_open_time_ms"], errors="raise").astype("int64")
    events["interval_minutes"] = pd.to_numeric(events["interval_minutes"], errors="raise").astype("int64")
    events["signal_to_fill_bars"] = pd.to_numeric(events["signal_to_fill_bars"], errors="raise").astype("int64")
    events["direction_sign"] = pd.to_numeric(events["direction_sign"], errors="raise").astype("int64")
    events["fill_close_time_ms"] = (
        pd.to_datetime(events["fill_open_time_utc"], utc=True).map(lambda value: int(value.timestamp() * 1000))
        + events["interval_minutes"] * 60_000
    ).astype("int64")
    numeric = ["signal_volume", *STATIC_FEATURE_COLUMNS]
    events[numeric] = events[numeric].apply(pd.to_numeric, errors="coerce")
    events = events.loc[(events["signal_to_fill_bars"] > 1) & (events["fill_close_time_ms"] > events["signal_open_time_ms"])].copy()
    if events.empty:
        raise RuntimeError("No valid completed 5m clean-fill episodes remain after integrity checks")
    return events.sort_values(["signal_open_time_ms", "episode_id"], kind="stable").reset_index(drop=True)


def load_paths(library_dir: Path, events: pd.DataFrame) -> pd.DataFrame:
    """Read the two referenced 5m path files, omitting fields not observable at a snapshot."""
    fields = [
        "episode_id",
        "offset_bars",
        "volume",
        "normalized_open_pct",
        "normalized_high_pct",
        "normalized_low_pct",
        "normalized_close_pct",
    ]
    episode_ids = set(events["episode_id"].astype(str))
    frames: list[pd.DataFrame] = []
    for path_file in sorted(events["path_file"].drop_duplicates()):
        path = library_dir / "paths" / str(path_file)
        if not path.exists():
            raise FileNotFoundError(f"Missing trajectory file referenced by episodes.csv: {path}")
        frame = pd.read_csv(path, compression="gzip", usecols=fields)
        frame = frame.loc[frame["episode_id"].astype(str).isin(episode_ids)].copy()
        frames.append(frame)
    if not frames:
        raise RuntimeError("No 5m trajectory files were available")
    paths = pd.concat(frames, ignore_index=True)
    numeric = [column for column in fields if column != "episode_id"]
    paths[numeric] = paths[numeric].apply(pd.to_numeric, errors="coerce")
    return paths


def build_snapshot_dataset(events: pd.DataFrame, paths: pd.DataFrame, offsets: list[int]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Build labels only after calculating each feature from candles at or before its snapshot."""
    event_fields = [
        "episode_id",
        "asset",
        "direction",
        "direction_sign",
        "signal_open_time_ms",
        "fill_close_time_ms",
        "interval_minutes",
        "signal_to_fill_bars",
        "signal_volume",
        *STATIC_FEATURE_COLUMNS,
    ]
    work = paths.merge(events[event_fields], on="episode_id", how="inner", validate="many_to_one")
    work = work.loc[
        work["offset_bars"].ge(0) & work["offset_bars"].le(work["signal_to_fill_bars"])
    ].copy()
    work = work.sort_values(["episode_id", "offset_bars"], kind="stable").reset_index(drop=True)
    if work.empty:
        raise RuntimeError("No trajectory rows remain through the recorded fill candles")

    directional_peak_at_bar = np.where(
        work["direction_sign"].to_numpy(dtype=int) == 1,
        work["normalized_high_pct"].to_numpy(dtype=float),
        work["normalized_low_pct"].to_numpy(dtype=float),
    )
    work["directional_peak_at_bar"] = directional_peak_at_bar
    groups = work.groupby("episode_id", sort=False)
    work["peak_move_pct"] = groups["directional_peak_at_bar"].cummax()
    work["current_move_pct"] = work["normalized_close_pct"].astype(float)
    work["drawdown_from_peak_pct"] = np.maximum(0.0, work["peak_move_pct"] - work["current_move_pct"])
    work["current_bar_range_pct"] = np.abs(
        work["normalized_high_pct"].astype(float) - work["normalized_low_pct"].astype(float)
    )
    work["current_bar_body_pct"] = np.abs(
        work["normalized_close_pct"].astype(float) - work["normalized_open_pct"].astype(float)
    )
    close_change = groups["normalized_close_pct"].diff().fillna(0.0).astype(float)
    work["recent_abs_close_change_mean_3_pct"] = close_change.groupby(work["episode_id"], sort=False).transform(
        lambda values: values.abs().rolling(window=3, min_periods=1).mean()
    )
    work["recent_bar_range_mean_3_pct"] = groups["current_bar_range_pct"].transform(
        lambda values: values.rolling(window=3, min_periods=1).mean()
    )
    safe_signal_volume = np.maximum(work["signal_volume"].to_numpy(dtype=float), 1e-12)
    safe_volume = np.maximum(work["volume"].to_numpy(dtype=float), 0.0)
    work["log_volume_ratio_to_signal"] = np.log1p(safe_volume / safe_signal_volume)
    work["elapsed_log_bars"] = np.log1p(work["offset_bars"].to_numpy(dtype=float))
    work["remaining_bars"] = work["signal_to_fill_bars"] - work["offset_bars"]
    work["future_max_away_pct"] = groups["directional_peak_at_bar"].transform(
        lambda values: values.iloc[::-1].cummax().iloc[::-1]
    )
    work["snapshot_open_time_ms"] = (
        work["signal_open_time_ms"] + work["offset_bars"] * work["interval_minutes"] * 60_000
    ).astype("int64")
    work["snapshot_close_time_ms"] = (
        work["snapshot_open_time_ms"] + work["interval_minutes"] * 60_000
    ).astype("int64")
    work["asset_is_btc"] = work["asset"].eq("BTCUSDT").astype(float)
    work["direction_is_lower"] = work["direction"].eq("lower_wick").astype(float)

    selected = work.loc[
        work["offset_bars"].isin(offsets)
        & work["remaining_bars"].gt(0)
        & work["current_move_pct"].gt(0)
    ].copy()
    required_numeric = [
        *SNAPSHOT_FEATURE_COLUMNS,
        *CATEGORICAL_FEATURE_COLUMNS,
        "remaining_bars",
        "future_max_away_pct",
    ]
    finite = np.isfinite(selected[required_numeric].to_numpy(dtype=float)).all(axis=1)
    selected = selected.loc[finite].copy()
    if selected.empty:
        raise RuntimeError("No finite, away-from-wick snapshots were available at the selected offsets")
    selected = selected.sort_values(["snapshot_close_time_ms", "episode_id", "offset_bars"], kind="stable").reset_index(drop=True)
    return selected, {
        "path_rows_through_fill": int(len(work)),
        "snapshot_rows_before_finite_filter": int((work["offset_bars"].isin(offsets) & work["remaining_bars"].gt(0) & work["current_move_pct"].gt(0)).sum()),
        "snapshot_rows_after_finite_filter": int(len(selected)),
    }


def chronological_split(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    holdout_months: int,
    embargo_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Create disjoint episode sets and require every training label to be resolved before holdout."""
    last_signal = pd.Timestamp(int(events["signal_open_time_ms"].max()), unit="ms", tz="UTC")
    holdout_start = last_signal - pd.DateOffset(months=holdout_months)
    holdout_start_ms = int(holdout_start.timestamp() * 1000)
    train_resolution_cutoff_ms = holdout_start_ms - embargo_bars * 5 * 60_000

    train = snapshots.loc[
        snapshots["signal_open_time_ms"].lt(holdout_start_ms)
        & snapshots["fill_close_time_ms"].le(train_resolution_cutoff_ms)
    ].copy()
    holdout = snapshots.loc[snapshots["signal_open_time_ms"].ge(holdout_start_ms)].copy()
    if train.empty or holdout.empty:
        raise RuntimeError("Chronological split produced an empty train or holdout population")
    train_episode_ids = set(train["episode_id"].astype(str))
    holdout_episode_ids = set(holdout["episode_id"].astype(str))
    overlap = train_episode_ids.intersection(holdout_episode_ids)
    if overlap:
        raise RuntimeError("Episode overlap across the chronological split would leak resolved outcomes")
    min_holdout_snapshot_close_ms = int(holdout["snapshot_close_time_ms"].min())
    max_train_fill_close_ms = int(train["fill_close_time_ms"].max())
    if max_train_fill_close_ms > min_holdout_snapshot_close_ms:
        raise RuntimeError("Resolved-before-snapshot guard failed before model fitting")
    return train, holdout, {
        "holdout_months": holdout_months,
        "embargo_bars": embargo_bars,
        "embargo_minutes": embargo_bars * 5,
        "holdout_start_utc": utc_iso(holdout_start_ms),
        "train_resolution_cutoff_utc": utc_iso(train_resolution_cutoff_ms),
        "max_train_fill_close_utc": utc_iso(max_train_fill_close_ms),
        "min_holdout_snapshot_close_utc": utc_iso(min_holdout_snapshot_close_ms),
        "minimum_resolved_before_snapshot_gap_minutes": round(
            (min_holdout_snapshot_close_ms - max_train_fill_close_ms) / 60_000.0, 2
        ),
        "train_episode_count": len(train_episode_ids),
        "holdout_episode_count": len(holdout_episode_ids),
        "same_episode_overlap_count": len(overlap),
        "all_train_labels_resolved_before_every_holdout_snapshot": True,
    }


def robust_feature_matrix(frame: pd.DataFrame, centers: np.ndarray | None = None, scales: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    numeric = frame.loc[:, SNAPSHOT_FEATURE_COLUMNS].to_numpy(dtype=float)
    if centers is None or scales is None:
        centers = np.nanmedian(numeric, axis=0)
        q25 = np.nanquantile(numeric, 0.25, axis=0)
        q75 = np.nanquantile(numeric, 0.75, axis=0)
        scales = q75 - q25
        fallback = np.nanstd(numeric, axis=0)
        scales = np.where(scales > 1e-12, scales, np.where(fallback > 1e-12, fallback, 1.0))
    transformed_numeric = np.clip((numeric - centers) / scales, -12.0, 12.0)
    categories = frame.loc[:, CATEGORICAL_FEATURE_COLUMNS].to_numpy(dtype=float)
    # These fixed penalties preserve asset/direction conditioning without forbidding cross-asset analogues outright.
    transformed_categories = categories * np.asarray((1.75, 1.25), dtype=float)
    return np.ascontiguousarray(np.hstack((transformed_numeric, transformed_categories)), dtype=np.float32), centers, scales


def fit_local_quantile_model(train: pd.DataFrame) -> dict[str, Any]:
    matrix, centers, scales = robust_feature_matrix(train)
    return {
        "schema_version": "2.0.0-prototype",
        "estimator": "robust-scaled weighted nearest-neighbour empirical conditional quantiles",
        "numeric_feature_columns": list(SNAPSHOT_FEATURE_COLUMNS),
        "categorical_feature_columns": list(CATEGORICAL_FEATURE_COLUMNS),
        "feature_centers": centers.astype(float),
        "feature_scales": scales.astype(float),
        "x_train": matrix,
        "y_remaining_log1p": np.log1p(train["remaining_bars"].to_numpy(dtype=float)),
        "y_future_max_away_pct": train["future_max_away_pct"].to_numpy(dtype=float),
        "quantiles": list(QUANTILES),
        "training_row_count": int(len(train)),
    }


def weighted_quantiles(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0:
        return np.quantile(ordered_values, quantiles)
    return np.interp(np.asarray(quantiles, dtype=float), cumulative / total, ordered_values)


def predict_quantiles(model: dict[str, Any], frame: pd.DataFrame, neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    x_test, _, _ = robust_feature_matrix(
        frame,
        np.asarray(model["feature_centers"], dtype=float),
        np.asarray(model["feature_scales"], dtype=float),
    )
    x_train = np.asarray(model["x_train"], dtype=np.float32)
    y_duration = np.asarray(model["y_remaining_log1p"], dtype=float)
    y_excursion = np.asarray(model["y_future_max_away_pct"], dtype=float)
    count = min(neighbors, len(x_train))
    if count < 12:
        raise RuntimeError("Fewer than 12 resolved training snapshots are available for conditional quantiles")
    duration = np.empty((len(x_test), len(QUANTILES)), dtype=float)
    excursion = np.empty_like(duration)
    for index, row in enumerate(x_test):
        delta = x_train - row
        distance = np.sqrt(np.einsum("ij,ij->i", delta, delta, optimize=True))
        nearest = np.argpartition(distance, count - 1)[:count]
        local_distance = distance[nearest]
        local_scale = max(float(np.median(local_distance)), 0.05)
        weights = np.exp(-np.minimum(local_distance / local_scale, 50.0))
        duration[index] = np.expm1(weighted_quantiles(y_duration[nearest], weights, QUANTILES))
        excursion[index] = weighted_quantiles(y_excursion[nearest], weights, QUANTILES)
    return np.maximum.accumulate(duration, axis=1), np.maximum.accumulate(excursion, axis=1)


def observable_state_frame(state: dict[str, Any]) -> pd.DataFrame:
    """Validate a live/display state and derive the transformed V2 feature fields."""
    missing = [name for name in ("asset", "direction", *OBSERVABLE_STATE_INPUT_COLUMNS) if name not in state]
    if missing:
        raise ValueError(f"Observable state is missing required fields: {', '.join(missing)}")
    asset = str(state["asset"])
    direction = str(state["direction"])
    if asset not in {"ETHUSDT", "BTCUSDT"}:
        raise ValueError("asset must be ETHUSDT or BTCUSDT")
    if direction not in {"upper_wick", "lower_wick"}:
        raise ValueError("direction must be upper_wick or lower_wick")
    values: dict[str, float] = {}
    for name in OBSERVABLE_STATE_INPUT_COLUMNS:
        try:
            values[name] = float(state[name])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Observable state field {name} must be numeric") from error
        if not np.isfinite(values[name]):
            raise ValueError(f"Observable state field {name} must be finite")
    if values["elapsed_bars"] < 0:
        raise ValueError("elapsed_bars must be non-negative")
    if values["current_move_pct"] <= 0:
        raise ValueError("current_move_pct must be positive while the wick is still away from fill")
    if values["drawdown_from_peak_pct"] < 0 or values["volume_ratio_to_signal"] < 0:
        raise ValueError("drawdown_from_peak_pct and volume_ratio_to_signal must be non-negative")
    row = {
        "elapsed_log_bars": float(np.log1p(values.pop("elapsed_bars"))),
        "log_volume_ratio_to_signal": float(np.log1p(values.pop("volume_ratio_to_signal"))),
        "asset_is_btc": float(asset == "BTCUSDT"),
        "direction_is_lower": float(direction == "lower_wick"),
        **values,
    }
    return pd.DataFrame([row])


def load_model_bundle(path: Path) -> dict[str, Any]:
    """Load a V2 bundle created by this script without importing any server code."""
    with path.open("rb") as handle:
        model = pickle.load(handle)
    required = {"schema_version", "x_train", "y_remaining_log1p", "y_future_max_away_pct", "feature_centers", "feature_scales"}
    missing = sorted(required.difference(model)) if isinstance(model, dict) else ["model dictionary"]
    if missing:
        raise ValueError(f"Invalid V2 model bundle: missing {', '.join(missing)}")
    if model["schema_version"] != "2.0.0-prototype":
        raise ValueError(f"Unsupported V2 model schema: {model['schema_version']}")
    return model


def predict_from_observable_state(
    model: dict[str, Any], state: dict[str, Any], neighbors: int | None = None
) -> dict[str, Any]:
    """Return V2 conditional risk quantiles from an already-observed snapshot state."""
    row = observable_state_frame(state)
    requested_neighbors = int(neighbors if neighbors is not None else model.get("default_neighbors", 192))
    if requested_neighbors < 12:
        raise ValueError("neighbors must be at least 12")
    remaining, away = predict_quantiles(model, row, requested_neighbors)
    labels = ("p10", "p50", "p90")
    return {
        "schema_version": "2.0.0-prototype",
        "conditional_population": model.get("population", "clean eventual-fill episodes only"),
        "neighbors": min(requested_neighbors, int(len(model["x_train"]))),
        "quantiles": {
            "remaining_bars": {label: round(float(remaining[0, index]), 6) for index, label in enumerate(labels)},
            "future_max_away_pct": {label: round(float(away[0, index]), 6) for index, label in enumerate(labels)},
        },
        "target_semantics": "future_max_away_pct includes the observed snapshot candle, matching V1 replay semantics",
        "warning": "Conditional clean-fill risk quantiles, not unconditional fill probability or predictive-edge evidence.",
    }


def observable_state_payload_from_snapshot(row: pd.Series) -> dict[str, Any]:
    """Create a valid contract example from one already-observed chronological holdout snapshot."""
    state: dict[str, Any] = {
        "asset": str(row["asset"]),
        "direction": str(row["direction"]),
        "elapsed_bars": int(row["offset_bars"]),
        "volume_ratio_to_signal": float(np.expm1(float(row["log_volume_ratio_to_signal"]))),
    }
    for name in OBSERVABLE_STATE_INPUT_COLUMNS:
        if name not in state:
            state[name] = float(row[name])
    return {
        "schema_version": "2.0.0-prototype",
        "note": "Example observable snapshot contract from a chronological holdout row. It contains no target labels.",
        "state": state,
    }


def pinball_loss(actual: np.ndarray, prediction: np.ndarray, quantile: float) -> float:
    residual = actual - prediction
    return float(np.mean(np.maximum(quantile * residual, (quantile - 1.0) * residual)))


def target_calibration(actual: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
    coverage = []
    losses = []
    for index, quantile in enumerate(QUANTILES):
        observed = float(np.mean(actual <= predictions[:, index]))
        coverage.append(
            {
                "quantile": quantile,
                "nominal_coverage": quantile,
                "observed_coverage": round(observed, 6),
                "coverage_error": round(observed - quantile, 6),
            }
        )
        losses.append(round(pinball_loss(actual, predictions[:, index], quantile), 6))
    interval_coverage = float(np.mean((actual >= predictions[:, 0]) & (actual <= predictions[:, -1])))
    return {
        "case_count": int(len(actual)),
        "coverage_by_quantile": coverage,
        "p10_to_p90_interval_nominal_coverage": 0.8,
        "p10_to_p90_interval_observed_coverage": round(interval_coverage, 6),
        "p10_to_p90_interval_coverage_error": round(interval_coverage - 0.8, 6),
        "median_absolute_error_at_p50": round(float(np.median(np.abs(actual - predictions[:, 1]))), 6),
        "mean_pinball_loss_by_quantile": losses,
    }


def grouped_calibration(predictions: pd.DataFrame, group_column: str) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for value, group in predictions.groupby(group_column, sort=True):
        duration = group[["remaining_bars_p10", "remaining_bars_p50", "remaining_bars_p90"]].to_numpy(dtype=float)
        excursion = group[["future_max_away_pct_p10", "future_max_away_pct_p50", "future_max_away_pct_p90"]].to_numpy(dtype=float)
        output[str(value)] = {
            "remaining_bars": target_calibration(group["actual_remaining_bars"].to_numpy(dtype=float), duration),
            "future_max_away_pct": target_calibration(group["actual_future_max_away_pct"].to_numpy(dtype=float), excursion),
        }
    return output


def compact_metric(metric: dict[str, Any]) -> str:
    return (
        f"{metric['p10_to_p90_interval_observed_coverage']:.1%} interval coverage "
        f"(nominal 80%); p50 median absolute error {metric['median_absolute_error_at_p50']:.4g}"
    )


def markdown_report(summary: dict[str, Any]) -> str:
    calibration = summary["calibration"]
    split = summary["chronological_split"]
    data = summary["data"]
    future_p10 = calibration["future_max_away_pct"]["coverage_by_quantile"][0]
    return "\n".join(
        [
            "# V2 Conditional Wick Risk Calibration",
            "",
            "**Status:** diagnostic prototype only. V1 real historical candle trajectories remain unchanged.",
            "",
            "## Holdout result",
            "",
            f"- Cases: {calibration['remaining_bars']['case_count']} chronological holdout snapshots from {split['holdout_episode_count']} disjoint episodes.",
            f"- Remaining-bars: {compact_metric(calibration['remaining_bars'])}.",
            f"- Future max move-away: {compact_metric(calibration['future_max_away_pct'])}.",
            f"- Lower-tail check: future-move p10 covered {future_p10['observed_coverage']:.1%} versus {future_p10['nominal_coverage']:.0%} nominal; do not use that lower bound as a calibrated risk limit yet.",
            f"- Evaluation window starts {split['holdout_start_utc']}; deterministic caps used {data['train_rows_used']} train and {data['holdout_rows_used']} holdout snapshots.",
            "",
            "## Leakage guard",
            "",
            f"- Train labels had resolved by {split['max_train_fill_close_utc']}; the earliest scored holdout snapshot closed at {split['min_holdout_snapshot_close_utc']} ({split['minimum_resolved_before_snapshot_gap_minutes']:.1f} minutes later).",
            f"- Embargo: {split['embargo_bars']} five-minute bars; same-episode overlap: {split['same_episode_overlap_count']}.",
            "- The split is chronological by episode signal time; there is no random row split.",
            "",
            "## Scope and target",
            "",
            "- Population: only 5m ETHUSDT/BTCUSDT strict-wick episodes that eventually made the library's clean fill. This is **not** an unconditional fill probability.",
            "- Snapshot features use signal-time fields plus completed candles up to the snapshot close. Labels are remaining bars and maximum direction-normalized move away from the wick through the fill (including the observed snapshot candle, matching V1 replay semantics).",
            "",
            "## Limitations",
            "",
            "- Conditional calibration on historical clean fills is not predictive-edge evidence and excludes no-fill, invalidated, and censored signals.",
            "- The local empirical quantiles are a transparent baseline; calibration can drift by regime, asset, direction, and sparse long-duration states.",
            "- V2 has no live selector, UI, or position-sizing integration. Treat the JSON/CSV as a risk diagnostic beside V1 paths.",
            "",
            "## Later live-display interface",
            "",
            "- `python train_conditional_wick_v2.py --print-input-schema` prints the required observable-state JSON contract.",
            f"- `python train_conditional_wick_v2.py --predict-json {summary['artifact_paths']['example_observable_state_json']} --model-path {summary['artifact_paths']['model_pickle']}` emits one quantile response without retraining.",
            "- Python callers can use `load_model_bundle(path)` and `predict_from_observable_state(model, state)` from this script; no server wiring is included.",
            "",
            f"Machine-readable summary: `{summary['artifact_paths']['summary_json']}`  ",
            f"Holdout predictions: `{summary['artifact_paths']['holdout_predictions_csv']}`  ",
            f"Reusable local model bundle: `{summary['artifact_paths']['model_pickle']}`",
            "",
        ]
    )


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library-dir", type=Path, default=root / "data" / "conditional_path_library_5y")
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "conditional_path_library_5y" / "v2_models")
    parser.add_argument("--markdown-output", type=Path, default=root / "V2_CALIBRATION.md")
    parser.add_argument("--model-path", type=Path, help="Existing V2 pickle used with --predict-json")
    parser.add_argument("--holdout-months", type=int, default=12)
    parser.add_argument("--embargo-bars", type=int, default=288, help="5m bars separating resolved training labels from holdout signals")
    parser.add_argument("--snapshot-offsets", default=DEFAULT_SNAPSHOT_OFFSETS)
    parser.add_argument("--neighbors", type=int, help="Neighbour count; training default is 192 and prediction defaults to the bundle value")
    parser.add_argument("--max-train-rows", type=int, default=18_000)
    parser.add_argument("--max-holdout-rows", type=int, default=1_200)
    parser.add_argument("--smoke", action="store_true", help="Use smaller deterministic caps for a bounded local check")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--predict-json", type=Path, help="Observable state JSON object (or {\"state\": object}) to score from an existing bundle")
    mode.add_argument("--print-input-schema", action="store_true", help="Print the small observable-state input contract and exit")
    args = parser.parse_args()
    if args.print_input_schema:
        print(json.dumps(observable_state_schema(), indent=2, sort_keys=True))
        return
    if args.predict_json:
        if args.neighbors is not None and args.neighbors < 12:
            parser.error("neighbors must be at least 12")
        try:
            payload = json.loads(args.predict_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"could not read --predict-json: {error}")
        state = payload.get("state", payload) if isinstance(payload, dict) else None
        if not isinstance(state, dict):
            parser.error("--predict-json must contain a JSON object or {\"state\": object}")
        model_path = (args.model_path or (args.output_dir / "conditional_wick_v2_model.pkl")).resolve()
        if not model_path.exists():
            parser.error(f"V2 model bundle does not exist: {model_path}")
        try:
            response = predict_from_observable_state(load_model_bundle(model_path), state, args.neighbors)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        print(json.dumps(response, sort_keys=True, allow_nan=False))
        return
    neighbors = 192 if args.neighbors is None else args.neighbors
    if args.holdout_months < 1 or args.embargo_bars < 0 or neighbors < 12:
        parser.error("holdout months must be positive; embargo bars non-negative; neighbors at least 12")
    if args.max_train_rows < neighbors or args.max_holdout_rows < 1:
        parser.error("max train rows must be at least neighbors; max holdout rows must be positive")
    try:
        offsets = parse_offsets(args.snapshot_offsets)
    except ValueError as error:
        parser.error(str(error))
    if args.smoke:
        args.max_train_rows = min(args.max_train_rows, 4_000)
        args.max_holdout_rows = min(args.max_holdout_rows, 300)
        neighbors = min(neighbors, 96)

    library_dir = args.library_dir.resolve()
    output_dir = args.output_dir.resolve()
    markdown_output = args.markdown_output.resolve()
    events = load_events(library_dir)
    paths = load_paths(library_dir, events)
    snapshots, construction = build_snapshot_dataset(events, paths, offsets)
    train_all, holdout_all, split = chronological_split(snapshots, events, args.holdout_months, args.embargo_bars)
    train = evenly_spaced(train_all.sort_values(["snapshot_close_time_ms", "episode_id", "offset_bars"], kind="stable"), args.max_train_rows)
    holdout = evenly_spaced(holdout_all.sort_values(["snapshot_close_time_ms", "episode_id", "offset_bars"], kind="stable"), args.max_holdout_rows)
    if len(train) < neighbors:
        raise RuntimeError(f"Only {len(train)} leakage-safe training snapshots remain; need at least {neighbors}")
    model = fit_local_quantile_model(train)
    model["default_neighbors"] = int(neighbors)
    model["input_schema"] = observable_state_schema()
    remaining_quantiles, away_quantiles = predict_quantiles(model, holdout, neighbors)

    predictions = holdout.loc[
        :,
        [
            "episode_id",
            "asset",
            "direction",
            "offset_bars",
            "signal_open_time_ms",
            "snapshot_open_time_ms",
            "snapshot_close_time_ms",
            "actual_remaining_bars" if "actual_remaining_bars" in holdout.columns else "remaining_bars",
            "future_max_away_pct",
        ],
    ].copy()
    predictions = predictions.rename(
        columns={"offset_bars": "snapshot_offset_bars", "remaining_bars": "actual_remaining_bars", "future_max_away_pct": "actual_future_max_away_pct"}
    )
    for index, label in enumerate(("p10", "p50", "p90")):
        predictions[f"remaining_bars_{label}"] = remaining_quantiles[:, index]
        predictions[f"future_max_away_pct_{label}"] = away_quantiles[:, index]
    for column in ("signal_open_time_ms", "snapshot_open_time_ms", "snapshot_close_time_ms"):
        predictions[column.replace("_ms", "_utc")] = predictions[column].map(utc_iso)
    predictions = predictions.drop(columns=["signal_open_time_ms", "snapshot_open_time_ms", "snapshot_close_time_ms"])

    duration_predictions = predictions[["remaining_bars_p10", "remaining_bars_p50", "remaining_bars_p90"]].to_numpy(dtype=float)
    away_predictions = predictions[["future_max_away_pct_p10", "future_max_away_pct_p50", "future_max_away_pct_p90"]].to_numpy(dtype=float)
    calibration = {
        "remaining_bars": target_calibration(predictions["actual_remaining_bars"].to_numpy(dtype=float), duration_predictions),
        "future_max_away_pct": target_calibration(predictions["actual_future_max_away_pct"].to_numpy(dtype=float), away_predictions),
        "by_asset": grouped_calibration(predictions, "asset"),
        "by_direction": grouped_calibration(predictions, "direction"),
    }

    summary_path = output_dir / "conditional_wick_v2_summary.json"
    model_path = output_dir / "conditional_wick_v2_model.pkl"
    predictions_path = output_dir / "conditional_wick_v2_holdout_predictions.csv"
    example_state_path = output_dir / "conditional_wick_v2_example_observable_state.json"
    summary: dict[str, Any] = {
        "schema_version": "2.0.0-prototype",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "purpose": "Leakage-safe V2 conditional risk quantiles beside V1 historical trajectories; not a fill-probability or trading-edge claim.",
        "conditional_population": "Only completed 5m ETHUSDT/BTCUSDT strict-wick episodes that eventually made the library clean fill; no-fill, invalidated, and censored signals are out of population.",
        "estimator": model["estimator"],
        "targets": {
            "remaining_bars": "5m bars from the snapshot candle to the eventual clean fill candle",
            "future_max_away_pct": "maximum direction-normalized move away from the wick across the snapshot candle through the eventual fill, matching V1 replay semantics",
        },
        "quantiles": list(QUANTILES),
        "snapshot_feature_columns": list(SNAPSHOT_FEATURE_COLUMNS),
        "categorical_feature_columns": list(CATEGORICAL_FEATURE_COLUMNS),
        "live_prediction_interface": {
            "callable": "load_model_bundle(path) then predict_from_observable_state(model, state, neighbors=None)",
            "cli": "python train_conditional_wick_v2.py --predict-json state.json --model-path conditional_wick_v2_model.pkl",
            "input_schema": observable_state_schema(),
            "server_integration": "none; this is a standalone artifact contract for a later display layer",
        },
        "chronological_split": split,
        "data": {
            "asset_timeframe": "ETHUSDT/BTCUSDT 5m",
            "clean_fill_episode_count": int(len(events)),
            "signal_start_utc": utc_iso(int(events["signal_open_time_ms"].min())),
            "signal_end_utc": utc_iso(int(events["signal_open_time_ms"].max())),
            "snapshot_offsets_bars": offsets,
            **construction,
            "train_snapshot_rows_before_cap": int(len(train_all)),
            "holdout_snapshot_rows_before_cap": int(len(holdout_all)),
            "train_rows_used": int(len(train)),
            "holdout_rows_used": int(len(holdout)),
            "neighbors": int(neighbors),
            "smoke_mode": bool(args.smoke),
        },
        "calibration": calibration,
        "limitations": [
            "Conditional clean-fill calibration is not an unconditional fill probability and does not establish predictive edge.",
            "The estimator is intentionally local and historical; regime drift and sparse long-duration states can degrade calibration.",
            "V2 does not replace V1 selected real candle paths and is not wired into a UI, server, or sizing rule.",
        ],
        "artifact_paths": {
            "summary_json": str(summary_path),
            "model_pickle": str(model_path),
            "holdout_predictions_csv": str(predictions_path),
            "example_observable_state_json": str(example_state_path),
            "markdown_report": str(markdown_output),
        },
    }
    model["training_window"] = {
        "max_train_fill_close_utc": split["max_train_fill_close_utc"],
        "holdout_start_utc": split["holdout_start_utc"],
        "training_episode_count": split["train_episode_count"],
    }
    model["population"] = summary["conditional_population"]
    atomic_write_pickle(model_path, model)
    atomic_write_csv(predictions_path, predictions)
    atomic_write_json(example_state_path, observable_state_payload_from_snapshot(holdout.iloc[0]))
    atomic_write_json(summary_path, summary)
    atomic_write_text(markdown_output, markdown_report(summary))
    print(
        json.dumps(
            {
                "status": "ok",
                "summary_json": str(summary_path),
                "markdown_report": str(markdown_output),
                "holdout_cases": calibration["remaining_bars"]["case_count"],
                "remaining_p10_p90_coverage": calibration["remaining_bars"]["p10_to_p90_interval_observed_coverage"],
                "future_away_p10_p90_coverage": calibration["future_max_away_pct"]["p10_to_p90_interval_observed_coverage"],
                "resolved_before_snapshot_gap_minutes": split["minimum_resolved_before_snapshot_gap_minutes"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
