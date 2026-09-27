"""Optional live V3 learned-route selection for the projection dashboard.

The neural model never generates chart candles.  It predicts an outcome
signature and embedding from the observable prefix, then selects one completed
historical continuation from the frozen fit index. Missing or incompatible
artifacts are handled by the dashboard as a V1 fallback rather than a service
failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from conditional_wick_assets import SUPPORTED_ASSETS, asset_indicator_column
from train_neural_path_v3 import (
    PRE_SIGNAL_LENGTH,
    RECENT_LENGTH,
    STATIC_COLUMNS,
    NeuralPathModel,
    SequenceScaler,
    StaticScaler,
    apply_sequence_scaler,
    apply_static_scaler,
    dedupe_alignments,
    direction_normalized_pct,
    episode_window,
    raw_window,
    select_real_medoid,
    v1_feature_scales,
    v1_scalar_distance,
    weighted_quantiles,
)

DEFAULT_VALIDATED_LIVE_MAX_AGE_BARS = 120


@dataclass
class NeuralPathV3LiveBundle:
    model: NeuralPathModel
    scalers: dict[str, Any]
    index: dict[str, np.ndarray]
    feature_scales: dict[str, float]
    fit_forecast_space: np.ndarray
    forecast_center: np.ndarray
    forecast_scale: np.ndarray
    candidate_pool: int
    neighbors: int
    snapshot_offsets_bars: tuple[int, ...]
    live_min_age_bars: int
    live_max_age_bars: int
    timeframe: str
    regime_label: str
    selector: str
    bar_minutes: int
    pre_signal_source_bars: int
    recent_source_bars: int
    future_horizon_bars: tuple[int, ...]
    summary: dict[str, Any]


def _sequence_scaler(payload: dict[str, Any]) -> SequenceScaler:
    return SequenceScaler(
        mean=np.asarray(payload["mean"], dtype=np.float32),
        scale=np.asarray(payload["scale"], dtype=np.float32),
    )


def _static_scaler(payload: dict[str, Any]) -> StaticScaler:
    return StaticScaler(
        center=np.asarray(payload["center"], dtype=np.float32),
        scale=np.asarray(payload["scale"], dtype=np.float32),
    )


def _required_index(index: Any, name: str) -> np.ndarray:
    if name not in index.files:
        raise ValueError(f"V3 retrieval index is missing {name}")
    return np.asarray(index[name])


def load_live_bundle(
    model_path: Path, index_path: Path, summary_path: Path
) -> NeuralPathV3LiveBundle:
    """Load and validate one CPU inference bundle from local trusted artifacts."""
    if not model_path.exists() or not index_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            "V3 model, retrieval index, or summary artifact is missing"
        )
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    static_columns = tuple(checkpoint.get("static_columns", ()))
    if static_columns != STATIC_COLUMNS:
        raise ValueError("V3 artifact static feature schema does not match the runtime")
    future_horizon_bars = tuple(
        int(value) for value in checkpoint.get("future_horizon_bars", ())
    )
    direction_horizon_bars = tuple(
        int(value) for value in checkpoint.get("direction_horizon_bars", (6, 12, 48))
    )
    if not future_horizon_bars or not direction_horizon_bars:
        raise ValueError("V3 artifact horizon schema is empty")

    config = checkpoint["config"]
    model = NeuralPathModel(
        static_features=len(STATIC_COLUMNS),
        sequence_width=int(config["sequence_width"]),
        embedding_dim=int(config["embedding_dim"]),
        dropout=float(config["dropout"]),
        future_horizon_count=len(future_horizon_bars),
        direction_horizon_count=len(direction_horizon_bars),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with np.load(index_path, allow_pickle=False) as stored:
        index = {
            name: _required_index(stored, name)
            for name in (
                "embedding",
                "curve_log_ratio",
                "log_remaining",
                "log_excursion",
                "raw_static",
                "episode_id",
                "asset",
                "direction",
                "offset_bars",
                "snapshot_close_time_ms",
                "fill_close_time_ms",
            )
        }
    row_count = len(index["episode_id"])
    if row_count < 12 or any(len(values) != row_count for values in index.values()):
        raise ValueError("V3 retrieval index arrays do not share a valid row count")
    index["embedding"] = index["embedding"].astype(np.float32)
    index["curve_log_ratio"] = index["curve_log_ratio"].astype(np.float32)
    index["log_remaining"] = index["log_remaining"].astype(np.float32)
    index["log_excursion"] = index["log_excursion"].astype(np.float32)
    index["raw_static"] = index["raw_static"].astype(np.float32)

    fit_signature = np.column_stack(
        (
            index["log_remaining"],
            index["log_excursion"],
            index["curve_log_ratio"],
        )
    ).astype(np.float64)
    forecast_center = np.median(fit_signature, axis=0)
    forecast_scale = np.maximum(
        np.quantile(fit_signature, 0.75, axis=0)
        - np.quantile(fit_signature, 0.25, axis=0),
        0.05,
    )
    snapshot_offsets = checkpoint.get("snapshot_offsets_bars") or summary.get(
        "data", {}
    ).get("snapshot_offsets_bars", [])
    deployment_gate = summary.get("deployment_gate", {})
    selector = str(
        deployment_gate.get("selector", "forecast_hybrid_real_path")
    )
    if selector not in {"forecast_hybrid_real_path", "neural_real_path"}:
        raise ValueError(f"unsupported V3 live selector: {selector}")
    return NeuralPathV3LiveBundle(
        model=model,
        scalers=checkpoint["scalers"],
        index=index,
        feature_scales=v1_feature_scales(index["raw_static"]),
        fit_forecast_space=(fit_signature - forecast_center) / forecast_scale,
        forecast_center=forecast_center,
        forecast_scale=forecast_scale,
        candidate_pool=int(config["candidate_pool"]),
        neighbors=int(config["retrieval_neighbors"]),
        snapshot_offsets_bars=tuple(int(value) for value in snapshot_offsets),
        live_min_age_bars=int(deployment_gate.get("min_elapsed_bars", 1)),
        live_max_age_bars=int(
            deployment_gate.get("max_elapsed_bars", DEFAULT_VALIDATED_LIVE_MAX_AGE_BARS)
        ),
        timeframe=str(deployment_gate.get("timeframe", "5m")),
        regime_label=str(deployment_gate.get("regime_label", "default")),
        selector=selector,
        bar_minutes=int(checkpoint.get("bar_minutes", 5)),
        pre_signal_source_bars=int(
            checkpoint.get("pre_signal_source_bars", PRE_SIGNAL_LENGTH)
        ),
        recent_source_bars=int(
            checkpoint.get("recent_source_bars", RECENT_LENGTH)
        ),
        future_horizon_bars=future_horizon_bars,
        summary=summary,
    )


def _raw_candles(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        column: frame[column].to_numpy(dtype=np.float64)
        for column in ("open_time", "open", "high", "low", "close", "volume")
    }


def _live_episode_path(
    frame: pd.DataFrame,
    signal_index: int,
    current_index: int,
    target: float,
    direction_sign: int,
) -> pd.DataFrame:
    observed = frame.iloc[signal_index : current_index + 1]
    result = pd.DataFrame(
        {
            "offset_bars": np.arange(len(observed), dtype=np.int64),
            "volume": observed["volume"].to_numpy(dtype=np.float64),
        }
    )
    for field in ("open", "high", "low", "close"):
        result[f"normalized_{field}_pct"] = direction_normalized_pct(
            observed[field].to_numpy(dtype=np.float64), target, direction_sign
        )
    return result


def _raw_static_row(observable_state: dict[str, Any]) -> np.ndarray:
    values: dict[str, float] = {
        key: float(observable_state[key])
        for key in STATIC_COLUMNS
        if key in observable_state
    }
    values["elapsed_log_bars"] = float(
        np.log1p(max(float(observable_state["elapsed_bars"]), 0.0))
    )
    values["log_volume_ratio_to_signal"] = float(
        np.log1p(max(float(observable_state["volume_ratio_to_signal"]), 0.0))
    )
    asset = str(observable_state["asset"])
    for supported_asset in SUPPORTED_ASSETS:
        values[asset_indicator_column(supported_asset)] = float(
            asset == supported_asset
        )
    values["direction_is_lower"] = float(
        str(observable_state["direction"]) == "lower_wick"
    )
    missing = [column for column in STATIC_COLUMNS if column not in values]
    if missing:
        raise ValueError(f"V3 live state is missing: {', '.join(missing)}")
    result = np.asarray(
        [[values[column] for column in STATIC_COLUMNS]], dtype=np.float32
    )
    if not np.isfinite(result).all():
        raise ValueError("V3 live static features must all be finite")
    return result


def _candidate_indices(distance: np.ndarray, maximum: int) -> np.ndarray:
    count = min(maximum, len(distance))
    if count == len(distance):
        return np.arange(len(distance), dtype=np.int64)
    return np.argpartition(distance, count - 1)[:count]


def live_age_bounds(bundle: NeuralPathV3LiveBundle) -> tuple[int, int]:
    """Intersect artifact age support with the validated deployment gate."""
    artifact_minimum = (
        min(bundle.snapshot_offsets_bars)
        if bundle.snapshot_offsets_bars
        else bundle.live_min_age_bars
    )
    artifact_maximum = (
        max(bundle.snapshot_offsets_bars)
        if bundle.snapshot_offsets_bars
        else bundle.live_max_age_bars
    )
    return max(artifact_minimum, bundle.live_min_age_bars), min(
        artifact_maximum, bundle.live_max_age_bars
    )


@torch.no_grad()
def select_live_route(
    bundle: NeuralPathV3LiveBundle,
    frame: pd.DataFrame,
    signal: pd.Series,
    current_index: int,
    observable_state: dict[str, Any],
) -> dict[str, Any]:
    """Select a real historical continuation from one observable live prefix."""
    signal_index = int(signal["bar_index"])
    if current_index <= signal_index:
        raise ValueError("V3 needs at least one completed candle after the signal")
    elapsed_bars = current_index - signal_index
    artifact_min_age, live_max_age = live_age_bounds(bundle)
    if not artifact_min_age <= elapsed_bars <= live_max_age:
        raise ValueError(
            f"elapsed age {elapsed_bars} bars is outside V3 support "
            f"{artifact_min_age}-{live_max_age}"
        )

    target = float(signal["wick_target"])
    direction_sign = int(signal["direction_sign"])
    signal_ms = int(signal["open_time"])
    signal_volume = float(signal["volume"])
    current_ms = int(frame["open_time"].iat[current_index])
    raw = _raw_candles(frame)
    live_episode = _live_episode_path(
        frame, signal_index, current_index, target, direction_sign
    )
    pre = raw_window(
        raw,
        signal_ms - bundle.bar_minutes * 60_000,
        PRE_SIGNAL_LENGTH,
        target,
        direction_sign,
        signal_volume,
        signal_ms,
        source_length=bundle.pre_signal_source_bars,
    )[None, :, :]
    recent = raw_window(
        raw,
        current_ms,
        RECENT_LENGTH,
        target,
        direction_sign,
        signal_volume,
        signal_ms,
        source_length=bundle.recent_source_bars,
    )[None, :, :]
    episode = episode_window(live_episode, elapsed_bars, signal_volume)[None, :, :]
    raw_static = _raw_static_row(observable_state)

    pre = apply_sequence_scaler(pre, _sequence_scaler(bundle.scalers["pre_signal"]))
    recent = apply_sequence_scaler(recent, _sequence_scaler(bundle.scalers["recent"]))
    episode = apply_sequence_scaler(
        episode, _sequence_scaler(bundle.scalers["episode"])
    )
    static = apply_static_scaler(raw_static, _static_scaler(bundle.scalers["static"]))
    outputs = bundle.model(
        torch.from_numpy(pre).transpose(1, 2),
        torch.from_numpy(recent).transpose(1, 2),
        torch.from_numpy(episode).transpose(1, 2),
        torch.from_numpy(static),
    )
    if bundle.selector == "neural_real_path":
        query_embedding = outputs["embedding"].numpy()[0]
        neural_distance = np.maximum(
            0.0, 1.0 - bundle.index["embedding"] @ query_embedding
        )
        pool = _candidate_indices(neural_distance, bundle.candidate_pool)
        selection_distance = neural_distance[pool]
    else:
        query_risk = np.sort(outputs["risk"].numpy(), axis=2)[0]
        query_curve = outputs["curve"].numpy()[0]
        query_signature = np.concatenate(
            (query_risk[:, 1], query_curve), axis=0
        ).astype(np.float64)
        query_forecast_space = (
            query_signature - bundle.forecast_center
        ) / bundle.forecast_scale
        forecast_distance = np.mean(
            np.abs(bundle.fit_forecast_space - query_forecast_space[None, :]), axis=1
        )
        scalar_distance = v1_scalar_distance(
            bundle.index["raw_static"],
            raw_static[0],
            bundle.index["asset"].astype(str),
            str(observable_state["asset"]),
            bundle.index["direction"].astype(str),
            str(observable_state["direction"]),
            bundle.feature_scales,
        )
        scalar_pool = _candidate_indices(scalar_distance, bundle.candidate_pool)
        forecast_pool = _candidate_indices(forecast_distance, bundle.candidate_pool)
        pool = np.unique(np.concatenate((scalar_pool, forecast_pool)))
        scalar_part = scalar_distance[pool]
        forecast_part = forecast_distance[pool]
        selection_distance = 0.30 * scalar_part / max(
            float(np.median(scalar_part)), 1e-6
        ) + 0.70 * forecast_part / max(float(np.median(forecast_part)), 1e-6)
    chosen, chosen_distance = dedupe_alignments(
        pool,
        selection_distance,
        bundle.index["episode_id"].astype(str),
        bundle.neighbors,
    )
    if len(chosen) < 3:
        raise ValueError("V3 live retrieval found fewer than three distinct episodes")
    temperature = max(float(np.median(chosen_distance)), 1e-4)
    weights = np.exp(-chosen_distance / temperature) + 1e-6
    medoid = select_real_medoid(bundle.index["curve_log_ratio"][chosen], weights)
    selected = int(chosen[medoid])
    remaining = np.expm1(
        weighted_quantiles(
            bundle.index["log_remaining"][chosen], weights, (0.1, 0.5, 0.9)
        )
    )
    excursion = np.expm1(
        weighted_quantiles(
            bundle.index["log_excursion"][chosen], weights, (0.1, 0.5, 0.9)
        )
    )
    return {
        "episode_id": str(bundle.index["episode_id"][selected]),
        "historical_asset": str(bundle.index["asset"][selected]),
        "alignment_offset_bars": int(bundle.index["offset_bars"][selected]),
        "hybrid_distance": float(chosen_distance[medoid]),
        "selector": bundle.selector,
        "candidate_pool_size": len(pool),
        "neighbor_episode_count": len(chosen),
        "remaining_bars": {
            key: float(value)
            for key, value in zip(("p10", "p50", "p90"), remaining, strict=True)
        },
        "future_max_away_pct": {
            key: float(value)
            for key, value in zip(("p10", "p50", "p90"), excursion, strict=True)
        },
        "elapsed_bars": elapsed_bars,
        "supported_age_min_bars": artifact_min_age,
        "supported_age_max_bars": live_max_age,
        "artifact_generated_at_utc": bundle.summary.get("generated_at_utc"),
        "regime_label": bundle.regime_label,
        "fit_rows": len(bundle.index["episode_id"]),
    }
