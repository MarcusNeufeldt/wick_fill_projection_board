"""Train a leakage-safe neural analogue retriever for completed 5m wick paths.

This is an isolated research challenger.  It does not alter the live V1
dashboard.  A compact temporal convolutional network (TCN) learns an embedding
from three observable views of a wick episode:

* fixed pre-signal market context;
* the most recent candles ending at the snapshot; and
* a resampled signal-to-snapshot trajectory.

The embedding is supervised by remaining time, future adverse excursion,
fixed-clock future distance-to-target, and next-leg direction.  It is then used
to retrieve *real* historical continuations.  A hybrid retriever uses the union
of neural and scalar candidates so that the old scalar gate cannot permanently
hide a useful neural analogue.

The population intentionally matches V2 for this first experiment: strict 5m
signals which made a clean fill.  Unresolved/censored episodes require a
separate all-outcome library and are explicitly not treated as negative fills.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
    raise SystemExit(
        "PyTorch is required. Create the project venv and install requirements-ml.txt."
    ) from exc

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except ImportError as exc:  # pragma: no cover - exercised by the CLI error path
    raise SystemExit(
        "scikit-learn is required. Create the project venv and install requirements-ml.txt."
    ) from exc

from build_conditional_path_scenarios import FEATURE_WEIGHTS, STATE_WEIGHTS
from conditional_wick_assets import default_library_dir
from train_conditional_wick_v2 import (
    CATEGORICAL_FEATURE_COLUMNS,
    SNAPSHOT_FEATURE_COLUMNS,
    build_snapshot_dataset,
    chronological_split,
    evenly_spaced,
    load_events,
    load_paths,
)

SCHEMA_VERSION = "3.0.0-neural-retrieval-research"
BAR_MINUTES = 5
PRE_SIGNAL_LENGTH = 96
RECENT_LENGTH = 256
EPISODE_LENGTH = 128
FUTURE_HORIZON_BARS = np.asarray([1, 3, 6, 12, 24, 48, 96, 192], dtype=np.int64)
DIRECTION_HORIZON_BARS = np.asarray([6, 12, 48], dtype=np.int64)
DEFAULT_OFFSETS = [1, 3, 6, 12, 24, 60, 120, 240, 480, 960]
RAW_SEQUENCE_CHANNELS = (
    "open_signed_log_distance",
    "high_signed_log_distance",
    "low_signed_log_distance",
    "close_signed_log_distance",
    "directional_log_return_pct",
    "log_range_pct",
    "log_volume_ratio_to_signal",
    "is_post_signal",
    "valid",
)
EPISODE_SEQUENCE_CHANNELS = (
    "open_signed_log_distance",
    "high_signed_log_distance",
    "low_signed_log_distance",
    "close_signed_log_distance",
    "log_range_pct",
    "log_volume_ratio_to_signal",
    "valid",
)
STATIC_COLUMNS = (*SNAPSHOT_FEATURE_COLUMNS, *CATEGORICAL_FEATURE_COLUMNS)


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat().replace("+00:00", "Z")


def package_version_or_unknown(name: str) -> str:
    try:
        return package_version(name)
    except PackageNotFoundError:
        return "unknown"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def signed_log1p(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.sign(values) * np.log1p(np.abs(values))


def direction_normalized_pct(
    prices: np.ndarray, target: float, direction_sign: int
) -> np.ndarray:
    if not np.isfinite(target) or target <= 0:
        raise ValueError("wick target must be finite and positive")
    return (
        direction_sign * (np.asarray(prices, dtype=np.float64) / target - 1.0) * 100.0
    )


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20260921
    train_cap: int = 18_000
    validation_cap: int = 2_000
    holdout_cap: int = 1_200
    epochs: int = 12
    patience: int = 3
    batch_size: int = 128
    learning_rate: float = 7e-4
    weight_decay: float = 1e-4
    embedding_dim: int = 64
    sequence_width: int = 32
    dropout: float = 0.10
    retrieval_neighbors: int = 32
    candidate_pool: int = 512
    validation_fraction: float = 0.15
    embargo_bars: int = 288
    holdout_months: int = 12


@dataclass
class SequenceScaler:
    mean: np.ndarray
    scale: np.ndarray


@dataclass
class StaticScaler:
    center: np.ndarray
    scale: np.ndarray


@dataclass
class SampleArrays:
    pre_signal: np.ndarray
    recent: np.ndarray
    episode: np.ndarray
    static: np.ndarray
    log_remaining: np.ndarray
    log_excursion: np.ndarray
    curve_log_ratio: np.ndarray
    direction_targets: np.ndarray
    current_move_pct: np.ndarray
    weights: np.ndarray
    metadata: pd.DataFrame


def parse_offsets(value: str) -> list[int]:
    offsets = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not offsets or offsets[0] < 1:
        raise ValueError("snapshot offsets must contain positive integers")
    return offsets


def load_raw_candles(data_dir: Path, asset: str) -> dict[str, np.ndarray]:
    path = data_dir / f"{asset}_5m_5y.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing raw 5m candles: {path}")
    columns = ["open_time", "open", "high", "low", "close", "volume"]
    frame = pd.read_csv(path, usecols=columns)
    frame[columns] = frame[columns].apply(pd.to_numeric, errors="coerce")
    frame = (
        frame.dropna()
        .sort_values("open_time", kind="stable")
        .drop_duplicates("open_time", keep="last")
    )
    open_time = frame["open_time"].to_numpy(dtype=np.int64)
    if len(open_time) < 2 or np.any(np.diff(open_time) <= 0):
        raise RuntimeError(f"Raw candles are not strictly chronological for {asset}")
    return {column: frame[column].to_numpy(dtype=np.float64) for column in columns}


def raw_window(
    raw: dict[str, np.ndarray],
    end_open_time_ms: int,
    length: int,
    target: float,
    direction_sign: int,
    signal_volume: float,
    signal_open_time_ms: int,
) -> np.ndarray:
    times = raw["open_time"]
    end_index = int(np.searchsorted(times, end_open_time_ms))
    if end_index >= len(times) or int(times[end_index]) != int(end_open_time_ms):
        raise RuntimeError(f"Raw candle {end_open_time_ms} is missing")
    start_index = max(0, end_index - length + 1)
    source = slice(start_index, end_index + 1)
    count = end_index - start_index + 1
    result = np.zeros((length, len(RAW_SEQUENCE_CHANNELS)), dtype=np.float32)
    destination = slice(length - count, length)

    normalized = {
        field: direction_normalized_pct(raw[field][source], target, direction_sign)
        for field in ("open", "high", "low", "close")
    }
    close = raw["close"][source]
    previous_close = np.concatenate(([close[0]], close[:-1]))
    log_return = (
        direction_sign
        * np.log(np.maximum(close, 1e-12) / np.maximum(previous_close, 1e-12))
        * 100.0
    )
    range_pct = np.abs(raw["high"][source] - raw["low"][source]) / target * 100.0
    volume_ratio = np.maximum(raw["volume"][source], 0.0) / max(
        float(signal_volume), 1e-12
    )

    result[destination, 0] = signed_log1p(normalized["open"])
    result[destination, 1] = signed_log1p(normalized["high"])
    result[destination, 2] = signed_log1p(normalized["low"])
    result[destination, 3] = signed_log1p(normalized["close"])
    result[destination, 4] = np.clip(log_return, -20.0, 20.0).astype(np.float32)
    result[destination, 5] = np.log1p(range_pct).astype(np.float32)
    result[destination, 6] = np.log1p(volume_ratio).astype(np.float32)
    result[destination, 7] = (times[source] >= signal_open_time_ms).astype(np.float32)
    result[destination, 8] = 1.0
    return result


def episode_window(
    path: pd.DataFrame, snapshot_offset: int, signal_volume: float
) -> np.ndarray:
    observed = path.loc[path["offset_bars"].le(snapshot_offset)].sort_values(
        "offset_bars", kind="stable"
    )
    if observed.empty or int(observed["offset_bars"].iloc[-1]) != int(snapshot_offset):
        raise RuntimeError(f"Episode path is missing snapshot offset {snapshot_offset}")
    source_x = observed["offset_bars"].to_numpy(dtype=np.float64)
    destination_x = np.linspace(float(source_x[0]), float(source_x[-1]), EPISODE_LENGTH)
    result = np.ones((EPISODE_LENGTH, len(EPISODE_SEQUENCE_CHANNELS)), dtype=np.float32)
    for column_index, field in enumerate(
        (
            "normalized_open_pct",
            "normalized_high_pct",
            "normalized_low_pct",
            "normalized_close_pct",
        )
    ):
        values = np.interp(
            destination_x, source_x, observed[field].to_numpy(dtype=np.float64)
        )
        result[:, column_index] = signed_log1p(values)
    ranges = np.abs(
        observed["normalized_high_pct"].to_numpy(dtype=np.float64)
        - observed["normalized_low_pct"].to_numpy(dtype=np.float64)
    )
    result[:, 4] = np.log1p(np.interp(destination_x, source_x, ranges)).astype(
        np.float32
    )
    volumes = np.maximum(observed["volume"].to_numpy(dtype=np.float64), 0.0) / max(
        signal_volume, 1e-12
    )
    result[:, 5] = np.log1p(np.interp(destination_x, source_x, volumes)).astype(
        np.float32
    )
    result[:, 6] = 1.0
    return result


def future_targets(
    path: pd.DataFrame,
    snapshot_offset: int,
    current_move_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = path.sort_values("offset_bars", kind="stable")
    offsets = ordered["offset_bars"].to_numpy(dtype=np.int64)
    closes = ordered["normalized_close_pct"].to_numpy(dtype=np.float64)
    fill_offset = int(offsets[-1])
    denominator = max(float(current_move_pct), 0.02)
    future_distances: list[float] = []
    for horizon in FUTURE_HORIZON_BARS:
        wanted = snapshot_offset + int(horizon)
        if wanted >= fill_offset:
            future_distances.append(0.0)
            continue
        position = int(np.searchsorted(offsets, wanted))
        if position >= len(offsets) or int(offsets[position]) != wanted:
            raise RuntimeError(f"Episode path is missing future offset {wanted}")
        future_distances.append(max(0.0, float(closes[position])))
    ratios = np.clip(
        np.asarray(future_distances, dtype=np.float32) / denominator, 0.0, 25.0
    )
    curve_log_ratio = np.log1p(ratios).astype(np.float32)
    directions = []
    for horizon in DIRECTION_HORIZON_BARS:
        curve_index = int(np.flatnonzero(FUTURE_HORIZON_BARS == horizon)[0])
        directions.append(float(future_distances[curve_index] < current_move_pct))
    return curve_log_ratio, np.asarray(directions, dtype=np.float32)


def inner_chronological_split(
    train_pool: pd.DataFrame,
    validation_fraction: float,
    embargo_bars: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    episodes = (
        train_pool[["episode_id", "signal_open_time_ms"]]
        .drop_duplicates("episode_id")
        .sort_values(["signal_open_time_ms", "episode_id"], kind="stable")
    )
    cut_index = min(
        len(episodes) - 1,
        max(1, int(len(episodes) * (1.0 - validation_fraction))),
    )
    validation_start_ms = int(episodes.iloc[cut_index]["signal_open_time_ms"])
    validation = train_pool.loc[
        train_pool["signal_open_time_ms"].ge(validation_start_ms)
    ].copy()
    if validation.empty:
        raise RuntimeError("Inner chronological validation split is empty")
    earliest_validation_snapshot_ms = int(validation["snapshot_close_time_ms"].min())
    resolution_cutoff_ms = (
        earliest_validation_snapshot_ms - embargo_bars * BAR_MINUTES * 60_000
    )
    fit = train_pool.loc[
        train_pool["signal_open_time_ms"].lt(validation_start_ms)
        & train_pool["fill_close_time_ms"].le(resolution_cutoff_ms)
    ].copy()
    if fit.empty:
        raise RuntimeError(
            "Inner chronological fit split is empty after label-availability filtering"
        )
    fit_ids = set(fit["episode_id"].astype(str))
    validation_ids = set(validation["episode_id"].astype(str))
    if fit_ids.intersection(validation_ids):
        raise RuntimeError("Episode overlap in inner chronological split")
    if int(fit["fill_close_time_ms"].max()) > earliest_validation_snapshot_ms:
        raise RuntimeError("Inner resolved-before-snapshot guard failed")
    details = {
        "validation_start_utc": pd.Timestamp(
            validation_start_ms, unit="ms", tz="UTC"
        ).isoformat(),
        "fit_episode_count_before_cap": len(fit_ids),
        "validation_episode_count_before_cap": len(validation_ids),
        "max_fit_fill_close_utc": pd.Timestamp(
            int(fit["fill_close_time_ms"].max()), unit="ms", tz="UTC"
        ).isoformat(),
        "min_validation_snapshot_close_utc": pd.Timestamp(
            earliest_validation_snapshot_ms, unit="ms", tz="UTC"
        ).isoformat(),
        "same_episode_overlap_count": 0,
        "all_fit_labels_resolved_before_every_validation_snapshot": True,
    }
    return fit, validation, details


def inverse_episode_frequency(frame: pd.DataFrame) -> np.ndarray:
    counts = (
        frame.groupby("episode_id")["episode_id"]
        .transform("size")
        .to_numpy(dtype=np.float32)
    )
    weights = 1.0 / np.maximum(counts, 1.0)
    return weights / max(float(weights.mean()), 1e-12)


def materialize_samples(
    snapshots: pd.DataFrame,
    events: pd.DataFrame,
    paths: pd.DataFrame,
    raw_by_asset: dict[str, dict[str, np.ndarray]],
    label: str,
) -> SampleArrays:
    event_lookup = events.set_index("episode_id", drop=False)
    selected_ids = set(snapshots["episode_id"].astype(str))
    selected_paths = paths.loc[
        paths["episode_id"].astype(str).isin(selected_ids)
    ].copy()
    path_lookup = {
        str(key): value
        for key, value in selected_paths.groupby("episode_id", sort=False)
    }
    count = len(snapshots)
    pre_signal = np.empty(
        (count, PRE_SIGNAL_LENGTH, len(RAW_SEQUENCE_CHANNELS)), dtype=np.float32
    )
    recent = np.empty(
        (count, RECENT_LENGTH, len(RAW_SEQUENCE_CHANNELS)), dtype=np.float32
    )
    episode = np.empty(
        (count, EPISODE_LENGTH, len(EPISODE_SEQUENCE_CHANNELS)), dtype=np.float32
    )
    curve = np.empty((count, len(FUTURE_HORIZON_BARS)), dtype=np.float32)
    direction = np.empty((count, len(DIRECTION_HORIZON_BARS)), dtype=np.float32)

    for output_index, row in enumerate(snapshots.itertuples(index=False)):
        event_id = str(row.episode_id)
        event = event_lookup.loc[event_id]
        target = float(event["wick_target"])
        sign = int(event["direction_sign"])
        signal_ms = int(event["signal_open_time_ms"])
        signal_volume = float(event["signal_volume"])
        raw = raw_by_asset[str(event["asset"])]
        pre_signal[output_index] = raw_window(
            raw,
            signal_ms - BAR_MINUTES * 60_000,
            PRE_SIGNAL_LENGTH,
            target,
            sign,
            signal_volume,
            signal_ms,
        )
        recent[output_index] = raw_window(
            raw,
            int(row.snapshot_open_time_ms),
            RECENT_LENGTH,
            target,
            sign,
            signal_volume,
            signal_ms,
        )
        episode_path = path_lookup[event_id]
        episode[output_index] = episode_window(
            episode_path, int(row.offset_bars), signal_volume
        )
        curve[output_index], direction[output_index] = future_targets(
            episode_path,
            int(row.offset_bars),
            float(row.current_move_pct),
        )
        if (output_index + 1) % 2_000 == 0 or output_index + 1 == count:
            print(f"materialized {label}: {output_index + 1}/{count}", flush=True)

    metadata_columns = [
        "episode_id",
        "asset",
        "direction",
        "signal_open_time_ms",
        "snapshot_open_time_ms",
        "snapshot_close_time_ms",
        "fill_close_time_ms",
        "offset_bars",
        "remaining_bars",
        "future_max_away_pct",
        "current_move_pct",
    ]
    return SampleArrays(
        pre_signal=pre_signal,
        recent=recent,
        episode=episode,
        static=snapshots.loc[:, STATIC_COLUMNS].to_numpy(dtype=np.float32),
        log_remaining=np.log1p(snapshots["remaining_bars"].to_numpy(dtype=np.float32)),
        log_excursion=np.log1p(
            np.maximum(snapshots["future_max_away_pct"].to_numpy(dtype=np.float32), 0.0)
        ),
        curve_log_ratio=curve,
        direction_targets=direction,
        current_move_pct=snapshots["current_move_pct"].to_numpy(dtype=np.float32),
        weights=inverse_episode_frequency(snapshots),
        metadata=snapshots.loc[:, metadata_columns].reset_index(drop=True),
    )


def fit_sequence_scaler(values: np.ndarray) -> SequenceScaler:
    valid = values[:, :, -1] > 0.5
    mean = np.zeros(values.shape[-1], dtype=np.float32)
    scale = np.ones(values.shape[-1], dtype=np.float32)
    for channel in range(values.shape[-1] - 1):
        selected = values[:, :, channel][valid]
        mean[channel] = float(np.mean(selected))
        scale[channel] = max(float(np.std(selected)), 1e-4)
    return SequenceScaler(mean=mean, scale=scale)


def apply_sequence_scaler(values: np.ndarray, scaler: SequenceScaler) -> np.ndarray:
    output = values.copy().astype(np.float32)
    valid = output[:, :, -1:] > 0.5
    output[:, :, :-1] = (output[:, :, :-1] - scaler.mean[:-1]) / scaler.scale[:-1]
    output[:, :, :-1] *= valid
    output[:, :, -1] = valid[:, :, 0].astype(np.float32)
    return output


def fit_static_scaler(values: np.ndarray) -> StaticScaler:
    center = np.nanmedian(values, axis=0).astype(np.float32)
    q25 = np.nanquantile(values, 0.25, axis=0).astype(np.float32)
    q75 = np.nanquantile(values, 0.75, axis=0).astype(np.float32)
    scale = np.maximum(q75 - q25, 1e-4).astype(np.float32)
    return StaticScaler(center=center, scale=scale)


def apply_static_scaler(values: np.ndarray, scaler: StaticScaler) -> np.ndarray:
    output = (values.astype(np.float32) - scaler.center) / scaler.scale
    return np.nan_to_num(output, nan=0.0, posinf=20.0, neginf=-20.0).clip(-20.0, 20.0)


class WickSequenceDataset(Dataset):
    def __init__(self, samples: SampleArrays) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples.static)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        return (
            torch.from_numpy(self.samples.pre_signal[index]).transpose(0, 1),
            torch.from_numpy(self.samples.recent[index]).transpose(0, 1),
            torch.from_numpy(self.samples.episode[index]).transpose(0, 1),
            torch.from_numpy(self.samples.static[index]),
            torch.tensor(self.samples.log_remaining[index], dtype=torch.float32),
            torch.tensor(self.samples.log_excursion[index], dtype=torch.float32),
            torch.from_numpy(self.samples.curve_log_ratio[index]),
            torch.from_numpy(self.samples.direction_targets[index]),
            torch.tensor(self.samples.weights[index], dtype=torch.float32),
        )


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            width, width, kernel_size=5, dilation=dilation, padding="same"
        )
        self.conv2 = nn.Conv1d(
            width, width, kernel_size=3, dilation=dilation, padding="same"
        )
        self.norm1 = nn.GroupNorm(4, width)
        self.norm2 = nn.GroupNorm(4, width)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.dropout(F.gelu(self.norm1(self.conv1(values))))
        values = self.dropout(self.norm2(self.conv2(values)))
        return F.gelu(values + residual)


class SequenceEncoder(nn.Module):
    def __init__(self, channels: int, width: int, dropout: float) -> None:
        super().__init__()
        self.input = nn.Conv1d(channels, width, kernel_size=5, padding="same")
        self.blocks = nn.Sequential(
            ResidualTemporalBlock(width, 1, dropout),
            ResidualTemporalBlock(width, 2, dropout),
            ResidualTemporalBlock(width, 4, dropout),
            ResidualTemporalBlock(width, 8, dropout),
        )
        self.output = nn.Linear(width * 2, width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = self.blocks(F.gelu(self.input(values)))
        pooled = torch.cat((encoded.mean(dim=-1), encoded[:, :, -1]), dim=1)
        return F.gelu(self.output(pooled))


class NeuralPathModel(nn.Module):
    def __init__(
        self,
        static_features: int,
        sequence_width: int = 32,
        embedding_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pre_encoder = SequenceEncoder(
            len(RAW_SEQUENCE_CHANNELS), sequence_width, dropout
        )
        self.recent_encoder = SequenceEncoder(
            len(RAW_SEQUENCE_CHANNELS), sequence_width, dropout
        )
        self.episode_encoder = SequenceEncoder(
            len(EPISODE_SEQUENCE_CHANNELS), sequence_width, dropout
        )
        self.static_encoder = nn.Sequential(
            nn.Linear(static_features, sequence_width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fusion = nn.Sequential(
            nn.Linear(sequence_width * 4, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.GELU(),
        )
        self.risk_head = nn.Linear(embedding_dim, 6)
        self.curve_head = nn.Linear(embedding_dim, len(FUTURE_HORIZON_BARS))
        self.direction_head = nn.Linear(embedding_dim, len(DIRECTION_HORIZON_BARS))

    def forward(
        self,
        pre_signal: torch.Tensor,
        recent: torch.Tensor,
        episode: torch.Tensor,
        static: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = torch.cat(
            (
                self.pre_encoder(pre_signal),
                self.recent_encoder(recent),
                self.episode_encoder(episode),
                self.static_encoder(static),
            ),
            dim=1,
        )
        latent = self.fusion(fused)
        risk = F.softplus(self.risk_head(latent)).reshape(-1, 2, 3)
        return {
            "embedding": F.normalize(latent, p=2, dim=1),
            "risk": risk,
            "curve": F.softplus(self.curve_head(latent)),
            "direction_logits": self.direction_head(latent),
        }


def pinball_per_sample(
    actual: torch.Tensor, predicted: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    error = actual[:, None] - predicted
    return torch.maximum(quantiles * error, (quantiles - 1.0) * error).mean(dim=1)


def metric_triplet_loss(
    embedding: torch.Tensor, target_signature: torch.Tensor
) -> torch.Tensor:
    if len(embedding) < 4:
        return embedding.new_tensor(0.0)
    with torch.no_grad():
        normalized_target = (
            target_signature - target_signature.mean(dim=0)
        ) / target_signature.std(dim=0).clamp_min(1e-4)
        distances = torch.cdist(normalized_target, normalized_target)
        distances.fill_diagonal_(float("inf"))
        positive = distances.argmin(dim=1)
        distances.fill_diagonal_(float("-inf"))
        negative = distances.argmax(dim=1)
    return F.triplet_margin_loss(
        embedding, embedding[positive], embedding[negative], margin=0.2
    )


def batch_loss(
    outputs: dict[str, torch.Tensor], batch: tuple[torch.Tensor, ...]
) -> tuple[torch.Tensor, dict[str, float]]:
    _, _, _, _, log_remaining, log_excursion, curve, direction, weights = batch
    quantiles = torch.tensor([0.1, 0.5, 0.9], device=log_remaining.device)
    risk = outputs["risk"]
    remaining_loss = pinball_per_sample(log_remaining, risk[:, 0, :], quantiles)
    excursion_loss = pinball_per_sample(log_excursion, risk[:, 1, :], quantiles)
    curve_loss = F.smooth_l1_loss(outputs["curve"], curve, reduction="none").mean(dim=1)
    direction_loss = F.binary_cross_entropy_with_logits(
        outputs["direction_logits"], direction, reduction="none"
    ).mean(dim=1)
    weighted = weights / weights.mean().clamp_min(1e-6)
    supervised = (
        (remaining_loss + excursion_loss + 0.60 * curve_loss + 0.20 * direction_loss)
        * weighted
    ).mean()
    crossing = (
        F.relu(risk[:, :, 0] - risk[:, :, 1]).mean()
        + F.relu(risk[:, :, 1] - risk[:, :, 2]).mean()
    )
    signature = torch.cat(
        (log_remaining[:, None], log_excursion[:, None], curve), dim=1
    )
    metric = metric_triplet_loss(outputs["embedding"], signature)
    total = supervised + 0.05 * crossing + 0.10 * metric
    return total, {
        "supervised": float(supervised.detach().cpu()),
        "crossing": float(crossing.detach().cpu()),
        "metric": float(metric.detach().cpu()),
    }


def move_batch(
    batch: tuple[torch.Tensor, ...], device: torch.device
) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device, non_blocking=True) for value in batch)


@torch.no_grad()
def predict_model(
    model: NeuralPathModel,
    samples: SampleArrays,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    loader = DataLoader(
        WickSequenceDataset(samples),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    collected: dict[str, list[np.ndarray]] = {
        key: [] for key in ("embedding", "risk", "curve", "direction")
    }
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch[0], batch[1], batch[2], batch[3])
        collected["embedding"].append(outputs["embedding"].cpu().numpy())
        collected["risk"].append(outputs["risk"].cpu().numpy())
        collected["curve"].append(outputs["curve"].cpu().numpy())
        collected["direction"].append(
            torch.sigmoid(outputs["direction_logits"]).cpu().numpy()
        )
    return {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}


def train_model(
    fit: SampleArrays,
    validation: SampleArrays,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[NeuralPathModel, list[dict[str, float]], int]:
    model = NeuralPathModel(
        static_features=fit.static.shape[1],
        sequence_width=config.sequence_width,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        WickSequenceDataset(fit),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_score = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        training_losses: list[float] = []
        for batch in loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch[0], batch[1], batch[2], batch[3])
            loss, _ = batch_loss(outputs, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            training_losses.append(float(loss.detach().cpu()))

        validation_predictions = predict_model(
            model, validation, config.batch_size, device
        )
        median_risk = np.sort(validation_predictions["risk"], axis=2)[:, :, 1]
        validation_score = float(
            np.median(np.abs(median_risk[:, 0] - validation.log_remaining))
            + np.median(np.abs(median_risk[:, 1] - validation.log_excursion))
            + 0.5
            * np.mean(
                np.abs(validation_predictions["curve"] - validation.curve_log_ratio)
            )
        )
        record = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(training_losses)),
            "validation_score": validation_score,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        if validation_score < best_score - 1e-4:
            best_score = validation_score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    if best_state is None:
        raise RuntimeError("Neural training did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, history, best_epoch


def summarize_sequences(samples: SampleArrays) -> np.ndarray:
    summaries: list[np.ndarray] = [samples.static.astype(np.float32)]
    for sequence in (samples.pre_signal, samples.recent, samples.episode):
        valid = sequence[:, :, -1:] > 0.5
        features = sequence[:, :, :-1]
        masked = np.where(valid, features, np.nan)
        valid_rows = valid[:, :, 0]
        first_positions = np.argmax(valid_rows, axis=1)
        first = features[np.arange(len(features)), first_positions]
        last = np.nan_to_num(masked[:, -1, :], nan=0.0)
        summaries.extend(
            [
                np.nan_to_num(np.nanmean(masked, axis=1), nan=0.0),
                np.nan_to_num(np.nanstd(masked, axis=1), nan=0.0),
                np.nan_to_num(np.nanmin(masked, axis=1), nan=0.0),
                np.nan_to_num(np.nanmax(masked, axis=1), nan=0.0),
                last,
                last - first,
            ]
        )
    return np.concatenate(summaries, axis=1).astype(np.float32)


def fit_tree_quantiles(
    fit: SampleArrays,
    holdout: SampleArrays,
    seed: int,
) -> dict[str, np.ndarray]:
    train_x = summarize_sequences(fit)
    holdout_x = summarize_sequences(holdout)
    output: dict[str, np.ndarray] = {}
    for target_name, train_y in (
        ("remaining", fit.log_remaining),
        ("excursion", fit.log_excursion),
    ):
        predictions: list[np.ndarray] = []
        for quantile in (0.1, 0.5, 0.9):
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=quantile,
                learning_rate=0.06,
                max_iter=160,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=0.1,
                random_state=seed,
            )
            model.fit(train_x, train_y, sample_weight=fit.weights)
            predictions.append(np.maximum(0.0, model.predict(holdout_x)))
        output[target_name] = np.sort(np.stack(predictions, axis=1), axis=1)
    return output


def log_distance(values: np.ndarray, target: float, floor: float) -> np.ndarray:
    safe_values = np.maximum(np.asarray(values, dtype=np.float64), 0.0) + floor
    safe_target = max(float(target), 0.0) + floor
    return np.abs(np.log(safe_values / safe_target))


def v1_feature_scales(fit_static: np.ndarray) -> dict[str, float]:
    scales: dict[str, float] = {}
    for name in FEATURE_WEIGHTS:
        values = fit_static[:, STATIC_COLUMNS.index(name)].astype(np.float64)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)) * 1.4826)
        fallback = float(np.std(values))
        scales[name] = mad if mad > 1e-12 else (fallback if fallback > 1e-12 else 1.0)
    return scales


def v1_scalar_distance(
    fit_static: np.ndarray,
    query_static: np.ndarray,
    fit_assets: np.ndarray,
    query_asset: str,
    fit_directions: np.ndarray,
    query_direction: str,
    feature_scales: dict[str, float],
) -> np.ndarray:
    current_index = STATIC_COLUMNS.index("current_move_pct")
    peak_index = STATIC_COLUMNS.index("peak_move_pct")
    drawdown_index = STATIC_COLUMNS.index("drawdown_from_peak_pct")
    elapsed_index = STATIC_COLUMNS.index("elapsed_log_bars")
    state_distance = (
        STATE_WEIGHTS["current_move_pct"]
        * log_distance(fit_static[:, current_index], query_static[current_index], 0.30)
        + STATE_WEIGHTS["peak_move_pct"]
        * log_distance(fit_static[:, peak_index], query_static[peak_index], 0.30)
        + STATE_WEIGHTS["drawdown_from_peak_pct"]
        * log_distance(
            fit_static[:, drawdown_index], query_static[drawdown_index], 0.30
        )
        + STATE_WEIGHTS["elapsed_bars"]
        * log_distance(
            np.expm1(fit_static[:, elapsed_index]),
            float(np.expm1(query_static[elapsed_index])),
            3.0,
        )
    )
    feature_distance = np.zeros(len(fit_static), dtype=np.float64)
    for name, weight in FEATURE_WEIGHTS.items():
        column = STATIC_COLUMNS.index(name)
        feature_distance += (
            weight
            * np.abs(fit_static[:, column] - query_static[column])
            / feature_scales[name]
        )
    category_distance = (fit_assets != query_asset).astype(np.float64) * 0.35 + (
        fit_directions != query_direction
    ).astype(np.float64) * 0.10
    return state_distance + 0.5 * feature_distance + category_distance


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, quantiles: Iterable[float]
) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = np.maximum(weights[order], 1e-12)
    cumulative = np.cumsum(sorted_weights)
    cumulative /= cumulative[-1]
    return np.interp(
        np.asarray(list(quantiles), dtype=float), cumulative, sorted_values
    )


def dedupe_alignments(
    indices: np.ndarray,
    distances: np.ndarray,
    episode_ids: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(distances, kind="stable")
    kept_indices: list[int] = []
    kept_distances: list[float] = []
    seen: set[str] = set()
    for position in order:
        candidate = int(indices[position])
        episode_id = str(episode_ids[candidate])
        if episode_id in seen:
            continue
        seen.add(episode_id)
        kept_indices.append(candidate)
        kept_distances.append(float(distances[position]))
        if len(kept_indices) >= maximum:
            break
    return np.asarray(kept_indices, dtype=np.int64), np.asarray(
        kept_distances, dtype=np.float64
    )


def select_real_medoid(curves: np.ndarray, weights: np.ndarray) -> int:
    consensus = np.average(curves, axis=0, weights=weights)
    distance = np.mean(np.abs(curves - consensus[None, :]), axis=1)
    return int(np.argmin(distance))


def retrieve_predictions(
    fit: SampleArrays,
    holdout: SampleArrays,
    fit_raw_static: np.ndarray,
    holdout_raw_static: np.ndarray,
    fit_embedding: np.ndarray,
    holdout_embedding: np.ndarray,
    holdout_neural_predictions: dict[str, np.ndarray],
    neighbors: int,
    candidate_pool: int,
    mode: str,
) -> dict[str, np.ndarray]:
    episode_ids = fit.metadata["episode_id"].astype(str).to_numpy()
    fit_assets = fit.metadata["asset"].astype(str).to_numpy()
    holdout_assets = holdout.metadata["asset"].astype(str).to_numpy()
    fit_directions = fit.metadata["direction"].astype(str).to_numpy()
    holdout_directions = holdout.metadata["direction"].astype(str).to_numpy()
    feature_scales = v1_feature_scales(fit_raw_static)
    fit_outcome_signature = np.column_stack(
        (
            fit.log_remaining,
            fit.log_excursion,
            fit.curve_log_ratio,
        )
    ).astype(np.float64)
    query_risk = np.sort(holdout_neural_predictions["risk"], axis=2)
    query_forecast_signature = np.column_stack(
        (
            query_risk[:, 0, 1],
            query_risk[:, 1, 1],
            holdout_neural_predictions["curve"],
        )
    ).astype(np.float64)
    forecast_center = np.median(fit_outcome_signature, axis=0)
    forecast_scale = np.maximum(
        np.quantile(fit_outcome_signature, 0.75, axis=0)
        - np.quantile(fit_outcome_signature, 0.25, axis=0),
        0.05,
    )
    fit_forecast_space = (fit_outcome_signature - forecast_center) / forecast_scale
    query_forecast_space = (query_forecast_signature - forecast_center) / forecast_scale
    result_remaining = np.empty((len(holdout.static), 3), dtype=np.float32)
    result_excursion = np.empty((len(holdout.static), 3), dtype=np.float32)
    result_curve = np.empty_like(holdout.curve_log_ratio)

    for start in range(0, len(holdout.static), 128):
        stop = min(len(holdout.static), start + 128)
        neural_distance = np.maximum(
            0.0, 1.0 - holdout_embedding[start:stop] @ fit_embedding.T
        )
        scalar_distance = np.stack(
            [
                v1_scalar_distance(
                    fit_raw_static,
                    holdout_raw_static[index],
                    fit_assets,
                    holdout_assets[index],
                    fit_directions,
                    holdout_directions[index],
                    feature_scales,
                )
                for index in range(start, stop)
            ],
            axis=0,
        )
        forecast_distance = np.mean(
            np.abs(
                query_forecast_space[start:stop, None, :]
                - fit_forecast_space[None, :, :]
            ),
            axis=2,
        )
        for local_index in range(stop - start):
            if mode == "scalar":
                pool = np.argpartition(
                    scalar_distance[local_index],
                    min(candidate_pool, len(fit_raw_static) - 1),
                )[:candidate_pool]
                distance = scalar_distance[local_index, pool]
            elif mode == "neural":
                pool = np.argpartition(
                    neural_distance[local_index],
                    min(candidate_pool, len(fit_raw_static) - 1),
                )[:candidate_pool]
                distance = neural_distance[local_index, pool]
            elif mode == "hybrid":
                scalar_pool = np.argpartition(
                    scalar_distance[local_index],
                    min(candidate_pool, len(fit_raw_static) - 1),
                )[:candidate_pool]
                neural_pool = np.argpartition(
                    neural_distance[local_index],
                    min(candidate_pool, len(fit_raw_static) - 1),
                )[:candidate_pool]
                pool = np.unique(np.concatenate((scalar_pool, neural_pool)))
                scalar_part = scalar_distance[local_index, pool]
                neural_part = neural_distance[local_index, pool]
                scalar_scale = max(float(np.median(scalar_part)), 1e-6)
                neural_scale = max(float(np.median(neural_part)), 1e-6)
                distance = (
                    0.35 * scalar_part / scalar_scale
                    + 0.65 * neural_part / neural_scale
                )
            elif mode == "forecast":
                pool = np.argpartition(
                    forecast_distance[local_index],
                    min(candidate_pool, len(fit_raw_static) - 1),
                )[:candidate_pool]
                distance = forecast_distance[local_index, pool]
            elif mode == "forecast_hybrid":
                scalar_pool = np.argpartition(
                    scalar_distance[local_index],
                    min(candidate_pool, len(fit_raw_static) - 1),
                )[:candidate_pool]
                forecast_pool = np.argpartition(
                    forecast_distance[local_index],
                    min(candidate_pool, len(fit_raw_static) - 1),
                )[:candidate_pool]
                pool = np.unique(np.concatenate((scalar_pool, forecast_pool)))
                scalar_part = scalar_distance[local_index, pool]
                forecast_part = forecast_distance[local_index, pool]
                scalar_scale = max(float(np.median(scalar_part)), 1e-6)
                forecast_scale_at_query = max(float(np.median(forecast_part)), 1e-6)
                distance = (
                    0.30 * scalar_part / scalar_scale
                    + 0.70 * forecast_part / forecast_scale_at_query
                )
            else:
                raise ValueError(f"Unknown retrieval mode: {mode}")

            chosen, chosen_distance = dedupe_alignments(
                pool, distance, episode_ids, neighbors
            )
            if len(chosen) < 3:
                raise RuntimeError(
                    "Retrieval produced fewer than three distinct historical episodes"
                )
            temperature = max(float(np.median(chosen_distance)), 1e-4)
            weights = np.exp(-chosen_distance / temperature) + 1e-6
            result_remaining[start + local_index] = weighted_quantiles(
                fit.log_remaining[chosen], weights, (0.1, 0.5, 0.9)
            )
            result_excursion[start + local_index] = weighted_quantiles(
                fit.log_excursion[chosen], weights, (0.1, 0.5, 0.9)
            )
            medoid = select_real_medoid(fit.curve_log_ratio[chosen], weights)
            result_curve[start + local_index] = fit.curve_log_ratio[chosen[medoid]]
    return {
        "remaining": result_remaining,
        "excursion": result_excursion,
        "curve": result_curve,
    }


def risk_metrics(actual_log: np.ndarray, predicted_log: np.ndarray) -> dict[str, float]:
    predictions = np.expm1(np.sort(predicted_log, axis=1))
    actual = np.expm1(actual_log)
    return {
        "p50_median_absolute_error": float(
            np.median(np.abs(actual - predictions[:, 1]))
        ),
        "p10_p90_coverage": float(
            np.mean((actual >= predictions[:, 0]) & (actual <= predictions[:, 2]))
        ),
        "p10_p90_mean_width": float(np.mean(predictions[:, 2] - predictions[:, 0])),
    }


def path_metrics(
    samples: SampleArrays,
    predicted_curve_log_ratio: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    selected = (
        np.ones(len(samples.current_move_pct), dtype=bool) if mask is None else mask
    )
    current_move = samples.current_move_pct[selected]
    actual_distance = (
        np.expm1(samples.curve_log_ratio[selected]) * current_move[:, None]
    )
    predicted_distance = (
        np.expm1(np.maximum(predicted_curve_log_ratio[selected], 0.0))
        * current_move[:, None]
    )
    per_horizon = np.mean(np.abs(actual_distance - predicted_distance), axis=0)
    direction_accuracy: dict[str, float] = {}
    for horizon in DIRECTION_HORIZON_BARS:
        curve_index = int(np.flatnonzero(FUTURE_HORIZON_BARS == horizon)[0])
        actual_toward = actual_distance[:, curve_index] < current_move
        predicted_toward = predicted_distance[:, curve_index] < current_move
        direction_accuracy[str(int(horizon))] = float(
            np.mean(actual_toward == predicted_toward)
        )
    return {
        "mean_absolute_distance_error_pct_points": float(
            np.mean(np.abs(actual_distance - predicted_distance))
        ),
        "mae_by_horizon_bars": {
            str(int(horizon)): float(value)
            for horizon, value in zip(FUTURE_HORIZON_BARS, per_horizon, strict=True)
        },
        "next_leg_direction_accuracy": direction_accuracy,
    }


def evaluate_predictions(
    holdout: SampleArrays,
    neural_direct: dict[str, np.ndarray],
    tree: dict[str, np.ndarray],
    retrievals: dict[str, dict[str, np.ndarray]],
    mask: np.ndarray | None = None,
) -> dict[str, Any]:
    selected = (
        np.ones(len(holdout.current_move_pct), dtype=bool) if mask is None else mask
    )
    direct_risk = np.sort(neural_direct["risk"], axis=2)
    models: dict[str, Any] = {
        "tree_richer_features": {
            "remaining_bars": risk_metrics(
                holdout.log_remaining[selected], tree["remaining"][selected]
            ),
            "future_max_away_pct": risk_metrics(
                holdout.log_excursion[selected], tree["excursion"][selected]
            ),
        },
        "neural_direct_diagnostic": {
            "remaining_bars": risk_metrics(
                holdout.log_remaining[selected], direct_risk[selected, 0, :]
            ),
            "future_max_away_pct": risk_metrics(
                holdout.log_excursion[selected], direct_risk[selected, 1, :]
            ),
            "path": path_metrics(holdout, neural_direct["curve"], selected),
        },
    }
    for name, predictions in retrievals.items():
        models[name] = {
            "remaining_bars": risk_metrics(
                holdout.log_remaining[selected], predictions["remaining"][selected]
            ),
            "future_max_away_pct": risk_metrics(
                holdout.log_excursion[selected], predictions["excursion"][selected]
            ),
            "path": path_metrics(holdout, predictions["curve"], selected),
        }
    return models


def episode_balanced_bootstrap_delta(
    baseline_error: np.ndarray,
    challenger_error: np.ndarray,
    episode_ids: np.ndarray,
    seed: int,
    repetitions: int = 2_000,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "episode_id": episode_ids.astype(str),
            "baseline": baseline_error.astype(float),
            "challenger": challenger_error.astype(float),
        }
    )
    episode = frame.groupby("episode_id", sort=False)[["baseline", "challenger"]].mean()
    differences = (episode["challenger"] - episode["baseline"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(
        differences, size=(repetitions, len(differences)), replace=True
    ).mean(axis=1)
    baseline_mean = float(episode["baseline"].mean())
    point_delta = float(differences.mean())
    return {
        "episode_count": len(episode),
        "baseline_episode_balanced_error": baseline_mean,
        "challenger_episode_balanced_error": float(episode["challenger"].mean()),
        "delta_challenger_minus_scalar": point_delta,
        "relative_improvement": float(-point_delta / baseline_mean)
        if baseline_mean > 0
        else 0.0,
        "bootstrap_95pct_low": float(np.quantile(draws, 0.025)),
        "bootstrap_95pct_high": float(np.quantile(draws, 0.975)),
        "probability_challenger_better": float(np.mean(draws < 0.0)),
    }


def comparison_vs_scalar(
    holdout: SampleArrays,
    neural_direct: dict[str, np.ndarray],
    retrievals: dict[str, dict[str, np.ndarray]],
    seed: int,
) -> dict[str, Any]:
    scalar = retrievals["scalar_real_path_retrieval"]
    actual_remaining = np.expm1(holdout.log_remaining)
    actual_excursion = np.expm1(holdout.log_excursion)
    actual_curve = np.expm1(holdout.curve_log_ratio) * holdout.current_move_pct[:, None]
    episode_ids = holdout.metadata["episode_id"].astype(str).to_numpy()
    scalar_errors = {
        "remaining_bars": np.abs(
            actual_remaining - np.expm1(scalar["remaining"][:, 1])
        ),
        "future_max_away_pct": np.abs(
            actual_excursion - np.expm1(scalar["excursion"][:, 1])
        ),
        "path": np.mean(
            np.abs(
                actual_curve
                - np.expm1(scalar["curve"]) * holdout.current_move_pct[:, None]
            ),
            axis=1,
        ),
    }
    direct_risk = np.sort(neural_direct["risk"], axis=2)
    challengers = {
        "neural_direct_diagnostic": {
            "remaining_bars": np.abs(actual_remaining - np.expm1(direct_risk[:, 0, 1])),
            "future_max_away_pct": np.abs(
                actual_excursion - np.expm1(direct_risk[:, 1, 1])
            ),
            "path": np.mean(
                np.abs(
                    actual_curve
                    - np.expm1(neural_direct["curve"])
                    * holdout.current_move_pct[:, None]
                ),
                axis=1,
            ),
        },
        **{
            name: {
                "remaining_bars": np.abs(
                    actual_remaining - np.expm1(value["remaining"][:, 1])
                ),
                "future_max_away_pct": np.abs(
                    actual_excursion - np.expm1(value["excursion"][:, 1])
                ),
                "path": np.mean(
                    np.abs(
                        actual_curve
                        - np.expm1(value["curve"]) * holdout.current_move_pct[:, None]
                    ),
                    axis=1,
                ),
            }
            for name, value in retrievals.items()
            if name != "scalar_real_path_retrieval"
        },
    }
    return {
        name: {
            metric: episode_balanced_bootstrap_delta(
                scalar_errors[metric],
                errors[metric],
                episode_ids,
                seed + challenger_index * 100 + metric_index,
            )
            for metric_index, metric in enumerate(
                ("remaining_bars", "future_max_away_pct", "path")
            )
        }
        for challenger_index, (name, errors) in enumerate(challengers.items())
    }


def subgroup_metrics(
    holdout: SampleArrays,
    neural_direct: dict[str, np.ndarray],
    tree: dict[str, np.ndarray],
    retrievals: dict[str, dict[str, np.ndarray]],
) -> dict[str, Any]:
    groups: dict[str, Any] = {"by_asset": {}, "by_direction": {}}
    for column, destination in (("asset", "by_asset"), ("direction", "by_direction")):
        values = holdout.metadata[column].astype(str).to_numpy()
        for value in sorted(set(values)):
            mask = values == value
            groups[destination][value] = {
                "case_count": int(mask.sum()),
                "episode_count": int(
                    holdout.metadata.loc[mask, "episode_id"].nunique()
                ),
                "metrics": evaluate_predictions(
                    holdout, neural_direct, tree, retrievals, mask
                ),
            }
    return groups


def markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Neural Historical-Path Retrieval Experiment",
        "",
        f"Generated: `{summary['generated_at_utc']}`.",
        "",
        "This is an isolated five-asset 5m challenger. It does not alter the live dashboard.",
        "Every displayed-retrieval candidate remains a real historical continuation; the neural network only learns retrieval relevance.",
        "",
        "## Population and split",
        "",
        f"- Fit snapshots: **{summary['data']['fit_rows']}** from **{summary['data']['fit_episodes']}** episodes.",
        f"- Validation snapshots: **{summary['data']['validation_rows']}** from **{summary['data']['validation_episodes']}** episodes.",
        f"- Untouched holdout snapshots: **{summary['data']['holdout_rows']}** from **{summary['data']['holdout_episodes']}** episodes.",
        "- Both chronological boundaries require every training label to have resolved before the next split starts.",
        "- Successive snapshots are episode-weighted so long episodes do not dominate merely by contributing more rows.",
        "",
        "## Holdout results",
        "",
        "| Model | Remaining p50 MAE (bars) | Excursion p50 MAE (pp) | Path MAE (pp) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, metrics in summary["holdout_metrics"].items():
        path_value = metrics.get("path", {}).get(
            "mean_absolute_distance_error_pct_points"
        )
        path_text = "—" if path_value is None else f"{path_value:.4f}"
        lines.append(
            f"| {name} | {metrics['remaining_bars']['p50_median_absolute_error']:.2f} | "
            f"{metrics['future_max_away_pct']['p50_median_absolute_error']:.4f} | {path_text} |"
        )
    lines.extend(
        [
            "",
            "Coverage is reported in the JSON together with interval width; coverage alone is not evidence of a sharper forecast.",
            "The neural direct path is diagnostic only. Candidate routes for product use come from the neural or hybrid real-history retrievers.",
            "",
            "## Scope boundary",
            "",
            "This first experiment remains conditional on clean completed fills because the current path library does not export equivalent unresolved histories.",
            "It must not be interpreted as an unconditional fill probability or a position-sizing rule.",
            "",
        ]
    )
    return "\n".join(lines)


def scale_samples(
    fit: SampleArrays,
    validation: SampleArrays,
    holdout: SampleArrays,
) -> tuple[SampleArrays, SampleArrays, SampleArrays, dict[str, Any]]:
    pre_scaler = fit_sequence_scaler(fit.pre_signal)
    recent_scaler = fit_sequence_scaler(fit.recent)
    episode_scaler = fit_sequence_scaler(fit.episode)
    static_scaler = fit_static_scaler(fit.static)

    def transform(samples: SampleArrays) -> SampleArrays:
        return SampleArrays(
            pre_signal=apply_sequence_scaler(samples.pre_signal, pre_scaler),
            recent=apply_sequence_scaler(samples.recent, recent_scaler),
            episode=apply_sequence_scaler(samples.episode, episode_scaler),
            static=apply_static_scaler(samples.static, static_scaler),
            log_remaining=samples.log_remaining,
            log_excursion=samples.log_excursion,
            curve_log_ratio=samples.curve_log_ratio,
            direction_targets=samples.direction_targets,
            current_move_pct=samples.current_move_pct,
            weights=samples.weights,
            metadata=samples.metadata,
        )

    serialized = {
        "pre_signal": {
            "mean": pre_scaler.mean.tolist(),
            "scale": pre_scaler.scale.tolist(),
        },
        "recent": {
            "mean": recent_scaler.mean.tolist(),
            "scale": recent_scaler.scale.tolist(),
        },
        "episode": {
            "mean": episode_scaler.mean.tolist(),
            "scale": episode_scaler.scale.tolist(),
        },
        "static": {
            "center": static_scaler.center.tolist(),
            "scale": static_scaler.scale.tolist(),
        },
    }
    return transform(fit), transform(validation), transform(holdout), serialized


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--library-dir", type=Path, default=default_library_dir(project_root)
    )
    parser.add_argument("--data-dir", type=Path, default=project_root / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "neural_path_v3",
    )
    parser.add_argument(
        "--snapshot-offsets", default=",".join(str(value) for value in DEFAULT_OFFSETS)
    )
    parser.add_argument("--train-cap", type=int, default=18_000)
    parser.add_argument("--validation-cap", type=int, default=2_000)
    parser.add_argument("--holdout-cap", type=int, default=1_200)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_output_dir = Path(__file__).resolve().parent / "data" / "neural_path_v3"
    output_dir = (
        default_output_dir.with_name("neural_path_v3_smoke")
        if args.smoke and args.output_dir.resolve() == default_output_dir.resolve()
        else args.output_dir
    )
    config = ExperimentConfig(
        train_cap=2_000 if args.smoke else args.train_cap,
        validation_cap=400 if args.smoke else args.validation_cap,
        holdout_cap=400 if args.smoke else args.holdout_cap,
        epochs=2 if args.smoke else args.epochs,
        batch_size=args.batch_size,
        patience=2 if args.smoke else 3,
    )
    if (
        min(
            config.train_cap,
            config.validation_cap,
            config.holdout_cap,
            config.epochs,
            config.batch_size,
        )
        < 1
    ):
        raise ValueError("row caps, epochs, and batch size must be positive")
    seed_everything(config.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(
        "cuda"
        if (
            args.device == "cuda"
            or (args.device == "auto" and torch.cuda.is_available())
        )
        else "cpu"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading completed 5m episode library", flush=True)
    events = load_events(args.library_dir)
    paths = load_paths(args.library_dir, events)
    snapshots, snapshot_counts = build_snapshot_dataset(
        events, paths, parse_offsets(args.snapshot_offsets)
    )
    outer_train, holdout_pool, outer_split = chronological_split(
        snapshots,
        events,
        holdout_months=config.holdout_months,
        embargo_bars=config.embargo_bars,
    )
    fit_pool, validation_pool, inner_split = inner_chronological_split(
        outer_train,
        validation_fraction=config.validation_fraction,
        embargo_bars=config.embargo_bars,
    )
    fit_frame = evenly_spaced(fit_pool, config.train_cap)
    validation_frame = evenly_spaced(validation_pool, config.validation_cap)
    holdout_frame = evenly_spaced(holdout_pool, config.holdout_cap)

    required_assets = sorted(
        set(fit_frame["asset"].astype(str))
        | set(validation_frame["asset"].astype(str))
        | set(holdout_frame["asset"].astype(str))
    )
    raw_by_asset = {
        asset: load_raw_candles(args.data_dir, asset) for asset in required_assets
    }
    selected_ids = set(
        pd.concat(
            (
                fit_frame["episode_id"],
                validation_frame["episode_id"],
                holdout_frame["episode_id"],
            )
        )
        .astype(str)
        .tolist()
    )
    selected_paths = paths.loc[
        paths["episode_id"].astype(str).isin(selected_ids)
    ].copy()
    del paths

    fit = materialize_samples(fit_frame, events, selected_paths, raw_by_asset, "fit")
    validation = materialize_samples(
        validation_frame, events, selected_paths, raw_by_asset, "validation"
    )
    holdout = materialize_samples(
        holdout_frame, events, selected_paths, raw_by_asset, "holdout"
    )
    raw_fit_for_tree = fit
    raw_holdout_for_tree = holdout
    fit, validation, holdout, scalers = scale_samples(fit, validation, holdout)

    print(f"training neural model on {device}", flush=True)
    model, history, best_epoch = train_model(fit, validation, config, device)
    fit_predictions = predict_model(model, fit, config.batch_size, device)
    holdout_predictions = predict_model(model, holdout, config.batch_size, device)

    print("training richer-feature quantile tree baseline", flush=True)
    tree_predictions = fit_tree_quantiles(
        raw_fit_for_tree, raw_holdout_for_tree, config.seed
    )
    retrievals = {
        f"{mode}_real_path_retrieval": retrieve_predictions(
            fit,
            holdout,
            raw_fit_for_tree.static,
            raw_holdout_for_tree.static,
            fit_predictions["embedding"],
            holdout_predictions["embedding"],
            holdout_predictions,
            config.retrieval_neighbors,
            config.candidate_pool,
            mode,
        )
        for mode in ("scalar", "neural", "hybrid", "forecast", "forecast_hybrid")
    }
    metrics = evaluate_predictions(
        holdout, holdout_predictions, tree_predictions, retrievals
    )
    groups = subgroup_metrics(
        holdout, holdout_predictions, tree_predictions, retrievals
    )
    paired_comparisons = comparison_vs_scalar(
        holdout,
        holdout_predictions,
        retrievals,
        config.seed,
    )

    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "purpose": "learn predictive prefix structure while preserving real historical continuations",
        "status": "research challenger; not connected to the live dashboard",
        "conditional_population": "strict five-asset 5m clean-fill episodes only",
        "config": asdict(config),
        "environment": {
            "python": sys.version.split()[0],
            "torch": package_version_or_unknown("torch"),
            "scikit_learn": package_version_or_unknown("scikit-learn"),
            "device": str(device),
        },
        "channels": {
            "pre_signal": list(RAW_SEQUENCE_CHANNELS),
            "recent": list(RAW_SEQUENCE_CHANNELS),
            "episode": list(EPISODE_SEQUENCE_CHANNELS),
            "static": list(STATIC_COLUMNS),
            "future_horizon_bars": FUTURE_HORIZON_BARS.tolist(),
        },
        "data": {
            **snapshot_counts,
            "fit_rows": len(fit_frame),
            "fit_episodes": int(fit_frame["episode_id"].nunique()),
            "validation_rows": len(validation_frame),
            "validation_episodes": int(validation_frame["episode_id"].nunique()),
            "holdout_rows": len(holdout_frame),
            "holdout_episodes": int(holdout_frame["episode_id"].nunique()),
            "assets": required_assets,
        },
        "outer_chronological_split": outer_split,
        "inner_chronological_split": inner_split,
        "training": {"best_epoch": best_epoch, "history": history},
        "holdout_metrics": metrics,
        "subgroup_holdout_metrics": groups,
        "episode_clustered_comparison_vs_scalar": paired_comparisons,
        "limitations": [
            "completed clean fills only; this is not an unconditional fill probability",
            "fixed sampled snapshot offsets rather than every possible alignment",
            "one chronological holdout period; promotion requires repeated walk-forward folds",
            "neural direct paths are diagnostics and are not dashboard display candidates",
        ],
    }

    checkpoint = {
        "schema_version": SCHEMA_VERSION,
        "config": asdict(config),
        "model_state_dict": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "scalers": scalers,
        "channels": summary["channels"],
    }
    torch.save(checkpoint, output_dir / "neural_path_v3_model.pt")
    np.savez_compressed(
        output_dir / "neural_path_v3_retrieval_index.npz",
        embedding=fit_predictions["embedding"].astype(np.float16),
        curve_log_ratio=fit.curve_log_ratio.astype(np.float16),
        log_remaining=fit.log_remaining.astype(np.float32),
        log_excursion=fit.log_excursion.astype(np.float32),
        episode_id=fit.metadata["episode_id"].astype(str).to_numpy(),
        offset_bars=fit.metadata["offset_bars"].to_numpy(dtype=np.int32),
    )
    np.savez_compressed(
        output_dir / "neural_path_v3_holdout_arrays.npz",
        actual_log_remaining=holdout.log_remaining.astype(np.float32),
        actual_log_excursion=holdout.log_excursion.astype(np.float32),
        actual_curve_log_ratio=holdout.curve_log_ratio.astype(np.float32),
        current_move_pct=holdout.current_move_pct.astype(np.float32),
        neural_direct_risk=holdout_predictions["risk"].astype(np.float32),
        neural_direct_curve=holdout_predictions["curve"].astype(np.float32),
        scalar_remaining=retrievals["scalar_real_path_retrieval"]["remaining"].astype(
            np.float32
        ),
        scalar_excursion=retrievals["scalar_real_path_retrieval"]["excursion"].astype(
            np.float32
        ),
        scalar_curve=retrievals["scalar_real_path_retrieval"]["curve"].astype(
            np.float32
        ),
        neural_remaining=retrievals["neural_real_path_retrieval"]["remaining"].astype(
            np.float32
        ),
        neural_excursion=retrievals["neural_real_path_retrieval"]["excursion"].astype(
            np.float32
        ),
        neural_curve=retrievals["neural_real_path_retrieval"]["curve"].astype(
            np.float32
        ),
        hybrid_remaining=retrievals["hybrid_real_path_retrieval"]["remaining"].astype(
            np.float32
        ),
        hybrid_excursion=retrievals["hybrid_real_path_retrieval"]["excursion"].astype(
            np.float32
        ),
        hybrid_curve=retrievals["hybrid_real_path_retrieval"]["curve"].astype(
            np.float32
        ),
        forecast_remaining=retrievals["forecast_real_path_retrieval"][
            "remaining"
        ].astype(np.float32),
        forecast_excursion=retrievals["forecast_real_path_retrieval"][
            "excursion"
        ].astype(np.float32),
        forecast_curve=retrievals["forecast_real_path_retrieval"]["curve"].astype(
            np.float32
        ),
        forecast_hybrid_remaining=retrievals["forecast_hybrid_real_path_retrieval"][
            "remaining"
        ].astype(np.float32),
        forecast_hybrid_excursion=retrievals["forecast_hybrid_real_path_retrieval"][
            "excursion"
        ].astype(np.float32),
        forecast_hybrid_curve=retrievals["forecast_hybrid_real_path_retrieval"][
            "curve"
        ].astype(np.float32),
        episode_id=holdout.metadata["episode_id"].astype(str).to_numpy(),
        asset=holdout.metadata["asset"].astype(str).to_numpy(),
        direction=holdout.metadata["direction"].astype(str).to_numpy(),
    )
    prediction_rows = holdout.metadata.copy()
    prediction_rows["actual_remaining_bars"] = np.expm1(holdout.log_remaining)
    prediction_rows["actual_future_max_away_pct"] = np.expm1(holdout.log_excursion)
    for model_name, predictions in {
        "tree": tree_predictions,
        **{
            name.replace("_real_path_retrieval", ""): value
            for name, value in retrievals.items()
        },
    }.items():
        prediction_rows[f"{model_name}_remaining_p50"] = np.expm1(
            predictions["remaining"][:, 1]
        )
        prediction_rows[f"{model_name}_excursion_p50"] = np.expm1(
            predictions["excursion"][:, 1]
        )
    prediction_rows.to_csv(
        output_dir / "neural_path_v3_holdout_predictions.csv", index=False
    )
    atomic_write_json(output_dir / "neural_path_v3_summary.json", summary)
    atomic_write_text(output_dir / "NEURAL_PATH_V3_REPORT.md", markdown_report(summary))
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "best_epoch": best_epoch,
                "metrics": metrics,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
