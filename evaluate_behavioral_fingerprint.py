"""Evaluate deterministic behavioral fingerprints for historical wick retrieval.

This proof of concept compares three observable-only similarity definitions:

* state_geometry: signal candle geometry plus the live wick state;
* context_summary: state/geometry plus pre-signal and recent context summaries;
* sequence_shape: both prior blocks plus mirrored, multi-resolution price paths.

Historical future outcomes are labels only. They never enter a query fingerprint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from build_conditional_path_library import FEATURE_COLUMNS, read_five_minute_file
from evaluate_prospective_entry_baseline import (
    chronological_signal_split,
    read_observations,
    stratified_sample,
)
from train_prospective_entry_model import build_feature_matrix, raw_path


HORIZONS_MINUTES = (1_440, 10_080, 43_200)
BASE_COLUMNS = (
    *FEATURE_COLUMNS,
    "log_entry_age_minutes",
    "log_entry_distance_pct",
    "log_peak_distance_pct",
    "drawdown_from_peak_pct",
    "log_departure_to_entry_bars",
    "entry_bar_range_pct",
    "entry_bar_body_pct",
)
CONTEXT_COLUMNS = (
    "signal_to_entry_aligned_return_pct",
    "pre_60m_aligned_return_pct",
    "pre_60m_realized_vol_pct",
    "pre_60m_mean_range_pct",
    "pre_60m_volume_ratio",
    "pre_240m_aligned_return_pct",
    "pre_240m_realized_vol_pct",
    "pre_240m_mean_range_pct",
    "pre_240m_volume_ratio",
    "pre_480m_aligned_return_pct",
    "pre_480m_realized_vol_pct",
    "pre_480m_mean_range_pct",
    "pre_480m_volume_ratio",
    "recent_15m_aligned_return_pct",
    "recent_15m_realized_vol_pct",
    "recent_15m_mean_range_pct",
    "recent_15m_volume_ratio",
    "recent_60m_aligned_return_pct",
    "recent_60m_realized_vol_pct",
    "recent_60m_mean_range_pct",
    "recent_60m_volume_ratio",
    "recent_240m_aligned_return_pct",
    "recent_240m_realized_vol_pct",
    "recent_240m_mean_range_pct",
    "recent_240m_volume_ratio",
    "recent_1280m_aligned_return_pct",
    "recent_1280m_realized_vol_pct",
    "recent_1280m_mean_range_pct",
    "recent_1280m_volume_ratio",
)
SEQUENCE_VIEWS = (
    ("pre", "pre_signal_start_index", "pre_signal_end_index", 12, "last"),
    ("recent", "recent_start_index", "recent_end_index", 12, "last"),
    ("episode", "signal_to_entry_start_index", "signal_to_entry_end_index", 16, "first"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def evenly_spaced_indices(start: int, end: int, points: int, length: int) -> np.ndarray:
    start = max(0, min(int(start), length - 1))
    end = max(start, min(int(end), length - 1))
    return np.rint(np.linspace(start, end, points)).astype(np.int64)


def mirrored_distance_pct(close: np.ndarray, target: float, direction_sign: int) -> np.ndarray:
    return float(direction_sign) * (close / max(float(target), 1e-12) - 1.0) * 100.0


def sequence_fingerprint_from_close(
    close: np.ndarray,
    observation: Mapping[str, Any],
) -> np.ndarray:
    """Create a fixed-width mirrored path fingerprint from observable candles only."""
    target = float(observation["wick_target"])
    direction_sign = int(observation["direction_sign"])
    blocks: list[np.ndarray] = []
    for _, start_field, end_field, points, anchor in SEQUENCE_VIEWS:
        indices = evenly_spaced_indices(
            int(observation[start_field]),
            int(observation[end_field]),
            points,
            len(close),
        )
        values = mirrored_distance_pct(close[indices], target, direction_sign)
        values = values - (values[-1] if anchor == "last" else values[0])
        blocks.append(values)
    return np.concatenate(blocks).astype(np.float32)


def sequence_column_names() -> tuple[str, ...]:
    columns: list[str] = []
    for name, _, _, points, _ in SEQUENCE_VIEWS:
        columns.extend(f"sequence_{name}_{index:02d}" for index in range(points))
    return tuple(columns)


def build_sequence_matrix(root: Path, observations: pd.DataFrame) -> pd.DataFrame:
    columns = sequence_column_names()
    matrix = np.empty((len(observations), len(columns)), dtype=np.float32)
    for asset, partition in observations.groupby("asset", sort=True, observed=True):
        frame = read_five_minute_file(raw_path(root, str(asset), "5m"))
        close = frame["close"].to_numpy(dtype=np.float64)
        for row_index, row in partition.iterrows():
            matrix[int(row_index)] = sequence_fingerprint_from_close(close, row)
    return pd.DataFrame(matrix, index=observations.index, columns=columns)


def robust_block_transform(
    fit: pd.DataFrame,
    query: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, dict[str, list[float]]]:
    fit_values = fit.to_numpy(dtype=np.float64)
    query_values = query.to_numpy(dtype=np.float64)
    median = np.nanmedian(fit_values, axis=0)
    absolute = np.abs(fit_values - median)
    scale = np.nanmedian(absolute, axis=0) * 1.4826
    standard_deviation = np.nanstd(fit_values, axis=0)
    scale = np.where(scale > 1e-8, scale, np.where(standard_deviation > 1e-8, standard_deviation, 1.0))
    fit_values = np.nan_to_num((fit_values - median) / scale, nan=0.0, posinf=8.0, neginf=-8.0)
    query_values = np.nan_to_num((query_values - median) / scale, nan=0.0, posinf=8.0, neginf=-8.0)
    fit_values = np.clip(fit_values, -8.0, 8.0) / math.sqrt(max(fit_values.shape[1], 1))
    query_values = np.clip(query_values, -8.0, 8.0) / math.sqrt(max(query_values.shape[1], 1))
    return (
        fit_values.astype(np.float32),
        query_values.astype(np.float32),
        {"median": median.tolist(), "scale": scale.tolist()},
    )


def fingerprint_matrices(
    blocks: Iterable[tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[np.ndarray, np.ndarray]:
    fit_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    for fit, query in blocks:
        fit_values, query_values, _ = robust_block_transform(fit, query)
        fit_parts.append(fit_values)
        query_parts.append(query_values)
    return np.concatenate(fit_parts, axis=1), np.concatenate(query_parts, axis=1)


def unique_episode_neighbors(
    neighbor_positions: np.ndarray,
    candidate_signal_ids: np.ndarray,
    neighbors: int,
) -> np.ndarray:
    selected: list[int] = []
    seen: set[str] = set()
    for position in neighbor_positions:
        signal_id = str(candidate_signal_ids[int(position)])
        if signal_id in seen:
            continue
        seen.add(signal_id)
        selected.append(int(position))
        if len(selected) == neighbors:
            break
    return np.asarray(selected, dtype=np.int64)


def retrieve_neighbors(
    candidate_matrix: np.ndarray,
    query_matrix: np.ndarray,
    candidate_signal_ids: np.ndarray,
    neighbors: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    search_neighbors = min(len(candidate_matrix), max(neighbors * 8, 128))
    index = NearestNeighbors(
        n_neighbors=search_neighbors,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    )
    index.fit(candidate_matrix)
    distances, positions = index.kneighbors(query_matrix)
    selected: list[np.ndarray] = []
    nearest_distances = np.empty(len(query_matrix), dtype=np.float64)
    for query_index, row in enumerate(positions):
        chosen = unique_episode_neighbors(row, candidate_signal_ids, neighbors)
        if len(chosen) < neighbors:
            raise RuntimeError(
                f"Only {len(chosen)} distinct historical episodes were available; {neighbors} required"
            )
        selected.append(chosen)
        nearest_distances[query_index] = float(distances[query_index, 0])
    return selected, nearest_distances


def evaluate_horizon(
    candidates: pd.DataFrame,
    queries: pd.DataFrame,
    selected_neighbors: list[np.ndarray],
    horizon_minutes: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    slug = f"{horizon_minutes}m"
    fill_column = f"target_hit_{slug}"
    adverse_column = f"max_adverse_pre_target_upper_{slug}_pct"
    candidate_fill = candidates[fill_column].to_numpy(dtype=bool)
    candidate_adverse = candidates[adverse_column].to_numpy(dtype=float)
    candidate_wait = (
        candidates["target_touch_bars_from_entry"].to_numpy(dtype=float) * 5.0
    )
    actual_fill = queries[fill_column].to_numpy(dtype=bool)
    actual_adverse = queries[adverse_column].to_numpy(dtype=float)
    actual_wait = queries["target_touch_bars_from_entry"].to_numpy(dtype=float) * 5.0
    predicted_fill = np.empty(len(queries), dtype=float)
    predicted_adverse = np.empty(len(queries), dtype=float)
    predicted_wait = np.empty(len(queries), dtype=float)
    for index, chosen in enumerate(selected_neighbors):
        fills = candidate_fill[chosen]
        predicted_fill[index] = float(np.mean(fills))
        predicted_adverse[index] = float(np.median(candidate_adverse[chosen]))
        filled_waits = candidate_wait[chosen][fills & np.isfinite(candidate_wait[chosen])]
        predicted_wait[index] = (
            float(np.median(filled_waits)) if len(filled_waits) else float(horizon_minutes)
        )
    valid_adverse = np.isfinite(actual_adverse) & np.isfinite(predicted_adverse)
    valid_wait = actual_fill & np.isfinite(actual_wait) & np.isfinite(predicted_wait)
    rows = pd.DataFrame(
        {
            "signal_id": queries["signal_id"].to_numpy(),
            "asset": queries["asset"].to_numpy(),
            "direction": queries["direction"].to_numpy(),
            "entry_age_minutes": queries["entry_age_minutes"].to_numpy(),
            "actual_fill": actual_fill.astype(float),
            "predicted_fill": predicted_fill,
            "actual_adverse_pct": actual_adverse,
            "predicted_adverse_pct": predicted_adverse,
            "actual_wait_minutes": actual_wait,
            "predicted_wait_minutes": predicted_wait,
        },
        index=queries.index,
    )
    metrics: dict[str, float | int] = {
        "queries": int(len(queries)),
        "filled_queries": int(np.sum(actual_fill)),
        "fill_brier": float(np.mean((predicted_fill - actual_fill.astype(float)) ** 2)),
        "adverse_p50_mae_pct": float(
            np.mean(np.abs(predicted_adverse[valid_adverse] - actual_adverse[valid_adverse]))
        ),
        "wait_p50_mae_minutes": float(
            np.mean(np.abs(predicted_wait[valid_wait] - actual_wait[valid_wait]))
        ),
    }
    return metrics, rows


def relative_improvement(baseline: float, challenger: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) < 1e-12:
        return float("nan")
    return (baseline - challenger) / baseline * 100.0


def episode_balanced_comparison(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    *,
    seed: int = 20260927,
    bootstrap_samples: int = 2_000,
) -> dict[str, Any]:
    """Compare paired query errors after giving each signal episode equal weight."""
    if not baseline.index.equals(challenger.index):
        raise ValueError("Paired fingerprint predictions must use identical query rows")
    errors = {
        "fill_brier": (
            (baseline["predicted_fill"] - baseline["actual_fill"]) ** 2,
            (challenger["predicted_fill"] - challenger["actual_fill"]) ** 2,
        ),
        "adverse_p50_absolute_error_pct": (
            np.abs(baseline["predicted_adverse_pct"] - baseline["actual_adverse_pct"]),
            np.abs(challenger["predicted_adverse_pct"] - challenger["actual_adverse_pct"]),
        ),
        "wait_p50_absolute_error_minutes": (
            np.abs(baseline["predicted_wait_minutes"] - baseline["actual_wait_minutes"]).where(
                baseline["actual_fill"].astype(bool)
            ),
            np.abs(
                challenger["predicted_wait_minutes"] - challenger["actual_wait_minutes"]
            ).where(challenger["actual_fill"].astype(bool)),
        ),
    }
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    for metric, (baseline_error, challenger_error) in errors.items():
        paired = pd.DataFrame(
            {
                "signal_id": baseline["signal_id"],
                "baseline": baseline_error,
                "challenger": challenger_error,
            }
        ).dropna()
        episodes = paired.groupby("signal_id", sort=True)[["baseline", "challenger"]].mean()
        delta = episodes["challenger"].to_numpy(dtype=float) - episodes["baseline"].to_numpy(
            dtype=float
        )
        if len(delta) == 0:
            raise RuntimeError(f"No paired episodes are available for {metric}")
        sample_positions = rng.integers(0, len(delta), size=(bootstrap_samples, len(delta)))
        bootstrap_delta = delta[sample_positions].mean(axis=1)
        baseline_mean = float(episodes["baseline"].mean())
        challenger_mean = float(episodes["challenger"].mean())
        output[metric] = {
            "episodes": int(len(episodes)),
            "baseline_error": baseline_mean,
            "challenger_error": challenger_mean,
            "relative_improvement_pct": relative_improvement(baseline_mean, challenger_mean),
            "delta_challenger_minus_baseline": challenger_mean - baseline_mean,
            "bootstrap_95pct_low": float(np.quantile(bootstrap_delta, 0.025)),
            "bootstrap_95pct_high": float(np.quantile(bootstrap_delta, 0.975)),
            "probability_challenger_better": float(np.mean(bootstrap_delta < 0.0)),
        }
    return output


def evaluate(root: Path, dataset_dir: Path, max_queries: int, neighbors: int) -> dict[str, Any]:
    observations, metadata = read_observations(dataset_dir, "5m")
    observations = observations.reset_index(drop=True)
    features = build_feature_matrix(root, observations)
    sequences = build_sequence_matrix(root, observations)
    fit, validation, holdout, boundaries = chronological_signal_split(observations)
    maximum_horizon = max(HORIZONS_MINUTES)
    full_column = f"horizon_{maximum_horizon}m_fully_observed"
    eligible_history = pd.concat([fit, validation]).sort_index()
    holdout_start_ms = int(holdout["entry_close_time_ms"].min())
    label_available_ms = (
        eligible_history["entry_close_time_ms"].to_numpy(dtype=np.int64)
        + maximum_horizon * 60_000
    )
    eligible_history = eligible_history.loc[
        eligible_history[full_column].astype(bool).to_numpy()
        & (label_available_ms <= holdout_start_ms)
    ].copy()
    eligible_holdout = holdout.loc[holdout[full_column].astype(bool)].copy()
    queries = stratified_sample(eligible_holdout, max_queries).sort_index()
    if len(eligible_history) < neighbors * 4:
        raise RuntimeError("Insufficient label-mature historical observations for fingerprint evaluation")
    if queries.empty:
        raise RuntimeError("No fully observed holdout queries are available")

    levels = {
        "state_geometry": (BASE_COLUMNS,),
        "context_summary": (BASE_COLUMNS, CONTEXT_COLUMNS),
        "sequence_shape": (BASE_COLUMNS, CONTEXT_COLUMNS, sequence_column_names()),
    }
    level_results: dict[str, Any] = {}
    prediction_rows: dict[tuple[str, int], pd.DataFrame] = {}
    for level, block_columns in levels.items():
        blocks: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for columns in block_columns:
            source = sequences if str(columns[0]).startswith("sequence_") else features
            blocks.append(
                (
                    source.loc[eligible_history.index, list(columns)],
                    source.loc[queries.index, list(columns)],
                )
            )
        candidate_matrix, query_matrix = fingerprint_matrices(blocks)
        selected, nearest_distances = retrieve_neighbors(
            candidate_matrix,
            query_matrix,
            eligible_history["signal_id"].to_numpy(),
            neighbors,
        )
        horizon_results: dict[str, Any] = {}
        for horizon in HORIZONS_MINUTES:
            metrics, rows = evaluate_horizon(eligible_history, queries, selected, horizon)
            horizon_results[str(horizon)] = metrics
            prediction_rows[(level, horizon)] = rows
        level_results[level] = {
            "dimensions": int(candidate_matrix.shape[1]),
            "blocks": [list(columns) for columns in block_columns],
            "nearest_distance": {
                "p10": float(np.quantile(nearest_distances, 0.10)),
                "p50": float(np.quantile(nearest_distances, 0.50)),
                "p90": float(np.quantile(nearest_distances, 0.90)),
            },
            "horizons": horizon_results,
        }

    comparisons: dict[str, Any] = {}
    episode_comparisons: dict[str, Any] = {}
    baseline = level_results["state_geometry"]
    for level in ("context_summary", "sequence_shape"):
        by_horizon: dict[str, Any] = {}
        episode_by_horizon: dict[str, Any] = {}
        for horizon in HORIZONS_MINUTES:
            slug = str(horizon)
            base_metrics = baseline["horizons"][slug]
            metrics = level_results[level]["horizons"][slug]
            by_horizon[slug] = {
                metric: relative_improvement(float(base_metrics[metric]), float(metrics[metric]))
                for metric in ("fill_brier", "adverse_p50_mae_pct", "wait_p50_mae_minutes")
            }
            episode_by_horizon[slug] = episode_balanced_comparison(
                prediction_rows[("state_geometry", horizon)],
                prediction_rows[(level, horizon)],
                seed=20260927 + horizon,
            )
        comparisons[level] = by_horizon
        episode_comparisons[level] = episode_by_horizon

    sequence_rows = prediction_rows[("sequence_shape", 10_080)].copy()
    baseline_rows = prediction_rows[("state_geometry", 10_080)]
    sequence_rows["baseline_fill_squared_error"] = (
        baseline_rows["predicted_fill"] - baseline_rows["actual_fill"]
    ) ** 2
    sequence_rows["sequence_fill_squared_error"] = (
        sequence_rows["predicted_fill"] - sequence_rows["actual_fill"]
    ) ** 2
    sequence_rows["baseline_adverse_absolute_error"] = np.abs(
        baseline_rows["predicted_adverse_pct"] - baseline_rows["actual_adverse_pct"]
    )
    sequence_rows["sequence_adverse_absolute_error"] = np.abs(
        sequence_rows["predicted_adverse_pct"] - sequence_rows["actual_adverse_pct"]
    )
    asset_comparison = {
        str(asset): {
            "queries": int(len(group)),
            "fill_brier_improvement_pct": relative_improvement(
                float(group["baseline_fill_squared_error"].mean()),
                float(group["sequence_fill_squared_error"].mean()),
            ),
            "adverse_mae_improvement_pct": relative_improvement(
                float(group["baseline_adverse_absolute_error"].mean()),
                float(group["sequence_adverse_absolute_error"].mean()),
            ),
        }
        for asset, group in sequence_rows.groupby("asset", observed=True, sort=True)
    }

    return {
        "schema_version": "behavioral-fingerprint-poc-v1.0.0",
        "generated_at_utc": utc_now(),
        "status": "research_only_not_promoted",
        "timeframe": "5m",
        "assets": sorted(observations["asset"].astype(str).unique().tolist()),
        "population": {
            "all_observations": int(len(observations)),
            "label_mature_history": int(len(eligible_history)),
            "fully_observed_holdout": int(len(eligible_holdout)),
            "evaluated_queries": int(len(queries)),
            "distinct_history_episodes": int(eligible_history["signal_id"].nunique()),
            "distinct_query_episodes": int(queries["signal_id"].nunique()),
            "neighbors_per_query": int(neighbors),
        },
        "chronological_split": boundaries,
        "label_availability": (
            "Every candidate's full 30-day outcome window ended before the first holdout entry close"
        ),
        "direction_policy": "upper and lower wick paths are mirrored into the same coordinates",
        "future_data_policy": "future candles are labels only and never enter a fingerprint",
        "horizons_minutes": list(HORIZONS_MINUTES),
        "levels": level_results,
        "improvement_vs_state_geometry_pct": comparisons,
        "episode_balanced_comparison_vs_state_geometry": episode_comparisons,
        "sequence_shape_7d_asset_comparison": asset_comparison,
        "dataset_schema_version": metadata.get("schema_version"),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/prospective_entry_outcomes_v1"),
    )
    parser.add_argument("--max-queries", type=int, default=1_500)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/behavioral_fingerprint_poc_5m.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else root / args.dataset_dir
    output = args.output if args.output.is_absolute() else root / args.output
    report = evaluate(root, dataset_dir, args.max_queries, args.neighbors)
    atomic_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "population": report["population"],
                "improvement_vs_state_geometry_pct": report[
                    "improvement_vs_state_geometry_pct"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
