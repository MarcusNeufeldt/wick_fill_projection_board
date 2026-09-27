#!/usr/bin/env python3
"""Leakage-safe rolling comparison of numerical and historical-retrieval forecasts.

Systems A-C share identical folds, queries, targets, and episode-balanced metrics:

* A: the existing 53-feature supervised ExtraTrees/HGB model;
* B: the same supervised model with 40 observable sequence coordinates;
* C: horizon-specific handcrafted behavioral-fingerprint retrieval.

System D (V3 learned retrieval) is deliberately rejected unless it is retrained
inside each fold.  A global V3 artifact is not a valid rolling-fold input.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from dynamic_neighborhood import (
    NeighborhoodConfig,
    candidate_neighborhood_configs,
    config_record,
    neighborhood_support,
    radius_threshold,
    resolve_with_nearest_fallback,
    retrieve_episode_neighbor_pool,
    weighted_quantile,
)
from evaluate_behavioral_fingerprint import (
    HORIZONS_MINUTES,
    atomic_json,
    build_sequence_matrix,
    retrieve_neighbors,
    sequence_column_names,
    utc_now,
)
from evaluate_prospective_entry_baseline import read_observations, stratified_sample
from optimize_behavioral_fingerprint import (
    FingerprintConfig,
    candidate_configs,
    compose_matrices,
    scaled_blocks,
)
from prospective_entry_outcomes import horizon_slug
from train_prospective_entry_model import (
    CONFIGS,
    ForestConfig,
    build_feature_matrix,
    class_probability_maps,
    fit_bundle,
)


MAXIMUM_HORIZON_MINUTES = max(HORIZONS_MINUTES)
MAXIMUM_HORIZON_MS = MAXIMUM_HORIZON_MINUTES * 60_000
METRIC_COLUMNS = (
    "fill_brier",
    "adverse_p50_absolute_error_pct",
    "wait_p50_absolute_error_minutes",
)


@dataclass(frozen=True)
class RollingFold:
    number: int
    train_end_signal_ms: int
    validation_end_signal_ms: int
    test_end_signal_ms: int


def rolling_signal_folds(
    observations: pd.DataFrame,
    folds: int,
    initial_train_fraction: float,
) -> list[RollingFold]:
    """Create expanding, non-overlapping validation/test signal windows."""
    signals = np.sort(observations["signal_open_time_ms"].unique()).astype(np.int64)
    if folds < 1:
        raise ValueError("folds must be positive")
    if not 0.20 <= initial_train_fraction < 0.80:
        raise ValueError("initial_train_fraction must be in [0.20, 0.80)")
    initial = max(20, int(len(signals) * initial_train_fraction))
    window = (len(signals) - initial) // (folds * 2)
    if window < 2:
        raise RuntimeError("Not enough distinct signals for the requested rolling folds")

    result: list[RollingFold] = []
    for number in range(folds):
        train_end = initial + number * 2 * window
        validation_end = train_end + window
        test_end = len(signals) if number == folds - 1 else validation_end + window
        result.append(
            RollingFold(
                number=number + 1,
                train_end_signal_ms=int(signals[train_end]),
                validation_end_signal_ms=int(signals[validation_end]),
                test_end_signal_ms=(
                    int(signals[-1]) + 1 if test_end == len(signals) else int(signals[test_end])
                ),
            )
        )
    return result


def signal_window(
    observations: pd.DataFrame,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    return observations.loc[
        observations["signal_open_time_ms"].ge(start_ms)
        & observations["signal_open_time_ms"].lt(end_ms)
    ].copy()


def eligible_history_and_queries(
    history: pd.DataFrame,
    query_window: pd.DataFrame,
    maximum_queries: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Apply query maturity and strict label-availability at the query cutoff."""
    fully_observed = f"horizon_{MAXIMUM_HORIZON_MINUTES}m_fully_observed"
    mature_queries = query_window.loc[query_window[fully_observed].astype(bool)].copy()
    if mature_queries.empty:
        raise RuntimeError("No fully observed queries are available in this fold window")
    queries = stratified_sample(mature_queries, maximum_queries).sort_index()
    first_query_close_ms = int(queries["entry_close_time_ms"].min())
    label_available_ms = (
        history["entry_close_time_ms"].to_numpy(dtype=np.int64) + MAXIMUM_HORIZON_MS
    )
    eligible = history.loc[
        history[fully_observed].astype(bool).to_numpy()
        & (label_available_ms <= first_query_close_ms)
    ].copy()
    if eligible.empty:
        raise RuntimeError("No label-mature history is available before the fold cutoff")
    return eligible.sort_index(), queries, {
        "first_query_close_ms": first_query_close_ms,
        "latest_history_label_available_ms": int(
            (eligible["entry_close_time_ms"] + MAXIMUM_HORIZON_MS).max()
        ),
    }


def common_prediction_rows(
    queries: pd.DataFrame,
    predicted_fill: np.ndarray,
    predicted_adverse: np.ndarray,
    predicted_wait: np.ndarray,
    horizon_minutes: int,
    extra: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    slug = horizon_slug(horizon_minutes)
    actual_fill = queries[f"target_hit_{slug}"].astype(bool).to_numpy()
    lower = queries[f"max_adverse_pre_target_lower_{slug}_pct"].to_numpy(dtype=float)
    upper = queries[f"max_adverse_pre_target_upper_{slug}_pct"].to_numpy(dtype=float)
    actual_wait = queries["target_touch_bars_from_entry"].to_numpy(dtype=float) * 5.0
    data: dict[str, Any] = {
        "observation_id": queries["observation_id"].astype(str).to_numpy(),
        "signal_id": queries["signal_id"].astype(str).to_numpy(),
        "asset": queries["asset"].astype(str).to_numpy(),
        "direction": queries["direction"].astype(str).to_numpy(),
        "entry_age_minutes": queries["entry_age_minutes"].to_numpy(dtype=int),
        "entry_distance_from_target_pct": queries[
            "entry_distance_from_target_pct"
        ].to_numpy(dtype=float),
        "actual_fill": actual_fill.astype(float),
        "predicted_fill": np.clip(predicted_fill.astype(float), 0.0, 1.0),
        "actual_adverse_pct": (lower + upper) / 2.0,
        "predicted_adverse_pct": np.maximum(predicted_adverse.astype(float), 0.0),
        "actual_wait_minutes": actual_wait,
        "predicted_wait_minutes": np.maximum(predicted_wait.astype(float), 0.0),
    }
    if "recent_60m_realized_vol_pct" in queries:
        data["recent_60m_realized_vol_pct"] = queries[
            "recent_60m_realized_vol_pct"
        ].to_numpy(dtype=float)
    if extra:
        data.update(extra)
    return pd.DataFrame(data, index=queries.index)


def supervised_prediction_rows(
    bundle: dict[str, Any],
    queries: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[int, pd.DataFrame]:
    matrix = features.loc[queries.index].to_numpy(dtype=np.float32)
    probabilities = class_probability_maps(bundle, matrix)
    risk = np.column_stack([model.predict(matrix) for model in bundle["risk_models"]])
    wait = np.column_stack(
        [np.expm1(model.predict(matrix)) for model in bundle["time_models"]]
    )
    rows: dict[int, pd.DataFrame] = {}
    for horizon_index, horizon in enumerate(HORIZONS_MINUTES):
        slug = horizon_slug(horizon)
        fill = np.asarray(
            [entry.get(1, 0.0) for entry in probabilities[f"fill_{slug}"]], dtype=float
        )
        rows[horizon] = common_prediction_rows(
            queries,
            fill,
            risk[:, horizon_index],
            wait[:, horizon_index],
            horizon,
        )
    return rows


def retrieval_prediction_rows(
    history: pd.DataFrame,
    queries: pd.DataFrame,
    selected: list[np.ndarray],
    history_matrix: np.ndarray,
    query_matrix: np.ndarray,
) -> dict[int, pd.DataFrame]:
    candidate_wait = history["target_touch_bars_from_entry"].to_numpy(dtype=float) * 5.0
    nearest = np.empty(len(queries), dtype=float)
    median_distance = np.empty(len(queries), dtype=float)
    same_asset_share = np.empty(len(queries), dtype=float)
    for index, chosen in enumerate(selected):
        distances = np.linalg.norm(history_matrix[chosen] - query_matrix[index], axis=1)
        nearest[index] = float(np.min(distances))
        median_distance[index] = float(np.median(distances))
        same_asset_share[index] = float(
            np.mean(
                history.iloc[chosen]["asset"].astype(str).to_numpy()
                == str(queries.iloc[index]["asset"])
            )
        )

    output: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS_MINUTES:
        slug = horizon_slug(horizon)
        candidate_fill = history[f"target_hit_{slug}"].astype(bool).to_numpy()
        lower = history[f"max_adverse_pre_target_lower_{slug}_pct"].to_numpy(dtype=float)
        upper = history[f"max_adverse_pre_target_upper_{slug}_pct"].to_numpy(dtype=float)
        midpoint = (lower + upper) / 2.0
        predicted_fill = np.empty(len(queries), dtype=float)
        predicted_adverse = np.empty(len(queries), dtype=float)
        predicted_wait = np.empty(len(queries), dtype=float)
        agreement = np.empty(len(queries), dtype=float)
        for index, chosen in enumerate(selected):
            fills = candidate_fill[chosen]
            predicted_fill[index] = float(np.mean(fills))
            predicted_adverse[index] = float(np.median(midpoint[chosen]))
            filled_waits = candidate_wait[chosen][fills & np.isfinite(candidate_wait[chosen])]
            predicted_wait[index] = (
                float(np.median(filled_waits)) if len(filled_waits) else float(horizon)
            )
            agreement[index] = max(predicted_fill[index], 1.0 - predicted_fill[index])
        output[horizon] = common_prediction_rows(
            queries,
            predicted_fill,
            predicted_adverse,
            predicted_wait,
            horizon,
            {
                "nearest_distance": nearest,
                "median_neighbor_distance": median_distance,
                "effective_neighbors": np.full(len(queries), len(selected[0]), dtype=float),
                "fill_agreement": agreement,
                "same_asset_neighbor_share": same_asset_share,
            },
        )
    return output


def dynamic_prediction_rows(
    history: pd.DataFrame,
    queries: pd.DataFrame,
    pool_positions: list[np.ndarray],
    pool_distances: list[np.ndarray],
    config: NeighborhoodConfig,
    maximum_distance: float | None,
    horizon: int,
) -> pd.DataFrame:
    slug = horizon_slug(horizon)
    candidate_fill = history[f"target_hit_{slug}"].astype(bool).to_numpy()
    lower = history[f"max_adverse_pre_target_lower_{slug}_pct"].to_numpy(dtype=float)
    upper = history[f"max_adverse_pre_target_upper_{slug}_pct"].to_numpy(dtype=float)
    adverse = (lower + upper) / 2.0
    candidate_wait = history["target_touch_bars_from_entry"].to_numpy(dtype=float) * 5.0
    candidate_assets = history["asset"].astype(str).to_numpy()

    predicted_fill = np.empty(len(queries), dtype=float)
    predicted_adverse = np.empty(len(queries), dtype=float)
    predicted_wait = np.empty(len(queries), dtype=float)
    support_fields: dict[str, list[Any]] = {
        "raw_neighbors": [],
        "effective_neighbors": [],
        "nearest_distance": [],
        "median_neighbor_distance": [],
        "distance_dispersion": [],
        "fill_agreement": [],
        "adverse_outcome_dispersion_pct": [],
        "wait_outcome_iqr_minutes": [],
        "radius_failed": [],
        "same_asset_neighbor_share": [],
    }
    for index, (positions, distances) in enumerate(
        zip(pool_positions, pool_distances, strict=True)
    ):
        chosen, chosen_distances, weights, radius_failed = resolve_with_nearest_fallback(
            positions, distances, config, maximum_distance
        )
        total_weight = max(float(np.sum(weights)), 1e-12)
        fills = candidate_fill[chosen]
        fill_probability = float(np.sum(weights * fills.astype(float)) / total_weight)
        predicted_fill[index] = fill_probability
        predicted_adverse[index] = weighted_quantile(adverse[chosen], weights, 0.5)
        filled = fills & np.isfinite(candidate_wait[chosen])
        predicted_wait[index] = (
            weighted_quantile(candidate_wait[chosen][filled], weights[filled], 0.5)
            if np.any(filled)
            else float(horizon)
        )
        support = neighborhood_support(
            chosen_distances,
            weights,
            fill_probability,
            adverse[chosen],
            np.where(fills, candidate_wait[chosen], np.nan),
            radius_failed,
        )
        for field, value in support.items():
            support_fields[field].append(value)
        support_fields["same_asset_neighbor_share"].append(
            float(
                np.sum(
                    weights
                    * (candidate_assets[chosen] == str(queries.iloc[index]["asset"])).astype(float)
                )
                / total_weight
            )
        )
    return common_prediction_rows(
        queries,
        predicted_fill,
        predicted_adverse,
        predicted_wait,
        horizon,
        {field: np.asarray(values) for field, values in support_fields.items()},
    )


def calibrate_support(
    validation_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    minimum_effective_neighbors: float = 8.0,
) -> pd.DataFrame:
    """Map distance to validation-period percentiles; test labels are never used."""
    output = test_rows.copy()
    for field in ("nearest_distance", "median_neighbor_distance"):
        reference = np.sort(validation_rows[field].to_numpy(dtype=float))
        values = output[field].to_numpy(dtype=float)
        output[f"{field}_percentile"] = (
            np.searchsorted(reference, values, side="right") / max(len(reference), 1)
        )
    median_percentile = output["median_neighbor_distance_percentile"]
    output["historical_support_bucket"] = pd.cut(
        median_percentile,
        bins=[-np.inf, 0.20, 0.40, 0.60, 0.80, np.inf],
        labels=["very_high", "high", "medium", "low", "very_low"],
        right=False,
    ).astype(str)
    output["historical_support_low"] = (
        output["effective_neighbors"].lt(minimum_effective_neighbors)
        | output["radius_failed"].astype(bool)
        | output["median_neighbor_distance_percentile"].ge(0.80)
    )
    return output


def episode_errors(rows: pd.DataFrame) -> pd.DataFrame:
    work = rows.copy()
    work["fill_brier"] = (work["predicted_fill"] - work["actual_fill"]) ** 2
    work["adverse_p50_absolute_error_pct"] = np.abs(
        work["predicted_adverse_pct"] - work["actual_adverse_pct"]
    )
    work["wait_p50_absolute_error_minutes"] = np.where(
        work["actual_fill"].astype(bool),
        np.abs(work["predicted_wait_minutes"] - work["actual_wait_minutes"]),
        np.nan,
    )
    return work.groupby("signal_id", sort=False)[list(METRIC_COLUMNS)].mean()


def calibration_error(rows: pd.DataFrame, bins: int = 10) -> float:
    grouped = rows.groupby("signal_id", sort=False)[["predicted_fill", "actual_fill"]].mean()
    bucket = np.minimum((grouped["predicted_fill"].to_numpy() * bins).astype(int), bins - 1)
    total = len(grouped)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if np.any(mask):
            error += float(np.sum(mask) / total) * abs(
                float(grouped.loc[mask, "predicted_fill"].mean())
                - float(grouped.loc[mask, "actual_fill"].mean())
            )
    return error


def summarize_rows(rows: pd.DataFrame) -> dict[str, float | int]:
    errors = episode_errors(rows)
    return {
        "queries": int(len(rows)),
        "episodes": int(len(errors)),
        **{column: float(errors[column].mean()) for column in METRIC_COLUMNS},
        "fill_calibration_ece": calibration_error(rows),
    }


def subgroup_summary(rows: pd.DataFrame) -> dict[str, Any]:
    """Expose where a model works; never use these test groups for selection."""
    work = rows.copy()
    work["age_group"] = pd.cut(
        work["entry_age_minutes"],
        bins=[-np.inf, 60, 360, 1_440, 10_080, 43_200, np.inf],
        labels=["under_1h", "1h_6h", "6h_1d", "1d_7d", "7d_30d", "over_30d"],
        right=False,
    ).astype(str)
    work["distance_group"] = pd.cut(
        work["entry_distance_from_target_pct"],
        bins=[-np.inf, 1, 3, 7, 15, 30, np.inf],
        labels=["under_1pct", "1_3pct", "3_7pct", "7_15pct", "15_30pct", "over_30pct"],
        right=False,
    ).astype(str)
    if "recent_60m_realized_vol_pct" in work:
        ranks = work["recent_60m_realized_vol_pct"].rank(method="average", pct=True)
        work["volatility_group"] = pd.cut(
            ranks,
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["low", "middle", "high"],
            include_lowest=True,
        ).astype(str)
    if "historical_support_bucket" not in work and "median_neighbor_distance" in work:
        ranks = work["median_neighbor_distance"].rank(method="average", pct=True)
        work["support_group"] = pd.cut(
            ranks,
            bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
            labels=["high", "middle", "low"],
            include_lowest=True,
        ).astype(str)

    output: dict[str, Any] = {}
    fields = ["asset", "direction", "age_group", "distance_group"]
    if "volatility_group" in work:
        fields.append("volatility_group")
    if "support_group" in work:
        fields.append("support_group")
    if "historical_support_bucket" in work:
        fields.append("historical_support_bucket")
    for field in fields:
        output[field] = {
            str(value): summarize_rows(group)
            for value, group in work.groupby(field, observed=True, sort=True)
        }
    return output


def paired_episode_bootstrap(
    baseline: pd.DataFrame,
    challenger: pd.DataFrame,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    left = episode_errors(baseline)
    right = episode_errors(challenger)
    shared = left.index.intersection(right.index)
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for column in METRIC_COLUMNS:
        delta = (right.loc[shared, column] - left.loc[shared, column]).dropna().to_numpy()
        if len(delta) == 0:
            continue
        draws = np.empty(samples, dtype=float)
        for index in range(samples):
            draws[index] = float(np.mean(rng.choice(delta, size=len(delta), replace=True)))
        baseline_error = float(left.loc[shared, column].mean())
        challenger_error = float(right.loc[shared, column].mean())
        result[column] = {
            "baseline_error": baseline_error,
            "challenger_error": challenger_error,
            "relative_improvement_pct": 100.0
            * (baseline_error - challenger_error)
            / max(baseline_error, 1e-12),
            "challenger_minus_baseline_ci95": [
                float(np.quantile(draws, 0.025)),
                float(np.quantile(draws, 0.975)),
            ],
            "probability_challenger_better": float(np.mean(draws < 0.0)),
        }
    return result


def aggregate_selection_score(rows_by_horizon: dict[int, pd.DataFrame]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for horizon, rows in rows_by_horizon.items():
        summary = summarize_rows(rows)
        for column in METRIC_COLUMNS:
            metrics[f"{horizon}:{column}"] = float(summary[column])
    return metrics


def choose_supervised_config(
    validation: dict[str, dict[int, pd.DataFrame]],
) -> tuple[str, dict[str, Any]]:
    names = sorted(validation)
    values = {name: aggregate_selection_score(validation[name]) for name in names}
    keys = tuple(values[names[0]])
    scale = {key: max(values[names[0]][key], 1e-12) for key in keys}
    scores = {
        name: float(np.mean([values[name][key] / scale[key] for key in keys]))
        for name in names
    }
    selected = min(names, key=lambda name: (scores[name], name))
    return selected, {"selected": selected, "scores": scores, "raw_metrics": values}


def choose_retrieval_configs(
    validation: dict[str, dict[int, pd.DataFrame]],
    configs: tuple[FingerprintConfig, ...],
    regression_tolerance: float,
) -> tuple[dict[int, FingerprintConfig], dict[str, Any]]:
    baseline = validation["state_only"]
    selected: dict[int, FingerprintConfig] = {}
    evidence: dict[str, Any] = {}
    for horizon in HORIZONS_MINUTES:
        base_metrics = summarize_rows(baseline[horizon])
        candidates: list[dict[str, Any]] = []
        for config in configs:
            metrics = summarize_rows(validation[config.name][horizon])
            ratios = {
                column: float(metrics[column]) / max(float(base_metrics[column]), 1e-12)
                for column in METRIC_COLUMNS
            }
            candidates.append(
                {
                    "name": config.name,
                    "mean_error_ratio": float(np.mean(list(ratios.values()))),
                    "worst_error_ratio": float(np.max(list(ratios.values()))),
                    "metric_error_ratios": ratios,
                }
            )
        eligible = [
            item
            for item in candidates
            if item["worst_error_ratio"] <= 1.0 + regression_tolerance
        ]
        winner = min(eligible, key=lambda item: (item["mean_error_ratio"], item["name"]))
        selected[horizon] = next(config for config in configs if config.name == winner["name"])
        evidence[str(horizon)] = {
            "selected": winner,
            "candidates": sorted(candidates, key=lambda item: item["mean_error_ratio"]),
        }
    return selected, evidence


def neighborhood_configs(smoke: bool, baseline_k: int) -> tuple[NeighborhoodConfig, ...]:
    if not smoke:
        return candidate_neighborhood_configs()
    values = (
        NeighborhoodConfig(baseline_k, "uniform", None),
        NeighborhoodConfig(baseline_k, "inverse", None),
        NeighborhoodConfig(min(16, max(baseline_k, 8)), "uniform", 0.50),
        NeighborhoodConfig(min(16, max(baseline_k, 8)), "inverse", 0.50),
    )
    return tuple({value.name: value for value in values}.values())


def choose_dynamic_neighborhood(
    baseline_rows: pd.DataFrame,
    baseline_name: str,
    candidates: dict[str, tuple[NeighborhoodConfig, float | None, pd.DataFrame]],
    regression_tolerance: float,
) -> tuple[NeighborhoodConfig, float | None, pd.DataFrame, dict[str, Any]]:
    baseline_metrics = summarize_rows(baseline_rows)
    records: list[dict[str, Any]] = []
    for name, (config, maximum_distance, rows) in candidates.items():
        metrics = summarize_rows(rows)
        ratios = {
            column: float(metrics[column]) / max(float(baseline_metrics[column]), 1e-12)
            for column in METRIC_COLUMNS
        }
        records.append(
            {
                **config_record(config, maximum_distance),
                "mean_error_ratio": float(np.mean(list(ratios.values()))),
                "worst_error_ratio": float(np.max(list(ratios.values()))),
                "metric_error_ratios": ratios,
            }
        )
    eligible = [
        record
        for record in records
        if record["worst_error_ratio"] <= 1.0 + regression_tolerance
    ]
    if not eligible:
        eligible = [record for record in records if record["name"] == baseline_name]
    winner = min(eligible, key=lambda item: (item["mean_error_ratio"], item["name"]))
    selected_config, maximum_distance, selected_rows = candidates[winner["name"]]
    ordered = sorted(records, key=lambda item: (item["mean_error_ratio"], item["name"]))
    return selected_config, maximum_distance, selected_rows, {
        "selected": winner,
        "candidate_count": len(records),
        "eligible_count": len(eligible),
        "top_candidates": ordered[:10],
        "regression_tolerance": regression_tolerance,
    }


def build_dynamic_pool(
    blocks: dict[str, tuple[np.ndarray, np.ndarray]],
    config: FingerprintConfig,
    history: pd.DataFrame,
    maximum_neighbors: int = 96,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    history_matrix, query_matrix = compose_matrices(blocks, config)
    return retrieve_episode_neighbor_pool(
        history_matrix,
        query_matrix,
        history["signal_id"].astype(str).to_numpy(),
        maximum_neighbors=maximum_neighbors,
    )


def attach_support_to_systems(
    systems: dict[str, dict[int, pd.DataFrame]],
    support_source: dict[int, pd.DataFrame],
) -> None:
    support_columns = (
        "raw_neighbors",
        "effective_neighbors",
        "nearest_distance",
        "median_neighbor_distance",
        "distance_dispersion",
        "fill_agreement",
        "adverse_outcome_dispersion_pct",
        "wait_outcome_iqr_minutes",
        "radius_failed",
        "same_asset_neighbor_share",
        "nearest_distance_percentile",
        "median_neighbor_distance_percentile",
        "historical_support_bucket",
        "historical_support_low",
    )
    for horizon, source in support_source.items():
        lookup = source.set_index("observation_id")
        for by_horizon in systems.values():
            rows = by_horizon[horizon]
            keys = rows["observation_id"].astype(str)
            for column in support_columns:
                rows[column] = lookup.loc[keys, column].to_numpy()


SUPPORT_STACK_COLUMNS = (
    "effective_neighbors",
    "nearest_distance",
    "median_neighbor_distance",
    "distance_dispersion",
    "fill_agreement",
    "adverse_outcome_dispersion_pct",
    "wait_outcome_iqr_minutes",
    "same_asset_neighbor_share",
    "nearest_distance_percentile",
    "median_neighbor_distance_percentile",
    "radius_failed",
    "historical_support_low",
)


def support_stack_matrix(
    a_rows: pd.DataFrame,
    b_rows: pd.DataFrame,
    include_support: bool = True,
) -> np.ndarray:
    if not np.array_equal(
        a_rows["observation_id"].astype(str).to_numpy(),
        b_rows["observation_id"].astype(str).to_numpy(),
    ):
        raise RuntimeError("Support stack requires identical A/B observation order")
    columns = [
        a_rows["predicted_fill"].to_numpy(dtype=float),
        a_rows["predicted_adverse_pct"].to_numpy(dtype=float),
        np.log1p(a_rows["predicted_wait_minutes"].to_numpy(dtype=float)),
        b_rows["predicted_fill"].to_numpy(dtype=float),
        b_rows["predicted_adverse_pct"].to_numpy(dtype=float),
        np.log1p(b_rows["predicted_wait_minutes"].to_numpy(dtype=float)),
    ]
    if include_support:
        for column in SUPPORT_STACK_COLUMNS:
            values = a_rows[column].astype(float).to_numpy()
            if column in {
                "effective_neighbors",
                "nearest_distance",
                "median_neighbor_distance",
                "distance_dispersion",
                "adverse_outcome_dispersion_pct",
                "wait_outcome_iqr_minutes",
            }:
                values = np.log1p(np.maximum(values, 0.0))
            columns.append(values)
    matrix = np.column_stack(columns)
    return np.nan_to_num(matrix, nan=0.0, posinf=20.0, neginf=-20.0).astype(np.float32)


def episode_sample_weights(rows: pd.DataFrame) -> np.ndarray:
    counts = rows.groupby("signal_id")["signal_id"].transform("size").to_numpy(dtype=float)
    return 1.0 / np.maximum(counts, 1.0)


def support_stacked_supervised_rows(
    validation_systems: dict[str, dict[int, pd.DataFrame]],
    test_systems: dict[str, dict[int, pd.DataFrame]],
    fold_number: int,
    include_support: bool = True,
) -> dict[int, pd.DataFrame]:
    """Fit a fixed small meta-model on validation, then score the later test period."""
    output: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS_MINUTES:
        train_a = validation_systems["A_supervised_53"][horizon]
        train_b = validation_systems["B_supervised_53_plus_sequence"][horizon]
        test_a = test_systems["A_supervised_53"][horizon]
        test_b = test_systems["B_supervised_53_plus_sequence"][horizon]
        train_matrix = support_stack_matrix(train_a, train_b, include_support)
        test_matrix = support_stack_matrix(test_a, test_b, include_support)
        seed_offset = 0 if include_support else 100_000
        weights = episode_sample_weights(train_a)
        fill_y = train_a["actual_fill"].to_numpy(dtype=int)
        if len(np.unique(fill_y)) == 1:
            predicted_fill = np.full(len(test_a), float(fill_y[0]), dtype=float)
        else:
            classifier = HistGradientBoostingClassifier(
                learning_rate=0.045,
                max_iter=160,
                max_leaf_nodes=10,
                min_samples_leaf=25,
                l2_regularization=3.0,
                random_state=61000 + seed_offset + fold_number * 10 + horizon,
            ).fit(train_matrix, fill_y, sample_weight=weights)
            predicted_fill = classifier.predict_proba(test_matrix)[:, 1]

        risk_model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.045,
            max_iter=160,
            max_leaf_nodes=10,
            min_samples_leaf=25,
            l2_regularization=3.0,
            random_state=62000 + seed_offset + fold_number * 10 + horizon,
        ).fit(
            train_matrix,
            train_a["actual_adverse_pct"].to_numpy(dtype=float),
            sample_weight=weights,
        )
        predicted_adverse = risk_model.predict(test_matrix)

        filled = train_a["actual_fill"].astype(bool).to_numpy()
        wait_model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.045,
            max_iter=160,
            max_leaf_nodes=10,
            min_samples_leaf=20,
            l2_regularization=3.0,
            random_state=63000 + seed_offset + fold_number * 10 + horizon,
        ).fit(
            train_matrix[filled],
            np.log1p(train_a.loc[filled, "actual_wait_minutes"].to_numpy(dtype=float)),
            sample_weight=weights[filled],
        )
        predicted_wait = np.expm1(wait_model.predict(test_matrix))
        rows = test_a.copy()
        rows["predicted_fill"] = np.clip(predicted_fill, 0.0, 1.0)
        rows["predicted_adverse_pct"] = np.maximum(predicted_adverse, 0.0)
        rows["predicted_wait_minutes"] = np.maximum(predicted_wait, 0.0)
        output[horizon] = rows
    return output


def evaluate_retrieval_config(
    config: FingerprintConfig,
    blocks: dict[str, tuple[np.ndarray, np.ndarray]],
    history: pd.DataFrame,
    queries: pd.DataFrame,
    neighbors: int,
) -> dict[int, pd.DataFrame]:
    history_matrix, query_matrix = compose_matrices(blocks, config)
    selected, _ = retrieve_neighbors(
        history_matrix,
        query_matrix,
        history["signal_id"].astype(str).to_numpy(),
        neighbors,
    )
    return retrieval_prediction_rows(
        history,
        queries,
        selected,
        history_matrix,
        query_matrix,
    )


def validate_v3_fold_provenance(metadata: dict[str, Any], fold_cutoff_ms: int) -> None:
    """Reject V3 artifacts without explicit fold-local outcome provenance."""
    cutoff = metadata.get("training_labels_available_through_ms")
    trained_before = metadata.get("training_completed_before_fold_cutoff", False)
    if cutoff is None or int(cutoff) > int(fold_cutoff_ms) or not bool(trained_before):
        raise RuntimeError(
            "V3 artifact is not proven fold-local; retrain embedding, normalizers, "
            "retrieval index, reranker, and calibration inside this fold"
        )


def tree_feature_sets(
    features: pd.DataFrame,
    sequences: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    return {
        "A_supervised_53": features,
        "B_supervised_53_plus_sequence": pd.concat([features, sequences], axis=1),
    }


def forest_configs(smoke: bool) -> tuple[ForestConfig, ...]:
    if smoke:
        return (ForestConfig("smoke", 40, 12, 20, 0.80),)
    return CONFIGS


def fingerprint_configs(smoke: bool) -> tuple[FingerprintConfig, ...]:
    configs = candidate_configs()
    if not smoke:
        return configs
    keep = {
        "state_only",
        "context_1",
        "context_1_recent_0.5",
        "context_1_recent_episode_0.5",
    }
    return tuple(config for config in configs if config.name in keep)


def run_fold(
    observations: pd.DataFrame,
    features: pd.DataFrame,
    sequences: pd.DataFrame,
    fold: RollingFold,
    validation_queries: int,
    test_queries: int,
    neighbors: int,
    regression_tolerance: float,
    bootstrap_samples: int,
    smoke: bool,
) -> dict[str, Any]:
    before_validation = observations.loc[
        observations["signal_open_time_ms"].lt(fold.train_end_signal_ms)
    ].copy()
    validation_window = signal_window(
        observations, fold.train_end_signal_ms, fold.validation_end_signal_ms
    )
    validation_history, validation, validation_availability = eligible_history_and_queries(
        before_validation, validation_window, validation_queries
    )

    before_test = observations.loc[
        observations["signal_open_time_ms"].lt(fold.validation_end_signal_ms)
    ].copy()
    test_window = signal_window(
        observations, fold.validation_end_signal_ms, fold.test_end_signal_ms
    )
    test_history, test, test_availability = eligible_history_and_queries(
        before_test, test_window, test_queries
    )
    for query_frame in (validation, test):
        query_frame["recent_60m_realized_vol_pct"] = features.loc[
            query_frame.index, "recent_60m_realized_vol_pct"
        ].to_numpy(dtype=float)
    feature_sets = tree_feature_sets(features, sequences)

    tree_selection: dict[str, Any] = {}
    selected_tree_configs: dict[str, ForestConfig] = {}
    selected_tree_validation: dict[str, dict[int, pd.DataFrame]] = {}
    for system, matrix in feature_sets.items():
        validation_outputs: dict[str, dict[int, pd.DataFrame]] = {}
        for config in forest_configs(smoke):
            print(
                json.dumps(
                    {"fold": fold.number, "stage": "validation", "system": system, "config": config.name}
                ),
                flush=True,
            )
            bundle = fit_bundle(
                validation_history,
                matrix,
                HORIZONS_MINUTES,
                (),
                config,
                "5m",
            )
            validation_outputs[config.name] = supervised_prediction_rows(
                bundle, validation, matrix
            )
        selected_name, evidence = choose_supervised_config(validation_outputs)
        selected_tree_configs[system] = next(
            config for config in forest_configs(smoke) if config.name == selected_name
        )
        selected_tree_validation[system] = validation_outputs[selected_name]
        tree_selection[system] = evidence

    retrieval_validation_blocks = scaled_blocks(
        validation_history, validation, features, sequences
    )
    configs = fingerprint_configs(smoke)
    retrieval_validation: dict[str, dict[int, pd.DataFrame]] = {}
    for config in configs:
        print(
            json.dumps(
                {"fold": fold.number, "stage": "validation", "system": "C_retrieval", "config": config.name}
            ),
            flush=True,
        )
        retrieval_validation[config.name] = evaluate_retrieval_config(
            config,
            retrieval_validation_blocks,
            validation_history,
            validation,
            neighbors,
        )
    selected_retrieval, retrieval_selection = choose_retrieval_configs(
        retrieval_validation, configs, regression_tolerance
    )

    dynamic_validation_pools: dict[
        str, tuple[list[np.ndarray], list[np.ndarray]]
    ] = {}
    for config in {item.name: item for item in selected_retrieval.values()}.values():
        print(
            json.dumps(
                {
                    "fold": fold.number,
                    "stage": "validation",
                    "system": "C2_dynamic_retrieval",
                    "config": config.name,
                }
            ),
            flush=True,
        )
        dynamic_validation_pools[config.name] = build_dynamic_pool(
            retrieval_validation_blocks,
            config,
            validation_history,
        )

    selected_dynamic: dict[int, tuple[NeighborhoodConfig, float | None]] = {}
    selected_dynamic_validation: dict[int, pd.DataFrame] = {}
    dynamic_selection: dict[str, Any] = {}
    baseline_neighborhood = NeighborhoodConfig(neighbors, "uniform", None)
    dynamic_configs = neighborhood_configs(smoke, neighbors)
    for horizon, fingerprint_config in selected_retrieval.items():
        pool_positions, pool_distances = dynamic_validation_pools[fingerprint_config.name]
        candidates: dict[
            str, tuple[NeighborhoodConfig, float | None, pd.DataFrame]
        ] = {}
        for neighborhood in dynamic_configs:
            maximum_distance = radius_threshold(
                pool_distances, neighborhood.radius_quantile
            )
            rows = dynamic_prediction_rows(
                validation_history,
                validation,
                pool_positions,
                pool_distances,
                neighborhood,
                maximum_distance,
                horizon,
            )
            candidates[neighborhood.name] = (neighborhood, maximum_distance, rows)
        baseline_rows = candidates[baseline_neighborhood.name][2]
        neighborhood, maximum_distance, validation_rows, evidence = (
            choose_dynamic_neighborhood(
                baseline_rows,
                baseline_neighborhood.name,
                candidates,
                regression_tolerance,
            )
        )
        selected_dynamic[horizon] = (neighborhood, maximum_distance)
        selected_dynamic_validation[horizon] = validation_rows
        dynamic_selection[str(horizon)] = {
            "fingerprint": {
                "name": fingerprint_config.name,
                "weights": fingerprint_config.weights,
            },
            "neighborhood": evidence,
        }
    calibrated_dynamic_validation = {
        horizon: calibrate_support(rows, rows)
        for horizon, rows in selected_dynamic_validation.items()
    }

    test_outputs: dict[str, dict[int, pd.DataFrame]] = {}
    for system, matrix in feature_sets.items():
        config = selected_tree_configs[system]
        print(
            json.dumps(
                {"fold": fold.number, "stage": "test", "system": system, "config": config.name}
            ),
            flush=True,
        )
        bundle = fit_bundle(test_history, matrix, HORIZONS_MINUTES, (), config, "5m")
        test_outputs[system] = supervised_prediction_rows(bundle, test, matrix)

    retrieval_test_blocks = scaled_blocks(test_history, test, features, sequences)
    retrieval_test_cache: dict[str, dict[int, pd.DataFrame]] = {}
    for config in {item.name: item for item in selected_retrieval.values()}.values():
        print(
            json.dumps(
                {"fold": fold.number, "stage": "test", "system": "C_retrieval", "config": config.name}
            ),
            flush=True,
        )
        retrieval_test_cache[config.name] = evaluate_retrieval_config(
            config,
            retrieval_test_blocks,
            test_history,
            test,
            neighbors,
        )
    test_outputs["C_horizon_specific_retrieval"] = {
        horizon: retrieval_test_cache[config.name][horizon]
        for horizon, config in selected_retrieval.items()
    }

    dynamic_test_pools: dict[str, tuple[list[np.ndarray], list[np.ndarray]]] = {}
    for config in {item.name: item for item in selected_retrieval.values()}.values():
        print(
            json.dumps(
                {
                    "fold": fold.number,
                    "stage": "test",
                    "system": "C2_dynamic_retrieval",
                    "config": config.name,
                }
            ),
            flush=True,
        )
        dynamic_test_pools[config.name] = build_dynamic_pool(
            retrieval_test_blocks,
            config,
            test_history,
        )
    dynamic_test_rows: dict[int, pd.DataFrame] = {}
    for horizon, fingerprint_config in selected_retrieval.items():
        neighborhood, maximum_distance = selected_dynamic[horizon]
        pool_positions, pool_distances = dynamic_test_pools[fingerprint_config.name]
        rows = dynamic_prediction_rows(
            test_history,
            test,
            pool_positions,
            pool_distances,
            neighborhood,
            maximum_distance,
            horizon,
        )
        dynamic_test_rows[horizon] = calibrate_support(
            selected_dynamic_validation[horizon], rows
        )
    test_outputs["C2_dynamic_support_retrieval"] = dynamic_test_rows
    attach_support_to_systems(test_outputs, dynamic_test_rows)
    attach_support_to_systems(selected_tree_validation, calibrated_dynamic_validation)
    test_outputs["E0_prediction_stack_supervised"] = support_stacked_supervised_rows(
        selected_tree_validation,
        test_outputs,
        fold.number,
        include_support=False,
    )
    test_outputs["E_support_stacked_supervised"] = support_stacked_supervised_rows(
        selected_tree_validation,
        test_outputs,
        fold.number,
        include_support=True,
    )

    metrics: dict[str, Any] = {}
    for system, by_horizon in test_outputs.items():
        metrics[system] = {
            str(horizon): summarize_rows(rows) for horizon, rows in by_horizon.items()
        }

    comparisons: dict[str, Any] = {}
    baseline = test_outputs["A_supervised_53"]
    for system, by_horizon in test_outputs.items():
        if system == "A_supervised_53":
            continue
        comparisons[system] = {
            str(horizon): paired_episode_bootstrap(
                baseline[horizon],
                by_horizon[horizon],
                bootstrap_samples,
                20260927 + fold.number * 100 + horizon,
            )
            for horizon in HORIZONS_MINUTES
        }

    return {
        "fold": asdict(fold),
        "population": {
            "validation_history": int(len(validation_history)),
            "validation_queries": int(len(validation)),
            "validation_query_episodes": int(validation["signal_id"].nunique()),
            "test_history": int(len(test_history)),
            "test_queries": int(len(test)),
            "test_query_episodes": int(test["signal_id"].nunique()),
        },
        "label_availability": {
            "validation": validation_availability,
            "test": test_availability,
        },
        "validation_selection": {
            "supervised": tree_selection,
            "retrieval": retrieval_selection,
            "dynamic_neighborhood": dynamic_selection,
            "support_stack": {
                "training_period": "validation only",
                "hyperparameters": "fixed before test",
                "episode_balanced_sample_weights": True,
                "support_columns": list(SUPPORT_STACK_COLUMNS),
                "ablation": "E0 uses identical A/B prediction inputs without support columns",
            },
        },
        "test_metrics": metrics,
        "test_comparison_vs_A": comparisons,
        "_test_outputs": test_outputs,
    }


def support_conditioned_report(
    combined_rows: dict[str, dict[int, pd.DataFrame]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    baseline = combined_rows["A_supervised_53"]
    bucket_order = ("very_high", "high", "medium", "low", "very_low")
    for horizon in HORIZONS_MINUTES:
        horizon_output: dict[str, Any] = {}
        for bucket in bucket_order:
            systems: dict[str, Any] = {}
            for system, by_horizon in combined_rows.items():
                rows = by_horizon[horizon]
                group = rows.loc[rows["historical_support_bucket"].eq(bucket)]
                if group.empty:
                    continue
                systems[system] = summarize_rows(group)
            comparisons: dict[str, Any] = {}
            baseline_group = baseline[horizon].loc[
                baseline[horizon]["historical_support_bucket"].eq(bucket)
            ]
            baseline_metrics = summarize_rows(baseline_group)
            for system in (
                "C_horizon_specific_retrieval",
                "C2_dynamic_support_retrieval",
                "E0_prediction_stack_supervised",
                "E_support_stacked_supervised",
            ):
                challenger_group = combined_rows[system][horizon].loc[
                    combined_rows[system][horizon]["historical_support_bucket"].eq(bucket)
                ]
                challenger_metrics = summarize_rows(challenger_group)
                comparisons[system] = {
                    column: 100.0
                    * (float(baseline_metrics[column]) - float(challenger_metrics[column]))
                    / max(float(baseline_metrics[column]), 1e-12)
                    for column in METRIC_COLUMNS
                }
            horizon_output[bucket] = {
                "systems": systems,
                "relative_improvement_vs_A_pct": comparisons,
            }
        output[str(horizon)] = horizon_output
    return output


def evaluate(
    root: Path,
    dataset_dir: Path,
    folds: int,
    initial_train_fraction: float,
    validation_queries: int,
    test_queries: int,
    neighbors: int,
    regression_tolerance: float,
    bootstrap_samples: int,
    smoke: bool,
) -> dict[str, Any]:
    observations, metadata = read_observations(dataset_dir, "5m")
    observations = observations.reset_index(drop=True)
    features = build_feature_matrix(root, observations)
    sequences = build_sequence_matrix(root, observations)
    raw_fold_results = [
        run_fold(
            observations,
            features,
            sequences,
            fold,
            validation_queries,
            test_queries,
            neighbors,
            regression_tolerance,
            bootstrap_samples,
            smoke,
        )
        for fold in rolling_signal_folds(observations, folds, initial_train_fraction)
    ]
    combined: dict[str, dict[int, list[pd.DataFrame]]] = {}
    for fold_result in raw_fold_results:
        for system, by_horizon in fold_result["_test_outputs"].items():
            combined.setdefault(system, {})
            for horizon, rows in by_horizon.items():
                combined[system].setdefault(horizon, []).append(rows)
    combined_rows = {
        system: {
            horizon: pd.concat(parts, ignore_index=True)
            for horizon, parts in by_horizon.items()
        }
        for system, by_horizon in combined.items()
    }
    aggregate_metrics = {
        system: {
            str(horizon): summarize_rows(rows) for horizon, rows in by_horizon.items()
        }
        for system, by_horizon in combined_rows.items()
    }
    aggregate_subgroups = {
        system: {
            str(horizon): subgroup_summary(rows) for horizon, rows in by_horizon.items()
        }
        for system, by_horizon in combined_rows.items()
    }
    aggregate_comparisons: dict[str, Any] = {}
    baseline = combined_rows["A_supervised_53"]
    for system, by_horizon in combined_rows.items():
        if system == "A_supervised_53":
            continue
        aggregate_comparisons[system] = {
            str(horizon): paired_episode_bootstrap(
                baseline[horizon],
                by_horizon[horizon],
                bootstrap_samples,
                20261027 + horizon,
            )
            for horizon in HORIZONS_MINUTES
        }
    aggregate_pairwise: dict[str, Any] = {
        "C2_dynamic_vs_C_fixed": {
            str(horizon): paired_episode_bootstrap(
                combined_rows["C_horizon_specific_retrieval"][horizon],
                combined_rows["C2_dynamic_support_retrieval"][horizon],
                bootstrap_samples,
                20262027 + horizon,
            )
            for horizon in HORIZONS_MINUTES
        },
        "E_support_stack_vs_B_sequence": {
            str(horizon): paired_episode_bootstrap(
                combined_rows["B_supervised_53_plus_sequence"][horizon],
                combined_rows["E_support_stacked_supervised"][horizon],
                bootstrap_samples,
                20263027 + horizon,
            )
            for horizon in HORIZONS_MINUTES
        },
        "E_support_stack_vs_E0_prediction_only": {
            str(horizon): paired_episode_bootstrap(
                combined_rows["E0_prediction_stack_supervised"][horizon],
                combined_rows["E_support_stacked_supervised"][horizon],
                bootstrap_samples,
                20264027 + horizon,
            )
            for horizon in HORIZONS_MINUTES
        },
    }
    support_conditioned = support_conditioned_report(combined_rows)
    fold_results = []
    for fold_result in raw_fold_results:
        fold_result.pop("_test_outputs")
        fold_results.append(fold_result)
    return {
        "schema_version": "rolling-forecast-comparison-v1.1.0",
        "generated_at_utc": utc_now(),
        "status": "smoke_only" if smoke else "research_only_not_promoted",
        "timeframe": "5m",
        "assets": sorted(observations["asset"].astype(str).unique().tolist()),
        "systems": {
            "A": "existing 53-feature supervised ExtraTrees/HGB",
            "B": "same supervised learner plus 40 observable sequence coordinates",
            "C": "validation-selected horizon-specific handcrafted fingerprint retrieval",
            "C2": (
                "same validation-selected fingerprint plus validation-selected k, distance "
                "weighting, and maximum radius with effective-N support diagnostics"
            ),
            "E": (
                "fixed small validation-trained stack over A/B predictions and observable "
                "retrieval-support variables"
            ),
            "E0": (
                "ablation of E using the same fixed validation-trained stack but only A/B "
                "predictions and no retrieval-support variables"
            ),
            "D": (
                "not run: V3 must retrain its encoder, normalization, retrieval population, "
                "and reranker inside every fold; the global artifact is intentionally rejected"
            ),
        },
        "fold_policy": {
            "folds": folds,
            "initial_train_fraction": initial_train_fraction,
            "validation_and_test_signal_windows_do_not_overlap": True,
            "all_training_labels_mature_before_first_query_close": True,
            "hyperparameters_selected_on_validation_only": True,
            "test_metrics_episode_balanced": True,
            "bootstrap_unit": "signal episode",
            "targets": (
                "fill Brier; midpoint adverse MAE; conditional-on-fill wait MAE"
            ),
        },
        "population": {
            "observations": int(len(observations)),
            "episodes": int(observations["signal_id"].nunique()),
            "sequence_dimensions": len(sequence_column_names()),
            "neighbors": neighbors,
        },
        "aggregate_test_metrics": aggregate_metrics,
        "aggregate_test_comparison_vs_A": aggregate_comparisons,
        "aggregate_pairwise_comparisons": aggregate_pairwise,
        "aggregate_test_subgroups": aggregate_subgroups,
        "support_conditioned_models": support_conditioned,
        "fold_results": fold_results,
        "dataset_schema_version": metadata.get("schema_version"),
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/prospective_entry_outcomes_v1"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--initial-train-fraction", type=float, default=0.40)
    parser.add_argument("--validation-queries", type=int, default=750)
    parser.add_argument("--test-queries", type=int, default=1_000)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--regression-tolerance", type=float, default=0.01)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/rolling_forecast_comparison_5m.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else root / args.dataset_dir
    output = args.output if args.output.is_absolute() else root / args.output
    report = evaluate(
        root,
        dataset_dir,
        args.folds,
        args.initial_train_fraction,
        args.validation_queries,
        args.test_queries,
        args.neighbors,
        args.regression_tolerance,
        args.bootstrap_samples,
        args.smoke,
    )
    atomic_json(output, report)
    print(json.dumps({"status": report["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
