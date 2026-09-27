#!/usr/bin/env python3
"""Chronologically validate pre-signal/recent-context analogue reranking."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from candle_archetype import SOFT_MATCHER_BASELINE_VERSION
from contextual_entry_matcher import (
    CONTEXTUAL_MATCHER_SCHEMA_VERSION,
    CONTEXT_FEATURES,
    context_scales,
    predict_from_ranked,
    rank_candidates,
    save_bundle,
)
from evaluate_prospective_entry_baseline import (
    chronological_signal_split,
    distance_bucket,
    interval_error,
    outcome_brier,
    read_observations,
    robust_feature_scales,
    stratified_sample,
    summarize_metrics,
)
from prospective_entry_outcomes import horizon_slug, threshold_slug
from train_prospective_entry_model import atomic_json, build_feature_matrix


REPORT_SCHEMA_VERSION = "contextual-entry-matcher-eval-v1.0.0"


def evaluate_weight(
    history: pd.DataFrame,
    queries: pd.DataFrame,
    features: pd.DataFrame,
    base_scales: dict[str, float],
    rich_scales: dict[str, float],
    weight: float,
    horizons: tuple[int, ...],
    thresholds: tuple[float, ...],
    top_k: int,
) -> pd.DataFrame:
    maximum_horizon_ms = max(horizons) * 60_000
    interval_minutes = int(str(queries["timeframe"].iloc[0]).removesuffix("m"))
    maximum_slug = horizon_slug(max(horizons))
    rows: list[dict[str, Any]] = []
    for query_tuple in queries.itertuples(index=True):
        query_index = int(query_tuple.Index)
        query = pd.Series(query_tuple._asdict()).drop(labels="Index")
        candidates = history.loc[
            history["entry_close_time_ms"].to_numpy(dtype=np.int64) + maximum_horizon_ms
            <= int(query["entry_close_time_ms"])
        ]
        candidates = candidates.loc[
            candidates[f"horizon_{maximum_slug}_fully_observed"].astype(bool)
        ]
        ranked = rank_candidates(
            candidates,
            features,
            query,
            features.loc[query_index],
            base_scales,
            rich_scales,
            weight,
            top_k,
        )
        if len(ranked) < max(12, top_k // 4):
            continue
        prediction = predict_from_ranked(ranked, horizons, thresholds, interval_minutes)
        for horizon in horizons:
            slug = horizon_slug(horizon)
            actual_known = bool(query[f"horizon_{slug}_fully_observed"]) or bool(
                query[f"target_hit_{slug}"]
            )
            if not actual_known or slug not in prediction["horizons"]:
                continue
            value = prediction["horizons"][slug]
            actual_fill = float(bool(query[f"target_hit_{slug}"]))
            actual_wait = (
                float(query["target_touch_bars_from_entry"]) * interval_minutes
                if actual_fill
                else np.nan
            )
            wait = value["remaining_time_minutes_if_filled_within_horizon"]
            predicted_wait = np.nan if wait is None else float(wait["p50"])
            lower_column = f"max_adverse_pre_target_lower_{slug}_pct"
            upper_column = f"max_adverse_pre_target_upper_{slug}_pct"
            base = {
                "config": f"context_{weight:.2f}",
                "observation_id": query["observation_id"],
                "direction": query["direction"],
                "entry_age_minutes": int(query["entry_age_minutes"]),
                "distance_bucket": str(
                    distance_bucket(pd.Series([query["entry_distance_from_target_pct"]])).iloc[0]
                ),
                "horizon_minutes": horizon,
                "risk_interval_error_pct": interval_error(
                    float(value["additional_adverse_pct"]["p50"]),
                    float(query[lower_column]),
                    float(query[upper_column]),
                ),
                "fill_brier": (float(value["fill_probability"]) - actual_fill) ** 2,
                "time_absolute_error_minutes": (
                    abs(predicted_wait - actual_wait)
                    if np.isfinite(predicted_wait) and np.isfinite(actual_wait)
                    else np.nan
                ),
                "outcome_brier": np.nan,
            }
            rows.append(base)
            for threshold in thresholds:
                key = threshold_slug(threshold)
                actual = str(query[f"outcome_{slug}_vs_{key}pct"])
                if actual == "right_censored":
                    continue
                rows.append(
                    {
                        **base,
                        "risk_interval_error_pct": np.nan,
                        "fill_brier": np.nan,
                        "time_absolute_error_minutes": np.nan,
                        "outcome_brier": outcome_brier(
                            value["competing_outcomes"][f"{key}pct"], actual
                        ),
                        "adverse_threshold_pct": threshold,
                    }
                )
    return pd.DataFrame(rows)


def average_rank(results: dict[float, dict[str, Any]]) -> list[float]:
    scores = {weight: 0 for weight in results}
    for field in ("risk_interval_mae_pct", "fill_brier", "time_mae_minutes", "outcome_brier"):
        for rank, weight in enumerate(
            sorted(results, key=lambda item: (float(results[item][field]), item)), start=1
        ):
            scores[weight] += rank
    return sorted(results, key=lambda weight: (scores[weight], weight))


def subgroup(metrics: pd.DataFrame, column: str) -> dict[str, Any]:
    return {
        str(value): summarize_metrics(group)
        for value, group in metrics.groupby(column, observed=True, sort=True)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/prospective_entry_outcomes_v1"))
    parser.add_argument("--timeframe", choices=("1m", "5m"), required=True)
    parser.add_argument("--weights", default="0,0.10,0.20,0.30")
    parser.add_argument("--validation-queries", type=int, default=1_000)
    parser.add_argument("--holdout-queries", type=int, default=2_000)
    parser.add_argument("--top-k", type=int, default=80)
    args = parser.parse_args()
    root = args.root.resolve()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else root / args.dataset_dir
    observations, metadata = read_observations(dataset_dir, args.timeframe)
    observations = observations.reset_index(drop=True)
    print(json.dumps({"stage": "features", "rows": len(observations)}), flush=True)
    features = build_feature_matrix(root, observations)
    fit, validation, holdout, boundaries = chronological_signal_split(observations)
    validation_sample = stratified_sample(validation, args.validation_queries)
    holdout_sample = stratified_sample(holdout, args.holdout_queries)
    horizons = tuple(int(value) for value in metadata["horizons_minutes"])
    thresholds = tuple(float(value) for value in metadata["adverse_thresholds_pct"])
    weights = tuple(float(value.strip()) for value in args.weights.split(",") if value.strip())
    base_scales = robust_feature_scales(fit)
    rich_scales = context_scales(features.loc[fit.index])
    validation_results: dict[float, dict[str, Any]] = {}
    for weight in weights:
        metrics = evaluate_weight(
            fit,
            validation_sample,
            features,
            base_scales,
            rich_scales,
            weight,
            horizons,
            thresholds,
            args.top_k,
        )
        validation_results[weight] = summarize_metrics(metrics)
        print(json.dumps({"stage": "validation", "weight": weight, **validation_results[weight]}), flush=True)
    selected = average_rank(validation_results)[0]
    history = pd.concat([fit, validation]).sort_index()
    holdout_base_scales = robust_feature_scales(history)
    holdout_rich_scales = context_scales(features.loc[history.index])
    fixed_weights = sorted({0.0, selected})
    holdout_results: dict[float, dict[str, Any]] = {}
    holdout_frames: dict[float, pd.DataFrame] = {}
    for weight in fixed_weights:
        metrics = evaluate_weight(
            history,
            holdout_sample,
            features,
            holdout_base_scales,
            holdout_rich_scales,
            weight,
            horizons,
            thresholds,
            args.top_k,
        )
        holdout_frames[weight] = metrics
        holdout_results[weight] = summarize_metrics(metrics)
        print(json.dumps({"stage": "holdout", "weight": weight, **holdout_results[weight]}), flush=True)
    baseline = holdout_results[0.0]
    challenger = holdout_results[selected]
    promotion = (
        selected > 0
        and challenger["distinct_observations"] >= min(500, len(holdout_sample))
        and challenger["risk_interval_mae_pct"] < baseline["risk_interval_mae_pct"]
        and challenger["time_mae_minutes"] < baseline["time_mae_minutes"]
        and challenger["fill_brier"] <= baseline["fill_brier"]
        and challenger["outcome_brier"] <= baseline["outcome_brier"]
    )
    report_dir = root / "data" / "contextual_entry_matcher" / args.timeframe
    artifact_path = report_dir / "matcher.joblib"
    if promotion:
        maximum_slug = horizon_slug(max(horizons))
        deploy = observations.loc[
            observations[f"horizon_{maximum_slug}_fully_observed"].astype(bool)
        ].copy().reset_index(drop=True)
        deploy_features = features.loc[
            observations[f"horizon_{maximum_slug}_fully_observed"].astype(bool), CONTEXT_FEATURES
        ].copy().reset_index(drop=True)
        label_columns = [
            column
            for column in deploy.columns
            if column.startswith(("horizon_", "target_hit_", "max_adverse_", "outcome_"))
        ]
        keep_columns = [
            "observation_id", "asset", "timeframe", "direction", "entry_close_time_ms",
            "entry_age_bars", "entry_age_minutes", "entry_distance_from_target_pct",
            "peak_distance_from_target_pct", "drawdown_from_peak_pct",
            "target_touch_bars_from_entry", *[column for column in fit.columns if column in (
                "body_pct_of_range", "dominant_wick_pct_of_range", "opposite_wick_pct_of_range",
                "range_pct_of_close", "range_vs_prior_20_median", "volume_vs_prior_20_mean",
                "aligned_prior_1h_return_pct")], *label_columns,
        ]
        deploy = deploy.loc[:, list(dict.fromkeys(keep_columns))]
        bundle = {
            "schema_version": CONTEXTUAL_MATCHER_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "baseline_version": SOFT_MATCHER_BASELINE_VERSION,
            "timeframe": args.timeframe,
            "context_weight": selected,
            "top_k": args.top_k,
            "horizons_minutes": list(horizons),
            "thresholds_pct": list(thresholds),
            "base_feature_scales": robust_feature_scales(observations),
            "context_feature_scales": context_scales(features),
            "context_feature_columns": list(CONTEXT_FEATURES),
            "context_feature_values": deploy_features.to_numpy(dtype=np.float32),
            "observations": deploy,
            "holdout_metrics": challenger,
            "baseline_holdout_metrics": baseline,
            "promotion_gate_passed": True,
            "support": {
                "entry_age_minutes_min": int(deploy["entry_age_minutes"].min()),
                "entry_age_minutes_max": int(deploy["entry_age_minutes"].max()),
                "age_counts": deploy["entry_age_minutes"].value_counts().sort_index().to_dict(),
                "assets": sorted(deploy["asset"].astype(str).unique()),
            },
        }
        save_bundle(bundle, artifact_path)
    selected_frame = holdout_frames[selected]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "timeframe": args.timeframe,
        "baseline_version": SOFT_MATCHER_BASELINE_VERSION,
        "context_features": list(CONTEXT_FEATURES),
        "split": boundaries,
        "validation": {str(key): value for key, value in validation_results.items()},
        "validation_selected_weight": selected,
        "holdout": {str(key): value for key, value in holdout_results.items()},
        "promotion_gate_passed": promotion,
        "artifact": str(artifact_path) if promotion else None,
        "subgroups": {
            "direction": subgroup(selected_frame, "direction"),
            "entry_age_minutes": subgroup(selected_frame, "entry_age_minutes"),
            "departure_distance": subgroup(selected_frame, "distance_bucket"),
        },
    }
    atomic_json(report_dir / "evaluation.json", report)
    print(json.dumps({"stage": "complete", "selected_weight": selected, "promoted": promotion, "artifact": report["artifact"]}), flush=True)


if __name__ == "__main__":
    main()
