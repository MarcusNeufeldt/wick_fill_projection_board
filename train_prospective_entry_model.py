#!/usr/bin/env python3
"""Train and chronologically gate the all-outcome numerical wick model."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import HistGradientBoostingRegressor

from build_conditional_path_library import read_five_minute_file, read_one_minute_file
from candle_archetype import ARCHETYPE_BLEND_WEIGHT, SOFT_MATCHER_BASELINE_VERSION
from evaluate_prospective_entry_baseline import (
    MatcherConfig,
    chronological_signal_split,
    distance_bucket,
    evaluate_config,
    interval_error,
    outcome_brier,
    read_observations,
    robust_feature_scales,
    stratified_sample,
    summarize_metrics,
)
from prospective_entry_model import (
    FEATURE_COLUMNS_V1,
    MODEL_SCHEMA_VERSION,
    OUTCOME_CLASSES,
    OUTCOME_TO_CODE,
    default_artifact_root,
    features_from_observations,
    save_bundle,
)
from prospective_entry_outcomes import horizon_slug, threshold_slug


TRAINING_SCHEMA_VERSION = "prospective-entry-training-v1.0.0"
TRAIN_JOBS = int(os.environ.get("CANDLE_PROJECTION_TRAIN_JOBS", "-1"))


@dataclass(frozen=True)
class ForestConfig:
    name: str
    n_estimators: int
    max_depth: int
    min_samples_leaf: int
    max_features: float


CONFIGS = (
    ForestConfig("local", 220, 20, 8, 0.90),
    ForestConfig("moderate", 220, 16, 20, 0.80),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def raw_path(root: Path, asset: str, timeframe: str) -> Path:
    return root / "data" / f"{asset}_{timeframe}_5y.csv"


def build_feature_matrix(root: Path, observations: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for (asset, timeframe), partition in observations.groupby(
        ["asset", "timeframe"], sort=True, observed=True
    ):
        reader = read_one_minute_file if timeframe == "1m" else read_five_minute_file
        frame = reader(raw_path(root, str(asset), str(timeframe)))
        pieces.append(features_from_observations(frame, partition))
    return pd.concat(pieces).sort_index().loc[observations.index, FEATURE_COLUMNS_V1]


def classifier_targets(
    observations: pd.DataFrame,
    horizons: Iterable[int],
    thresholds: Iterable[float],
) -> tuple[np.ndarray, list[str]]:
    values: list[np.ndarray] = []
    names: list[str] = []
    for horizon in horizons:
        slug = horizon_slug(int(horizon))
        values.append(observations[f"target_hit_{slug}"].astype(int).to_numpy())
        names.append(f"fill_{slug}")
    for horizon in horizons:
        slug = horizon_slug(int(horizon))
        for threshold in thresholds:
            key = threshold_slug(float(threshold))
            labels = observations[f"outcome_{slug}_vs_{key}pct"].astype(str)
            unknown = set(labels.unique()).difference(OUTCOME_TO_CODE)
            if unknown:
                raise RuntimeError(f"Unexpected outcome labels for {slug}/{key}: {sorted(unknown)}")
            values.append(labels.map(OUTCOME_TO_CODE).to_numpy(dtype=np.int16))
            names.append(f"outcome_{slug}_vs_{key}pct")
    return np.column_stack(values).astype(np.int16), names


def risk_targets(observations: pd.DataFrame, horizons: Iterable[int]) -> np.ndarray:
    values: list[np.ndarray] = []
    for horizon in horizons:
        slug = horizon_slug(int(horizon))
        lower = observations[f"max_adverse_pre_target_lower_{slug}_pct"].to_numpy(dtype=float)
        upper = observations[f"max_adverse_pre_target_upper_{slug}_pct"].to_numpy(dtype=float)
        values.append((lower + upper) / 2.0)
    return np.column_stack(values).astype(np.float32)


def fit_bundle(
    observations: pd.DataFrame,
    features: pd.DataFrame,
    horizons: tuple[int, ...],
    thresholds: tuple[float, ...],
    config: ForestConfig,
    timeframe: str,
) -> dict[str, Any]:
    maximum_slug = horizon_slug(max(horizons))
    known = observations[f"horizon_{maximum_slug}_fully_observed"].astype(bool)
    train = observations.loc[known]
    matrix = features.loc[train.index].to_numpy(dtype=np.float32)
    class_y, class_names = classifier_targets(train, horizons, thresholds)
    classifier = ExtraTreesClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=12345,
        n_jobs=TRAIN_JOBS,
    ).fit(matrix, class_y)
    risk_models: list[HistGradientBoostingRegressor] = []
    time_models: list[HistGradientBoostingRegressor] = []
    interval_minutes = int(timeframe.removesuffix("m"))
    for horizon in horizons:
        slug = horizon_slug(int(horizon))
        horizon_known = observations[f"horizon_{slug}_fully_observed"].astype(bool) | observations[
            f"target_hit_{slug}"
        ].astype(bool)
        horizon_known &= observations.index.isin(train.index)
        lower = observations.loc[horizon_known, f"max_adverse_pre_target_lower_{slug}_pct"].to_numpy(dtype=float)
        upper = observations.loc[horizon_known, f"max_adverse_pre_target_upper_{slug}_pct"].to_numpy(dtype=float)
        risk_models.append(
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.055,
                max_iter=180,
                max_leaf_nodes=31 if config.name == "local" else 20,
                max_depth=None,
                min_samples_leaf=config.min_samples_leaf,
                l2_regularization=1.0,
                random_state=23456 + int(horizon),
            ).fit(
                features.loc[horizon_known.index[horizon_known]].to_numpy(dtype=np.float32),
                (lower + upper) / 2.0,
            )
        )
        filled = horizon_known & observations[f"target_hit_{slug}"].astype(bool)
        time_y = np.log1p(
            observations.loc[filled, "target_touch_bars_from_entry"].to_numpy(dtype=float)
            * interval_minutes
        )
        time_models.append(
            HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.055,
                max_iter=180,
                max_leaf_nodes=31 if config.name == "local" else 20,
                min_samples_leaf=config.min_samples_leaf,
                l2_regularization=1.0,
                random_state=34567 + int(horizon),
            ).fit(
                features.loc[filled.index[filled]].to_numpy(dtype=np.float32), time_y
            )
        )
    return {
        "schema_version": MODEL_SCHEMA_VERSION,
        "timeframe": timeframe,
        "feature_columns": list(features.columns),
        "horizons_minutes": list(horizons),
        "thresholds_pct": list(thresholds),
        "classifier": classifier,
        "classifier_output_names": class_names,
        "risk_models": risk_models,
        "time_models": time_models,
        "configuration": config.__dict__,
        "training_rows": int(known.sum()),
        "training_filled_rows": int(observations.loc[known, "target_touch_bars_from_entry"].notna().sum()),
    }


def class_probability_maps(bundle: dict[str, Any], matrix: np.ndarray) -> dict[str, list[dict[int, float]]]:
    raw = bundle["classifier"].predict_proba(matrix)
    if not isinstance(raw, list):
        raw = [raw]
    result: dict[str, list[dict[int, float]]] = {}
    for name, classes, probabilities in zip(
        bundle["classifier_output_names"], bundle["classifier"].classes_, raw, strict=True
    ):
        result[name] = [
            {int(label): float(value) for label, value in zip(classes, row, strict=True)}
            for row in probabilities
        ]
    return result


def evaluate_bundle(
    bundle: dict[str, Any],
    observations: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    if observations.empty:
        return pd.DataFrame()
    matrix = features.loc[observations.index].to_numpy(dtype=np.float32)
    probabilities = class_probability_maps(bundle, matrix)
    risk_predictions = np.column_stack(
        [model.predict(matrix) for model in bundle["risk_models"]]
    )
    wait_predictions = np.column_stack(
        [np.expm1(model.predict(matrix)) for model in bundle["time_models"]]
    )
    horizons = tuple(int(value) for value in bundle["horizons_minutes"])
    thresholds = tuple(float(value) for value in bundle["thresholds_pct"])
    rows: list[dict[str, Any]] = []
    for position, (_, query) in enumerate(observations.iterrows()):
        for horizon_index, horizon in enumerate(horizons):
            slug = horizon_slug(horizon)
            actual_known = bool(query[f"horizon_{slug}_fully_observed"]) or bool(
                query[f"target_hit_{slug}"]
            )
            if not actual_known:
                continue
            actual_fill = float(bool(query[f"target_hit_{slug}"]))
            fill_probability = probabilities[f"fill_{slug}"][position].get(1, 0.0)
            actual_wait = (
                float(query["target_touch_bars_from_entry"])
                * int(str(query["timeframe"]).removesuffix("m"))
                if actual_fill
                else np.nan
            )
            lower_column = f"max_adverse_pre_target_lower_{slug}_pct"
            upper_column = f"max_adverse_pre_target_upper_{slug}_pct"
            base = {
                "config": bundle["configuration"]["name"],
                "observation_id": query["observation_id"],
                "direction": query["direction"],
                "entry_age_minutes": int(query["entry_age_minutes"]),
                "distance_bucket": str(
                    distance_bucket(pd.Series([query["entry_distance_from_target_pct"]])).iloc[0]
                ),
                "horizon_minutes": horizon,
                "risk_interval_error_pct": interval_error(
                    float(risk_predictions[position, horizon_index]),
                    float(query[lower_column]),
                    float(query[upper_column]),
                ),
                "fill_brier": (fill_probability - actual_fill) ** 2,
                "time_absolute_error_minutes": (
                    abs(float(wait_predictions[position, horizon_index]) - actual_wait)
                    if np.isfinite(actual_wait)
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
                encoded = probabilities[f"outcome_{slug}_vs_{key}pct"][position]
                outcome_probabilities = {
                    label: encoded.get(code, 0.0) for label, code in OUTCOME_TO_CODE.items()
                }
                rows.append(
                    {
                        **base,
                        "risk_interval_error_pct": np.nan,
                        "fill_brier": np.nan,
                        "time_absolute_error_minutes": np.nan,
                        "outcome_brier": outcome_brier(outcome_probabilities, actual),
                        "adverse_threshold_pct": threshold,
                    }
                )
    return pd.DataFrame(rows)


def metric_rank(results: dict[str, dict[str, Any]]) -> list[str]:
    names = list(results)
    scores = {name: 0.0 for name in names}
    fields = ("risk_interval_mae_pct", "fill_brier", "time_mae_minutes", "outcome_brier")
    for field in fields:
        ordered = sorted(names, key=lambda name: (float(results[name][field]), name))
        for rank, name in enumerate(ordered, start=1):
            scores[name] += rank
    return sorted(names, key=lambda name: (scores[name], name))


def calibrate_ranges(
    bundle: dict[str, Any],
    observations: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    matrix = features.loc[observations.index].to_numpy(dtype=np.float32)
    risk_prediction = np.column_stack(
        [model.predict(matrix) for model in bundle["risk_models"]]
    )
    adjustments: dict[str, dict[str, float]] = {}
    for index, horizon in enumerate(bundle["horizons_minutes"]):
        slug = horizon_slug(int(horizon))
        known = observations[f"horizon_{slug}_fully_observed"].astype(bool) | observations[
            f"target_hit_{slug}"
        ].astype(bool)
        upper = observations.loc[known, f"max_adverse_pre_target_upper_{slug}_pct"].to_numpy(dtype=float)
        residual = upper - risk_prediction[known.to_numpy(), index]
        adjustments[slug] = {
            "p80": float(max(0.0, np.quantile(residual, 0.80))),
            "p90": float(max(0.0, np.quantile(residual, 0.90))),
        }
    bundle["risk_upper_adjustments"] = adjustments
    interval_minutes = int(bundle["timeframe"].removesuffix("m"))
    time_calibration: dict[str, dict[str, float]] = {}
    for index, horizon in enumerate(bundle["horizons_minutes"]):
        slug = horizon_slug(int(horizon))
        filled = observations[f"target_hit_{slug}"].astype(bool)
        actual_log = np.log1p(
            observations.loc[filled, "target_touch_bars_from_entry"].to_numpy(dtype=float)
            * interval_minutes
        )
        predicted_log = bundle["time_models"][index].predict(
            features.loc[filled.index[filled]].to_numpy(dtype=np.float32)
        )
        residual = actual_log - predicted_log
        time_calibration[slug] = {
            "p10": float(np.quantile(residual, 0.10)),
            "p50": float(np.quantile(residual, 0.50)),
            "p90": float(np.quantile(residual, 0.90)),
        }
    bundle["time_log_residual_quantiles"] = time_calibration


def subgroup_metrics(metrics: pd.DataFrame, column: str) -> dict[str, Any]:
    return {
        str(value): summarize_metrics(group)
        for value, group in metrics.groupby(column, observed=True, sort=True)
    }


def train_timeframe(
    root: Path,
    dataset_dir: Path,
    artifact_root: Path,
    timeframe: str,
    validation_queries: int,
    holdout_queries: int,
) -> dict[str, Any]:
    observations, metadata = read_observations(dataset_dir, timeframe)
    observations = observations.reset_index(drop=True)
    print(json.dumps({"stage": "features", "timeframe": timeframe, "rows": len(observations)}), flush=True)
    features = build_feature_matrix(root, observations)
    fit, validation, holdout, boundaries = chronological_signal_split(observations)
    validation_sample = stratified_sample(validation, validation_queries)
    holdout_sample = stratified_sample(holdout, holdout_queries)
    horizons = tuple(int(value) for value in metadata["horizons_minutes"])
    thresholds = tuple(float(value) for value in metadata["adverse_thresholds_pct"])

    validation_results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, dict[str, Any]] = {}
    for config in CONFIGS:
        print(json.dumps({"stage": "fit_config", "timeframe": timeframe, "config": config.name}), flush=True)
        bundle = fit_bundle(fit, features, horizons, thresholds, config, timeframe)
        metrics = evaluate_bundle(bundle, validation_sample, features)
        validation_results[config.name] = summarize_metrics(metrics)
        fitted[config.name] = bundle
        print(json.dumps({"stage": "validation", "timeframe": timeframe, "config": config.name, **validation_results[config.name]}), flush=True)
    selected_name = metric_rank(validation_results)[0]
    selected_config = next(config for config in CONFIGS if config.name == selected_name)
    calibrate_ranges(fitted[selected_name], validation_sample, features)

    history = pd.concat([fit, validation]).sort_index()
    holdout_bundle = fit_bundle(history, features, horizons, thresholds, selected_config, timeframe)
    calibrate_ranges(holdout_bundle, validation_sample, features)
    holdout_metrics_frame = evaluate_bundle(holdout_bundle, holdout_sample, features)
    model_metrics = summarize_metrics(holdout_metrics_frame)
    baseline_frame, _ = evaluate_config(
        history,
        holdout_sample,
        MatcherConfig(ARCHETYPE_BLEND_WEIGHT, True),
        robust_feature_scales(fit),
        horizons,
        thresholds,
        80,
    )
    frozen_baseline = summarize_metrics(baseline_frame)
    promotion = (
        model_metrics["distinct_observations"] >= min(500, len(holdout_sample))
        and model_metrics["risk_interval_mae_pct"] < frozen_baseline["risk_interval_mae_pct"]
        and model_metrics["time_mae_minutes"] < frozen_baseline["time_mae_minutes"]
        and model_metrics["fill_brier"] <= frozen_baseline["fill_brier"]
        and model_metrics["outcome_brier"] <= frozen_baseline["outcome_brier"]
    )
    print(json.dumps({"stage": "holdout", "timeframe": timeframe, "selected": selected_name, "promoted": promotion, **model_metrics}), flush=True)

    artifact_path = artifact_root / timeframe / "model.joblib"
    if promotion:
        maximum_slug = horizon_slug(max(horizons))
        deployment_rows = observations.loc[
            observations[f"horizon_{maximum_slug}_fully_observed"].astype(bool)
        ]
        deployment_bundle = fit_bundle(
            deployment_rows,
            features,
            horizons,
            thresholds,
            selected_config,
            timeframe,
        )
        calibrate_ranges(deployment_bundle, holdout_sample, features)
        deployment_bundle.update(
            {
                "generated_at_utc": utc_now(),
                "baseline_version": SOFT_MATCHER_BASELINE_VERSION,
                "holdout_metrics": model_metrics,
                "baseline_holdout_metrics": frozen_baseline,
                "promoted": True,
                "support": {
                    "entry_age_minutes_min": int(deployment_rows["entry_age_minutes"].min()),
                    "entry_age_minutes_max": int(deployment_rows["entry_age_minutes"].max()),
                    "training_assets": sorted(deployment_rows["asset"].astype(str).unique()),
                    "direction_counts": deployment_rows["direction"].value_counts().to_dict(),
                    "age_counts": deployment_rows["entry_age_minutes"].value_counts().sort_index().to_dict(),
                },
            }
        )
        save_bundle(deployment_bundle, artifact_path)
    report = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "timeframe": timeframe,
        "baseline_version": SOFT_MATCHER_BASELINE_VERSION,
        "split": {**boundaries, "fit_rows": len(fit), "validation_rows": len(validation), "holdout_rows": len(holdout)},
        "validation": validation_results,
        "selected_configuration": selected_config.__dict__,
        "holdout": {
            "model": model_metrics,
            "frozen_baseline": frozen_baseline,
            "subgroups": {
                "direction": subgroup_metrics(holdout_metrics_frame, "direction"),
                "entry_age_minutes": subgroup_metrics(holdout_metrics_frame, "entry_age_minutes"),
                "departure_distance": subgroup_metrics(holdout_metrics_frame, "distance_bucket"),
                "horizon_minutes": subgroup_metrics(holdout_metrics_frame, "horizon_minutes"),
            },
        },
        "promotion_gate_passed": bool(promotion),
        "artifact": str(artifact_path) if promotion else None,
        "population": "all observable departed and still-unfilled entries; unresolved outcomes retained",
        "risk_measurement": "additional adverse percentage from prospective entry close",
        "intrabar_ordering": "ambiguous target/adverse same-bar cases remain a separate outcome class",
    }
    report_path = root / "data" / "prospective_entry_models" / timeframe / "training_report.json"
    atomic_json(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/prospective_entry_outcomes_v1"))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--timeframes", default="1m,5m")
    parser.add_argument("--validation-queries", type=int, default=1_000)
    parser.add_argument("--holdout-queries", type=int, default=2_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else root / args.dataset_dir
    artifact_root = (
        args.artifact_dir.resolve()
        if args.artifact_dir is not None
        else default_artifact_root(root)
    )
    reports = []
    for timeframe in [value.strip() for value in args.timeframes.split(",") if value.strip()]:
        validation_queries = min(args.validation_queries, 500 if timeframe == "1m" else args.validation_queries)
        holdout_queries = min(args.holdout_queries, 1_000 if timeframe == "1m" else args.holdout_queries)
        reports.append(
            train_timeframe(
                root,
                dataset_dir,
                artifact_root,
                timeframe,
                validation_queries,
                holdout_queries,
            )
        )
    print(
        json.dumps(
            {
                "stage": "complete",
                "results": [
                    {
                        "timeframe": report["timeframe"],
                        "promoted": report["promotion_gate_passed"],
                        "artifact": report["artifact"],
                    }
                    for report in reports
                ],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
