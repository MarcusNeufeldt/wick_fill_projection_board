#!/usr/bin/env python3
"""Chronologically evaluate wick matching on prospective-entry outcomes.

The evaluator never scores how convincing a projected candle route looks.  It
scores entry-relative adverse excursion, fill-by-horizon probability, remaining
waiting time for observed fills, and target-first/adverse-first outcomes.

Blend weights and direction pooling are selected on a validation period.  Only
the frozen winner, the versioned 70/30 pooled baseline, and fixed controls are
reported on the later holdout period.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from build_conditional_path_scenarios import (
    CATEGORY_PENALTIES,
    FEATURE_WEIGHTS,
    STATE_WEIGHTS,
    log_distance,
)
from build_conditional_path_library import FEATURE_COLUMNS
from candle_archetype import (
    ARCHETYPE_BLEND_WEIGHT,
    SOFT_MATCHER_BASELINE_VERSION,
    archetype_distance,
)
from prospective_entry_outcomes import horizon_slug, threshold_slug


EVALUATION_SCHEMA_VERSION = "prospective-entry-baseline-eval-v1.0.0"
OUTCOME_CLASSES = ("target_first", "adverse_first", "ambiguous_intrabar", "neither")
DEFAULT_BLEND_WEIGHTS = (0.0, 0.15, 0.30, 0.50, 0.70)


@dataclass(frozen=True)
class MatcherConfig:
    blend_weight: float
    pool_directions: bool

    @property
    def name(self) -> str:
        direction = "pooled" if self.pool_directions else "separated"
        return f"blend_{self.blend_weight:.2f}_{direction}"


def parse_float_list(value: str) -> tuple[float, ...]:
    result = tuple(sorted({float(item.strip()) for item in value.split(",") if item.strip()}))
    if not result or result[0] < 0 or result[-1] > 1:
        raise argparse.ArgumentTypeError("blend weights must be between 0 and 1")
    return result


def read_observations(dataset_dir: Path, timeframe: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    files = [
        Path(item["path"])
        for item in metadata["files"]
        if item["kind"] == "observations" and item["timeframe"] == timeframe
    ]
    if not files:
        raise RuntimeError(f"No {timeframe} observation partitions in {dataset_dir}")
    frame = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    # Partitions begin with their own RangeIndex.  Reset after concatenation so
    # stratified sampling cannot accidentally treat equal row numbers from
    # different assets as the same observation.
    return (
        frame.sort_values(["signal_open_time_ms", "observation_id"], kind="stable")
        .reset_index(drop=True),
        metadata,
    )


def chronological_signal_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    signals = np.sort(frame["signal_open_time_ms"].unique())
    if len(signals) < 20:
        raise RuntimeError("At least 20 distinct signals are required for chronological evaluation")
    fit_boundary = int(signals[max(1, int(len(signals) * 0.60))])
    holdout_boundary = int(signals[max(2, int(len(signals) * 0.80))])
    fit = frame.loc[frame["signal_open_time_ms"].lt(fit_boundary)].copy()
    validation = frame.loc[
        frame["signal_open_time_ms"].ge(fit_boundary)
        & frame["signal_open_time_ms"].lt(holdout_boundary)
    ].copy()
    holdout = frame.loc[frame["signal_open_time_ms"].ge(holdout_boundary)].copy()
    return fit, validation, holdout, {
        "fit_boundary_signal_open_time_ms": fit_boundary,
        "holdout_boundary_signal_open_time_ms": holdout_boundary,
    }


def distance_bucket(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 1.0, 3.0, 7.0, 15.0, 30.0, np.inf],
        labels=["<1%", "1-3%", "3-7%", "7-15%", "15-30%", "30%+"],
        right=False,
    ).astype(str)


def stratified_sample(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame.copy()
    work = frame.copy()
    work["_distance_bucket"] = distance_bucket(work["entry_distance_from_target_pct"])
    groups = list(work.groupby(["direction", "entry_age_minutes", "_distance_bucket"], observed=True, sort=True))
    quota = max(1, maximum // max(len(groups), 1))
    selected: list[pd.DataFrame] = []
    for _, group in groups:
        ordered = group.sort_values("entry_close_time_ms", kind="stable")
        positions = np.linspace(0, len(ordered) - 1, min(quota, len(ordered))).round().astype(int)
        selected.append(ordered.iloc[np.unique(positions)])
    output = pd.concat(selected).drop_duplicates("observation_id")
    if len(output) < maximum:
        remainder = work.loc[~work.index.isin(output.index)].sort_values("entry_close_time_ms", kind="stable")
        positions = np.linspace(0, len(remainder) - 1, min(maximum - len(output), len(remainder))).round().astype(int)
        output = pd.concat([output, remainder.iloc[np.unique(positions)]])
    return output.drop(columns="_distance_bucket", errors="ignore").head(maximum).copy()


def robust_feature_scales(fit: pd.DataFrame) -> dict[str, float]:
    scales: dict[str, float] = {}
    for field in FEATURE_COLUMNS:
        values = fit[field].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)) * 1.4826)
        scales[field] = max(mad, float(np.std(values)), 1e-6)
    return scales


def score_candidates(
    candidates: pd.DataFrame,
    query: pd.Series,
    config: MatcherConfig,
    feature_scales: dict[str, float],
) -> pd.DataFrame:
    value = candidates
    # The live 1m path libraries are intentionally isolated per asset, whereas
    # the 5m matcher uses the pooled five-asset library.
    if str(query["timeframe"]) == "1m":
        value = value.loc[value["asset"].eq(str(query["asset"]))]
    if not config.pool_directions:
        value = value.loc[value["direction"].eq(str(query["direction"]))]
    if value.empty:
        return value.copy()
    adaptive = np.zeros(len(value), dtype=float)
    for field, weight in FEATURE_WEIGHTS.items():
        adaptive += (
            weight
            * np.abs(value[field].to_numpy(dtype=float) - float(query[field]))
            / feature_scales[field]
        )
    shape = archetype_distance(value, query)
    scaled_shape = shape * max(float(np.median(adaptive)), 1e-12) / max(float(np.median(shape)), 1e-12)
    adaptive_weighted = (1.0 - config.blend_weight) * adaptive
    archetype_weighted = config.blend_weight * scaled_shape
    components = {
        "state_current_move_contribution": STATE_WEIGHTS["current_move_pct"]
        * log_distance(
            value["entry_distance_from_target_pct"].to_numpy(dtype=float),
            float(query["entry_distance_from_target_pct"]),
            0.30,
        ),
        "state_peak_move_contribution": STATE_WEIGHTS["peak_move_pct"]
        * log_distance(
            value["peak_distance_from_target_pct"].to_numpy(dtype=float),
            float(query["peak_distance_from_target_pct"]),
            0.30,
        ),
        "state_drawdown_contribution": STATE_WEIGHTS["drawdown_from_peak_pct"]
        * log_distance(
            value["drawdown_from_peak_pct"].to_numpy(dtype=float),
            float(query["drawdown_from_peak_pct"]),
            0.30,
        ),
        "state_age_contribution": STATE_WEIGHTS["elapsed_bars"]
        * log_distance(
            value["entry_age_bars"].to_numpy(dtype=float),
            float(query["entry_age_bars"]),
            3.0,
        ),
        "adaptive_signal_contribution": 0.5 * adaptive_weighted,
        "archetype_signal_contribution": 0.5 * archetype_weighted,
        "category_contribution": (
            value["asset"].ne(str(query["asset"])).to_numpy(dtype=float)
            * CATEGORY_PENALTIES["asset"]
            * (1.0 - config.blend_weight)
        ),
    }
    output = value.copy()
    for name, values in components.items():
        output[name] = values
    output["match_score"] = sum(components.values())
    return output.sort_values("match_score", kind="stable")


def interval_error(prediction: float, lower: float, upper: float) -> float:
    if prediction < lower:
        return lower - prediction
    if prediction > upper:
        return prediction - upper
    return 0.0


def outcome_brier(probabilities: dict[str, float], actual: str) -> float:
    return float(
        sum((probabilities.get(label, 0.0) - float(label == actual)) ** 2 for label in OUTCOME_CLASSES)
        / len(OUTCOME_CLASSES)
    )


def evaluate_config(
    history: pd.DataFrame,
    queries: pd.DataFrame,
    config: MatcherConfig,
    feature_scales: dict[str, float],
    horizons_minutes: Iterable[int],
    thresholds_pct: Iterable[float],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    maximum_horizon = max(int(value) for value in horizons_minutes)
    interval_minutes = int(str(queries["timeframe"].iloc[0]).removesuffix("m"))
    maximum_horizon_ms = maximum_horizon * 60_000
    metric_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for query in queries.itertuples(index=False):
        row = pd.Series(query._asdict())
        candidates = history.loc[
            history["entry_close_time_ms"].to_numpy(dtype=np.int64) + maximum_horizon_ms
            <= int(row["entry_close_time_ms"])
        ]
        candidates = candidates.loc[
            candidates[f"horizon_{horizon_slug(maximum_horizon)}_fully_observed"].astype(bool)
        ]
        ranked = score_candidates(candidates, row, config, feature_scales).head(top_k)
        if len(ranked) < max(12, top_k // 4):
            continue
        component_rows.append(
            {
                "observation_id": row["observation_id"],
                **{
                    column: float(ranked[column].mean())
                    for column in (
                        "state_current_move_contribution",
                        "state_peak_move_contribution",
                        "state_drawdown_contribution",
                        "state_age_contribution",
                        "adaptive_signal_contribution",
                        "archetype_signal_contribution",
                        "category_contribution",
                    )
                },
            }
        )
        for horizon in horizons_minutes:
            horizon = int(horizon)
            slug = horizon_slug(horizon)
            actual_known = bool(row[f"horizon_{slug}_fully_observed"]) or bool(row[f"target_hit_{slug}"])
            if not actual_known:
                continue
            neighbor_known = ranked[f"horizon_{slug}_fully_observed"].astype(bool) | ranked[f"target_hit_{slug}"].astype(bool)
            task_neighbors = ranked.loc[neighbor_known]
            if len(task_neighbors) < 12:
                continue
            lower_column = f"max_adverse_pre_target_lower_{slug}_pct"
            upper_column = f"max_adverse_pre_target_upper_{slug}_pct"
            risk_prediction = float(
                np.median(
                    (
                        task_neighbors[lower_column].to_numpy(dtype=float)
                        + task_neighbors[upper_column].to_numpy(dtype=float)
                    )
                    / 2.0
                )
            )
            fill_probability = float(task_neighbors[f"target_hit_{slug}"].mean())
            actual_fill = float(bool(row[f"target_hit_{slug}"]))
            fill_neighbors = task_neighbors.loc[task_neighbors[f"target_hit_{slug}"].astype(bool)]
            predicted_wait = (
                float(np.median(fill_neighbors["target_touch_bars_from_entry"].dropna().to_numpy(dtype=float)))
                if len(fill_neighbors)
                else np.nan
            )
            actual_wait = (
                float(row["target_touch_bars_from_entry"])
                if actual_fill
                else np.nan
            )
            base = {
                "config": config.name,
                "observation_id": row["observation_id"],
                "direction": row["direction"],
                "entry_age_minutes": int(row["entry_age_minutes"]),
                "distance_bucket": str(
                    distance_bucket(pd.Series([row["entry_distance_from_target_pct"]])).iloc[0]
                ),
                "horizon_minutes": horizon,
                "risk_interval_error_pct": interval_error(
                    risk_prediction,
                    float(row[lower_column]),
                    float(row[upper_column]),
                ),
                "fill_brier": (fill_probability - actual_fill) ** 2,
                "time_absolute_error_minutes": (
                    abs(predicted_wait - actual_wait) * interval_minutes
                    if np.isfinite(predicted_wait) and np.isfinite(actual_wait)
                    else np.nan
                ),
                "outcome_brier": np.nan,
            }
            metric_rows.append(base)
            for threshold in thresholds_pct:
                outcome_column = f"outcome_{slug}_vs_{threshold_slug(float(threshold))}pct"
                actual_outcome = str(row[outcome_column])
                if actual_outcome == "right_censored":
                    continue
                counts = task_neighbors[outcome_column].value_counts(normalize=True)
                probabilities = {label: float(counts.get(label, 0.0)) for label in OUTCOME_CLASSES}
                metric_rows.append(
                    {
                        **base,
                        "risk_interval_error_pct": np.nan,
                        "fill_brier": np.nan,
                        "time_absolute_error_minutes": np.nan,
                        "outcome_brier": outcome_brier(probabilities, actual_outcome),
                        "adverse_threshold_pct": float(threshold),
                    }
                )
    return pd.DataFrame(metric_rows), pd.DataFrame(component_rows)


def summarize_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "distinct_observations": int(frame["observation_id"].nunique()) if len(frame) else 0,
        "risk_interval_mae_pct": float(frame["risk_interval_error_pct"].mean()),
        "fill_brier": float(frame["fill_brier"].mean()),
        "time_mae_minutes": float(frame["time_absolute_error_minutes"].mean()),
        "outcome_brier": float(frame["outcome_brier"].mean()),
    }


def validation_rank(metrics_by_config: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, frame in metrics_by_config.items():
        rows.append({"config": name, **summarize_metrics(frame)})
    table = pd.DataFrame(rows)
    metric_columns = ["risk_interval_mae_pct", "fill_brier", "time_mae_minutes", "outcome_brier"]
    ranks = table[metric_columns].rank(method="average", ascending=True)
    table["selection_rank"] = ranks.mean(axis=1)
    return table.sort_values(["selection_rank", "config"], kind="stable").reset_index(drop=True)


def subgroup_report(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    return {
        str(key): summarize_metrics(group)
        for key, group in frame.groupby(column, observed=True, sort=True)
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("data/prospective_entry_outcomes_v1"))
    parser.add_argument("--timeframe", choices=("1m", "5m"), default="5m")
    parser.add_argument("--blend-weights", type=parse_float_list, default=DEFAULT_BLEND_WEIGHTS)
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--validation-queries", type=int, default=1_000)
    parser.add_argument("--holdout-queries", type=int, default=2_000)
    parser.add_argument("--output", type=Path, default=Path("data/prospective_entry_baseline_evaluation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    observations, metadata = read_observations(args.dataset_dir, args.timeframe)
    fit, validation, holdout, boundaries = chronological_signal_split(observations)
    validation_queries = stratified_sample(validation, args.validation_queries)
    holdout_queries = stratified_sample(holdout, args.holdout_queries)
    feature_scales = robust_feature_scales(fit)
    horizons = tuple(int(value) for value in metadata["horizons_minutes"])
    thresholds = tuple(float(value) for value in metadata["adverse_thresholds_pct"])
    configs = [
        MatcherConfig(weight, pooled)
        for weight in args.blend_weights
        for pooled in (False, True)
    ]
    validation_metrics: dict[str, pd.DataFrame] = {}
    for config in configs:
        metrics, _ = evaluate_config(
            fit,
            validation_queries,
            config,
            feature_scales,
            horizons,
            thresholds,
            args.top_k,
        )
        validation_metrics[config.name] = metrics
        print(json.dumps({"stage": "validation", "config": config.name, **summarize_metrics(metrics)}), flush=True)
    ranking = validation_rank(validation_metrics)
    winner_name = str(ranking.iloc[0]["config"])
    baseline = MatcherConfig(ARCHETYPE_BLEND_WEIGHT, True)
    fixed_controls = {
        baseline.name: baseline,
        winner_name: next(config for config in configs if config.name == winner_name),
        MatcherConfig(ARCHETYPE_BLEND_WEIGHT, False).name: MatcherConfig(ARCHETYPE_BLEND_WEIGHT, False),
        MatcherConfig(0.0, True).name: MatcherConfig(0.0, True),
    }
    history = pd.concat([fit, validation], ignore_index=True)
    holdout_metrics: dict[str, pd.DataFrame] = {}
    holdout_components: dict[str, pd.DataFrame] = {}
    for config in fixed_controls.values():
        metrics, components = evaluate_config(
            history,
            holdout_queries,
            config,
            feature_scales,
            horizons,
            thresholds,
            args.top_k,
        )
        holdout_metrics[config.name] = metrics
        holdout_components[config.name] = components
        print(json.dumps({"stage": "holdout", "config": config.name, **summarize_metrics(metrics)}), flush=True)
    baseline_metrics = summarize_metrics(holdout_metrics[baseline.name])
    winner_metrics = summarize_metrics(holdout_metrics[winner_name])
    promotion = (
        winner_name != baseline.name
        and winner_metrics["distinct_observations"] >= 500
        and winner_metrics["risk_interval_mae_pct"] < baseline_metrics["risk_interval_mae_pct"]
        and winner_metrics["time_mae_minutes"] < baseline_metrics["time_mae_minutes"]
        and winner_metrics["fill_brier"] <= baseline_metrics["fill_brier"]
        and winner_metrics["outcome_brier"] <= baseline_metrics["outcome_brier"]
    )
    baseline_frame = holdout_metrics[baseline.name]
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "baseline_version": SOFT_MATCHER_BASELINE_VERSION,
        "timeframe": args.timeframe,
        "asset_pool_policy": (
            "same asset only, matching the isolated live 1m libraries"
            if args.timeframe == "1m"
            else "five-asset pooled, matching the live 5m library"
        ),
        "objective": "entry-relative adverse risk and remaining waiting time; route appearance is not scored",
        "dataset": {
            "schema_version": metadata.get("schema_version"),
            "entry_ages_minutes": metadata.get("entry_ages_minutes"),
            "horizons_minutes": metadata.get("horizons_minutes"),
            "adverse_thresholds_pct": metadata.get("adverse_thresholds_pct"),
            "series": metadata.get("series"),
        },
        "split": {
            **boundaries,
            "fit_observations": int(len(fit)),
            "validation_observations": int(len(validation)),
            "holdout_observations": int(len(holdout)),
            "validation_queries_sampled": int(len(validation_queries)),
            "holdout_queries_sampled": int(len(holdout_queries)),
            "label_availability": "candidate maximum-horizon labels must be fully observed before each query entry time",
        },
        "validation_ranking": ranking.to_dict(orient="records"),
        "validation_selected_config": winner_name,
        "holdout": {name: summarize_metrics(frame) for name, frame in holdout_metrics.items()},
        "baseline_holdout_subgroups": {
            "direction": subgroup_report(baseline_frame, "direction"),
            "entry_age_minutes": subgroup_report(baseline_frame, "entry_age_minutes"),
            "departure_distance": subgroup_report(baseline_frame, "distance_bucket"),
        },
        "score_component_contributions": {
            name: {
                column: float(frame[column].mean())
                for column in frame.columns
                if column != "observation_id"
            }
            for name, frame in holdout_components.items()
            if len(frame)
        },
        "promotion_gate": {
            "promote_validation_winner": bool(promotion),
            "rule": "requires at least 500 held-out observations and simultaneous improvement in adverse risk and time with no fill/outcome Brier regression",
            "decision": (
                "eligible for a separate live challenger review"
                if promotion
                else "keep the versioned 70/30 pooled baseline"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(json_safe(report), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
