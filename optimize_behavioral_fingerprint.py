"""Select behavioral-fingerprint weights on validation, then score holdout once."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_behavioral_fingerprint import (
    BASE_COLUMNS,
    CONTEXT_COLUMNS,
    HORIZONS_MINUTES,
    atomic_json,
    build_sequence_matrix,
    episode_balanced_comparison,
    evaluate_horizon,
    retrieve_neighbors,
    robust_block_transform,
    sequence_column_names,
    utc_now,
)
from evaluate_prospective_entry_baseline import (
    chronological_signal_split,
    read_observations,
    stratified_sample,
)
from train_prospective_entry_model import build_feature_matrix


MAXIMUM_HORIZON_MINUTES = max(HORIZONS_MINUTES)
BLOCK_ORDER = ("state", "context", "pre", "recent", "episode")


@dataclass(frozen=True)
class FingerprintConfig:
    name: str
    weights: dict[str, float]


def candidate_configs() -> tuple[FingerprintConfig, ...]:
    configs = [FingerprintConfig("state_only", {"state": 1.0})]
    for context_weight in (0.25, 0.5, 1.0, 2.0):
        configs.append(
            FingerprintConfig(
                f"context_{context_weight:g}",
                {"state": 1.0, "context": context_weight},
            )
        )
    for context_weight in (0.5, 1.0):
        for sequence_weight in (0.25, 0.5, 1.0, 2.0):
            configs.append(
                FingerprintConfig(
                    f"context_{context_weight:g}_all_sequence_{sequence_weight:g}",
                    {
                        "state": 1.0,
                        "context": context_weight,
                        "pre": sequence_weight,
                        "recent": sequence_weight,
                        "episode": sequence_weight,
                    },
                )
            )
    for view in ("pre", "recent", "episode"):
        for sequence_weight in (0.5, 1.0):
            configs.append(
                FingerprintConfig(
                    f"context_1_{view}_{sequence_weight:g}",
                    {"state": 1.0, "context": 1.0, view: sequence_weight},
                )
            )
    for sequence_weight in (0.5, 1.0):
        configs.append(
            FingerprintConfig(
                f"context_1_recent_episode_{sequence_weight:g}",
                {
                    "state": 1.0,
                    "context": 1.0,
                    "recent": sequence_weight,
                    "episode": sequence_weight,
                },
            )
        )
    return tuple(configs)


def split_block_columns() -> dict[str, tuple[str, ...]]:
    sequence_columns = sequence_column_names()
    return {
        "state": tuple(BASE_COLUMNS),
        "context": tuple(CONTEXT_COLUMNS),
        "pre": tuple(column for column in sequence_columns if column.startswith("sequence_pre_")),
        "recent": tuple(
            column for column in sequence_columns if column.startswith("sequence_recent_")
        ),
        "episode": tuple(
            column for column in sequence_columns if column.startswith("sequence_episode_")
        ),
    }


def eligible_populations(
    history: pd.DataFrame,
    future: pd.DataFrame,
    maximum_queries: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fully_observed = f"horizon_{MAXIMUM_HORIZON_MINUTES}m_fully_observed"
    future = future.loc[future[fully_observed].astype(bool)].copy()
    if future.empty:
        raise RuntimeError("No fully observed future queries are available")
    first_query_close = int(future["entry_close_time_ms"].min())
    available_at = (
        history["entry_close_time_ms"].to_numpy(dtype=np.int64)
        + MAXIMUM_HORIZON_MINUTES * 60_000
    )
    history = history.loc[
        history[fully_observed].astype(bool).to_numpy() & (available_at <= first_query_close)
    ].copy()
    queries = stratified_sample(future, maximum_queries).sort_index()
    return history.sort_index(), queries


def scaled_blocks(
    history: pd.DataFrame,
    queries: pd.DataFrame,
    features: pd.DataFrame,
    sequences: pd.DataFrame,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for block, columns in split_block_columns().items():
        source = features if block in {"state", "context"} else sequences
        fit_values, query_values, _ = robust_block_transform(
            source.loc[history.index, list(columns)],
            source.loc[queries.index, list(columns)],
        )
        output[block] = fit_values, query_values
    return output


def compose_matrices(
    blocks: dict[str, tuple[np.ndarray, np.ndarray]],
    config: FingerprintConfig,
) -> tuple[np.ndarray, np.ndarray]:
    fit_parts: list[np.ndarray] = []
    query_parts: list[np.ndarray] = []
    for block in BLOCK_ORDER:
        weight = float(config.weights.get(block, 0.0))
        if weight <= 0.0:
            continue
        fit, query = blocks[block]
        fit_parts.append(fit * weight)
        query_parts.append(query * weight)
    return np.concatenate(fit_parts, axis=1), np.concatenate(query_parts, axis=1)


def evaluate_config(
    config: FingerprintConfig,
    blocks: dict[str, tuple[np.ndarray, np.ndarray]],
    history: pd.DataFrame,
    queries: pd.DataFrame,
    neighbors: int,
) -> dict[str, Any]:
    history_matrix, query_matrix = compose_matrices(blocks, config)
    selected, distances = retrieve_neighbors(
        history_matrix,
        query_matrix,
        history["signal_id"].to_numpy(),
        neighbors,
    )
    horizons: dict[str, Any] = {}
    rows: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS_MINUTES:
        metrics, predictions = evaluate_horizon(history, queries, selected, horizon)
        horizons[str(horizon)] = metrics
        rows[horizon] = predictions
    return {
        "config": {"name": config.name, "weights": config.weights},
        "dimensions": int(history_matrix.shape[1]),
        "nearest_distance_p50": float(np.median(distances)),
        "horizons": horizons,
        "rows": rows,
    }


def selection_record(
    baseline_rows: pd.DataFrame,
    challenger_rows: pd.DataFrame,
) -> dict[str, Any]:
    comparison = episode_balanced_comparison(
        baseline_rows,
        challenger_rows,
        bootstrap_samples=250,
    )
    ratios = {
        metric: float(values["challenger_error"]) / max(float(values["baseline_error"]), 1e-12)
        for metric, values in comparison.items()
    }
    return {
        "mean_error_ratio": float(np.mean(list(ratios.values()))),
        "worst_error_ratio": float(np.max(list(ratios.values()))),
        "metric_error_ratios": ratios,
    }


def choose_configs(
    results: dict[str, dict[str, Any]],
    configs: tuple[FingerprintConfig, ...],
    regression_tolerance: float,
) -> tuple[dict[int, FingerprintConfig], dict[str, Any]]:
    baseline = results["state_only"]
    selected: dict[int, FingerprintConfig] = {}
    evidence: dict[str, Any] = {}
    for horizon in HORIZONS_MINUTES:
        records: list[dict[str, Any]] = []
        for config in configs:
            if config.name == "state_only":
                record = {
                    "name": config.name,
                    "weights": config.weights,
                    "mean_error_ratio": 1.0,
                    "worst_error_ratio": 1.0,
                    "metric_error_ratios": {
                        "fill_brier": 1.0,
                        "adverse_p50_absolute_error_pct": 1.0,
                        "wait_p50_absolute_error_minutes": 1.0,
                    },
                }
            else:
                score = selection_record(
                    baseline["rows"][horizon],
                    results[config.name]["rows"][horizon],
                )
                record = {"name": config.name, "weights": config.weights, **score}
            records.append(record)
        eligible = [
            record
            for record in records
            if float(record["worst_error_ratio"]) <= 1.0 + regression_tolerance
        ]
        winner = min(eligible, key=lambda item: (float(item["mean_error_ratio"]), item["name"]))
        selected[horizon] = next(config for config in configs if config.name == winner["name"])
        evidence[str(horizon)] = {
            "selected": winner,
            "all_candidates": sorted(records, key=lambda item: float(item["mean_error_ratio"])),
            "regression_tolerance": regression_tolerance,
        }
    return selected, evidence


def public_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key != "rows"}


def optimize(
    root: Path,
    dataset_dir: Path,
    validation_queries: int,
    holdout_queries: int,
    neighbors: int,
    regression_tolerance: float,
) -> dict[str, Any]:
    observations, metadata = read_observations(dataset_dir, "5m")
    observations = observations.reset_index(drop=True)
    features = build_feature_matrix(root, observations)
    sequences = build_sequence_matrix(root, observations)
    fit, validation, holdout, boundaries = chronological_signal_split(observations)
    validation_history, validation_sample = eligible_populations(fit, validation, validation_queries)
    validation_blocks = scaled_blocks(validation_history, validation_sample, features, sequences)
    configs = candidate_configs()
    validation_results: dict[str, dict[str, Any]] = {}
    for config in configs:
        print(json.dumps({"stage": "validation", "config": config.name}), flush=True)
        validation_results[config.name] = evaluate_config(
            config,
            validation_blocks,
            validation_history,
            validation_sample,
            neighbors,
        )
    selected, selection_evidence = choose_configs(
        validation_results,
        configs,
        regression_tolerance,
    )

    holdout_history, holdout_sample = eligible_populations(
        pd.concat([fit, validation]).sort_index(),
        holdout,
        holdout_queries,
    )
    holdout_blocks = scaled_blocks(holdout_history, holdout_sample, features, sequences)
    holdout_configs = {"state_only": next(config for config in configs if config.name == "state_only")}
    holdout_configs.update({config.name: config for config in selected.values()})
    holdout_results: dict[str, dict[str, Any]] = {}
    for config in holdout_configs.values():
        print(json.dumps({"stage": "holdout", "config": config.name}), flush=True)
        holdout_results[config.name] = evaluate_config(
            config,
            holdout_blocks,
            holdout_history,
            holdout_sample,
            neighbors,
        )

    baseline = holdout_results["state_only"]
    final_comparison: dict[str, Any] = {}
    for horizon, config in selected.items():
        challenger = holdout_results[config.name]
        final_comparison[str(horizon)] = {
            "selected_config": {"name": config.name, "weights": config.weights},
            "row_weighted_baseline": baseline["horizons"][str(horizon)],
            "row_weighted_challenger": challenger["horizons"][str(horizon)],
            "episode_balanced": episode_balanced_comparison(
                baseline["rows"][horizon],
                challenger["rows"][horizon],
                seed=20260927 + horizon,
            ),
        }

    return {
        "schema_version": "behavioral-fingerprint-optimization-v1.0.0",
        "generated_at_utc": utc_now(),
        "status": "research_only_not_promoted",
        "timeframe": "5m",
        "assets": sorted(observations["asset"].astype(str).unique().tolist()),
        "population": {
            "all_observations": int(len(observations)),
            "validation_history": int(len(validation_history)),
            "validation_queries": int(len(validation_sample)),
            "holdout_history": int(len(holdout_history)),
            "holdout_queries": int(len(holdout_sample)),
            "holdout_query_episodes": int(holdout_sample["signal_id"].nunique()),
            "neighbors": int(neighbors),
            "candidate_configs": int(len(configs)),
        },
        "chronological_split": boundaries,
        "selection_policy": {
            "source": "validation period only",
            "score": "mean of episode-balanced challenger/baseline error ratios",
            "constraint": (
                "no fill, adverse, or waiting-time error ratio may exceed "
                f"{1.0 + regression_tolerance:.3f} on validation"
            ),
        },
        "validation_selection": selection_evidence,
        "holdout_results": {
            name: public_result(result) for name, result in holdout_results.items()
        },
        "final_holdout_comparison": final_comparison,
        "label_availability": (
            "For each split, candidate 30-day labels ended before its first query close"
        ),
        "future_data_policy": "future candles are labels only and never enter a fingerprint",
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
    parser.add_argument("--validation-queries", type=int, default=1_000)
    parser.add_argument("--holdout-queries", type=int, default=1_500)
    parser.add_argument("--neighbors", type=int, default=32)
    parser.add_argument("--regression-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/behavioral_fingerprint_optimized_5m.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else root / args.dataset_dir
    output = args.output if args.output.is_absolute() else root / args.output
    report = optimize(
        root,
        dataset_dir,
        args.validation_queries,
        args.holdout_queries,
        args.neighbors,
        args.regression_tolerance,
    )
    atomic_json(output, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": str(output),
                "population": report["population"],
                "selected": {
                    horizon: result["selected_config"]
                    for horizon, result in report["final_holdout_comparison"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
