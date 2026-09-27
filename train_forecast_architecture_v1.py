#!/usr/bin/env python3
"""Build the frozen production artifact for the validated 5-minute architecture."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from dynamic_neighborhood import NeighborhoodConfig
from evaluate_behavioral_fingerprint import (
    BASE_COLUMNS,
    CONTEXT_COLUMNS,
    HORIZONS_MINUTES,
    build_sequence_matrix,
    robust_block_transform,
    sequence_column_names,
)
from evaluate_prospective_entry_baseline import read_observations, stratified_sample
from evaluate_rolling_forecast_systems import (
    build_dynamic_pool,
    dynamic_prediction_rows,
    episode_sample_weights,
    supervised_prediction_rows,
    support_stack_matrix,
)
from forecast_architecture_v1 import (
    ARCHITECTURE_ID,
    ARCHITECTURE_SCHEMA_VERSION,
    save_architecture,
)
from forecast_artifact_manifest import active_manifest_path, load_active_manifest
from optimize_behavioral_fingerprint import FingerprintConfig, scaled_blocks
from prospective_entry_model import default_artifact_root
from prospective_entry_outcomes import horizon_slug
from train_prospective_entry_model import (
    CONFIGS,
    atomic_json,
    build_feature_matrix,
    calibrate_ranges,
    fit_bundle,
)


CALIBRATION_FRACTION = 0.20
CALIBRATION_QUERIES = 750
FIXED_FOREST_CONFIG = "moderate"
FROZEN_RETRIEVAL = {
    1_440: {
        "fingerprint": FingerprintConfig(
            "context_1_recent_episode_0.5",
            {"state": 1.0, "context": 1.0, "recent": 0.5, "episode": 0.5},
        ),
        "neighborhood": NeighborhoodConfig(48, "inverse_square", None),
    },
    10_080: {
        "fingerprint": FingerprintConfig(
            "context_0.5_all_sequence_0.5",
            {
                "state": 1.0,
                "context": 0.5,
                "pre": 0.5,
                "recent": 0.5,
                "episode": 0.5,
            },
        ),
        "neighborhood": NeighborhoodConfig(32, "uniform", None),
    },
    43_200: {
        "fingerprint": FingerprintConfig(
            "context_0.5_all_sequence_0.5",
            {
                "state": 1.0,
                "context": 0.5,
                "pre": 0.5,
                "recent": 0.5,
                "episode": 0.5,
            },
        ),
        "neighborhood": NeighborhoodConfig(48, "inverse_square", None),
    },
}
BLOCK_ORDER = ("state", "context", "pre", "recent", "episode")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def calibration_split(observations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    signals = np.sort(observations["signal_open_time_ms"].unique()).astype(np.int64)
    boundary = int(signals[max(1, int(len(signals) * (1.0 - CALIBRATION_FRACTION)))])
    window = observations.loc[observations["signal_open_time_ms"].ge(boundary)].copy()
    maximum_slug = horizon_slug(max(HORIZONS_MINUTES))
    window = window.loc[window[f"horizon_{maximum_slug}_fully_observed"].astype(bool)]
    queries = stratified_sample(window, CALIBRATION_QUERIES).sort_index()
    if queries.empty:
        raise RuntimeError("No fully matured calibration queries are available")
    first_query_close_ms = int(queries["entry_close_time_ms"].min())
    available_ms = (
        observations["entry_close_time_ms"].to_numpy(dtype=np.int64)
        + max(HORIZONS_MINUTES) * 60_000
    )
    history = observations.loc[
        observations["signal_open_time_ms"].lt(boundary).to_numpy()
        & observations[f"horizon_{maximum_slug}_fully_observed"].astype(bool).to_numpy()
        & (available_ms <= first_query_close_ms)
    ].copy()
    if history.empty:
        raise RuntimeError("No label-mature history exists before calibration")
    return history.sort_index(), queries, boundary


def _fit_e0_heads(
    a_rows: dict[int, pd.DataFrame],
    b_rows: dict[int, pd.DataFrame],
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for horizon in HORIZONS_MINUTES:
        train_a = a_rows[horizon]
        train_b = b_rows[horizon]
        matrix = support_stack_matrix(train_a, train_b, include_support=False)
        weights = episode_sample_weights(train_a)
        fill_y = train_a["actual_fill"].to_numpy(dtype=int)
        fill_model = None
        fill_constant = None
        if len(np.unique(fill_y)) == 1:
            fill_constant = float(fill_y[0])
        else:
            fill_model = HistGradientBoostingClassifier(
                learning_rate=0.045,
                max_iter=160,
                max_leaf_nodes=10,
                min_samples_leaf=25,
                l2_regularization=3.0,
                random_state=161_000 + int(horizon),
            ).fit(matrix, fill_y, sample_weight=weights)
        risk_model = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.045,
            max_iter=160,
            max_leaf_nodes=10,
            min_samples_leaf=25,
            l2_regularization=3.0,
            random_state=162_000 + int(horizon),
        ).fit(
            matrix,
            train_a["actual_adverse_pct"].to_numpy(dtype=float),
            sample_weight=weights,
        )
        output[horizon_slug(horizon)] = {
            "fill_model": fill_model,
            "fill_constant": fill_constant,
            "risk_model": risk_model,
            "training_rows": int(len(train_a)),
            "training_episodes": int(train_a["signal_id"].nunique()),
        }
    return output


def _block_columns() -> dict[str, list[str]]:
    sequence_columns = sequence_column_names()
    return {
        "state": list(BASE_COLUMNS),
        "context": list(CONTEXT_COLUMNS),
        "pre": [name for name in sequence_columns if name.startswith("sequence_pre_")],
        "recent": [
            name for name in sequence_columns if name.startswith("sequence_recent_")
        ],
        "episode": [
            name for name in sequence_columns if name.startswith("sequence_episode_")
        ],
    }


def _fit_support_artifact(
    history: pd.DataFrame,
    features: pd.DataFrame,
    sequences: pd.DataFrame,
    horizon: int,
    config: FingerprintConfig,
    neighborhood: NeighborhoodConfig,
    calibration_rows: pd.DataFrame,
) -> dict[str, Any]:
    columns_by_block = _block_columns()
    parts: list[np.ndarray] = []
    blocks: dict[str, Any] = {}
    active_order: list[str] = []
    for block in BLOCK_ORDER:
        weight = float(config.weights.get(block, 0.0))
        if weight <= 0.0:
            continue
        columns = columns_by_block[block]
        source = features if block in {"state", "context"} else sequences
        transformed, _, scaler = robust_block_transform(
            source.loc[history.index, columns], source.loc[history.index[:1], columns]
        )
        parts.append(transformed * weight)
        active_order.append(block)
        blocks[block] = {"columns": columns, "scaler": scaler, "weight": weight}
    slug = horizon_slug(horizon)
    lower = history[f"max_adverse_pre_target_lower_{slug}_pct"].to_numpy(dtype=float)
    upper = history[f"max_adverse_pre_target_upper_{slug}_pct"].to_numpy(dtype=float)
    waits = history["target_touch_bars_from_entry"].to_numpy(dtype=float) * 5.0
    return {
        "fingerprint_name": config.name,
        "block_order": active_order,
        "blocks": blocks,
        "neighborhood": {
            "name": neighborhood.name,
            "k": neighborhood.k,
            "weighting": neighborhood.weighting,
            "radius_quantile": neighborhood.radius_quantile,
            "maximum_distance": None,
        },
        "history_matrix": np.concatenate(parts, axis=1).astype(np.float32),
        "history_signal_ids": history["signal_id"].astype(str).to_numpy(),
        "history_assets": history["asset"].astype(str).to_numpy(),
        "history_fill": history[f"target_hit_{slug}"].astype(float).to_numpy(),
        "history_adverse_pct": ((lower + upper) / 2.0).astype(np.float32),
        "history_wait_minutes": waits.astype(np.float32),
        "calibration_nearest_neighbor_distances": np.sort(
            calibration_rows["nearest_distance"].to_numpy(dtype=np.float32)
        ),
        "calibration_median_neighbor_distances": np.sort(
            calibration_rows["median_neighbor_distance"].to_numpy(dtype=np.float32)
        ),
        "history_rows": int(len(history)),
        "history_episodes": int(history["signal_id"].nunique()),
    }


def _development_evidence(root: Path) -> dict[str, Any] | None:
    path = root / "data" / "rolling_forecast_comparison_5m.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "source": str(path),
        "status": "development evidence; prospective validation pending",
        "comparison_vs_a": payload.get("aggregate_test_comparison_vs_A", {}).get(
            "E0_prediction_stack_supervised"
        ),
    }


def train(
    root: Path,
    dataset_dir: Path,
    output: Path,
    architecture_id: str = ARCHITECTURE_ID,
) -> dict[str, Any]:
    manifest_path = active_manifest_path(output.parent)
    if manifest_path.exists():
        manifest = load_active_manifest(output.parent)
        active_output = (output.parent / str(manifest["artifact_file"])).resolve()
        if bool(manifest["frozen"]) and output.resolve() == active_output:
            raise RuntimeError(
                "Refusing to overwrite the active frozen forecast artifact; "
                "train a uniquely versioned candidate instead"
            )
    observations, metadata = read_observations(dataset_dir, "5m")
    observations = observations.reset_index(drop=True)
    print(json.dumps({"stage": "features", "rows": len(observations)}), flush=True)
    features = build_feature_matrix(root, observations)
    sequences = build_sequence_matrix(root, observations)
    feature_sets = {
        "A": features,
        "B": pd.concat([features, sequences], axis=1),
    }
    history, calibration, boundary = calibration_split(observations)
    config = next(value for value in CONFIGS if value.name == FIXED_FOREST_CONFIG)

    calibration_bundles: dict[str, dict[str, Any]] = {}
    calibration_predictions: dict[str, dict[int, pd.DataFrame]] = {}
    for name, matrix in feature_sets.items():
        print(json.dumps({"stage": "calibration_base", "system": name}), flush=True)
        bundle = fit_bundle(history, matrix, HORIZONS_MINUTES, (), config, "5m")
        calibration_bundles[name] = bundle
        calibration_predictions[name] = supervised_prediction_rows(
            bundle, calibration, matrix
        )
    e0_heads = _fit_e0_heads(calibration_predictions["A"], calibration_predictions["B"])

    print(json.dumps({"stage": "support_calibration"}), flush=True)
    retrieval_blocks = scaled_blocks(history, calibration, features, sequences)
    calibrated_support: dict[int, pd.DataFrame] = {}
    for horizon, specification in FROZEN_RETRIEVAL.items():
        fingerprint = specification["fingerprint"]
        neighborhood = specification["neighborhood"]
        pool_positions, pool_distances = build_dynamic_pool(
            retrieval_blocks, fingerprint, history
        )
        calibrated_support[horizon] = dynamic_prediction_rows(
            history,
            calibration,
            pool_positions,
            pool_distances,
            neighborhood,
            None,
            horizon,
        )

    maximum_slug = horizon_slug(max(HORIZONS_MINUTES))
    deployment = observations.loc[
        observations[f"horizon_{maximum_slug}_fully_observed"].astype(bool)
    ].copy()
    thresholds = tuple(float(value) for value in metadata["adverse_thresholds_pct"])
    print(json.dumps({"stage": "deployment_base", "rows": len(deployment)}), flush=True)
    base_a = fit_bundle(deployment, features, HORIZONS_MINUTES, thresholds, config, "5m")
    base_b = fit_bundle(deployment, feature_sets["B"], HORIZONS_MINUTES, (), config, "5m")
    calibrate_ranges(base_a, calibration, features)
    calibrate_ranges(base_b, calibration, feature_sets["B"])

    support_artifacts: dict[str, Any] = {}
    for horizon, specification in FROZEN_RETRIEVAL.items():
        print(json.dumps({"stage": "support_index", "horizon": horizon}), flush=True)
        support_artifacts[horizon_slug(horizon)] = _fit_support_artifact(
            deployment,
            features,
            sequences,
            horizon,
            specification["fingerprint"],
            specification["neighborhood"],
            calibrated_support[horizon],
        )

    generated_at = utc_now()
    latest_label_ms = int(
        (deployment["entry_close_time_ms"] + max(HORIZONS_MINUTES) * 60_000).max()
    )
    bundle = {
        "schema_version": ARCHITECTURE_SCHEMA_VERSION,
        "architecture_id": architecture_id,
        "generated_at_utc": generated_at,
        "timeframe": "5m",
        "horizons_minutes": list(HORIZONS_MINUTES),
        "base_a": base_a,
        "base_b": base_b,
        "e0_heads": e0_heads,
        "support_artifacts": support_artifacts,
        "ownership": {
            "fill_probability": "E0 prediction-only numerical stack",
            "adverse_p50": "E0 prediction-only numerical stack",
            "adverse_p80_p90": "A calibrated tail",
            "waiting_time": "A supervised model",
            "historical_support": "C2 dynamic weighted retrieval",
            "routes": "separate historical illustration layer",
        },
        "training_rows": int(len(deployment)),
        "training_episodes": int(deployment["signal_id"].nunique()),
        "support": {
            "entry_age_minutes_min": int(deployment["entry_age_minutes"].min()),
            "entry_age_minutes_max": int(deployment["entry_age_minutes"].max()),
            "training_assets": sorted(deployment["asset"].astype(str).unique()),
        },
        "calibration_rows": int(len(calibration)),
        "calibration_episodes": int(calibration["signal_id"].nunique()),
        "calibration_boundary_signal_open_time_ms": boundary,
        "training_label_cutoff_utc": datetime.fromtimestamp(
            latest_label_ms / 1000, tz=timezone.utc
        ).isoformat().replace("+00:00", "Z"),
        "development_evidence": _development_evidence(root),
        "frozen": True,
        "prospective_validation_status": "collecting",
    }
    save_architecture(bundle, output)
    report = {
        key: value
        for key, value in bundle.items()
        if key
        not in {"base_a", "base_b", "e0_heads", "support_artifacts"}
    }
    report["artifact"] = str(output)
    report["artifact_bytes"] = int(output.stat().st_size)
    report["frozen_retrieval"] = {
        str(horizon): {
            "fingerprint": specification["fingerprint"].name,
            "weights": specification["fingerprint"].weights,
            "neighborhood": specification["neighborhood"].name,
        }
        for horizon, specification in FROZEN_RETRIEVAL.items()
    }
    return report


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--architecture-id", default=ARCHITECTURE_ID)
    parser.add_argument(
        "--dataset-dir", type=Path, default=root / "data" / "prospective_entry_outcomes_v1"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "data" / "forecast_architecture_v1" / "5m" / "training_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or (
        default_artifact_root(args.root) / "5m" / f"{args.architecture_id}.joblib"
    )
    report = train(
        args.root.resolve(),
        args.dataset_dir.resolve(),
        output.resolve(),
        architecture_id=str(args.architecture_id),
    )
    atomic_json(args.report.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
