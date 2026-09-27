"""Frozen 5-minute forecast architecture composed from validated research systems.

The numerical forecast and historical-support diagnostic deliberately remain
separate.  E0 combines the A/B supervised predictions for fill and central
adverse risk.  A continues to own waiting-time ranges and adverse tail ranges.
C2 supplies historical support plus eligible historical route candidates;
support is not interpreted as confidence and does not change E0/A predictions.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from dynamic_neighborhood import (
    NeighborhoodConfig,
    neighborhood_support,
    resolve_with_nearest_fallback,
    retrieve_episode_neighbor_pool,
    weighted_quantile,
)
from evaluate_behavioral_fingerprint import (
    sequence_column_names,
    sequence_fingerprint_from_close,
)
from prospective_entry_model import predict_bundle as predict_prospective_bundle
from prospective_entry_outcomes import horizon_slug


ARCHITECTURE_SCHEMA_VERSION = "forecast-architecture-v1.0.0"
ARCHITECTURE_ID = "forecast_architecture_v1_5m_2026-09-27"
SUPPORT_LABELS = ("very_high", "high", "medium", "low", "very_low")


def sequence_features_from_live(
    frame: pd.DataFrame,
    observation: Mapping[str, Any],
) -> pd.DataFrame:
    values = sequence_fingerprint_from_close(
        frame["close"].to_numpy(dtype=np.float64), observation
    )
    return pd.DataFrame([values], columns=sequence_column_names())


def prediction_stack_row(
    a_prediction: Mapping[str, Any],
    b_prediction: Mapping[str, Any],
    slug: str,
) -> np.ndarray:
    return np.asarray(
        [[
            float(a_prediction["fill_probability"][slug]),
            float(a_prediction["additional_adverse_pct"][slug]["p50"]),
            math.log1p(
                float(
                    a_prediction["remaining_time_minutes_if_filled_within_horizon"][
                        slug
                    ]["p50"]
                )
            ),
            float(b_prediction["fill_probability"][slug]),
            float(b_prediction["additional_adverse_pct"][slug]["p50"]),
            math.log1p(
                float(
                    b_prediction["remaining_time_minutes_if_filled_within_horizon"][
                        slug
                    ]["p50"]
                )
            ),
        ]],
        dtype=np.float32,
    )


def _fill_probability(model: Any | None, constant: float | None, matrix: np.ndarray) -> float:
    if model is None:
        if constant is None:
            raise RuntimeError("Frozen fill head has neither a model nor a constant")
        return float(constant)
    classes = np.asarray(model.classes_)
    probabilities = model.predict_proba(matrix)[0]
    matches = np.flatnonzero(classes == 1)
    return float(probabilities[int(matches[0])]) if len(matches) else 0.0


def _scaled_block(
    source: pd.DataFrame,
    columns: list[str],
    scaler: Mapping[str, Any],
) -> np.ndarray:
    values = source.loc[:, columns].to_numpy(dtype=np.float64)
    median = np.asarray(scaler["median"], dtype=np.float64)
    scale = np.asarray(scaler["scale"], dtype=np.float64)
    transformed = np.nan_to_num(
        (values - median) / scale,
        nan=0.0,
        posinf=8.0,
        neginf=-8.0,
    )
    return (np.clip(transformed, -8.0, 8.0) / math.sqrt(max(len(columns), 1))).astype(
        np.float32
    )


def _support_for_horizon(
    artifact: Mapping[str, Any],
    features: pd.DataFrame,
    sequences: pd.DataFrame,
    asset: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query_parts: list[np.ndarray] = []
    for block in artifact["block_order"]:
        block_spec = artifact["blocks"][block]
        source = features if block in {"state", "context"} else sequences
        query_parts.append(
            _scaled_block(source, list(block_spec["columns"]), block_spec["scaler"])
            * float(block_spec["weight"])
        )
    query_matrix = np.concatenate(query_parts, axis=1)
    history_matrix = np.asarray(artifact["history_matrix"], dtype=np.float32)
    pool_positions, pool_distances = retrieve_episode_neighbor_pool(
        history_matrix,
        query_matrix,
        np.asarray(artifact["history_signal_ids"], dtype=str),
        maximum_neighbors=96,
    )
    config_data = artifact["neighborhood"]
    config = NeighborhoodConfig(
        int(config_data["k"]),
        str(config_data["weighting"]),
        config_data.get("radius_quantile"),
    )
    chosen, distances, weights, radius_failed = resolve_with_nearest_fallback(
        pool_positions[0],
        pool_distances[0],
        config,
        config_data.get("maximum_distance"),
    )
    fill_values = np.asarray(artifact["history_fill"], dtype=float)[chosen]
    adverse_values = np.asarray(artifact["history_adverse_pct"], dtype=float)[chosen]
    wait_values = np.asarray(artifact["history_wait_minutes"], dtype=float)[chosen]
    total_weight = max(float(np.sum(weights)), 1e-12)
    fill_probability = float(np.sum(weights * fill_values) / total_weight)
    support = neighborhood_support(
        distances,
        weights,
        fill_probability,
        adverse_values,
        wait_values,
        radius_failed,
    )
    median_reference = np.asarray(
        artifact["calibration_median_neighbor_distances"], dtype=float
    )
    nearest_reference = np.asarray(
        artifact["calibration_nearest_neighbor_distances"], dtype=float
    )
    median_percentile = float(
        np.searchsorted(median_reference, support["median_neighbor_distance"], side="right")
        / max(len(median_reference), 1)
    )
    nearest_percentile = float(
        np.searchsorted(nearest_reference, support["nearest_distance"], side="right")
        / max(len(nearest_reference), 1)
    )
    bucket_index = min(4, max(0, int(median_percentile * 5.0)))
    history_assets = np.asarray(artifact["history_assets"], dtype=str)[chosen]
    same_asset_share = float(
        np.sum(weights * (history_assets == str(asset)).astype(float)) / total_weight
    )
    route_candidates: list[dict[str, Any]] = []
    all_signal_ids = np.asarray(artifact["history_signal_ids"], dtype=str)
    all_assets = np.asarray(artifact["history_assets"], dtype=str)
    all_adverse = np.asarray(artifact["history_adverse_pct"], dtype=float)
    all_wait = np.asarray(artifact["history_wait_minutes"], dtype=float)
    for rank, (position, distance) in enumerate(
        zip(pool_positions[0], pool_distances[0], strict=True), start=1
    ):
        wait_minutes = float(all_wait[int(position)])
        adverse_pct = float(all_adverse[int(position)])
        if not np.isfinite(wait_minutes) or wait_minutes <= 0.0:
            continue
        route_candidates.append(
            {
                "episode_id": str(all_signal_ids[int(position)]),
                "asset": str(all_assets[int(position)]),
                "distance": float(distance),
                "neighbor_rank": rank,
                "wait_minutes": wait_minutes,
                "adverse_pct": adverse_pct if np.isfinite(adverse_pct) else None,
            }
        )
    result = {
        "level": SUPPORT_LABELS[bucket_index],
        "low_support": bool(
            float(support["effective_neighbors"]) < 8.0
            or bool(support["radius_failed"])
            or median_percentile >= 0.80
        ),
        "raw_neighbors": int(support["raw_neighbors"]),
        "effective_neighbors": round(float(support["effective_neighbors"]), 2),
        "nearest_distance_percentile": round(nearest_percentile, 4),
        "median_distance_percentile": round(median_percentile, 4),
        "fill_agreement": round(float(support["fill_agreement"]), 4),
        "same_asset_neighbor_share": round(same_asset_share, 4),
        "adverse_outcome_dispersion_pct": round(
            float(support["adverse_outcome_dispersion_pct"]), 4
        ),
        "wait_outcome_iqr_minutes": round(
            float(support["wait_outcome_iqr_minutes"]), 2
        ),
        "retrieval_fill_probability": round(fill_probability, 6),
        "retrieval_adverse_p50_pct": round(
            weighted_quantile(adverse_values, weights, 0.5), 6
        ),
        "method": "C2 dynamic weighted historical retrieval",
    }
    return result, route_candidates


def predict_architecture(
    bundle: Mapping[str, Any],
    features: pd.DataFrame,
    sequences: pd.DataFrame,
    asset: str,
) -> dict[str, Any]:
    if bundle.get("schema_version") != ARCHITECTURE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported forecast architecture schema: {bundle.get('schema_version')}"
        )
    a_prediction = predict_prospective_bundle(bundle["base_a"], features)
    b_features = pd.concat(
        [features.reset_index(drop=True), sequences.reset_index(drop=True)], axis=1
    )
    b_prediction = predict_prospective_bundle(bundle["base_b"], b_features)

    fill: dict[str, float] = {}
    adverse = {
        slug: dict(values)
        for slug, values in a_prediction["additional_adverse_pct"].items()
    }
    support: dict[str, Any] = {}
    route_candidates: dict[str, list[dict[str, Any]]] = {}
    running_fill = 0.0
    running_adverse = 0.0
    for horizon in bundle["horizons_minutes"]:
        slug = horizon_slug(int(horizon))
        matrix = prediction_stack_row(a_prediction, b_prediction, slug)
        head = bundle["e0_heads"][slug]
        running_fill = max(
            running_fill,
            _fill_probability(head.get("fill_model"), head.get("fill_constant"), matrix),
        )
        fill[slug] = min(1.0, running_fill)
        running_adverse = max(
            running_adverse,
            max(0.0, float(head["risk_model"].predict(matrix)[0])),
        )
        adverse[slug]["p50"] = running_adverse
        adverse[slug]["p80"] = max(float(adverse[slug]["p80"]), running_adverse)
        adverse[slug]["p90"] = max(float(adverse[slug]["p90"]), running_adverse)
        support[slug], route_candidates[slug] = _support_for_horizon(
            bundle["support_artifacts"][slug], features, sequences, asset
        )

    ownership = dict(bundle["ownership"])
    ownership["routes"] = "C2 real continuations aligned to frozen E0/A targets"
    return {
        "fill_probability": fill,
        "additional_adverse_pct": adverse,
        "remaining_time_minutes_if_filled_within_horizon": a_prediction[
            "remaining_time_minutes_if_filled_within_horizon"
        ],
        "competing_outcomes": a_prediction["competing_outcomes"],
        "historical_support": support,
        "historical_route_candidates": route_candidates,
        "ownership": ownership,
    }


def save_architecture(bundle: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(dict(bundle), temporary, compress=3)
    temporary.replace(path)


def load_architecture(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle.get("schema_version") != ARCHITECTURE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported forecast architecture schema: {bundle.get('schema_version')}"
        )
    return bundle
