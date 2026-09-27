#!/usr/bin/env python3
"""Context-aware all-outcome matcher used by research and live inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

from candle_archetype import ARCHETYPE_BLEND_WEIGHT, SOFT_MATCHER_BASELINE_VERSION
from evaluate_prospective_entry_baseline import (
    MatcherConfig,
    OUTCOME_CLASSES,
    robust_feature_scales,
    score_candidates,
)
from prospective_entry_model import FEATURE_COLUMNS_V1
from prospective_entry_outcomes import horizon_slug, threshold_slug


CONTEXTUAL_MATCHER_SCHEMA_VERSION = "contextual-entry-matcher-v1.0.0"
CONTEXT_FEATURES = tuple(
    column
    for column in FEATURE_COLUMNS_V1
    if column.startswith(("pre_", "recent_"))
    or column
    in {
        "entry_bar_range_pct",
        "entry_bar_body_pct",
        "signal_to_entry_aligned_return_pct",
    }
)


def context_scales(features: pd.DataFrame) -> dict[str, float]:
    result: dict[str, float] = {}
    for column in CONTEXT_FEATURES:
        values = features[column].to_numpy(dtype=float)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)) * 1.4826)
        result[column] = max(mad, float(np.std(values)), 1e-6)
    return result


def rank_candidates(
    candidates: pd.DataFrame,
    candidate_features: pd.DataFrame,
    query: pd.Series,
    query_features: pd.Series,
    base_scales: Mapping[str, float],
    rich_scales: Mapping[str, float],
    context_weight: float,
    top_k: int,
) -> pd.DataFrame:
    base = score_candidates(
        candidates,
        query,
        MatcherConfig(ARCHETYPE_BLEND_WEIGHT, True),
        dict(base_scales),
    )
    if base.empty or context_weight <= 0:
        return base.head(top_k)
    aligned = candidate_features.loc[base.index, CONTEXT_FEATURES]
    scale = np.asarray([rich_scales[column] for column in CONTEXT_FEATURES], dtype=float)
    difference = np.abs(
        aligned.to_numpy(dtype=float)
        - query_features.loc[list(CONTEXT_FEATURES)].to_numpy(dtype=float)[None, :]
    ) / scale[None, :]
    # A few extreme context fields should not drown out broad sequence agreement.
    context_distance = np.mean(np.minimum(difference, 12.0), axis=1)
    base_score = base["match_score"].to_numpy(dtype=float)
    scaled_context = context_distance * max(float(np.median(base_score)), 1e-9) / max(
        float(np.median(context_distance)), 1e-9
    )
    output = base.copy()
    output["base_match_score"] = base_score
    output["context_match_score"] = scaled_context
    output["match_score"] = (
        (1.0 - float(context_weight)) * base_score
        + float(context_weight) * scaled_context
    )
    return output.sort_values("match_score", kind="stable").head(top_k)


def predict_from_ranked(
    ranked: pd.DataFrame,
    horizons_minutes: list[int] | tuple[int, ...],
    thresholds_pct: list[float] | tuple[float, ...],
    interval_minutes: int,
) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    for horizon in horizons_minutes:
        slug = horizon_slug(int(horizon))
        known = ranked[f"horizon_{slug}_fully_observed"].astype(bool) | ranked[
            f"target_hit_{slug}"
        ].astype(bool)
        task = ranked.loc[known]
        if len(task) < 12:
            continue
        lower = task[f"max_adverse_pre_target_lower_{slug}_pct"].to_numpy(dtype=float)
        upper = task[f"max_adverse_pre_target_upper_{slug}_pct"].to_numpy(dtype=float)
        midpoint = (lower + upper) / 2.0
        hit = task[f"target_hit_{slug}"].astype(bool)
        hit_rows = task.loc[hit]
        outcomes: dict[str, dict[str, float]] = {}
        for threshold in thresholds_pct:
            key = threshold_slug(float(threshold))
            counts = task[f"outcome_{slug}_vs_{key}pct"].value_counts(normalize=True)
            outcomes[f"{key}pct"] = {
                label: float(counts.get(label, 0.0)) for label in OUTCOME_CLASSES
            }
        horizons[slug] = {
            "neighbors": int(len(task)),
            "fill_probability": float(hit.mean()),
            "additional_adverse_pct": {
                "p50": float(np.quantile(midpoint, 0.50)),
                "p80": float(np.quantile(upper, 0.80)),
                "p90": float(np.quantile(upper, 0.90)),
            },
            "remaining_time_minutes_if_filled_within_horizon": (
                None
                if hit_rows.empty
                else {
                    "p10": float(
                        np.quantile(
                            hit_rows["target_touch_bars_from_entry"].to_numpy(dtype=float)
                            * interval_minutes,
                            0.10,
                        )
                    ),
                    "p50": float(
                        np.quantile(
                            hit_rows["target_touch_bars_from_entry"].to_numpy(dtype=float)
                            * interval_minutes,
                            0.50,
                        )
                    ),
                    "p90": float(
                        np.quantile(
                            hit_rows["target_touch_bars_from_entry"].to_numpy(dtype=float)
                            * interval_minutes,
                            0.90,
                        )
                    ),
                }
            ),
            "competing_outcomes": outcomes,
        }
    return {"horizons": horizons}


def predict_bundle(
    bundle: Mapping[str, Any],
    query: pd.Series,
    query_features: pd.Series,
) -> dict[str, Any]:
    observations: pd.DataFrame = bundle["observations"]
    features = pd.DataFrame(
        bundle["context_feature_values"],
        columns=bundle["context_feature_columns"],
        index=observations.index,
    )
    maximum_horizon_ms = max(bundle["horizons_minutes"]) * 60_000
    eligible = observations.loc[
        observations["entry_close_time_ms"].to_numpy(dtype=np.int64) + maximum_horizon_ms
        <= int(query["entry_close_time_ms"])
    ]
    maximum_slug = horizon_slug(max(bundle["horizons_minutes"]))
    eligible = eligible.loc[eligible[f"horizon_{maximum_slug}_fully_observed"].astype(bool)]
    ranked = rank_candidates(
        eligible,
        features,
        query,
        query_features,
        bundle["base_feature_scales"],
        bundle["context_feature_scales"],
        float(bundle["context_weight"]),
        int(bundle["top_k"]),
    )
    if len(ranked) < 12:
        raise ValueError("Not enough past-only all-outcome analogues for this entry")
    prediction = predict_from_ranked(
        ranked,
        bundle["horizons_minutes"],
        bundle["thresholds_pct"],
        int(str(query["timeframe"]).removesuffix("m")),
    )
    prediction.update(
        {
            "eligible_observations": int(len(eligible)),
            "selected_neighbors": int(len(ranked)),
            "context_weight": float(bundle["context_weight"]),
            "baseline_version": bundle["baseline_version"],
        }
    )
    return prediction


def save_bundle(bundle: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(dict(bundle), temporary, compress=3)
    temporary.replace(path)


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if bundle.get("schema_version") != CONTEXTUAL_MATCHER_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported contextual matcher schema: {bundle.get('schema_version')}")
    if bundle.get("baseline_version") != SOFT_MATCHER_BASELINE_VERSION:
        raise RuntimeError("Contextual matcher does not match the frozen soft matcher baseline")
    return bundle
