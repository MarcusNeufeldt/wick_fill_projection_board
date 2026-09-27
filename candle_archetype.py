"""Direction-agnostic signal-candle archetype matching.

Upper and lower wicks are already represented by dominant/opposite wick shares
and a direction-aligned prior return.  This matcher therefore treats direction
as a mirror rather than a category penalty and gives the pinned signal candle
an explicit, inspectable neighbourhood before live-path state is compared.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd


ARCHETYPE_COMPONENT_SCALES = {
    "body_pct_of_range_pp": 0.75,
    "dominant_wick_pct_of_range_pp": 3.0,
    "opposite_wick_pct_of_range_pp": 3.0,
    "range_pct_of_close_factor": 1.5,
    "range_vs_prior_20_median_factor": 1.5,
    "volume_vs_prior_20_mean_factor": 2.0,
    "aligned_prior_1h_return_pct": 1.0,
}
ARCHETYPE_POOL_SIZE = 240
ARCHETYPE_FEATURE_MULTIPLIER = 1.5
ARCHETYPE_BLEND_WEIGHT = 0.30
SOFT_MATCHER_BASELINE_VERSION = "soft-state-v1.0.0"
SOFT_MATCHER_DIRECTION_POLICY = "upper/lower mirrored"


def soft_matcher_baseline_contract() -> dict[str, Any]:
    """Return the immutable research contract for the current soft baseline."""
    return {
        "version": SOFT_MATCHER_BASELINE_VERSION,
        "archetype_blend_weight": ARCHETYPE_BLEND_WEIGHT,
        "adaptive_feature_weight": 1.0 - ARCHETYPE_BLEND_WEIGHT,
        "direction_policy": SOFT_MATCHER_DIRECTION_POLICY,
        "route_population": "completed clean fills",
        "promotion_target": "prospective-entry adverse movement and remaining waiting time",
    }


def _target_value(target: Any, name: str) -> float:
    if isinstance(target, Mapping):
        return float(target[name])
    try:
        return float(target[name])
    except (KeyError, TypeError):
        return float(getattr(target, name))


def _factor_distance(values: np.ndarray, target: float, factor: float) -> np.ndarray:
    safe_values = np.maximum(np.asarray(values, dtype=float), 1e-12)
    safe_target = max(float(target), 1e-12)
    return np.abs(np.log(safe_values / safe_target)) / np.log(factor)


def archetype_component_distances(
    events: pd.DataFrame, target: Any
) -> dict[str, np.ndarray]:
    """Return normalized distances; one unit equals one documented tolerance."""
    return {
        "body_pct_of_range_pp": np.abs(
            events["body_pct_of_range"].to_numpy(dtype=float) * 100.0
            - _target_value(target, "body_pct_of_range") * 100.0
        )
        / ARCHETYPE_COMPONENT_SCALES["body_pct_of_range_pp"],
        "dominant_wick_pct_of_range_pp": np.abs(
            events["dominant_wick_pct_of_range"].to_numpy(dtype=float) * 100.0
            - _target_value(target, "dominant_wick_pct_of_range") * 100.0
        )
        / ARCHETYPE_COMPONENT_SCALES["dominant_wick_pct_of_range_pp"],
        "opposite_wick_pct_of_range_pp": np.abs(
            events["opposite_wick_pct_of_range"].to_numpy(dtype=float) * 100.0
            - _target_value(target, "opposite_wick_pct_of_range") * 100.0
        )
        / ARCHETYPE_COMPONENT_SCALES["opposite_wick_pct_of_range_pp"],
        "range_pct_of_close_factor": _factor_distance(
            events["range_pct_of_close"].to_numpy(dtype=float),
            _target_value(target, "range_pct_of_close"),
            ARCHETYPE_COMPONENT_SCALES["range_pct_of_close_factor"],
        ),
        "range_vs_prior_20_median_factor": _factor_distance(
            events["range_vs_prior_20_median"].to_numpy(dtype=float),
            _target_value(target, "range_vs_prior_20_median"),
            ARCHETYPE_COMPONENT_SCALES["range_vs_prior_20_median_factor"],
        ),
        "volume_vs_prior_20_mean_factor": _factor_distance(
            events["volume_vs_prior_20_mean"].to_numpy(dtype=float),
            _target_value(target, "volume_vs_prior_20_mean"),
            ARCHETYPE_COMPONENT_SCALES["volume_vs_prior_20_mean_factor"],
        ),
        "aligned_prior_1h_return_pct": np.abs(
            events["aligned_prior_1h_return_pct"].to_numpy(dtype=float)
            - _target_value(target, "aligned_prior_1h_return_pct")
        )
        / ARCHETYPE_COMPONENT_SCALES["aligned_prior_1h_return_pct"],
    }


def archetype_distance(events: pd.DataFrame, target: Any) -> np.ndarray:
    """Root-mean-square candle-configuration distance, independent of direction."""
    components = archetype_component_distances(events, target)
    values = np.column_stack(tuple(components.values()))
    return np.sqrt(np.mean(np.square(values), axis=1))


def archetype_event_scores(
    events: pd.DataFrame,
    target: Any,
    pool_size: int = ARCHETYPE_POOL_SIZE,
) -> pd.DataFrame:
    """Return the closest candle configurations for later live-state reranking."""
    if pool_size < 12:
        raise ValueError("archetype pool size must be at least 12")
    fields = [
        "episode_id",
        "asset",
        "timeframe",
        "direction",
        "signal_to_fill_bars",
    ]
    value = events.loc[:, fields].copy()
    value["archetype_distance"] = archetype_distance(events, target)
    value = value.sort_values(
        ["archetype_distance", "episode_id"], kind="stable"
    ).head(min(pool_size, len(value)))
    # Production matching applies 0.5 * feature_distance. The multiplier keeps
    # the archetype contribution explicit while the hard pool remains primary.
    value["feature_distance"] = (
        value["archetype_distance"] * ARCHETYPE_FEATURE_MULTIPLIER
    )
    # Asset and wick direction intentionally carry no penalty in this mode.
    value["category_distance"] = 0.0
    return value.reset_index(drop=True)


def archetype_profile(target: Any) -> dict[str, float]:
    return {
        "body_pct_of_range": _target_value(target, "body_pct_of_range"),
        "dominant_wick_pct_of_range": _target_value(
            target, "dominant_wick_pct_of_range"
        ),
        "opposite_wick_pct_of_range": _target_value(
            target, "opposite_wick_pct_of_range"
        ),
        "range_pct_of_close": _target_value(target, "range_pct_of_close"),
        "range_vs_prior_20_median": _target_value(
            target, "range_vs_prior_20_median"
        ),
        "volume_vs_prior_20_mean": _target_value(
            target, "volume_vs_prior_20_mean"
        ),
        "aligned_prior_1h_return_pct": _target_value(
            target, "aligned_prior_1h_return_pct"
        ),
    }
