"""Observable-only dynamic historical neighborhoods for wick retrieval research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass(frozen=True)
class NeighborhoodConfig:
    k: int
    weighting: str
    radius_quantile: float | None

    @property
    def name(self) -> str:
        radius = "all" if self.radius_quantile is None else f"q{int(self.radius_quantile * 100)}"
        return f"k{self.k}_{self.weighting}_{radius}"


def candidate_neighborhood_configs() -> tuple[NeighborhoodConfig, ...]:
    return tuple(
        NeighborhoodConfig(k, weighting, radius)
        for k in (8, 16, 24, 32, 48, 64, 96)
        for weighting in ("uniform", "inverse", "inverse_square", "exponential")
        for radius in (None, 0.25, 0.50, 0.75)
    )


def default_neighborhood_config() -> NeighborhoodConfig:
    return NeighborhoodConfig(32, "uniform", None)


def retrieve_episode_neighbor_pool(
    candidate_matrix: np.ndarray,
    query_matrix: np.ndarray,
    candidate_signal_ids: np.ndarray,
    maximum_neighbors: int = 96,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Retrieve up to maximum_neighbors distinct historical signal episodes."""
    search_neighbors = min(
        len(candidate_matrix),
        max(maximum_neighbors * 8, 512),
    )
    index = NearestNeighbors(
        n_neighbors=search_neighbors,
        metric="euclidean",
        algorithm="brute",
        n_jobs=-1,
    ).fit(candidate_matrix)
    distances, positions = index.kneighbors(query_matrix)
    selected_positions: list[np.ndarray] = []
    selected_distances: list[np.ndarray] = []
    for row_positions, row_distances in zip(positions, distances, strict=True):
        chosen_positions: list[int] = []
        chosen_distances: list[float] = []
        seen: set[str] = set()
        for position, distance in zip(row_positions, row_distances, strict=True):
            signal_id = str(candidate_signal_ids[int(position)])
            if signal_id in seen:
                continue
            seen.add(signal_id)
            chosen_positions.append(int(position))
            chosen_distances.append(float(distance))
            if len(chosen_positions) >= maximum_neighbors:
                break
        if not chosen_positions:
            raise RuntimeError("No historical episode neighbours were available")
        selected_positions.append(np.asarray(chosen_positions, dtype=np.int64))
        selected_distances.append(np.asarray(chosen_distances, dtype=np.float64))
    return selected_positions, selected_distances


def radius_threshold(
    pool_distances: list[np.ndarray],
    quantile: float | None,
) -> float | None:
    if quantile is None:
        return None
    values = np.concatenate(pool_distances)
    return float(np.quantile(values, quantile))


def distance_weights(distances: np.ndarray, weighting: str) -> np.ndarray:
    if len(distances) == 0:
        raise ValueError("distances may not be empty")
    if weighting == "uniform":
        return np.ones(len(distances), dtype=np.float64)
    scale = max(float(np.median(distances)), 1e-9)
    normalized = distances / scale
    if weighting == "inverse":
        return 1.0 / (normalized + 0.05)
    if weighting == "inverse_square":
        return 1.0 / np.square(normalized + 0.05)
    if weighting == "exponential":
        return np.exp(-normalized)
    raise ValueError(f"Unknown weighting: {weighting}")


def resolve_with_nearest_fallback(
    positions: np.ndarray,
    distances: np.ndarray,
    config: NeighborhoodConfig,
    maximum_distance: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    capped_positions = positions[: config.k]
    capped_distances = distances[: config.k]
    if maximum_distance is None:
        keep = np.ones(len(capped_positions), dtype=bool)
    else:
        keep = capped_distances <= maximum_distance
    radius_failed = not bool(np.any(keep))
    if radius_failed:
        chosen_positions = capped_positions[:1]
        chosen_distances = capped_distances[:1]
    else:
        chosen_positions = capped_positions[keep]
        chosen_distances = capped_distances[keep]
    weights = distance_weights(chosen_distances, config.weighting)
    return chosen_positions, chosen_distances, weights, radius_failed


def effective_sample_size(weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return total * total / max(float(np.sum(np.square(weights))), 1e-12)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if len(values) == 0:
        return float("nan")
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    cutoff = float(quantile) * float(cumulative[-1])
    return float(ordered_values[min(int(np.searchsorted(cumulative, cutoff, side="left")), len(values) - 1)])


def neighborhood_support(
    distances: np.ndarray,
    weights: np.ndarray,
    fill_probability: float,
    adverse_values: np.ndarray,
    wait_values: np.ndarray,
    radius_failed: bool,
) -> dict[str, float | int | bool]:
    normalized_weights = weights / max(float(np.sum(weights)), 1e-12)
    adverse_mean = float(np.sum(normalized_weights * adverse_values))
    adverse_dispersion = float(
        np.sqrt(np.sum(normalized_weights * np.square(adverse_values - adverse_mean)))
    )
    finite_wait = np.isfinite(wait_values)
    wait_dispersion = (
        weighted_quantile(wait_values[finite_wait], weights[finite_wait], 0.75)
        - weighted_quantile(wait_values[finite_wait], weights[finite_wait], 0.25)
        if np.any(finite_wait)
        else float("nan")
    )
    return {
        "raw_neighbors": int(len(weights)),
        "effective_neighbors": effective_sample_size(weights),
        "nearest_distance": float(np.min(distances)),
        "median_neighbor_distance": float(np.median(distances)),
        "distance_dispersion": float(np.std(distances)),
        "fill_agreement": max(fill_probability, 1.0 - fill_probability),
        "adverse_outcome_dispersion_pct": adverse_dispersion,
        "wait_outcome_iqr_minutes": float(wait_dispersion),
        "radius_failed": bool(radius_failed),
    }


def config_record(config: NeighborhoodConfig, maximum_distance: float | None) -> dict[str, Any]:
    return {
        "name": config.name,
        "k": config.k,
        "weighting": config.weighting,
        "radius_quantile": config.radius_quantile,
        "maximum_distance": maximum_distance,
    }
