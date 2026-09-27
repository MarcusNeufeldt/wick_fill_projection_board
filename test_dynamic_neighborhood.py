from __future__ import annotations

import unittest

import numpy as np

from dynamic_neighborhood import (
    NeighborhoodConfig,
    distance_weights,
    effective_sample_size,
    resolve_with_nearest_fallback,
    retrieve_episode_neighbor_pool,
    weighted_quantile,
)


class DynamicNeighborhoodTests(unittest.TestCase):
    def test_pool_keeps_one_observation_per_signal_episode(self) -> None:
        candidates = np.asarray([[0.0], [0.01], [0.02], [1.0], [2.0]], dtype=np.float32)
        queries = np.asarray([[0.0]], dtype=np.float32)
        signals = np.asarray(["a", "a", "b", "c", "d"])
        positions, distances = retrieve_episode_neighbor_pool(
            candidates, queries, signals, maximum_neighbors=3
        )
        self.assertEqual(positions[0].tolist(), [0, 2, 3])
        self.assertEqual(len(distances[0]), 3)

    def test_effective_n_reflects_weight_concentration(self) -> None:
        self.assertAlmostEqual(effective_sample_size(np.ones(8)), 8.0)
        concentrated = effective_sample_size(np.asarray([100.0, 1.0, 1.0, 1.0]))
        self.assertLess(concentrated, 1.1)

    def test_radius_fails_closed_to_nearest_and_marks_it(self) -> None:
        positions = np.asarray([4, 5, 6])
        distances = np.asarray([0.4, 0.5, 0.6])
        config = NeighborhoodConfig(3, "uniform", 0.25)
        chosen, chosen_distances, weights, failed = resolve_with_nearest_fallback(
            positions, distances, config, maximum_distance=0.1
        )
        self.assertTrue(failed)
        self.assertEqual(chosen.tolist(), [4])
        self.assertEqual(chosen_distances.tolist(), [0.4])
        self.assertEqual(weights.tolist(), [1.0])

    def test_distance_weighting_is_monotonic(self) -> None:
        distances = np.asarray([0.2, 0.4, 0.8])
        for method in ("inverse", "inverse_square", "exponential"):
            weights = distance_weights(distances, method)
            self.assertGreater(weights[0], weights[1])
            self.assertGreater(weights[1], weights[2])

    def test_weighted_quantile_tracks_dominant_analogue(self) -> None:
        values = np.asarray([1.0, 5.0, 10.0])
        weights = np.asarray([1.0, 10.0, 1.0])
        self.assertEqual(weighted_quantile(values, weights, 0.5), 5.0)


if __name__ == "__main__":
    unittest.main()
