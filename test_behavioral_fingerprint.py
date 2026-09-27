from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from evaluate_behavioral_fingerprint import (
    episode_balanced_comparison,
    sequence_fingerprint_from_close,
    unique_episode_neighbors,
)
from optimize_behavioral_fingerprint import (
    FingerprintConfig,
    choose_configs,
    compose_matrices,
)


class BehavioralFingerprintTests(unittest.TestCase):
    def observation(self, *, direction_sign: int, target: float) -> dict[str, int | float]:
        return {
            "wick_target": target,
            "direction_sign": direction_sign,
            "pre_signal_start_index": 0,
            "pre_signal_end_index": 19,
            "recent_start_index": 10,
            "recent_end_index": 29,
            "signal_to_entry_start_index": 20,
            "signal_to_entry_end_index": 29,
        }

    def test_fingerprint_never_reads_candles_after_entry(self) -> None:
        close = np.linspace(100.0, 120.0, 60)
        observation = self.observation(direction_sign=1, target=95.0)
        before = sequence_fingerprint_from_close(close, observation)
        changed = close.copy()
        changed[30:] *= 100.0
        after = sequence_fingerprint_from_close(changed, observation)
        np.testing.assert_allclose(before, after)

    def test_upper_and_lower_paths_are_mirrored(self) -> None:
        lower = 100.0 + np.linspace(0.0, 8.0, 60)
        upper = 100.0 - np.linspace(0.0, 8.0, 60)
        lower_fingerprint = sequence_fingerprint_from_close(
            lower,
            self.observation(direction_sign=1, target=100.0),
        )
        upper_fingerprint = sequence_fingerprint_from_close(
            upper,
            self.observation(direction_sign=-1, target=100.0),
        )
        np.testing.assert_allclose(lower_fingerprint, upper_fingerprint, atol=1e-6)

    def test_neighbors_keep_only_one_snapshot_per_episode(self) -> None:
        positions = np.asarray([0, 1, 2, 3, 4, 5])
        signals = np.asarray(["a", "a", "b", "c", "c", "d"])
        selected = unique_episode_neighbors(positions, signals, 3)
        np.testing.assert_array_equal(selected, np.asarray([0, 2, 3]))

    def test_episode_bootstrap_detects_uniformly_better_predictions(self) -> None:
        actual = pd.DataFrame(
            {
                "signal_id": ["a", "a", "b", "c"],
                "actual_fill": [1.0, 0.0, 1.0, 1.0],
                "actual_adverse_pct": [2.0, 3.0, 4.0, 5.0],
                "actual_wait_minutes": [60.0, np.nan, 180.0, 240.0],
            }
        )
        baseline = actual.assign(
            predicted_fill=[0.0, 1.0, 0.0, 0.0],
            predicted_adverse_pct=[4.0, 5.0, 6.0, 7.0],
            predicted_wait_minutes=[180.0, 60.0, 300.0, 360.0],
        )
        challenger = actual.assign(
            predicted_fill=[0.8, 0.2, 0.8, 0.8],
            predicted_adverse_pct=[2.2, 3.2, 4.2, 5.2],
            predicted_wait_minutes=[70.0, 60.0, 190.0, 250.0],
        )
        comparison = episode_balanced_comparison(
            baseline,
            challenger,
            bootstrap_samples=200,
        )
        for metric in comparison.values():
            self.assertGreater(metric["relative_improvement_pct"], 0.0)
            self.assertLess(metric["bootstrap_95pct_high"], 0.0)

    def test_block_weights_change_similarity_coordinates(self) -> None:
        blocks = {
            "state": (np.ones((2, 2)), np.ones((1, 2))),
            "context": (np.full((2, 1), 3.0), np.full((1, 1), 4.0)),
        }
        config = FingerprintConfig("weighted", {"state": 1.0, "context": 0.5})
        fit, query = compose_matrices(blocks, config)
        np.testing.assert_array_equal(fit[:, -1], np.asarray([1.5, 1.5]))
        np.testing.assert_array_equal(query[:, -1], np.asarray([2.0]))

    def test_validation_selector_rejects_material_metric_regression(self) -> None:
        rows = pd.DataFrame(
            {
                "signal_id": ["a", "b", "c"],
                "actual_fill": [1.0, 0.0, 1.0],
                "predicted_fill": [0.8, 0.2, 0.8],
                "actual_adverse_pct": [2.0, 3.0, 4.0],
                "predicted_adverse_pct": [2.2, 3.2, 4.2],
                "actual_wait_minutes": [60.0, np.nan, 180.0],
                "predicted_wait_minutes": [70.0, 60.0, 190.0],
            }
        )
        worse = rows.copy()
        worse["predicted_fill"] = [0.0, 1.0, 0.0]
        configs = (
            FingerprintConfig("state_only", {"state": 1.0}),
            FingerprintConfig("worse", {"state": 1.0, "context": 1.0}),
        )
        results = {
            "state_only": {"rows": {horizon: rows for horizon in (1_440, 10_080, 43_200)}},
            "worse": {"rows": {horizon: worse for horizon in (1_440, 10_080, 43_200)}},
        }
        selected, _ = choose_configs(results, configs, regression_tolerance=0.01)
        self.assertTrue(all(config.name == "state_only" for config in selected.values()))


if __name__ == "__main__":
    unittest.main()
