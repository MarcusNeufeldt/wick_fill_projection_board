from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from train_neural_path_v3 import (
    EPISODE_LENGTH,
    EPISODE_SEQUENCE_CHANNELS,
    FUTURE_HORIZON_BARS,
    PRE_SIGNAL_LENGTH,
    RAW_SEQUENCE_CHANNELS,
    RECENT_LENGTH,
    NeuralPathModel,
    apply_sequence_scaler,
    direction_normalized_pct,
    episode_balanced_bootstrap_delta,
    fit_sequence_scaler,
    future_targets,
    inner_chronological_split,
)


class NeuralPathV3Tests(unittest.TestCase):
    def test_direction_normalization_mirrors_upper_and_lower_wicks(self) -> None:
        lower = direction_normalized_pct(np.asarray([105.0]), 100.0, 1)
        upper = direction_normalized_pct(np.asarray([95.0]), 100.0, -1)
        np.testing.assert_allclose(lower, upper, rtol=0, atol=1e-8)

    def test_sequence_scaler_keeps_padding_zero(self) -> None:
        values = np.zeros((2, 4, 3), dtype=np.float32)
        values[0, 2:, 0] = [2.0, 4.0]
        values[0, 2:, 1] = [10.0, 14.0]
        values[0, 2:, 2] = 1.0
        values[1, 1:, 0] = [1.0, 3.0, 5.0]
        values[1, 1:, 1] = [8.0, 12.0, 16.0]
        values[1, 1:, 2] = 1.0
        transformed = apply_sequence_scaler(values, fit_sequence_scaler(values))
        self.assertTrue(
            np.all(transformed[:, :, :-1][transformed[:, :, -1] == 0.0] == 0.0)
        )
        self.assertTrue(np.all(transformed[:, :, -1] == values[:, :, -1]))

    def test_future_targets_stop_at_fill_touch(self) -> None:
        path = pd.DataFrame(
            {
                "offset_bars": np.arange(6),
                "normalized_close_pct": [0.2, 1.0, 1.4, 0.8, 0.3, 0.1],
            }
        )
        curve, direction = future_targets(path, snapshot_offset=2, current_move_pct=1.4)
        distances = np.expm1(curve) * 1.4
        self.assertAlmostEqual(float(distances[0]), 0.8, places=5)
        self.assertEqual(float(distances[1]), 0.0)
        self.assertTrue(np.all(distances[2:] == 0.0))
        self.assertTrue(np.all(direction == 1.0))

    def test_inner_split_is_episode_disjoint_and_label_available(self) -> None:
        rows = []
        day_ms = 24 * 60 * 60 * 1000
        for episode in range(20):
            signal = episode * 10 * day_ms
            rows.append(
                {
                    "episode_id": f"episode-{episode}",
                    "signal_open_time_ms": signal,
                    "snapshot_close_time_ms": signal + day_ms,
                    "fill_close_time_ms": signal + 2 * day_ms,
                }
            )
        fit, validation, details = inner_chronological_split(
            pd.DataFrame(rows), validation_fraction=0.2, embargo_bars=1
        )
        self.assertFalse(set(fit["episode_id"]).intersection(validation["episode_id"]))
        self.assertLessEqual(
            int(fit["fill_close_time_ms"].max()),
            int(validation["snapshot_close_time_ms"].min()),
        )
        self.assertTrue(
            details["all_fit_labels_resolved_before_every_validation_snapshot"]
        )

    def test_model_shapes(self) -> None:
        model = NeuralPathModel(
            static_features=7, sequence_width=8, embedding_dim=16, dropout=0.0
        )
        batch = 3
        output = model(
            torch.zeros(batch, len(RAW_SEQUENCE_CHANNELS), PRE_SIGNAL_LENGTH),
            torch.zeros(batch, len(RAW_SEQUENCE_CHANNELS), RECENT_LENGTH),
            torch.zeros(batch, len(EPISODE_SEQUENCE_CHANNELS), EPISODE_LENGTH),
            torch.zeros(batch, 7),
        )
        self.assertEqual(tuple(output["embedding"].shape), (batch, 16))
        self.assertEqual(tuple(output["risk"].shape), (batch, 2, 3))
        self.assertEqual(
            tuple(output["curve"].shape), (batch, len(FUTURE_HORIZON_BARS))
        )
        norms = torch.linalg.vector_norm(output["embedding"], dim=1)
        torch.testing.assert_close(norms, torch.ones_like(norms))

    def test_episode_bootstrap_detects_uniform_improvement(self) -> None:
        baseline = np.asarray([2.0, 4.0, 3.0, 5.0], dtype=float)
        challenger = baseline - 1.0
        episodes = np.asarray(["a", "a", "b", "c"])
        result = episode_balanced_bootstrap_delta(
            baseline, challenger, episodes, seed=7, repetitions=200
        )
        self.assertAlmostEqual(result["delta_challenger_minus_scalar"], -1.0)
        self.assertEqual(result["probability_challenger_better"], 1.0)
        self.assertLess(result["bootstrap_95pct_high"], 0.0)


if __name__ == "__main__":
    unittest.main()
