from __future__ import annotations

import numpy as np
import pandas as pd
import unittest

from evaluate_rolling_forecast_systems import (
    MAXIMUM_HORIZON_MS,
    calibration_error,
    episode_sample_weights,
    eligible_history_and_queries,
    rolling_signal_folds,
    SUPPORT_STACK_COLUMNS,
    support_stack_matrix,
    tree_feature_sets,
    validate_v3_fold_provenance,
)


def observation_frame(signal_count: int = 100) -> pd.DataFrame:
    rows = []
    for index in range(signal_count):
        signal_ms = index * 100_000_000
        rows.append(
            {
                "signal_open_time_ms": signal_ms,
                "entry_close_time_ms": signal_ms + 300_000,
                "observation_id": f"obs-{index}",
                "signal_id": f"signal-{index}",
                "direction": "lower" if index % 2 else "upper",
                "entry_age_minutes": 60,
                "entry_distance_from_target_pct": 2.0,
                "horizon_43200m_fully_observed": True,
            }
        )
    return pd.DataFrame(rows)


class RollingForecastSystemTests(unittest.TestCase):
    def test_rolling_folds_are_strictly_ordered_and_non_overlapping(self) -> None:
        folds = rolling_signal_folds(observation_frame(), folds=5, initial_train_fraction=0.40)
        self.assertEqual(len(folds), 5)
        for fold in folds:
            self.assertLess(fold.train_end_signal_ms, fold.validation_end_signal_ms)
            self.assertLess(fold.validation_end_signal_ms, fold.test_end_signal_ms)
        for previous, current in zip(folds, folds[1:], strict=False):
            self.assertEqual(previous.test_end_signal_ms, current.train_end_signal_ms)

    def test_history_labels_mature_before_first_query(self) -> None:
        history = observation_frame(4)
        query = observation_frame(2).copy()
        query.index = [10, 11]
        query["entry_close_time_ms"] = [
            MAXIMUM_HORIZON_MS + 500_000,
            MAXIMUM_HORIZON_MS + 600_000,
        ]
        history.loc[0, "entry_close_time_ms"] = 100_000
        history.loc[1, "entry_close_time_ms"] = 200_000
        history.loc[2, "entry_close_time_ms"] = 600_000
        history.loc[3, "entry_close_time_ms"] = 700_000
        eligible, queries, evidence = eligible_history_and_queries(
            history, query, maximum_queries=10
        )
        self.assertEqual(eligible.index.tolist(), [0, 1])
        self.assertEqual(len(queries), 2)
        self.assertLessEqual(
            evidence["latest_history_label_available_ms"],
            evidence["first_query_close_ms"],
        )

    def test_tree_feature_sets_add_exact_sequence_columns(self) -> None:
        features = pd.DataFrame(np.zeros((3, 53)), columns=[f"f{i}" for i in range(53)])
        sequences = pd.DataFrame(np.zeros((3, 40)), columns=[f"s{i}" for i in range(40)])
        systems = tree_feature_sets(features, sequences)
        self.assertEqual(systems["A_supervised_53"].shape[1], 53)
        self.assertEqual(systems["B_supervised_53_plus_sequence"].shape[1], 93)

    def test_calibration_error_is_episode_balanced(self) -> None:
        rows = pd.DataFrame(
            {
                "signal_id": ["a", "a", "a", "b"],
                "predicted_fill": [1.0, 1.0, 1.0, 0.0],
                "actual_fill": [1.0, 1.0, 1.0, 1.0],
            }
        )
        self.assertAlmostEqual(calibration_error(rows), 0.5)

    def test_v3_requires_explicit_fold_local_provenance(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not proven fold-local"):
            validate_v3_fold_provenance({}, 1_000)
        with self.assertRaisesRegex(RuntimeError, "not proven fold-local"):
            validate_v3_fold_provenance(
                {
                    "training_labels_available_through_ms": 1_001,
                    "training_completed_before_fold_cutoff": True,
                },
                1_000,
            )
        validate_v3_fold_provenance(
            {
                "training_labels_available_through_ms": 999,
                "training_completed_before_fold_cutoff": True,
            },
            1_000,
        )

    def test_episode_weights_prevent_snapshot_heavy_signals_from_dominating(self) -> None:
        rows = pd.DataFrame({"signal_id": ["a", "a", "a", "b"]})
        weights = episode_sample_weights(rows)
        self.assertAlmostEqual(float(weights[:3].sum()), 1.0)
        self.assertAlmostEqual(float(weights[3]), 1.0)

    def test_support_stack_ablation_removes_only_support_columns(self) -> None:
        rows = pd.DataFrame(
            {
                "observation_id": ["a", "b"],
                "predicted_fill": [0.2, 0.8],
                "predicted_adverse_pct": [1.0, 2.0],
                "predicted_wait_minutes": [60.0, 120.0],
                **{column: [1.0, 2.0] for column in SUPPORT_STACK_COLUMNS},
            }
        )
        without_support = support_stack_matrix(rows, rows, include_support=False)
        with_support = support_stack_matrix(rows, rows, include_support=True)
        self.assertEqual(without_support.shape, (2, 6))
        self.assertEqual(with_support.shape, (2, 6 + len(SUPPORT_STACK_COLUMNS)))


if __name__ == "__main__":
    unittest.main()
