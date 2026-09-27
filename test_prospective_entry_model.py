from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from build_conditional_path_library import FEATURE_COLUMNS
from evaluate_prospective_entry_baseline import resolve_dataset_file
from prospective_entry_model import FEATURE_COLUMNS_V1, features_from_observations


class ProspectiveEntryFeatureTests(unittest.TestCase):
    def test_promoted_dataset_recovers_legacy_staging_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset_dir = Path(temporary) / "prospective_entry_outcomes_v1"
            installed = dataset_dir / "observations" / "ETHUSDT_5m.parquet"
            installed.parent.mkdir(parents=True)
            installed.write_bytes(b"partition")
            stale = Path(temporary) / ".staging" / "observations" / installed.name

            resolved = resolve_dataset_file(
                dataset_dir,
                {"kind": "observations", "path": str(stale)},
            )

            self.assertEqual(resolved, installed)

    def test_dataset_relative_paths_are_resolved_from_dataset_root(self) -> None:
        dataset_dir = Path("dataset")
        resolved = resolve_dataset_file(
            dataset_dir,
            {"kind": "observations", "path": "observations/ETHUSDT_5m.parquet"},
        )
        self.assertEqual(resolved, dataset_dir / "observations" / "ETHUSDT_5m.parquet")

    def test_features_do_not_change_when_future_candles_change(self) -> None:
        rows = 700
        close = 100.0 + np.linspace(0.0, 7.0, rows) + np.sin(np.arange(rows) / 11.0)
        frame = pd.DataFrame(
            {
                "open_time": np.arange(rows, dtype=np.int64) * 300_000,
                "close_time": (np.arange(rows, dtype=np.int64) + 1) * 300_000 - 1,
                "open": close - 0.05,
                "high": close + 0.25,
                "low": close - 0.25,
                "close": close,
                "volume": 1_000.0 + np.arange(rows, dtype=float),
            }
        )
        row = {
            "asset": "ETHUSDT",
            "timeframe": "5m",
            "direction": "lower_wick",
            "direction_sign": 1,
            "signal_index": 300,
            "entry_index": 550,
            "entry_age_minutes": 1_250,
            "entry_distance_from_target_pct": 3.0,
            "peak_distance_from_target_pct": 4.0,
            "drawdown_from_peak_pct": 1.0,
            "departure_to_entry_bars": 240,
        }
        for index, column in enumerate(FEATURE_COLUMNS, start=1):
            row[column] = float(index)
        observations = pd.DataFrame([row])
        before = features_from_observations(frame, observations)
        changed = frame.copy()
        changed.loc[551:, ["open", "high", "low", "close", "volume"]] *= 50.0
        after = features_from_observations(changed, observations)
        pd.testing.assert_frame_equal(before, after)
        self.assertEqual(tuple(before.columns), FEATURE_COLUMNS_V1)
        self.assertFalse(before.isna().any().any())


if __name__ == "__main__":
    unittest.main()
