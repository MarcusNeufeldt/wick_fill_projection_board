from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from conditional_wick_assets import SUPPORTED_ASSETS
from wick_update import (
    UpdateRunner,
    copy_unmanaged_library_entries,
    promote_directory,
    validate_library,
    validate_outcomes,
)


class WickUpdateTests(unittest.TestCase):
    def test_dry_run_plans_complete_scope_without_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner = UpdateRunner(root, dry_run=True, skip_refresh=False)
            runner.execute()

            command_stages = [
                item["stage"]
                for item in runner.manifest["stages"]
                if "command" in item
            ]
            self.assertEqual(command_stages[:2], ["refresh_1m", "refresh_5m"])
            self.assertIn("build_routes_5m", command_stages)
            for asset in SUPPORTED_ASSETS:
                slug = asset.lower()
                self.assertIn(f"build_routes_1m_{slug}", command_stages)
                self.assertIn(f"cache_routes_1m_{slug}", command_stages)
            self.assertIn("build_all_outcomes", command_stages)
            self.assertIn("train_and_gate_risk_models", command_stages)
            self.assertIn("train_forecast_v1_candidate_5m", command_stages)
            candidate_event = next(
                item
                for item in runner.manifest["stages"]
                if item["stage"] == "save_forecast_v1_candidate_5m"
            )
            self.assertEqual(candidate_event["active_manifest_change"], "forbidden")
            self.assertFalse((root / "data").exists())

    def test_validate_library_and_preserve_unmanaged_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            current = root / "current"
            staged = root / "staged"
            destination = root / "library"
            (current / "v2_models").mkdir(parents=True)
            (current / "v2_models" / "model.pkl").write_bytes(b"v2")
            (staged / "paths").mkdir(parents=True)
            (staged / "episodes.csv").write_text("episode_id\n1\n", encoding="utf-8")
            (staged / "summary.json").write_text(
                json.dumps({"completed_episode_count": 1}), encoding="utf-8"
            )
            (staged / "paths" / "ETHUSDT_1m_paths.csv.gz").write_bytes(b"path")

            self.assertEqual(validate_library(staged, "1m", ("ETHUSDT",)), 1)
            copy_unmanaged_library_entries(current, staged)
            promote_directory(staged, destination, "test")
            self.assertEqual((destination / "v2_models" / "model.pkl").read_bytes(), b"v2")

    def test_validate_outcomes_requires_every_asset_and_timeframe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            series = []
            for asset in SUPPORTED_ASSETS:
                for timeframe in ("1m", "5m"):
                    stem = f"{asset}_{timeframe}"
                    for category in ("signals", "observations"):
                        path = root / category / f"{stem}.parquet"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"partition")
                    series.append(
                        {
                            "asset": asset,
                            "timeframe": timeframe,
                            "signal_count": 10,
                            "observation_count": 2,
                            "resolution_status_counts": {"filled": 7, "unfilled": 2, "right_censored": 1},
                        }
                    )
            (root / "metadata.json").write_text(json.dumps({"series": series}), encoding="utf-8")

            counts = validate_outcomes(root)
            self.assertEqual(counts["signals"], 100)
            self.assertEqual(counts["observations"], 20)
            self.assertEqual(counts["filled"], 70)


if __name__ == "__main__":
    unittest.main()
