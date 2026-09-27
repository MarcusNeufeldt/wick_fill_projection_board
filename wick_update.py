"""Manually refresh wick data, route libraries, and promoted risk models.

The dashboard refresh loop keeps raw candles current, but it deliberately does
not rebuild historical evidence or retrain models.  This command is the manual
maintenance boundary: run it periodically when newly resolved wicks should be
absorbed into the application.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from conditional_wick_assets import (
    SUPPORTED_ASSETS,
    default_library_dir,
    one_minute_library_dir,
)
from forecast_artifact_manifest import active_manifest_path, load_active_manifest, sha256_file
from prospective_entry_model import default_artifact_root


TIMEFRAMES = ("1m", "5m")
OWNED_LIBRARY_ENTRIES = {
    "DEFINITION.md",
    "episodes.csv",
    "native_runtime_cache_v1",
    "paths",
    "summary.json",
}


def cargo_executable() -> Path:
    configured = os.environ.get("CARGO")
    candidates = [Path(configured)] if configured else []
    discovered = shutil.which("cargo")
    if discovered:
        candidates.append(Path(discovered))
    if os.name == "nt":
        program_files = Path(os.environ.get("ProgramFiles", r"C:\Program Files"))
        candidates.extend(sorted(program_files.glob("Rust stable MSVC */bin/cargo.exe"), reverse=True))
        candidates.extend(sorted(program_files.glob("Rust stable GNU */bin/cargo.exe"), reverse=True))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError("Rust cargo executable not found; set CARGO to its full path")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def completed_episode_count(library_dir: Path) -> int:
    summary_path = library_dir / "summary.json"
    if not summary_path.exists():
        return 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return int(summary.get("completed_episode_count", 0))


def validate_library(library_dir: Path, timeframe: str, assets: Iterable[str]) -> int:
    required = (library_dir / "episodes.csv", library_dir / "summary.json", library_dir / "paths")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Built library is incomplete; missing: {', '.join(missing)}")
    count = completed_episode_count(library_dir)
    if count <= 0:
        raise RuntimeError(f"Built library reports no completed episodes: {library_dir}")
    for asset in assets:
        path_file = library_dir / "paths" / f"{asset}_{timeframe}_paths.csv.gz"
        if not path_file.exists() or path_file.stat().st_size == 0:
            raise RuntimeError(f"Built library path file is missing or empty: {path_file}")
    return count


def validate_outcomes(dataset_dir: Path) -> dict[str, int]:
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        raise RuntimeError(f"Outcome dataset metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    series = metadata.get("series")
    if not isinstance(series, list) or len(series) != len(SUPPORTED_ASSETS) * len(TIMEFRAMES):
        raise RuntimeError("Outcome dataset does not contain all five assets at 1m and 5m")
    for asset in SUPPORTED_ASSETS:
        for timeframe in TIMEFRAMES:
            stem = f"{asset}_{timeframe}"
            for category in ("signals", "observations"):
                path = dataset_dir / category / f"{stem}.parquet"
                if not path.exists() or path.stat().st_size == 0:
                    raise RuntimeError(f"Outcome partition is missing or empty: {path}")
    counts = {"signals": 0, "observations": 0, "filled": 0, "unfilled": 0, "right_censored": 0}
    for item in series:
        counts["signals"] += int(item.get("signal_count", 0))
        counts["observations"] += int(item.get("observation_count", 0))
        statuses = item.get("resolution_status_counts", {})
        for key in ("filled", "unfilled", "right_censored"):
            counts[key] += int(statuses.get(key, 0))
    return counts


def copy_unmanaged_library_entries(current: Path, staged: Path) -> None:
    """Carry forward V2 and research extras that the route builder does not own."""
    if not current.exists():
        return
    for source in current.iterdir():
        if source.name in OWNED_LIBRARY_ENTRIES:
            continue
        destination = staged / source.name
        if source.is_dir():
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)


def promote_directory(staged: Path, destination: Path, run_id: str) -> Path | None:
    """Swap one derived directory, restoring the prior version if promotion fails."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    backup = destination.with_name(f".{destination.name}.pre-update-{run_id}")
    if backup.exists():
        raise RuntimeError(f"Refusing to overwrite an existing update backup: {backup}")
    had_destination = destination.exists()
    if had_destination:
        destination.replace(backup)
    try:
        staged.replace(destination)
    except BaseException:
        if had_destination and backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    if not had_destination:
        return None
    try:
        shutil.rmtree(backup)
        return None
    except OSError:
        return backup


def promote_file(staged: Path, destination: Path, run_id: str) -> None:
    """Copy across drives and atomically replace the installed model file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{run_id}.tmp")
    shutil.copy2(staged, temporary)
    os.replace(temporary, destination)


class UpdateRunner:
    def __init__(self, root: Path, *, dry_run: bool, skip_refresh: bool) -> None:
        self.root = root.resolve()
        self.dry_run = dry_run
        self.skip_refresh = skip_refresh
        self.run_id = run_id_now()
        self.stage_root = self.root / "data" / ".wick_update_staging" / self.run_id
        self.manifest_path = self.root / "data" / "wick_update_runs" / f"{self.run_id}.json"
        self.manifest: dict[str, Any] = {
            "schema_version": "wick-manual-update-v1.0.0",
            "run_id": self.run_id,
            "started_at_utc": utc_now(),
            "status": "planned" if dry_run else "running",
            "scope": {"assets": list(SUPPORTED_ASSETS), "timeframes": list(TIMEFRAMES)},
            "stages": [],
            "notes": [
                "V3 neural age experts are not retrained or promoted by this command.",
                "Prospective-entry models replace installed artifacts only after their chronological gate passes.",
                "Forecast Architecture V1 retrains are saved as candidates and never repoint the frozen active manifest.",
            ],
        }

    def save(self) -> None:
        if not self.dry_run:
            atomic_json(self.manifest_path, self.manifest)

    def event(self, stage: str, status: str, **details: Any) -> None:
        entry = {"stage": stage, "status": status, "at_utc": utc_now(), **details}
        self.manifest["stages"].append(entry)
        self.save()

    def external_command(self, stage: str, command: list[str]) -> None:
        print(f"\n[{stage}]\n{subprocess.list2cmdline(command)}", flush=True)
        if self.dry_run:
            self.event(stage, "planned", command=command)
            return
        self.event(stage, "running", command=command)
        started = datetime.now(timezone.utc)
        try:
            subprocess.run(command, cwd=self.root, check=True)
        except BaseException as error:
            elapsed = (datetime.now(timezone.utc) - started).total_seconds()
            self.event(stage, "failed", elapsed_seconds=round(elapsed, 3), error=str(error))
            raise
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        self.event(stage, "complete", elapsed_seconds=round(elapsed, 3))

    def command(self, stage: str, script: str, *arguments: object) -> None:
        command = [sys.executable, str(self.root / script), *(str(value) for value in arguments)]
        self.external_command(stage, command)

    def build_rust_kernel(self) -> Path:
        crate = self.root / "experiments" / "rust_training_throughput"
        binary = crate / "target" / "release" / (
            "wick-throughput-poc.exe" if os.name == "nt" else "wick-throughput-poc"
        )
        cargo = cargo_executable()
        self.external_command(
            "build_rust_kernel",
            [
                str(cargo),
                "build",
                "--release",
                "--manifest-path",
                str(crate / "Cargo.toml"),
            ],
        )
        if not self.dry_run and not binary.is_file():
            raise RuntimeError(f"Rust kernel build did not produce {binary}")
        return binary

    def refresh_sources(self) -> None:
        if self.skip_refresh:
            self.event("refresh_sources", "skipped", reason="--skip-refresh")
            return
        for timeframe in TIMEFRAMES:
            self.command(
                f"refresh_{timeframe}",
                "refresh_futures_klines.py",
                "--interval",
                timeframe,
            )

    def rebuild_one_library(
        self, timeframe: str, rust_binary: Path, asset: str | None = None
    ) -> None:
        if timeframe == "5m":
            destination = default_library_dir(self.root)
            stage = self.stage_root / destination.name
            before_count = completed_episode_count(destination)
            self.command(
                "build_routes_5m",
                "build_conditional_path_library.py",
                "--out-dir",
                stage,
                "--engine",
                "rust",
                "--rust-binary",
                rust_binary,
            )
            if self.dry_run:
                self.event("promote_routes_5m", "planned", destination=str(destination))
                self.command(
                    "cache_routes_5m",
                    "build_native_runtime_cache.py",
                    "--timeframe",
                    "5m",
                    "--library-dir",
                    destination,
                    "--force",
                )
                return
            after_count = validate_library(stage, "5m", SUPPORTED_ASSETS)
            copy_unmanaged_library_entries(destination, stage)
            leftover = promote_directory(stage, destination, self.run_id)
            self.event(
                "promote_routes_5m",
                "complete",
                destination=str(destination),
                completed_episodes_before=before_count,
                completed_episodes_after=after_count,
                new_completed_episodes=after_count - before_count,
                undeleted_backup=str(leftover) if leftover else None,
            )
            self.command(
                "cache_routes_5m",
                "build_native_runtime_cache.py",
                "--timeframe",
                "5m",
                "--library-dir",
                destination,
                "--force",
            )
            return

        if asset is None:
            raise ValueError("A 1m route-library build requires an asset")
        destination = one_minute_library_dir(self.root, asset)
        stage = self.stage_root / destination.name
        before_count = completed_episode_count(destination)
        slug = asset.lower()
        self.command(
            f"build_routes_1m_{slug}",
            "build_sol_one_minute_path_library.py",
            "--asset",
            asset,
            "--out-dir",
            stage,
            "--engine",
            "rust",
            "--rust-binary",
            rust_binary,
        )
        if self.dry_run:
            self.event(f"promote_routes_1m_{slug}", "planned", destination=str(destination))
            self.command(
                f"cache_routes_1m_{slug}",
                "build_native_runtime_cache.py",
                "--timeframe",
                "1m",
                "--asset",
                asset,
                "--library-dir",
                destination,
                "--force",
            )
            return
        after_count = validate_library(stage, "1m", (asset,))
        copy_unmanaged_library_entries(destination, stage)
        leftover = promote_directory(stage, destination, self.run_id)
        self.event(
            f"promote_routes_1m_{slug}",
            "complete",
            destination=str(destination),
            completed_episodes_before=before_count,
            completed_episodes_after=after_count,
            new_completed_episodes=after_count - before_count,
            undeleted_backup=str(leftover) if leftover else None,
        )
        self.command(
            f"cache_routes_1m_{slug}",
            "build_native_runtime_cache.py",
            "--timeframe",
            "1m",
            "--asset",
            asset,
            "--library-dir",
            destination,
            "--force",
        )

    def rebuild_routes(self, rust_binary: Path) -> None:
        self.rebuild_one_library("5m", rust_binary)
        for asset in SUPPORTED_ASSETS:
            self.rebuild_one_library("1m", rust_binary, asset)

    def rebuild_outcomes(self, rust_binary: Path) -> Path:
        destination = self.root / "data" / "prospective_entry_outcomes_v1"
        stage = self.stage_root / destination.name
        self.command(
            "build_all_outcomes",
            "prospective_entry_outcomes.py",
            "--data-dir",
            self.root / "data",
            "--output-dir",
            stage,
            "--assets",
            ",".join(SUPPORTED_ASSETS),
            "--timeframes",
            ",".join(TIMEFRAMES),
            "--engine",
            "rust",
            "--rust-binary",
            rust_binary,
        )
        if self.dry_run:
            self.event("promote_all_outcomes", "planned", destination=str(destination))
            return destination
        counts = validate_outcomes(stage)
        leftover = promote_directory(stage, destination, self.run_id)
        self.event(
            "promote_all_outcomes",
            "complete",
            destination=str(destination),
            counts=counts,
            undeleted_backup=str(leftover) if leftover else None,
        )
        return destination

    def retrain_risk_models(self, dataset_dir: Path) -> None:
        candidate_root = self.stage_root / "prospective_entry_models"
        self.command(
            "train_and_gate_risk_models",
            "train_prospective_entry_model.py",
            "--root",
            self.root,
            "--dataset-dir",
            dataset_dir,
            "--artifact-dir",
            candidate_root,
            "--timeframes",
            ",".join(TIMEFRAMES),
        )
        installed_root = default_artifact_root(self.root).resolve()
        for timeframe in TIMEFRAMES:
            stage = f"promote_risk_model_{timeframe}"
            candidate = candidate_root / timeframe / "model.joblib"
            destination = installed_root / timeframe / "model.joblib"
            if self.dry_run:
                self.event(stage, "planned", destination=str(destination), condition="chronological gate passes")
                continue
            report_path = self.root / "data" / "prospective_entry_models" / timeframe / "training_report.json"
            report = json.loads(report_path.read_text(encoding="utf-8"))
            passed = bool(report.get("promotion_gate_passed"))
            if not passed:
                self.event(stage, "kept_existing", reason="chronological promotion gate did not pass")
                continue
            if not candidate.exists():
                raise RuntimeError(f"The {timeframe} gate passed but no candidate model was written: {candidate}")
            promote_file(candidate, destination, self.run_id)
            promote_file(report_path, destination.with_name("training_report.json"), self.run_id)
            self.event(
                stage,
                "complete",
                destination=str(destination),
                generated_at_utc=report.get("generated_at_utc"),
                holdout=report.get("holdout"),
            )

    def retrain_frozen_forecast_v1(self, dataset_dir: Path) -> None:
        architecture_id = f"forecast_architecture_v1_5m_candidate_{self.run_id}"
        candidate = (
            self.stage_root
            / "forecast_architecture_v1"
            / "5m"
            / f"{architecture_id}.joblib"
        )
        report = self.stage_root / "forecast_architecture_v1" / "5m" / "training_report.json"
        installed_dir = default_artifact_root(self.root).resolve() / "5m"
        destination = (
            installed_dir / "candidates" / f"{architecture_id}.joblib"
        )
        report_destination = destination.with_suffix(".json")
        self.command(
            "train_forecast_v1_candidate_5m",
            "train_forecast_architecture_v1.py",
            "--root",
            self.root,
            "--dataset-dir",
            dataset_dir,
            "--architecture-id",
            architecture_id,
            "--output",
            candidate,
            "--report",
            report,
        )
        if self.dry_run:
            self.event(
                "save_forecast_v1_candidate_5m",
                "planned",
                destination=str(destination),
                active_manifest=str(active_manifest_path(installed_dir)),
                active_manifest_change="forbidden",
            )
            return
        if not candidate.exists() or not report.exists():
            raise RuntimeError("Frozen 5m forecast trainer did not produce its artifact and report")
        metadata = json.loads(report.read_text(encoding="utf-8"))
        if metadata.get("architecture_id") != architecture_id:
            raise RuntimeError("5m forecast candidate has an unexpected architecture id")
        active = None
        if active_manifest_path(installed_dir).exists():
            active = load_active_manifest(installed_dir)
        promote_file(candidate, destination, self.run_id)
        promote_file(report, report_destination, self.run_id)
        self.event(
            "save_forecast_v1_candidate_5m",
            "candidate_saved_active_unchanged",
            destination=str(destination),
            report=str(report_destination),
            candidate_architecture_version=architecture_id,
            candidate_artifact_sha256=sha256_file(destination),
            active_architecture_version=(active or {}).get("architecture_version"),
            active_frozen=(active or {}).get("frozen"),
            generated_at_utc=metadata.get("generated_at_utc"),
            training_label_cutoff_utc=metadata.get("training_label_cutoff_utc"),
        )

    def execute(self) -> None:
        if self.dry_run:
            print("Manual wick update plan (no files or models will be changed).", flush=True)
        else:
            free_gb = shutil.disk_usage(self.root).free / (1024**3)
            self.stage_root.mkdir(parents=True, exist_ok=False)
            self.manifest["free_space_at_start_gb"] = round(free_gb, 2)
            self.save()
        try:
            self.refresh_sources()
            rust_binary = self.build_rust_kernel()
            self.rebuild_routes(rust_binary)
            dataset_dir = self.rebuild_outcomes(rust_binary)
            self.retrain_risk_models(dataset_dir)
            self.retrain_frozen_forecast_v1(dataset_dir)
        except KeyboardInterrupt:
            self.manifest["status"] = "interrupted"
            self.manifest["finished_at_utc"] = utc_now()
            self.save()
            raise
        except BaseException as error:
            self.manifest["status"] = "failed"
            self.manifest["finished_at_utc"] = utc_now()
            self.manifest["error"] = str(error)
            self.save()
            raise
        self.manifest["status"] = "planned" if self.dry_run else "complete"
        self.manifest["finished_at_utc"] = utc_now()
        self.save()
        if not self.dry_run:
            try:
                shutil.rmtree(self.stage_root)
            except OSError as error:
                self.manifest["staging_cleanup_warning"] = str(error)
                self.save()
            print(f"\nUpdate complete. Manifest: {self.manifest_path}", flush=True)
        print(
            "V3 neural route experts were intentionally left unchanged; fresh episodes are now available "
            "to the historical route engine, and the gated numerical risk models were evaluated.",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    update = subparsers.add_parser("update", help="Run the complete manual 1m/5m update")
    update.add_argument("--dry-run", action="store_true", help="Print the full plan without changing anything")
    update.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Use the local candle snapshots as-is; useful only for deterministic research",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "update":
        UpdateRunner(Path(__file__).resolve().parent, dry_run=args.dry_run, skip_refresh=args.skip_refresh).execute()


if __name__ == "__main__":
    main()
