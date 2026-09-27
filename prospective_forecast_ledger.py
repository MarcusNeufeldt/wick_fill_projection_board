"""Append-only local ledger for frozen prospective forecast observations."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


def default_ledger_path(project_root: Path) -> Path:
    if override := os.environ.get("CANDLE_PROJECTION_LEDGER"):
        return Path(override)
    if local_app_data := os.environ.get("LOCALAPPDATA"):
        return (
            Path(local_app_data)
            / "candle_projection_algo"
            / "prospective_validation"
            / "forecasts.sqlite"
        )
    return project_root / "data" / "prospective_validation" / "forecasts.sqlite"


def _ensure_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS forecasts (
            architecture_id TEXT NOT NULL,
            architecture_version TEXT,
            artifact_hash TEXT,
            forecast_source TEXT,
            asset TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            signal_open_time_ms INTEGER NOT NULL,
            observation_close_time_ms INTEGER NOT NULL,
            direction TEXT NOT NULL,
            wick_target REAL NOT NULL,
            entry_price REAL NOT NULL,
            prediction_json TEXT NOT NULL,
            recorded_at_utc TEXT NOT NULL,
            outcomes_json TEXT,
            resolved_at_utc TEXT,
            PRIMARY KEY (
                architecture_id,
                asset,
                timeframe,
                signal_open_time_ms,
                observation_close_time_ms
            )
        )
        """
    )
    existing = {
        str(row[1]) for row in connection.execute("PRAGMA table_info(forecasts)")
    }
    for column in ("architecture_version", "artifact_hash", "forecast_source"):
        if column not in existing:
            connection.execute(f"ALTER TABLE forecasts ADD COLUMN {column} TEXT")
    connection.execute(
        """
        UPDATE forecasts
        SET architecture_version = architecture_id
        WHERE architecture_version IS NULL
        """
    )
    connection.execute(
        """
        CREATE VIEW IF NOT EXISTS forecast_v1_evaluation AS
        SELECT *
        FROM forecasts
        WHERE forecast_source = 'forecast_v1'
          AND artifact_hash IS NOT NULL
        """
    )


def record_forecast(
    path: Path,
    *,
    architecture_version: str,
    artifact_hash: str,
    forecast_source: str,
    asset: str,
    timeframe: str,
    signal_open_time_ms: int,
    observation_close_time_ms: int,
    direction: str,
    wick_target: float,
    entry_price: float,
    payload: Mapping[str, Any],
) -> bool:
    """Insert one immutable forecast. Return False when the same snapshot already exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    connection = sqlite3.connect(path, timeout=5.0)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        _ensure_schema(connection)
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO forecasts (
                architecture_id, architecture_version, artifact_hash,
                forecast_source, asset, timeframe, signal_open_time_ms,
                observation_close_time_ms, direction, wick_target, entry_price,
                prediction_json, recorded_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                architecture_version,
                architecture_version,
                artifact_hash,
                forecast_source,
                asset,
                timeframe,
                int(signal_open_time_ms),
                int(observation_close_time_ms),
                direction,
                float(wick_target),
                float(entry_price),
                json.dumps(dict(payload), sort_keys=True, separators=(",", ":")),
                now,
            ),
        )
        connection.commit()
        return cursor.rowcount == 1
    finally:
        connection.close()


def backfill_forecast_provenance(
    path: Path,
    *,
    architecture_version: str,
    artifact_hash: str,
    forecast_source: str = "forecast_v1",
) -> int:
    """Attach verified provenance to pre-manifest rows from one architecture."""
    if not path.exists():
        return 0
    connection = sqlite3.connect(path, timeout=5.0)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        _ensure_schema(connection)
        cursor = connection.execute(
            """
            UPDATE forecasts
            SET architecture_version = ?, artifact_hash = ?, forecast_source = ?
            WHERE architecture_id = ?
              AND (artifact_hash IS NULL OR forecast_source IS NULL)
            """,
            (
                architecture_version,
                artifact_hash,
                forecast_source,
                architecture_version,
            ),
        )
        connection.commit()
        return int(cursor.rowcount)
    finally:
        connection.close()
