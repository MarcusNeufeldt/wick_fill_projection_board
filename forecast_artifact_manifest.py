"""Immutable artifact manifest and integrity helpers for prospective forecasts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


ACTIVE_MANIFEST_SCHEMA_VERSION = "forecast-active-manifest-v1.0.0"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def active_manifest_path(model_dir: Path) -> Path:
    return model_dir / "active.json"


def write_active_manifest(
    model_dir: Path,
    artifact_path: Path,
    *,
    architecture_version: str,
    training_labels_through: str,
    frozen: bool,
    allow_replace_frozen: bool = False,
) -> dict[str, Any]:
    model_dir = model_dir.resolve()
    artifact_path = artifact_path.resolve()
    if artifact_path.parent != model_dir:
        raise ValueError("Active artifact must live directly inside its timeframe model directory")
    payload = {
        "schema_version": ACTIVE_MANIFEST_SCHEMA_VERSION,
        "architecture_version": architecture_version,
        "artifact_file": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "frozen": bool(frozen),
        "training_labels_through": training_labels_through,
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    destination = active_manifest_path(model_dir)
    if destination.exists():
        existing = load_active_manifest(model_dir)
        if existing == payload:
            return existing
        if bool(existing["frozen"]) and not allow_replace_frozen:
            raise RuntimeError(
                "Refusing to replace a frozen active forecast manifest without "
                "an explicit promotion override"
            )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="active.", suffix=".tmp", dir=model_dir
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_active_manifest(model_dir: Path) -> dict[str, Any]:
    path = active_manifest_path(model_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != ACTIVE_MANIFEST_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported active forecast manifest: {payload.get('schema_version')}")
    required = {
        "architecture_version",
        "artifact_file",
        "artifact_sha256",
        "frozen",
        "training_labels_through",
    }
    missing = required.difference(payload)
    if missing:
        raise RuntimeError(f"Active forecast manifest is missing: {sorted(missing)}")
    return payload


def resolve_active_artifact(model_dir: Path) -> tuple[Path, dict[str, Any]]:
    model_dir = model_dir.resolve()
    manifest = load_active_manifest(model_dir)
    artifact = (model_dir / str(manifest["artifact_file"])).resolve()
    if artifact.parent != model_dir:
        raise RuntimeError("Active forecast artifact escapes its timeframe model directory")
    if not artifact.is_file():
        raise RuntimeError(f"Active forecast artifact does not exist: {artifact.name}")
    actual_hash = sha256_file(artifact)
    expected_hash = str(manifest["artifact_sha256"]).lower()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Active forecast artifact hash mismatch: expected {expected_hash}, got {actual_hash}"
        )
    return artifact, manifest
