"""Optional Rust accelerator for the hot historical-state match operation.

The public dashboard continues to use the Python implementation as its
reference fallback.  When the local release DLL is present, this module keeps
the observable matching contract intact while moving the full state scan and
per-episode reduction into native code.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
NATIVE_LIBRARY = ROOT / "native_wick_matcher" / "target" / "release" / "wick_fast_matcher.dll"
RUNTIME_CACHE_VERSION = 1
RUNTIME_CACHE_NAME = "native_runtime_cache_v1"
RUNTIME_STATE_COLUMNS = [
    "episode_id",
    "asset",
    "timeframe",
    "direction",
    "direction_sign",
    "offset_bars",
    "normalized_open_pct",
    "normalized_high_pct",
    "normalized_low_pct",
    "normalized_close_pct",
    "signal_to_departure_bars",
    "signal_to_fill_bars",
    "alignment_peak_move_pct",
    "alignment_current_move_pct",
    "alignment_drawdown_pct",
    "remaining_to_fill_bars",
    "future_peak_move_pct",
    "candidate_after_departure",
]


def runtime_cache_dir(library_dir: Path, timeframe: str) -> Path:
    return library_dir / RUNTIME_CACHE_NAME / timeframe


def library_fingerprint(library_dir: Path, timeframe: str) -> dict[str, Any]:
    """Cheap freshness identity for source episodes and compressed path files."""
    from conditional_wick_assets import assets_for_timeframe

    files = [library_dir / "episodes.csv"]
    if timeframe == "1m":
        # One-minute libraries are intentionally isolated per asset so only one
        # dense cache is loaded at a time. Fingerprint the paths actually owned
        # by this library instead of assuming the five-asset 5m/15m layout.
        files.extend(sorted((library_dir / "paths").glob(f"*_{timeframe}_paths.csv.gz")))
    else:
        files.extend(
            library_dir / "paths" / f"{asset}_{timeframe}_paths.csv.gz"
            for asset in assets_for_timeframe(timeframe)
        )
    entries: list[dict[str, int | str]] = []
    for path in files:
        stat = path.stat()
        entries.append({"name": str(path.relative_to(library_dir)), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return {"timeframe": timeframe, "files": entries}


class _NativeLibrary:
    """Small, dependency-free ctypes wrapper around the Rust CDylib."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.library = ctypes.CDLL(str(path))
        function = self.library.wick_match_top_k
        function.argtypes = [
            ctypes.POINTER(ctypes.c_uint32),  # state episode code
            ctypes.POINTER(ctypes.c_double),  # state current move
            ctypes.POINTER(ctypes.c_double),  # state peak move
            ctypes.POINTER(ctypes.c_double),  # state drawdown
            ctypes.POINTER(ctypes.c_int32),  # state offset
            ctypes.c_size_t,  # state count
            ctypes.POINTER(ctypes.c_int64),  # per-episode fill close
            ctypes.POINTER(ctypes.c_double),  # per-episode feature distance
            ctypes.POINTER(ctypes.c_double),  # per-episode category distance
            ctypes.c_size_t,  # episode count
            ctypes.c_int64,  # snapshot close
            ctypes.c_double,  # current move
            ctypes.c_double,  # peak move
            ctypes.c_double,  # drawdown
            ctypes.c_double,  # elapsed bars
            ctypes.c_double,  # current state weight
            ctypes.c_double,  # peak state weight
            ctypes.c_double,  # drawdown state weight
            ctypes.c_double,  # elapsed state weight
            ctypes.c_size_t,  # requested top k
            ctypes.POINTER(ctypes.c_uint32),  # output episode codes
            ctypes.POINTER(ctypes.c_uint64),  # output state rows
            ctypes.POINTER(ctypes.c_double),  # output scores
        ]
        function.restype = ctypes.c_size_t
        self._match = function

    @staticmethod
    def _pointer(values: np.ndarray, type_: type[ctypes._SimpleCData]) -> Any:
        return values.ctypes.data_as(ctypes.POINTER(type_))

    def top_k(
        self,
        *,
        state_episode_codes: np.ndarray,
        state_current_move: np.ndarray,
        state_peak_move: np.ndarray,
        state_drawdown: np.ndarray,
        state_offsets: np.ndarray,
        fill_close_times: np.ndarray,
        feature_distance: np.ndarray,
        category_distance: np.ndarray,
        snapshot_close_time_ms: int,
        current_state: dict[str, float],
        state_weights: dict[str, float],
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        capacity = min(int(top_k), len(fill_close_times))
        if capacity <= 0 or not len(state_episode_codes):
            return (
                np.empty(0, dtype=np.uint32),
                np.empty(0, dtype=np.uint64),
                np.empty(0, dtype=np.float64),
            )
        output_episodes = np.empty(capacity, dtype=np.uint32)
        output_states = np.empty(capacity, dtype=np.uint64)
        output_scores = np.empty(capacity, dtype=np.float64)
        count = int(
            self._match(
                self._pointer(state_episode_codes, ctypes.c_uint32),
                self._pointer(state_current_move, ctypes.c_double),
                self._pointer(state_peak_move, ctypes.c_double),
                self._pointer(state_drawdown, ctypes.c_double),
                self._pointer(state_offsets, ctypes.c_int32),
                len(state_episode_codes),
                self._pointer(fill_close_times, ctypes.c_int64),
                self._pointer(feature_distance, ctypes.c_double),
                self._pointer(category_distance, ctypes.c_double),
                len(fill_close_times),
                int(snapshot_close_time_ms),
                float(current_state["current_move_pct"]),
                float(current_state["peak_move_pct"]),
                float(current_state["drawdown_from_peak_pct"]),
                float(current_state["elapsed_bars"]),
                float(state_weights["current_move_pct"]),
                float(state_weights["peak_move_pct"]),
                float(state_weights["drawdown_from_peak_pct"]),
                float(state_weights["elapsed_bars"]),
                capacity,
                self._pointer(output_episodes, ctypes.c_uint32),
                self._pointer(output_states, ctypes.c_uint64),
                self._pointer(output_scores, ctypes.c_double),
            )
        )
        if count > capacity:
            raise RuntimeError("Native matcher returned more rows than its output capacity")
        return output_episodes[:count], output_states[:count], output_scores[:count]


def native_library_status() -> tuple[_NativeLibrary | None, str]:
    """Load the local accelerator if it has been compiled, without failing serving."""
    if not NATIVE_LIBRARY.exists():
        return None, "python_fallback"
    try:
        return _NativeLibrary(NATIVE_LIBRARY), "rust_native"
    except OSError:
        return None, "python_fallback"


@dataclass(frozen=True)
class NativeStateIndex:
    """Immutable candidate-state arrays built once per loaded trajectory library."""

    episode_ids: np.ndarray
    fill_close_times: np.ndarray
    state_rows: np.ndarray
    state_episode_codes: np.ndarray
    state_current_move: np.ndarray
    state_peak_move: np.ndarray
    state_drawdown: np.ndarray
    state_offsets: np.ndarray
    library: _NativeLibrary | None
    backend: str

    def close(self) -> None:
        """Release Windows memory-map handles when replacing a persisted cache."""
        for values in (
            self.episode_ids,
            self.fill_close_times,
            self.state_rows,
            self.state_episode_codes,
            self.state_current_move,
            self.state_peak_move,
            self.state_drawdown,
            self.state_offsets,
        ):
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()

    def save(self, directory: Path) -> None:
        """Persist only dense numerical candidate arrays; rows remain in Feather."""
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {
            "episode_ids.npy": self.episode_ids,
            "fill_close_times.npy": self.fill_close_times,
            "state_rows.npy": self.state_rows,
            "state_episode_codes.npy": self.state_episode_codes,
            "state_current_move.npy": self.state_current_move,
            "state_peak_move.npy": self.state_peak_move,
            "state_drawdown.npy": self.state_drawdown,
            "state_offsets.npy": self.state_offsets,
        }
        for name, values in arrays.items():
            np.save(directory / name, values, allow_pickle=False)

    @classmethod
    def load(cls, directory: Path) -> "NativeStateIndex":
        library, backend = native_library_status()
        return cls(
            episode_ids=np.load(directory / "episode_ids.npy", mmap_mode="r", allow_pickle=False),
            fill_close_times=np.load(directory / "fill_close_times.npy", mmap_mode="r", allow_pickle=False),
            state_rows=np.load(directory / "state_rows.npy", mmap_mode="r", allow_pickle=False),
            state_episode_codes=np.load(directory / "state_episode_codes.npy", mmap_mode="r", allow_pickle=False),
            state_current_move=np.load(directory / "state_current_move.npy", mmap_mode="r", allow_pickle=False),
            state_peak_move=np.load(directory / "state_peak_move.npy", mmap_mode="r", allow_pickle=False),
            state_drawdown=np.load(directory / "state_drawdown.npy", mmap_mode="r", allow_pickle=False),
            state_offsets=np.load(directory / "state_offsets.npy", mmap_mode="r", allow_pickle=False),
            library=library,
            backend=backend,
        )

    @classmethod
    def from_frames(
        cls,
        events: pd.DataFrame,
        path_states: pd.DataFrame,
        timeframe: str,
    ) -> "NativeStateIndex":
        """Create dense candidate arrays without changing render-path data."""
        from build_conditional_path_scenarios import sampled_matching_states

        event_rows = events.sort_values("episode_id", kind="stable").reset_index(drop=True)
        episode_ids = np.ascontiguousarray(event_rows["episode_id"].astype(str).to_numpy(dtype=str))
        if len(set(episode_ids.tolist())) != len(episode_ids):
            raise RuntimeError("Native matcher requires unique completed episode IDs")
        episode_codes = {episode_id: code for code, episode_id in enumerate(episode_ids.tolist())}
        candidate_source = sampled_matching_states(path_states, timeframe)
        candidate_source = candidate_source.loc[candidate_source["candidate_after_departure"]]
        row_ids = np.ascontiguousarray(candidate_source.index.to_numpy(dtype=np.int64))
        candidate_ids = candidate_source["episode_id"].astype(str).to_numpy(dtype=str)
        try:
            state_episode_codes = np.fromiter(
                (episode_codes[episode_id] for episode_id in candidate_ids),
                dtype=np.uint32,
                count=len(candidate_ids),
            )
        except KeyError as error:  # pragma: no cover - invalid library boundary
            raise RuntimeError("A candidate path state has no completed episode record") from error
        library, backend = native_library_status()
        return cls(
            episode_ids=episode_ids,
            fill_close_times=np.ascontiguousarray(event_rows["fill_close_time_ms"].to_numpy(dtype=np.int64)),
            state_rows=row_ids,
            state_episode_codes=np.ascontiguousarray(state_episode_codes),
            state_current_move=np.ascontiguousarray(
                candidate_source["alignment_current_move_pct"].to_numpy(dtype=np.float64)
            ),
            state_peak_move=np.ascontiguousarray(candidate_source["alignment_peak_move_pct"].to_numpy(dtype=np.float64)),
            state_drawdown=np.ascontiguousarray(candidate_source["alignment_drawdown_pct"].to_numpy(dtype=np.float64)),
            state_offsets=np.ascontiguousarray(candidate_source["offset_bars"].to_numpy(dtype=np.int32)),
            library=library,
            backend=backend,
        )

    def choose(
        self,
        *,
        events: pd.DataFrame,
        path_states: pd.DataFrame,
        event_scores: pd.DataFrame,
        snapshot_close_time_ms: int,
        current_state: dict[str, float],
        state_weights: dict[str, float],
        top_k: int,
    ) -> pd.DataFrame | None:
        """Return the same top candidate rows as the native scan, or None for fallback."""
        if self.library is None:
            return None
        score_by_id = event_scores.set_index("episode_id")
        ids = self.episode_ids.astype(str)
        positions = pd.Index(ids).get_indexer(score_by_id.index.astype(str))
        feature_distance = np.full(len(ids), np.inf, dtype=np.float64)
        category_distance = np.full(len(ids), np.inf, dtype=np.float64)
        present = positions >= 0
        if not np.any(present):
            return pd.DataFrame()
        feature_distance[positions[present]] = score_by_id.loc[
            score_by_id.index[present], "feature_distance"
        ].to_numpy(dtype=np.float64)
        category_distance[positions[present]] = score_by_id.loc[
            score_by_id.index[present], "category_distance"
        ].to_numpy(dtype=np.float64)
        episode_codes, state_positions, scores = self.library.top_k(
            state_episode_codes=self.state_episode_codes,
            state_current_move=self.state_current_move,
            state_peak_move=self.state_peak_move,
            state_drawdown=self.state_drawdown,
            state_offsets=self.state_offsets,
            fill_close_times=self.fill_close_times,
            feature_distance=feature_distance,
            category_distance=category_distance,
            snapshot_close_time_ms=snapshot_close_time_ms,
            current_state=current_state,
            state_weights=state_weights,
            top_k=top_k,
        )
        if not len(state_positions):
            return pd.DataFrame()
        original_rows = self.state_rows[state_positions.astype(np.intp, copy=False)]
        value = path_states.loc[original_rows].copy()
        value["feature_distance"] = feature_distance[episode_codes]
        value["category_distance"] = category_distance[episode_codes]
        value["match_score"] = scores
        value["state_distance"] = value["match_score"] - 0.5 * value["feature_distance"] - value["category_distance"]
        return value.reset_index(drop=True)


def write_runtime_cache(
    library_dir: Path,
    timeframe: str,
    states: pd.DataFrame,
    native_state_index: NativeStateIndex,
    *,
    force: bool = False,
) -> Path:
    """Atomically materialize a local binary cache after an offline library build."""
    target = runtime_cache_dir(library_dir, timeframe)
    if target.exists() and not force:
        raise FileExistsError(f"Runtime cache already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{timeframe}_runtime_cache_", dir=target.parent))
    try:
        cache_states = states.loc[:, RUNTIME_STATE_COLUMNS].reset_index(drop=True)
        cache_states.to_feather(temporary / "states.feather", compression="lz4")
        native_state_index.save(temporary / "native")
        metadata = {
            "schema_version": RUNTIME_CACHE_VERSION,
            "source_fingerprint": library_fingerprint(library_dir, timeframe),
            "state_row_count": int(len(cache_states)),
            "candidate_state_count": int(len(native_state_index.state_rows)),
            "native_backend_when_built": native_state_index.backend,
        }
        (temporary / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if target.exists():
            backup = target.with_name(f"{target.name}.replaced")
            if backup.exists():
                shutil.rmtree(backup)
            os.replace(target, backup)
            try:
                os.replace(temporary, target)
            except Exception:
                os.replace(backup, target)
                raise
            shutil.rmtree(backup)
        else:
            os.replace(temporary, target)
        return target
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_runtime_cache(library_dir: Path, timeframe: str) -> tuple[pd.DataFrame, NativeStateIndex] | None:
    """Return a matching fresh cache, otherwise leave the caller on the CSV path."""
    target = runtime_cache_dir(library_dir, timeframe)
    metadata_path = target / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != RUNTIME_CACHE_VERSION:
            return None
        if metadata.get("source_fingerprint") != library_fingerprint(library_dir, timeframe):
            return None
        states = pd.read_feather(target / "states.feather")
        native_state_index = NativeStateIndex.load(target / "native")
        if len(states) != int(metadata.get("state_row_count", -1)):
            return None
        if len(native_state_index.state_rows) != int(metadata.get("candidate_state_count", -1)):
            return None
        return states, native_state_index
    except (OSError, ValueError, json.JSONDecodeError):
        return None
