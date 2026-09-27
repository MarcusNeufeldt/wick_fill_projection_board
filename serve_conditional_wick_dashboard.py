#!/usr/bin/env python3
"""Serve the local pin-and-project dashboard for conditional wick trajectories.

The service intentionally runs separately from the existing generic Rust
nearest-neighbour dashboard. It never overwrites its files or binds its port.
The service caches the immutable trajectory-state library by timeframe, so a
new pinned signal reuses the same validated V1 selector without repeatedly
re-reading millions of historical path rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from build_conditional_path_library import (
    FEATURE_COLUMNS,
    detect_strict_signals,
    read_five_minute_file,
    read_one_minute_file,
    resample_to_fifteen_minutes,
    utc_iso,
)
from build_conditional_path_scenarios import (
    CATEGORY_PENALTIES,
    FEATURE_WEIGHTS,
    MATCH_SCORE_COMPONENT_COLUMNS,
    STATE_WEIGHTS,
    actual_candles,
    has_confirmed_departure,
    normalized_price,
    parse_utc,
    prepare_path_states,
    project_at,
    projected_candles,
    projected_path_metrics,
    read_library_paths,
    state_for_live_path,
    with_fill_close_time_ms,
)
from candle_archetype import (
    ARCHETYPE_BLEND_WEIGHT,
    ARCHETYPE_COMPONENT_SCALES,
    ARCHETYPE_POOL_SIZE,
    SOFT_MATCHER_BASELINE_VERSION,
    archetype_profile,
)
from conditional_wick_assets import (
    SUPPORTED_ASSETS,
    SUPPORTED_DASHBOARD_TIMEFRAMES,
    assets_for_timeframe,
    default_library_dir,
    one_minute_library_dir,
)
from download_futures_klines import CSV_HEADER
from native_wick_matcher import (
    RUNTIME_STATE_COLUMNS,
    NativeStateIndex,
    load_runtime_cache,
    runtime_cache_dir,
)
from refresh_futures_klines import (
    RefreshResult,
    exchange_server_time_ms,
    refresh_source,
    source_snapshot,
)
from forecast_architecture_v1 import (
    ARCHITECTURE_SCHEMA_VERSION,
    load_architecture,
    predict_architecture,
    sequence_features_from_live,
)
from forecast_artifact_manifest import (
    active_manifest_path,
    load_active_manifest,
    resolve_active_artifact,
    sha256_file,
)
from prospective_entry_model import (
    DISPLAY_HORIZONS_MINUTES,
    DISPLAY_THRESHOLDS_PCT,
    default_artifact_root as prospective_artifact_root,
    live_observation as prospective_live_observation,
    load_bundle as load_prospective_bundle,
    predict_bundle as predict_prospective_bundle,
)
from prospective_forecast_ledger import default_ledger_path, record_forecast
from train_conditional_wick_v2 import load_model_bundle, predict_from_observable_state

ROOT = Path(__file__).resolve().parent
ASSETS = SUPPORTED_ASSETS
TIMEFRAMES = SUPPORTED_DASHBOARD_TIMEFRAMES
DEFAULT_SIGNAL = {
    "asset": "ETHUSDT",
    "timeframe": "5m",
    "signal_time": "2026-09-16T18:35:00Z",
}
LIVE_SOURCE_SPECS = tuple((asset, "5m", asset) for asset in ASSETS) + tuple(
    (asset, "1m", f"{asset}_1m") for asset in ASSETS
)
VISUAL_TIMEFRAMES = ("1m", "5m", "15m", "30m")
MATCHING_MODES = ("adaptive", "legacy_v3", "blended", "archetype")
V3_LIVE_ARTIFACT_DIRS = {
    "1m": ("neural_path_v3_1m_fresh", "neural_path_v3_1m_mature"),
    "5m": ("neural_path_v3_5m_fresh", "neural_path_v3_5m_mature"),
}
SIGNAL_STABLE_LOOKBACK_BARS = 64
MAX_INCREMENTAL_APPEND_ROWS = 64


def validate_asset_timeframe(asset: str, timeframe: str) -> None:
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"timeframe must be {'/'.join(TIMEFRAMES)}")
    if asset not in assets_for_timeframe(timeframe):
        raise ValueError(f"{timeframe} does not support {asset}")


def validate_matching_mode(matching_mode: str) -> str:
    if matching_mode not in MATCHING_MODES:
        raise ValueError(f"matching_mode must be {'/'.join(MATCHING_MODES)}")
    return matching_mode


def display_timeframe_minutes(
    display_timeframe: str | None, source_timeframe: str
) -> tuple[str, int]:
    """Validate a display-only aggregation timeframe and return its minute width."""
    display = source_timeframe if not display_timeframe else display_timeframe
    if display not in VISUAL_TIMEFRAMES:
        raise ValueError(f"display_timeframe must be {'/'.join(VISUAL_TIMEFRAMES)}")
    source_minutes = int(source_timeframe.removesuffix("m"))
    display_minutes = int(display.removesuffix("m"))
    if display_minutes < source_minutes:
        raise ValueError(
            "display_timeframe cannot be smaller than the source timeframe"
        )
    return display, display_minutes


def aggregate_visual_candles(
    actual: list[dict[str, float | int]],
    projected: list[dict[str, float | int]],
    display_minutes: int,
) -> tuple[list[dict[str, float | int | str]], bool]:
    """Aggregate observed and projected candles without mixing their route semantics."""
    buckets: dict[int, dict[str, float | int]] = {}
    bucket_seconds = int(display_minutes) * 60
    for candles, phase in ((actual, "actual"), (projected, "projected")):
        for candle in candles:
            bucket_time = (int(candle["time"]) // bucket_seconds) * bucket_seconds
            bucket = buckets.get(bucket_time)
            if bucket is None:
                bucket = {
                    "time": bucket_time,
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                    "actual": 0,
                    "projected": 0,
                }
                buckets[bucket_time] = bucket
            else:
                bucket["high"] = max(float(bucket["high"]), float(candle["high"]))
                bucket["low"] = min(float(bucket["low"]), float(candle["low"]))
                bucket["close"] = float(candle["close"])
            bucket[phase] = int(bucket[phase]) + 1
    has_transition = False
    rows: list[dict[str, float | int | str]] = []
    for bucket in buckets.values():
        phase = (
            "transition"
            if bucket["actual"] and bucket["projected"]
            else "projected"
            if bucket["projected"]
            else "actual"
        )
        has_transition = has_transition or phase == "transition"
        rows.append(
            {
                "time": int(bucket["time"]),
                "open": float(bucket["open"]),
                "high": float(bucket["high"]),
                "low": float(bucket["low"]),
                "close": float(bucket["close"]),
                "phase": phase,
            }
        )
    return rows, has_transition


def incremental_strict_signals(
    frame: pd.DataFrame,
    previous_signals: pd.DataFrame,
    asset: str,
    timeframe: str,
    previous_row_count: int,
) -> pd.DataFrame:
    """Recompute every appended row, plus the history needed for its features."""
    if not 0 < previous_row_count <= len(frame):
        raise ValueError("Incremental signal cache has an invalid prior row count")
    recompute_start = max(0, previous_row_count - SIGNAL_STABLE_LOOKBACK_BARS)
    recomputed_signals = detect_strict_signals(
        frame.iloc[recompute_start:].copy(), asset, timeframe
    )
    recomputed_signals["bar_index"] = (
        recomputed_signals["bar_index"].astype("int64") + recompute_start
    )
    stable_from = recompute_start + SIGNAL_STABLE_LOOKBACK_BARS
    preserved = previous_signals.loc[
        previous_signals["bar_index"].lt(stable_from)
    ].copy()
    return (
        pd.concat(
            [
                preserved,
                recomputed_signals.loc[recomputed_signals["bar_index"].ge(stable_from)],
            ],
            ignore_index=True,
        )
        .drop_duplicates("open_time", keep="last")
        .sort_values("bar_index", kind="stable")
        .reset_index(drop=True)
    )


def episode_row_spans(path_states: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Index contiguous, episode-sorted path-state rows for direct route extraction."""
    episode_ids = path_states["episode_id"].astype(str).to_numpy()
    if not len(episode_ids):
        return {}
    starts = np.flatnonzero(
        np.concatenate((np.asarray([True]), episode_ids[1:] != episode_ids[:-1]))
    )
    ends = np.concatenate((starts[1:], np.asarray([len(episode_ids)])))
    return {
        str(episode_ids[start]): (int(start), int(end))
        for start, end in zip(starts, ends, strict=True)
    }


def appended_candle_rows(
    path: Path, row_count: int, interval_minutes: int
) -> pd.DataFrame:
    """Read and validate the small append tail without reparsing the five-year CSV."""
    if row_count <= 0 or row_count > MAX_INCREMENTAL_APPEND_ROWS:
        raise ValueError(
            "Incremental source append must contain a small positive number of rows"
        )
    size = path.stat().st_size
    read_size = min(size, max(16_384, row_count * 512))
    lines: list[bytes] = []
    while True:
        with path.open("rb") as handle:
            handle.seek(max(0, size - read_size))
            lines = [line for line in handle.read().splitlines() if line.strip()]
        if len(lines) >= row_count or read_size >= size:
            break
        read_size = min(size, read_size * 2)
    raw_rows = [next(csv.reader([line.decode("utf-8")])) for line in lines[-row_count:]]
    if len(raw_rows) != row_count or any(
        len(row) != len(CSV_HEADER) for row in raw_rows
    ):
        raise RuntimeError(
            "Source tail did not contain the expected complete candle rows"
        )
    frame = pd.DataFrame(raw_rows, columns=CSV_HEADER)
    columns = ["open_time", "open", "high", "low", "close", "volume", "close_time"]
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["open_time"] = frame["open_time"].astype("int64")
    frame["close_time"] = frame["close_time"].astype("int64")
    timestamps = frame["open_time"].to_numpy(dtype=np.int64)
    if len(timestamps) > 1 and not np.all(
        np.diff(timestamps) == int(interval_minutes) * 60_000
    ):
        raise RuntimeError("Incremental source tail has a timestamp gap")
    return frame.loc[:, columns]


def _optional_iso_utc(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value if parsed.tzinfo is not None else None


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def snapshot_age_support(
    elapsed_bars: int, snapshot_offsets_bars: list[int]
) -> dict[str, Any]:
    """Classify exact, interpolated, or unsupported V2 snapshot ages without overstating calibration."""
    offsets = sorted(
        {
            parsed
            for value in snapshot_offsets_bars
            if (parsed := _optional_nonnegative_int(value)) is not None and parsed > 0
        }
    )
    elapsed = int(elapsed_bars)
    if not offsets:
        return {
            "status": "unknown",
            "snapshot_offsets_bars": [],
            "nearest_lower_offset_bars": None,
            "nearest_upper_offset_bars": None,
        }
    lower = max((value for value in offsets if value <= elapsed), default=None)
    upper = min((value for value in offsets if value >= elapsed), default=None)
    if elapsed in offsets:
        status = "exact_sampled_age"
    elif lower is not None and upper is not None:
        status = "between_sampled_ages"
    else:
        status = "outside_sampled_age"
    return {
        "status": status,
        "snapshot_offsets_bars": offsets,
        "nearest_lower_offset_bars": lower,
        "nearest_upper_offset_bars": upper,
    }


def v2_artifact_metadata(summary: dict[str, Any]) -> dict[str, Any]:
    """Expose static-artifact provenance without inferring freshness from source CSVs."""
    split = summary.get("chronological_split", {})
    data = summary.get("data", {})
    return {
        "schema_version": summary.get("schema_version")
        if isinstance(summary.get("schema_version"), str)
        else None,
        "generated_at_utc": _optional_iso_utc(summary.get("generated_at_utc")),
        "training_label_cutoff_utc": _optional_iso_utc(
            split.get("max_train_fill_close_utc")
        ),
        "holdout_start_utc": _optional_iso_utc(split.get("holdout_start_utc")),
        "training_episode_count": _optional_nonnegative_int(
            split.get("train_episode_count")
        ),
        "training_snapshot_rows": _optional_nonnegative_int(
            data.get("train_rows_used")
        ),
        "conditional_population": (
            summary.get("conditional_population")
            if isinstance(summary.get("conditional_population"), str)
            else None
        ),
    }


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Conditional Wick Paths</title>
  <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root { --bg:#0b1118; --panel:#111b26; --line:#26394c; --text:#e7eef5; --muted:#93a5b7; --amber:#eca846; --teal:#38b7a6; --violet:#9497f8; --rust:#dc7954; --danger:#f08a7a; }
    * { box-sizing:border-box; }
    body { margin:0; background:radial-gradient(ellipse 100% 70% at 50% -20%,#1a2b39 0%,var(--bg) 58%); color:var(--text); font:14px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; min-height:100vh; }
    main { width:min(1450px,calc(100% - 32px)); margin:0 auto; padding:28px 0 36px; }
    header { display:flex; justify-content:space-between; align-items:start; gap:20px; padding:4px 0 22px; }
    .eyebrow { color:var(--amber); font-size:11px; font-weight:800; letter-spacing:.14em; text-transform:uppercase; }
    h1 { margin:5px 0 5px; font-size:clamp(25px,3vw,40px); line-height:1.08; letter-spacing:-.035em; }
    .sub { color:var(--muted); max-width:760px; margin:0; }
    .stamp { border:1px solid var(--line); border-radius:999px; padding:7px 11px; color:var(--muted); white-space:nowrap; font-size:12px; }
    .control { border:1px solid var(--line); background:rgba(17,27,38,.92); border-radius:15px; padding:16px; display:grid; grid-template-columns:112px 100px 126px minmax(180px,1fr) minmax(220px,1.1fr) auto; gap:10px; align-items:end; box-shadow:0 18px 45px rgba(0,0,0,.18); }
    label { color:var(--muted); display:grid; gap:6px; font-size:11px; font-weight:750; letter-spacing:.06em; text-transform:uppercase; }
    select,input,button { font:inherit; }
    select,input { color:var(--text); background:#0c151f; border:1px solid #30465a; border-radius:9px; min-height:40px; padding:0 10px; outline:none; }
    select:focus,input:focus { border-color:var(--amber); box-shadow:0 0 0 3px rgba(236,168,70,.13); }
    button { cursor:pointer; border:0; border-radius:9px; min-height:40px; padding:0 15px; color:#101820; font-weight:800; background:var(--amber); transition:transform .16s ease,filter .16s ease; }
    button:hover { transform:translateY(-1px); filter:brightness(1.06); }
    button:disabled { cursor:wait; opacity:.6; transform:none; }
    .status { min-height:24px; color:var(--muted); margin:12px 0 12px; }
    .status.error { color:var(--danger); }
    .layout { display:grid; grid-template-columns:minmax(0,1fr) 310px; gap:16px; }
    .panel { border:1px solid var(--line); border-radius:15px; background:rgba(17,27,38,.9); overflow:hidden; }
    .chart-head { padding:14px 17px 11px; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:16px; align-items:start; }
    .chart-head h2 { margin:0; font-size:15px; letter-spacing:-.015em; }
    .chart-head p { color:var(--muted); font-size:12px; margin:3px 0 0; }
    #chart { height:550px; width:100%; }
    .sidebar { display:grid; gap:16px; align-content:start; }
    .risk { padding:15px; }
    .risk h2 { margin:0 0 11px; font-size:13px; color:var(--muted); letter-spacing:.07em; text-transform:uppercase; }
    .route-grid { display:grid; gap:9px; }
    .route { --route:#fff; position:relative; text-align:left; color:var(--text); background:#0d1721; border:1px solid #2b4053; overflow:hidden; padding:12px; min-height:121px; }
    .route[aria-pressed="true"] { border-color:var(--route); box-shadow:0 0 0 1px color-mix(in srgb,var(--route) 40%,transparent); }
    .route::after { content:""; position:absolute; bottom:0; left:0; height:3px; width:var(--rail); background:var(--route); }
    .route-top { display:flex; justify-content:space-between; gap:8px; align-items:baseline; }
    .route-top b { font-size:15px; } .route-top span { color:var(--route); font-size:11px; font-weight:800; }
    .route p { color:var(--muted); font-size:11px; margin:5px 0 9px; min-height:29px; }
    .route-metrics { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:7px; }
    .route-metrics label { font-size:9px; letter-spacing:.04em; } .route-metrics strong { display:block; font-size:13px; margin-top:1px; }
    .route-risk { color:var(--muted); font-size:10px; line-height:1.35; margin:8px 0 0; }
    .state { padding:15px; } .state h2 { margin:0 0 10px; font-size:13px; color:var(--muted); letter-spacing:.07em; text-transform:uppercase; }
    .metric { display:flex; align-items:baseline; justify-content:space-between; border-top:1px solid #233448; padding:9px 0; gap:12px; }
    .metric span { color:var(--muted); font-size:12px; } .metric b { font-size:13px; text-align:right; }
    .v2 { padding:15px; }
    .v2 h2 { margin:0 0 5px; font-size:13px; color:var(--muted); letter-spacing:.07em; text-transform:uppercase; }
    .v2-kicker { color:#bdc9d5; font-size:11px; margin:0 0 10px; }
    .v2-block { border-top:1px solid #233448; padding:9px 0; display:grid; gap:3px; }
    .v2-block span { color:var(--muted); font-size:11px; }
    .v2-block em { color:#70859a; font-style:normal; }
    .v2-block strong { color:#eef3f7; font-size:13px; }
    .v2-block i { color:#7e91a4; font-style:normal; }
    .v2-calibration { color:#9db0c2; font-size:11px; line-height:1.45; padding-top:9px; border-top:1px solid #233448; }
    .v2-caution { color:#f0c183; font-size:11px; line-height:1.42; margin-top:9px; }
    .v2-unavailable { color:#93a5b7; font-size:12px; line-height:1.5; padding-top:7px; }
    .notice { border-left:3px solid var(--amber); margin-top:16px; padding:11px 13px; background:rgba(236,168,70,.08); color:#c9d4de; font-size:12px; }
    .source { color:var(--muted); font-size:11px; margin:10px 0 0; }
    footer { color:#71869a; font-size:11px; margin-top:15px; text-align:center; }
    a { color:#aeb4ff; }
    @media (max-width:1050px) { .layout { grid-template-columns:1fr; } .sidebar { grid-template-columns:repeat(2,minmax(0,1fr)); } .risk { grid-row:span 2; } }
    @media (max-width:700px) { main { width:min(100% - 20px,1450px); padding-top:18px; } header { display:block; } .stamp { display:inline-block; margin-top:12px; } .control { grid-template-columns:1fr 1fr; } .control label:last-of-type { grid-column:1/-1; } .control button { grid-column:1/-1; } .sidebar { grid-template-columns:1fr; } #chart { height:420px; } }
    @media (prefers-reduced-motion:reduce) { * { transition:none!important; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div><div class="eyebrow">conditional wick-fill instrument · v1</div><h1>Pin a wick. Inspect real historical paths.</h1><p class="sub">Choose an unfilled strict wick candle. The engine matches its live state against completed historical episodes, then draws real rescaled candle paths—including a dedicated adverse-first risk route—to the wick target.</p></div>
      <div class="stamp" id="asOf">Loading data snapshot…</div>
    </header>
    <form class="control" id="controls">
      <label>Asset<select id="asset"><option>ETHUSDT</option><option>BTCUSDT</option><option>SOLUSDT</option><option>UNIUSDT</option><option>NEARUSDT</option></select></label>
      <label>Timeframe<select id="timeframe"><option>5m</option><option>15m</option><option value="1m">1m</option></select></label>
      <label>Display candles<select id="displayTimeframe"><option value="5m">5m native</option><option value="15m">15m visual</option><option value="30m">30m visual</option></select></label>
      <label>Matcher<select id="matchingMode"><option value="adaptive">Adaptive live path</option><option value="legacy_v3">Legacy V3 comparison</option><option value="blended">Soft archetype blend</option><option value="archetype">Hard candle archetype</option></select></label>
      <label>Recent unfilled wick<select id="signalPicker" aria-label="Recent unfilled strict wick signals"><option value="">Loading candidates…</option></select></label>
      <label>Signal time (UTC)<input id="signalTime" spellcheck="false" value="2026-09-16T18:35:00Z" aria-describedby="timeHelp"></label>
      <button id="project" type="submit">Project paths</button>
    </form>
    <p class="source" id="timeHelp">Pick a current unfilled strict wick or paste any ISO UTC strict-wick candle. Soft blend keeps the full path pool while adding direction-mirrored candle-shape influence; hard archetype restricts the pool. Display candles only changes chart compression.</p>
    <div class="status" id="status" aria-live="polite">Loading the default pinned signal…</div>
    <section class="layout">
      <article class="panel"><div class="chart-head"><div><h2 id="chartTitle">Conditional route</h2><p id="chartNote">Historical candles left; selected conditional path right.</p></div><span class="stamp" id="targetBadge">Wick target —</span></div><div id="chart"></div></article>
      <aside class="sidebar"><section class="panel risk"><h2>Scenario routes</h2><div class="route-grid" id="routes"></div></section><section class="panel state"><h2>Observed state</h2><div id="state"></div></section><section class="panel v2"><h2>Risk from current price</h2><div id="v2"></div></section></aside>
    </section>
    <div class="notice">These paths are conditional on eventual clean wick fill. They are scenario references for time and risk sizing—not an unconditional probability, price target, or trade instruction.</div>
    <footer>Chart engine: <a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">TradingView Lightweight Charts™</a> · separate local service; no modification to the existing generic Rust dashboard.</footer>
  </main>
  <script>
  (() => {
    const colors={fast:'#38b7a6',normal:'#9497f8',adverse:'#e8b44f',extreme:'#dc7954'};
    const labels={fast:'Fast',normal:'Normal',adverse:'Adverse first',extreme:'Extreme'};
    let data=null, chart=null, actual=null, projected=null, aggregated=null, liveCandleSeries=null, targetLine=null, targetLineSeries=null, selectedScenario='normal', lastDisplayBarCount=0, pendingFitFrame=0;
    let refreshStatus=null, knownSelectedSourceVersion=null, projectionInFlight=false, projectionRefreshQueued=false, liveSocket=null, liveSocketKey=null, liveSocketGeneration=0, liveReconnectTimer=0, liveCloseRefreshTimer=0, liveCandle=null, liveFeedState='idle';
    const q=id=>document.getElementById(id);
    const selectedSourceKey=()=>q('timeframe').value==='1m'?q('asset').value+'_1m':q('asset').value;
    const selectedSourceVersion=payload=>payload?.sources?.[selectedSourceKey()]?.source_mtime_ns??null;
    const liveFeedKey=()=>`${q('asset').value}:${q('timeframe').value}`;
    const money=value=>'$'+Number(value).toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});
    const count=value=>{const number=Number(value);return Number.isFinite(number)?number.toLocaleString():'—';};
    const pct=value=>`${Number(value).toFixed(2)}%`;
    const share=value=>`${(Number(value)*100).toFixed(0)}%`;
    const duration=bars=>{ const minutes=Number(bars)*Number(data.pinned_signal.timeframe.replace('m','')); if(minutes>=1440)return `${(minutes/1440).toFixed(1)}d`; if(minutes>=60)return `${Math.round(minutes/60)}h`; return `${Math.max(1,Math.round(minutes))}m`; };
    const durationMinutes=minutes=>{minutes=Number(minutes);if(!Number.isFinite(minutes))return 'â€”';if(minutes>=1440)return `${(minutes/1440).toFixed(1)}d`;if(minutes>=60)return `${Math.round(minutes/60)}h`;return `${Math.max(1,Math.round(minutes))}m`;};
    const berlin=seconds=>new Date(Number(seconds)*1000).toLocaleString('en-GB',{timeZone:'Europe/Berlin',dateStyle:'medium',timeStyle:'short'});
    function setStatus(message,error=false){ q('status').textContent=message; q('status').className='status'+(error?' error':''); }
    function refreshCountdown(milliseconds){const seconds=Math.max(0,Math.ceil(milliseconds/1000)),minutes=Math.floor(seconds/60);return `${String(minutes).padStart(2,'0')}:${String(seconds%60).padStart(2,'0')}`;}
    function renderRefreshStamp(){
      const node=q('asOf'), source=refreshStatus?.sources?.[selectedSourceKey()];
      const asOf=source?.last_closed_candle_open_time_utc||data?.current_state?.as_of_open_time_utc;
      const snapshot=asOf?`Snapshot closes ${asOf.replace('T',' ').replace('Z',' UTC')}`:'Snapshot loading';
      const feed=liveFeedState==='live'?' · Binance Futures stream live':liveFeedState==='connecting'||liveFeedState==='reconnecting'?' · Binance Futures stream reconnecting':'';
      if(!refreshStatus){node.textContent=`${snapshot}${feed}`;return;}
      let cycle='refresh scheduling…';
      if(!refreshStatus.enabled)cycle='live refresh paused';
      else if(refreshStatus.in_progress)cycle='refreshing completed candles…';
      else if(refreshStatus.next_refresh_utc)cycle=`next refresh in ${refreshCountdown(Date.parse(refreshStatus.next_refresh_utc)-Date.now())}`;
      if(refreshStatus.last_error)cycle+=' · last refresh needs retry';
      node.textContent=`${snapshot} · ${cycle}${feed}`;
    }
    function ensureChart(){
      if(chart) return;
      if(!window.LightweightCharts){ setStatus('Chart library did not load. Check internet access and refresh.',true); return; }
      const node=q('chart');
      chart=LightweightCharts.createChart(node,{layout:{background:{color:'#111b26'},textColor:'#aab8c5',fontFamily:getComputedStyle(document.body).fontFamily},grid:{vertLines:{color:'rgba(48,64,82,.48)'},horzLines:{color:'rgba(48,64,82,.48)'}},rightPriceScale:{borderColor:'#304052'},timeScale:{borderColor:'#304052',timeVisible:true,secondsVisible:false},crosshair:{vertLine:{color:'#60748a',labelBackgroundColor:'#263545'},horzLine:{color:'#60748a',labelBackgroundColor:'#263545'}}});
      actual=chart.addSeries(LightweightCharts.CandlestickSeries,{upColor:'#68c4b4',downColor:'#d27571',borderVisible:false,wickUpColor:'#68c4b4',wickDownColor:'#d27571'});
      projected=chart.addSeries(LightweightCharts.CandlestickSeries,{upColor:colors.normal,downColor:colors.normal,borderUpColor:colors.normal,borderDownColor:colors.normal,wickUpColor:colors.normal,wickDownColor:colors.normal});
      aggregated=chart.addSeries(LightweightCharts.CandlestickSeries,{upColor:'#eca846',downColor:'#eca846',borderUpColor:'#eca846',borderDownColor:'#eca846',wickUpColor:'#eca846',wickDownColor:'#eca846'});
      liveCandleSeries=chart.addSeries(LightweightCharts.CandlestickSeries,{upColor:'#f6c85f',downColor:'#f6c85f',borderUpColor:'#f6c85f',borderDownColor:'#f6c85f',wickUpColor:'#f6c85f',wickDownColor:'#f6c85f',lastValueVisible:false,priceLineVisible:false});
      new ResizeObserver(entries=>{for(const entry of entries){chart.applyOptions({width:entry.contentRect.width,height:entry.contentRect.height});scheduleContentFit();}}).observe(node);
    }
    const intervalMinutes=value=>Number(String(value).replace('m',''));
    function syncDisplayTimeframe(){
      const node=q('displayTimeframe'), sourceMinutes=intervalMinutes(q('timeframe').value), currentMinutes=intervalMinutes(node.value||q('timeframe').value);
      const options=[1,5,15,30].filter(value=>value>=sourceMinutes);
      node.replaceChildren(...options.map(value=>new Option(value===sourceMinutes?String(value)+'m native':String(value)+'m visual',String(value)+'m')));
      node.value=String(options.includes(currentMinutes)?currentMinutes:sourceMinutes)+'m';
    }
    function setTargetLine(series){
      if(targetLine&&targetLineSeries)targetLineSeries.removePriceLine(targetLine);
      targetLine=series.createPriceLine({price:data.pinned_signal.wick_target,color:'#eca846',lineWidth:2,lineStyle:LightweightCharts.LineStyle.Dashed,axisLabelVisible:true,title:'wick target'});
      targetLineSeries=series;
    }
    function fitChartToFullRange(barCount){
      if(!chart)return;
      lastDisplayBarCount=Math.max(1,Number(barCount)||1);
      const width=Math.max(1,Number(q('chart').clientWidth)||1);
      const barSpacing=Math.max(.12,Math.min(9,(width-48)/lastDisplayBarCount));
      chart.timeScale().applyOptions({minBarSpacing:.12,barSpacing});
      chart.timeScale().fitContent();
    }
    function captureChartViewport(){
      if(!chart)return null;
      const scale=chart.timeScale();
      try{
        const range=scale.getVisibleRange();
        if(range)return {range};
      }catch(error){}
      try{
        const logicalRange=scale.getVisibleLogicalRange();
        if(logicalRange)return {logicalRange};
      }catch(error){}
      return null;
    }
    function restoreChartViewport(viewport){
      if(!chart||!viewport)return false;
      const scale=chart.timeScale();
      try{
        if(viewport.range){scale.setVisibleRange(viewport.range);return true;}
      }catch(error){}
      try{
        if(viewport.logicalRange){scale.setVisibleLogicalRange(viewport.logicalRange);return true;}
      }catch(error){}
      return false;
    }
    function cancelPendingContentFit(){
      if(!pendingFitFrame)return;
      const cancelFrame=window.cancelAnimationFrame||window.clearTimeout;
      cancelFrame(pendingFitFrame);
      pendingFitFrame=0;
    }
    function scheduleContentFit(){
      if(!chart||!lastDisplayBarCount||pendingFitFrame)return;
      const requestFrame=window.requestAnimationFrame||((callback)=>window.setTimeout(callback,0));
      pendingFitFrame=requestFrame(()=>{pendingFitFrame=0;fitChartToFullRange(lastDisplayBarCount);});
    }
    function clearLiveCandle(){
      if(liveCandleSeries)liveCandleSeries.setData([]);
    }
    function renderLiveCandle(){
      if(!liveCandleSeries||!data||!liveCandle||liveCandle.key!==liveFeedKey()||data.pinned_signal.asset!==q('asset').value||data.pinned_signal.timeframe!==q('timeframe').value||intervalMinutes(q('displayTimeframe').value)!==intervalMinutes(data.pinned_signal.timeframe)){
        clearLiveCandle();
        return;
      }
      const asOfOpen=Date.parse(data.current_state?.as_of_open_time_utc||'')/1000;
      if(!Number.isFinite(asOfOpen)||liveCandle.candle.time<=asOfOpen){
        clearLiveCandle();
        return;
      }
      liveCandleSeries.update(liveCandle.candle);
    }
    function scheduleClosedCandleRefresh(){
      if(liveCloseRefreshTimer)return;
      liveCloseRefreshTimer=window.setTimeout(async()=>{
        liveCloseRefreshTimer=0;
        await pollRefresh();
      },8000);
    }
    function stopLiveFeed(){
      liveSocketGeneration+=1;
      if(liveReconnectTimer){window.clearTimeout(liveReconnectTimer);liveReconnectTimer=0;}
      if(liveCloseRefreshTimer){window.clearTimeout(liveCloseRefreshTimer);liveCloseRefreshTimer=0;}
      const socket=liveSocket;
      liveSocket=null;
      liveSocketKey=null;
      liveCandle=null;
      liveFeedState='idle';
      if(socket){socket.onclose=null;socket.onerror=null;try{socket.close();}catch(error){}}
      clearLiveCandle();
    }
    function startLiveFeed(){
      const key=liveFeedKey(), [asset,timeframe]=key.split(':');
      if(liveSocketKey===key&&liveSocket&&(liveSocket.readyState===WebSocket.OPEN||liveSocket.readyState===WebSocket.CONNECTING))return;
      stopLiveFeed();
      if(!window.WebSocket){liveFeedState='unavailable';renderRefreshStamp();return;}
      liveSocketKey=key;
      const generation=++liveSocketGeneration;
      const connect=()=>{
        if(generation!==liveSocketGeneration||liveSocketKey!==key)return;
        liveFeedState='connecting';
        renderRefreshStamp();
        const socket=new WebSocket(`wss://fstream.binance.com/market/ws/${asset.toLowerCase()}@kline_${timeframe}`);
        liveSocket=socket;
        socket.onmessage=event=>{
          if(generation!==liveSocketGeneration||liveSocket!==socket)return;
          let payload;
          try{payload=JSON.parse(event.data);}catch(error){return;}
          const kline=payload?.k;
          if(!kline||String(kline.s).toUpperCase()!==asset||String(kline.i)!==timeframe)return;
          const candle={time:Math.floor(Number(kline.t)/1000),open:Number(kline.o),high:Number(kline.h),low:Number(kline.l),close:Number(kline.c)};
          if(!Number.isFinite(candle.time)||!Number.isFinite(candle.open)||!Number.isFinite(candle.high)||!Number.isFinite(candle.low)||!Number.isFinite(candle.close))return;
          const becameLive=liveFeedState!=='live';
          liveFeedState='live';
          liveCandle={key,candle};
          renderLiveCandle();
          if(becameLive)renderRefreshStamp();
          if(kline.x)scheduleClosedCandleRefresh();
        };
        socket.onerror=()=>{
          if(generation!==liveSocketGeneration||liveSocket!==socket)return;
          liveFeedState='reconnecting';
          renderRefreshStamp();
        };
        socket.onclose=()=>{
          if(generation!==liveSocketGeneration||liveSocket!==socket)return;
          liveSocket=null;
          liveFeedState='reconnecting';
          renderRefreshStamp();
          liveReconnectTimer=window.setTimeout(connect,2000);
        };
      };
      connect();
    }
    function aggregateVisualCandles(actualCandles,projectedCandles,displayMinutes,projectionColor){
      const buckets=new Map(), bucketSeconds=displayMinutes*60;
      const add=(candles,phase)=>candles.forEach(candle=>{
        const time=Math.floor(Number(candle.time)/bucketSeconds)*bucketSeconds;
        let bucket=buckets.get(time);
        if(!bucket){bucket={time,open:Number(candle.open),high:Number(candle.high),low:Number(candle.low),close:Number(candle.close),actual:0,projected:0};buckets.set(time,bucket);}
        else{bucket.high=Math.max(bucket.high,Number(candle.high));bucket.low=Math.min(bucket.low,Number(candle.low));bucket.close=Number(candle.close);}
        bucket[phase]+=1;
      });
      add(actualCandles,'actual'); add(projectedCandles,'projected');
      let hasTransition=false;
      const candles=Array.from(buckets.values()).map(bucket=>{
        const phase=bucket.actual&&bucket.projected?'transition':bucket.projected?'projected':'actual';
        hasTransition=hasTransition||phase==='transition';
        const color=phase==='projected'?projectionColor:phase==='transition'?'#eca846':bucket.close>=bucket.open?'#68c4b4':'#d27571';
        return {time:bucket.time,open:bucket.open,high:bucket.high,low:bucket.low,close:bucket.close,color,borderColor:color,wickColor:color};
      });
      return {candles,hasTransition};
    }
    function renderCandleSeries(item,projectionColor){
      const sourceMinutes=intervalMinutes(data.pinned_signal.timeframe), displayMinutes=intervalMinutes(q('displayTimeframe').value);
      if(data.visualization?.server_aggregated){
        const visualCandles=(item.visual_candles||[]).map(candle=>{
          const color=candle.phase==='projected'?projectionColor:candle.phase==='transition'?'#eca846':Number(candle.close)>=Number(candle.open)?'#68c4b4':'#d27571';
          return {...candle,color,borderColor:color,wickColor:color};
        });
        actual.setData([]); projected.setData([]); aggregated.setData(visualCandles); setTargetLine(aggregated);
        return {isAggregated:true,displayMinutes:Number(data.visualization.display_minutes)||displayMinutes,hasTransition:Boolean(item.visual_has_transition),barCount:visualCandles.length};
      }
      if(displayMinutes<=sourceMinutes){
        aggregated.setData([]); actual.setData(data.actual_candles); projected.setData(item.projected_candles); setTargetLine(actual);
        const firstTime=Number(data.actual_candles[0].time), lastTime=Number(item.projected_candles.at(-1).time);
        return {isAggregated:false,displayMinutes:sourceMinutes,hasTransition:false,barCount:Math.max(1,Math.round((lastTime-firstTime)/(sourceMinutes*60))+1)};
      }
      const result=aggregateVisualCandles(data.actual_candles,item.projected_candles,displayMinutes,projectionColor);
      actual.setData([]); projected.setData([]); aggregated.setData(result.candles); setTargetLine(aggregated);
      return {isAggregated:true,displayMinutes,hasTransition:result.hasTransition,barCount:result.candles.length};
    }
    function selectScenario(name,{viewport=null}={}){
      selectedScenario=name;
      if(!data) return; const item=data.scenarios.find(x=>x.name===name); if(!item) return;
      document.querySelectorAll('.route').forEach(el=>el.setAttribute('aria-pressed',String(el.dataset.name===name)));
      const color=colors[name]; projected.applyOptions({upColor:color,downColor:color,borderUpColor:color,borderDownColor:color,wickUpColor:color,wickDownColor:color}); const display=renderCandleSeries(item,color);
      q('chartTitle').textContent=`${labels[name]} route — conditional historical trajectory`;
      const terminalSeconds=item.projected_terminal_fill_candle_utc?Date.parse(item.projected_terminal_fill_candle_utc)/1000:item.projected_candles.at(-1).time+Number(data.pinned_signal.timeframe.replace('m',''))*60;
      const adverseNote=name==='adverse'&&item.projected_adverse_threshold_candle_utc?` · p80 adverse reached ${berlin(Date.parse(item.projected_adverse_threshold_candle_utc)/1000)} before fill`:'';
      q('chartNote').textContent=`${duration(item.remaining_to_fill_bars)} remaining · ${item.historical_asset} ${item.historical_timeframe} analogue${adverseNote} · terminal wick touch ${berlin(terminalSeconds)}`;
      if(display.isAggregated){
        const boundaryNote=display.hasTransition?' · amber boundary candle blends observed and projected '+data.pinned_signal.timeframe+' candles.':'';
        q('chartTitle').textContent=labels[name]+' route — '+display.displayMinutes+'m visual aggregation';
        q('chartNote').textContent=duration(item.remaining_to_fill_bars)+' remaining · display-only aggregation; engine stays '+data.pinned_signal.timeframe+'.'+boundaryNote;
      }
      lastDisplayBarCount=Math.max(1,Number(display.barCount)||1);
      if(!restoreChartViewport(viewport))fitChartToFullRange(display.barCount);
    }
    function renderV2(risk){
      const node=q('v2');
      queueMicrotask(()=>appendV2Support(node,risk));
      if(!risk || !risk.available){node.innerHTML=`<div class="v2-unavailable">${risk?.reason||'V2a diagnostic is unavailable for this pin.'}</div>`;return;}
      const prediction=risk.prediction?.quantiles, remaining=prediction?.remaining_bars, away=prediction?.future_max_away_pct;
      if(!remaining||!away){node.innerHTML='<div class="v2-unavailable">V2a returned an incomplete diagnostic response.</div>';return;}
      const calibration=risk.calibration||{}, coverage=risk.state_coverage||{}, ratio=value=>value==null?'—':`${(Number(value)*100).toFixed(1)}%`;
      const coverageWarning=coverage.elapsed_within_training_snapshot_range===false?`<div class="v2-caution">This pin is ${Number(coverage.elapsed_bars).toLocaleString()} bars after the signal, beyond V2a's largest trained snapshot offset of ${Number(coverage.max_training_snapshot_elapsed_bars).toLocaleString()} bars. Treat its ranges as an extrapolation, not a calibrated live range.</div>`:'';
      node.innerHTML=`<p class="v2-kicker">${risk.estimator}. Quantiles are conditional on historical clean fills.</p><div class="v2-block"><span>Remaining time to wick touch <em>p10 / p50 / p90</em></span><strong>${duration(remaining.p10)} <i>/</i> ${duration(remaining.p50)} <i>/</i> ${duration(remaining.p90)}</strong></div><div class="v2-block"><span>Future max move-away <em>p10 / p50 / p90</em></span><strong>${pct(away.p10)} <i>/</i> ${pct(away.p50)} <i>/</i> ${pct(away.p90)}</strong></div><div class="v2-calibration">Chronological holdout: ${Number(calibration.holdout_snapshot_count||0).toLocaleString()} snapshots. Time p10–p90 captured ${ratio(calibration.remaining_p10_p90_observed_coverage)}; move-away p10–p90 captured ${ratio(calibration.future_away_p10_p90_observed_coverage)} (80% nominal).</div><div class="v2-caution">Do not treat future move-away p10 as a sizing floor: it realized only ${ratio(calibration.future_away_p10_observed_coverage)} of later cases against a 10% target.</div>`;
    }
    function renderProspective(risk,fallback){
      const node=q('v2');
      if(!risk || !risk.available){renderV2(fallback);return;}
      const labels={'1440m':'1 day','10080m':'7 days','43200m':'30 days'};
      const blocks=Object.entries(labels).map(([key,label])=>{
        const fill=risk.fill_probability?.[key], adverse=risk.additional_adverse_pct?.[key], wait=risk.remaining_time_minutes_if_filled_within_horizon?.[key];
        if(fill==null||!adverse||!wait)return '';
        return `<div class="v2-block"><span>${label}: fill odds / additional adverse <em>p50 / p80 / p90</em></span><strong>${share(fill)} <i>Â·</i> ${pct(adverse.p50)} <i>/</i> ${pct(adverse.p80)} <i>/</i> ${pct(adverse.p90)}</strong><i>Wait if filled inside ${label}: ${durationMinutes(wait.p10)} / ${durationMinutes(wait.p50)} / ${durationMinutes(wait.p90)} (p10 / p50 / p90)</i></div>`;
      }).join('');
      const historicalSupport=risk.historical_support||{};
      const supportBlocks=Object.entries(labels).map(([key,label])=>{
        const item=historicalSupport[key];
        if(!item)return '';
        const level=String(item.level||'unknown').replaceAll('_',' ').toUpperCase();
        const effective=Number(item.effective_neighbors||0).toFixed(1);
        const agreement=share(item.fill_agreement||0);
        const familiarity=Math.max(0,100-Math.round(Number(item.median_distance_percentile||0)*100));
        return `<div class="v2-block"><span>${label} historical support</span><strong>${level}</strong><i>${effective} effective analogues; ${agreement} outcome agreement; familiarity percentile ${familiarity}%.</i></div>`;
      }).join('');
      const contest2=risk.competing_outcomes?.['1440m_vs_2pct']||{}, contest5=risk.competing_outcomes?.['43200m_vs_5pct']||{};
      const contestLine=(contest,label)=>contest.target_first==null?'':`<div class="v2-block"><span>${label}</span><strong>${share(contest.target_first)} target first <i>/</i> ${share(contest.adverse_first||0)} adverse first</strong><i>${share(contest.ambiguous_intrabar||0)} same-candle ordering unknown; ${share(contest.neither||0)} neither inside the horizon.</i></div>`;
      const contestBlock=contestLine(contest2,'Within 1d: wick target versus another 2% adverse move')+contestLine(contest5,'Within 30d: wick target versus another 5% adverse move');
      const support=risk.support||{}, supportWarning=support.status==='outside_trained_age_range'?`<div class="v2-caution">This pin is ${durationMinutes(risk.entry_age_minutes)} old, outside the trained age range ${durationMinutes(support.entry_age_minutes_min)}â€“${durationMinutes(support.entry_age_minutes_max)}. Treat this as extrapolation.</div>`:'';
      const model=risk.holdout?.model||{}, baseline=risk.holdout?.baseline||{}, improvement=field=>model[field]!=null&&baseline[field]?`${((baseline[field]-model[field])/baseline[field]*100).toFixed(1)}%`:null;
      const evidence=[improvement('risk_interval_mae_pct')&&`risk ${improvement('risk_interval_mae_pct')}`,improvement('time_mae_minutes')&&`wait ${improvement('time_mae_minutes')}`,improvement('fill_brier')&&`fill odds ${improvement('fill_brier')}`].filter(Boolean).join(' Â· ');
      const artifact=risk.artifact||{};
      const evidenceBlock=artifact.architecture_id?`<div class="v2-calibration">${artifact.architecture_id} &middot; ${artifact.frozen?'frozen':'mutable'} &middot; prospective validation ${artifact.prospective_validation_status||'pending'}.</div>`:`<div class="v2-calibration">Chronological holdout improvement versus ${artifact.baseline_version||'frozen matcher'}: ${evidence||'validated artifact'}.</div>`;
      const provenanceWarning=artifact.forecast_source==='legacy_fallback'?'<div class="v2-caution">Frozen V1 failed its integrity/load check. This card is using the legacy fallback and is excluded from V1 prospective scoring.</div>':'';
      node.innerHTML=`<p class="v2-kicker">All resolved and unresolved historical entries; measured from the current close of ${money(risk.entry_price)}.</p>${blocks}${supportBlocks}${contestBlock}${evidenceBlock}${provenanceWarning}${supportWarning}<div class="v2-caution">${risk.warning}</div>`;
    }
    function appendV2Support(node,risk){
      if(!risk || !risk.available) return;
      const add=text=>{const element=document.createElement('div');element.className='v2-caution';element.textContent=text;node.append(element);};
      const support=risk.age_support||{}, coverage=risk.state_coverage||{}, elapsed=Number(coverage.elapsed_bars);
      if(support.status==='exact_sampled_age') add('Snapshot-age support: exact sampled age at '+elapsed.toLocaleString()+' bars.');
      if(support.status==='between_sampled_ages') add('Snapshot-age support: '+elapsed.toLocaleString()+' bars falls between sampled ages '+Number(support.nearest_lower_offset_bars).toLocaleString()+' and '+Number(support.nearest_upper_offset_bars).toLocaleString()+'; treat this as interpolation, not exact-age calibration.');
      if(support.status==='outside_sampled_age') add('Snapshot-age support: '+elapsed.toLocaleString()+' bars is outside the sampled-age range; treat V2 ranges as extrapolation, not calibrated live ranges.');
      if(support.status==='unknown') add('Snapshot-age support is unavailable because this V2 artifact has no usable sampled offsets.');
      if(coverage.elapsed_within_training_snapshot_range===false) add('This pin is beyond V2a\'s largest trained snapshot offset of '+Number(coverage.max_training_snapshot_elapsed_bars).toLocaleString()+' bars.');
      const artifact=risk.artifact||{}, details=[];
      if(artifact.generated_at_utc) details.push('generated '+artifact.generated_at_utc.replace('T',' ').replace('Z',' UTC'));
      if(artifact.training_label_cutoff_utc) details.push('training labels resolved through '+artifact.training_label_cutoff_utc.replace('T',' ').replace('Z',' UTC'));
      if(details.length) add('Static V2 artifact: '+details.join('; ')+'. Source refresh does not retrain it.');
    }
    function appendRouteRiskDetails(){
      if(!data) return;
      document.querySelectorAll('.route').forEach(button=>{
        const item=data.scenarios.find(candidate=>candidate.name===button.dataset.name);
        if(!item || item.projected_future_max_away_move_pct==null) return;
        const detail=document.createElement('p');
        detail.className='route-risk';
        detail.textContent='Projected peak away: '+pct(item.projected_future_max_away_move_pct)+'; additional adverse from current close: '+pct(item.projected_additional_adverse_move_pct||0)+'.';
        button.append(detail);
      });
    }
    function render(next,{preserveViewport=false}={}){
      ensureChart(); if(!chart) return;
      const viewport=preserveViewport?captureChartViewport():null;
      if(viewport)cancelPendingContentFit();
      data=next; syncDisplayTimeframe();
      renderRefreshStamp();
      q('targetBadge').textContent=`Wick target ${money(data.pinned_signal.wick_target)}`;
      actual.setData(data.actual_candles); setTargetLine(actual);
      const routes=q('routes'); routes.replaceChildren();
      const preferredScenario=data.scenarios.some(item=>item.name===selectedScenario)?selectedScenario:'normal';
      data.scenarios.forEach((item,index)=>{const button=document.createElement('button');button.type='button';button.className='route';button.dataset.name=item.name;button.style.setProperty('--route',colors[item.name]||'#ffffff');button.style.setProperty('--rail',`${Math.max(8,Math.min(100,item.joint_risk_score_percentile*100))}%`);button.setAttribute('aria-pressed',String(index===1));const cohortCount=Math.round((item.matched_cohort_tail_at_or_above_fraction||0)*(item.matched_cohort_size||0));const thirdMetric=item.name==='adverse'?`<div><label>p80 adverse reached</label><strong>${item.projected_adverse_threshold_bars?duration(item.projected_adverse_threshold_bars):'not reached'}</strong></div>`:`<div><label>cohort slower/riskier</label><strong>${cohortCount} / ${count(item.matched_cohort_size)} (${share(item.matched_cohort_tail_at_or_above_fraction)})</strong></div>`;const routeBadge=item.name==='adverse'?'p80 adverse-first example':`risk score p${Math.round(item.joint_risk_score_percentile*100)}`;button.innerHTML=`<div class="route-top"><b>${labels[item.name]||item.name}</b><span>${routeBadge}</span></div><p>${item.description}</p><div class="route-metrics"><div><label>to wick touch</label><strong>${duration(item.remaining_to_fill_bars)}</strong></div><div><label>additional adverse</label><strong>${pct(item.projected_additional_adverse_move_pct||0)}</strong></div>${thirdMetric}</div>`;button.addEventListener('click',()=>selectScenario(item.name));routes.append(button);});
      const v3=data.v3_route||{}, routeEngine=data.route_engine||{}, matching=data.matching||{}, normalSelector=matching.mode==='archetype'?`Hard archetype · ${count(matching.signal_configuration_pool_size)} candles`:matching.mode==='blended'?`Soft shape blend · ${Math.round((matching.archetype_blend_weight||0)*100)}%`:routeEngine.active?`C2 + E0/A · ${count(routeEngine.normal_candidate_count)} routes`:v3.active?`Legacy V3 ${v3.regime_label||'neural'} · ${count(v3.fit_rows)} fit states`:'V1 historical';
      const stateRows=[['Direction',data.pinned_signal.direction.replace('_',' ')],['Current move away',pct(data.current_state.current_move_pct)],['Peak move away',pct(data.current_state.peak_move_pct)],['Drawdown from peak',pct(data.current_state.drawdown_from_peak_pct)],['Elapsed',`${Math.round(data.current_state.elapsed_minutes/60)}h`],['Normal selector',normalSelector],['Wick direction rule',matching.direction_policy||'category-aware'],[`Matched ${data.pinned_signal.timeframe} paths`,count(data.library?.top_k_state_matched_episodes)],['Cohort P50 remaining',`${Math.round(data.cohort_distribution.remaining_time_to_fill_minutes.p50/60)}h`]];
      q('state').innerHTML=stateRows.map(([k,v])=>`<div class="metric"><span>${k}</span><b>${v}</b></div>`).join('');
      renderProspective(data.prospective_risk,data.v2_risk);
      selectScenario(preferredScenario,{viewport});
      renderLiveCandle();
      queueMicrotask(appendRouteRiskDetails);
    }
    async function request({sourceRefresh=false}={}){
      if(projectionInFlight){projectionRefreshQueued=projectionRefreshQueued||sourceRefresh;return;}
      projectionInFlight=true;
      const asset=q('asset').value, timeframe=q('timeframe').value, display_timeframe=q('displayTimeframe').value, matching_mode=q('matchingMode').value, signal_time=q('signalTime').value.trim();
      q('project').disabled=true; setStatus('Searching resolved historical paths and rebuilding real candle scenarios…');
      try { const response=await fetch(`/api/scenarios?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}&display_timeframe=${encodeURIComponent(display_timeframe)}&matching_mode=${encodeURIComponent(matching_mode)}&signal_time=${encodeURIComponent(signal_time)}`,{cache:'no-store'}); const payload=await response.json(); if(!response.ok)throw new Error(payload.error||'Request failed'); render(payload,{preserveViewport:sourceRefresh}); const asOf=payload.current_state?.as_of_close_time_utc||'the latest completed candle'; const verb=sourceRefresh?'Refreshed path':'Loaded'; setStatus(`${verb} ${payload.pinned_signal.asset} ${payload.pinned_signal.timeframe} ${payload.pinned_signal.direction.replace('_',' ')} pinned at ${payload.pinned_signal.signal_open_time_utc}; projection as of ${asOf}.`); }
      catch(error){setStatus(error.message||String(error),true);}
      finally{q('project').disabled=false;projectionInFlight=false;if(projectionRefreshQueued){projectionRefreshQueued=false;request({sourceRefresh:true});}}
    }
    async function loadSignals({preferFirst=false}={}){
      const asset=q('asset').value,timeframe=q('timeframe').value,picker=q('signalPicker'); picker.disabled=true; picker.replaceChildren(new Option('Loading candidates…',''));
      try { const response=await fetch(`/api/signals?asset=${encodeURIComponent(asset)}&timeframe=${encodeURIComponent(timeframe)}`,{cache:'no-store'}); const payload=await response.json(); if(!response.ok)throw new Error(payload.error||'Could not load candidate signals');
        picker.replaceChildren(); if(!payload.signals.length){picker.append(new Option('No unfilled strict signals in recent scan','')); return;}
        payload.signals.forEach((signal,index)=>{const label=`${signal.signal_time_utc} · ${signal.direction.replace('_',' ')} · ${pct(signal.current_move_pct)} away`;picker.append(new Option(label,signal.signal_time_utc,index===0,index===0));});
        const input=q('signalTime'); const matching=payload.signals.find(signal=>signal.signal_time_utc===input.value.trim());
        if(matching){picker.value=matching.signal_time_utc;} else if(preferFirst){input.value=payload.signals[0].signal_time_utc;}
        return payload;
      } catch(error){picker.replaceChildren(new Option('Candidate list unavailable — paste time manually','')); setStatus(error.message||String(error),true);return null;}
      finally{picker.disabled=false;}
    }
    async function pollRefresh(){
      try{
        const response=await fetch('/api/refresh-status',{cache:'no-store'}); const payload=await response.json(); if(!response.ok)throw new Error(payload.error||'Live refresh status is unavailable');
        const sourceVersion=selectedSourceVersion(payload);
        const changed=knownSelectedSourceVersion!==null&&sourceVersion!==null&&sourceVersion!==knownSelectedSourceVersion;
        refreshStatus=payload; knownSelectedSourceVersion=sourceVersion; renderRefreshStamp();
        if(!changed)return;
        await loadSignals();
        request({sourceRefresh:true});
      }catch(error){q('asOf').textContent='Live refresh status unavailable';}
    }
    q('controls').addEventListener('submit',event=>{event.preventDefault();request();});
    q('signalPicker').addEventListener('change',()=>{if(q('signalPicker').value){q('signalTime').value=q('signalPicker').value;request();}});
    q('asset').addEventListener('change',async()=>{syncDisplayTimeframe();knownSelectedSourceVersion=selectedSourceVersion(refreshStatus);startLiveFeed();renderRefreshStamp();await loadSignals({preferFirst:true});request();});
    q('timeframe').addEventListener('change',async()=>{q('displayTimeframe').value=q('timeframe').value;syncDisplayTimeframe();knownSelectedSourceVersion=selectedSourceVersion(refreshStatus);startLiveFeed();await loadSignals({preferFirst:true});request();});
    q('displayTimeframe').addEventListener('change',()=>{if(data)request();});
    window.addEventListener('beforeunload',stopLiveFeed);
    (async()=>{syncDisplayTimeframe();await pollRefresh();await loadSignals();await request();startLiveFeed();window.setInterval(pollRefresh,15000);window.setInterval(renderRefreshStamp,1000);})();
  })();
  </script>
</body></html>"""


class Engine:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock = threading.Lock()
        self.library_load_lock = threading.Lock()
        self.scenario_cache: dict[
            tuple[str, str, str, str, tuple[int, ...]], dict[str, Any]
        ] = {}
        self.frame_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
        self.signal_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
        self.signal_cache_metadata: dict[tuple[str, str, int], tuple[int, int]] = {}
        self.library_dir = default_library_dir(root)
        self.library_state_cache: dict[
            tuple[str, str, tuple[int, ...]],
            tuple[
                pd.DataFrame, pd.DataFrame, NativeStateIndex, dict[str, tuple[int, int]]
            ],
        ] = {}
        self.v2_model_path = (
            self.library_dir / "v2_models" / "conditional_wick_v2_model.pkl"
        )
        self.v2_summary_path = (
            self.library_dir / "v2_models" / "conditional_wick_v2_summary.json"
        )
        self.v2_model: dict[str, Any] | None = None
        self.v2_model_mtime_ns: int | None = None
        self.v2_load_error: str | None = None
        self.v2_summary: dict[str, Any] | None = None
        self.v2_summary_mtime_ns: int | None = None
        self.v3_bundle_cache: dict[str, Any] = {}
        self.v3_artifact_stamps: dict[str, tuple[int, int, int]] = {}
        self.v3_load_errors: dict[str, str] = {}
        self.prospective_artifact_root = prospective_artifact_root(root)
        self.prospective_ledger_path = default_ledger_path(root)
        self.prospective_model: dict[str, Any] | None = None
        self.prospective_model_timeframe: str | None = None
        self.prospective_model_cache_key: tuple[Any, ...] | None = None
        self.prospective_load_error: str | None = None

    @staticmethod
    def _source_timeframe(timeframe: str) -> str:
        return "1m" if timeframe == "1m" else "5m"

    def _source_path(self, asset: str, source_timeframe: str = "5m") -> Path:
        if source_timeframe not in {"1m", "5m"}:
            raise ValueError(f"Unsupported raw source timeframe: {source_timeframe}")
        if asset not in assets_for_timeframe(source_timeframe):
            raise ValueError(f"{source_timeframe} raw source does not support {asset}")
        return self.root / "data" / f"{asset}_{source_timeframe}_5y.csv"

    def _library_dir_for_timeframe(self, asset: str, timeframe: str) -> Path:
        return (
            one_minute_library_dir(self.root, asset)
            if timeframe == "1m"
            else self.library_dir
        )

    def _library_stamp(self, asset: str, timeframe: str) -> tuple[int, ...]:
        library_dir = self._library_dir_for_timeframe(asset, timeframe)
        episodes = library_dir / "episodes.csv"
        library_assets = (
            (asset,) if timeframe == "1m" else assets_for_timeframe(timeframe)
        )
        path_stamps = tuple(
            (library_dir / "paths" / f"{asset}_{timeframe}_paths.csv.gz")
            .stat()
            .st_mtime_ns
            for asset in library_assets
        )
        cache_metadata = runtime_cache_dir(library_dir, timeframe) / "metadata.json"
        cache_stamp = (
            cache_metadata.stat().st_mtime_ns if cache_metadata.exists() else 0
        )
        return (episodes.stat().st_mtime_ns, *path_stamps, cache_stamp)

    def _stamp(self, asset: str, timeframe: str) -> tuple[int, ...]:
        source = self._source_path(asset, self._source_timeframe(timeframe))
        model_stamp = (
            self.v2_model_path.stat().st_mtime_ns
            if timeframe == "5m" and self.v2_model_path.exists()
            else 0
        )
        summary_stamp = (
            self.v2_summary_path.stat().st_mtime_ns
            if timeframe == "5m" and self.v2_summary_path.exists()
            else 0
        )
        v3_stamps = tuple(
            path.stat().st_mtime_ns if path.exists() else 0
            for directory in V3_LIVE_ARTIFACT_DIRS.get(timeframe, ())
            for path in (
                self.root / "data" / directory / "neural_path_v3_model.pt",
                self.root / "data" / directory / "neural_path_v3_retrieval_index.npz",
                self.root / "data" / directory / "neural_path_v3_summary.json",
            )
        )
        prospective_model_dir = self.prospective_artifact_root / timeframe
        prospective_paths = [
            prospective_model_dir / "model.joblib",
            active_manifest_path(prospective_model_dir),
        ]
        if timeframe == "5m" and prospective_paths[1].exists():
            try:
                manifest = load_active_manifest(prospective_model_dir)
                active_artifact = (
                    prospective_model_dir / str(manifest["artifact_file"])
                ).resolve()
                if active_artifact.parent == prospective_model_dir.resolve():
                    prospective_paths.append(active_artifact)
            except (OSError, ValueError, KeyError, json.JSONDecodeError, RuntimeError):
                pass
        prospective_stamp = tuple(
            path.stat().st_mtime_ns if path.exists() else 0
            for path in prospective_paths
        )
        return (
            source.stat().st_mtime_ns,
            *self._library_stamp(asset, timeframe),
            model_stamp,
            summary_stamp,
            *prospective_stamp,
            *v3_stamps,
        )

    def _library_states(
        self, asset: str, timeframe: str
    ) -> tuple[
        pd.DataFrame, pd.DataFrame, NativeStateIndex, dict[str, tuple[int, int]]
    ]:
        """Load one immutable trajectory-state cache, serializing the expensive cold path."""
        library_asset = asset if timeframe == "1m" else "all"
        stamp = self._library_stamp(asset, timeframe)
        key = (library_asset, timeframe, stamp)
        with self.lock:
            cached = self.library_state_cache.get(key)
        if cached is not None:
            return cached
        with self.library_load_lock:
            stamp = self._library_stamp(asset, timeframe)
            key = (library_asset, timeframe, stamp)
            with self.lock:
                cached = self.library_state_cache.get(key)
            if cached is not None:
                return cached
            library_dir = self._library_dir_for_timeframe(asset, timeframe)
            episodes_path = library_dir / "episodes.csv"
            episodes = with_fill_close_time_ms(pd.read_csv(episodes_path))
            episodes["signal_open_time_ms"] = pd.to_numeric(
                episodes["signal_open_time_ms"], errors="raise"
            ).astype("int64")
            trajectory_events = episodes.loc[episodes["timeframe"].eq(timeframe)].copy()
            if trajectory_events.empty:
                raise RuntimeError(
                    f"No completed {timeframe} trajectory episodes are available"
                )
            runtime_cache = load_runtime_cache(library_dir, timeframe)
            if runtime_cache is not None:
                states, native_state_index = runtime_cache
                cached_ids = set(native_state_index.episode_ids.astype(str).tolist())
                expected_ids = set(trajectory_events["episode_id"].astype(str).tolist())
                if cached_ids != expected_ids:
                    runtime_cache = None
            if runtime_cache is None:
                paths = read_library_paths(
                    library_dir / "paths", trajectory_events["path_file"].tolist()
                )
                states = prepare_path_states(trajectory_events, paths)
                # The state frame still has every normalized OHLC field needed to
                # render a selected suffix, so keeping it avoids a second raw-path cache.
                states = states.loc[:, RUNTIME_STATE_COLUMNS].reset_index(drop=True)
                native_state_index = NativeStateIndex.from_frames(
                    trajectory_events, states, timeframe
                )
            value = (episodes, states, native_state_index, episode_row_spans(states))
            with self.lock:
                # The native one-minute library has substantially more path
                # rows than the 5m/15m libraries. Keep it exclusive so a
                # switch to a 1m asset cannot coexist with every warmed five-asset
                # cache and exhaust the local dashboard process.
                if timeframe == "1m":
                    self.library_state_cache = {}
                else:
                    self.library_state_cache = {
                        cache_key: cached_value
                        for cache_key, cached_value in self.library_state_cache.items()
                        if cache_key[1] not in {timeframe, "1m"}
                    }
                self.library_state_cache[key] = value
            return value

    def _frame(self, asset: str, timeframe: str) -> pd.DataFrame:
        source = self._source_path(asset, self._source_timeframe(timeframe))
        # The updater swaps CSVs atomically.  Recheck its generation after the
        # read so an API request never stores a new frame under an old key.
        for _ in range(2):
            source_mtime_ns = source.stat().st_mtime_ns
            key = (asset, timeframe, source_mtime_ns)
            with self.lock:
                cached = self.frame_cache.get(key)
            if cached is not None:
                return cached
            raw = (
                read_one_minute_file if timeframe == "1m" else read_five_minute_file
            )(source)
            if source.stat().st_mtime_ns != source_mtime_ns:
                continue
            frame = (
                raw if timeframe in {"1m", "5m"} else resample_to_fifteen_minutes(raw)
            )
            with self.lock:
                self.frame_cache[key] = frame
            return frame
        raise RuntimeError(
            f"{asset} source changed while the dashboard was loading it; retry once refresh completes"
        )

    def apply_source_append(
        self, asset: str, interval: str, result: RefreshResult
    ) -> bool:
        """Advance an already-loaded raw frame from a verified small CSV append."""
        if not result.changed or result.rows_added <= 0:
            return False
        direct_timeframe = "1m" if interval == "1m" else "5m"
        source = self._source_path(asset, interval)
        source_mtime_ns = int(result.source_mtime_ns)
        with self.lock:
            cached = [
                (key, frame)
                for key, frame in self.frame_cache.items()
                if key[:2] == (asset, direct_timeframe)
            ]
        if not cached:
            if interval == "5m":
                with self.lock:
                    self.frame_cache = {
                        key: frame
                        for key, frame in self.frame_cache.items()
                        if key[:2] != (asset, "15m")
                    }
            return False
        previous_key, previous_frame = max(cached, key=lambda item: item[0][2])
        try:
            if source.stat().st_mtime_ns != source_mtime_ns:
                raise RuntimeError(
                    "source generation changed before its incremental frame update"
                )
            appended = appended_candle_rows(
                source, int(result.rows_added), int(interval.removesuffix("m"))
            )
            expected_open_time = (
                int(previous_frame["open_time"].iat[-1])
                + int(interval.removesuffix("m")) * 60_000
            )
            if int(appended["open_time"].iat[0]) != expected_open_time:
                raise RuntimeError("source append does not continue the cached frame")
            updated = pd.concat([previous_frame, appended], ignore_index=True)
            if source.stat().st_mtime_ns != source_mtime_ns:
                raise RuntimeError(
                    "source generation changed during its incremental frame update"
                )
        except Exception:
            # A large catch-up, external rewrite, or malformed tail is safer to
            # reload fully on demand than to merge into an uncertain generation.
            updated = None
        with self.lock:
            self.frame_cache = {
                key: frame
                for key, frame in self.frame_cache.items()
                if key[:2] != (asset, direct_timeframe)
                and not (interval == "5m" and key[:2] == (asset, "15m"))
            }
            if updated is not None:
                self.frame_cache[(asset, direct_timeframe, source_mtime_ns)] = updated
        return updated is not None

    def _signals(
        self, asset: str, timeframe: str, frame: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Detect strict signals once for each immutable source generation."""
        source = self._source_path(asset, self._source_timeframe(timeframe))
        source_mtime_ns = source.stat().st_mtime_ns
        key = (asset, timeframe, source_mtime_ns)
        with self.lock:
            cached = self.signal_cache.get(key)
            prior_entries = [
                (prior_key, signals, self.signal_cache_metadata.get(prior_key))
                for prior_key, signals in self.signal_cache.items()
                if prior_key[:2] == (asset, timeframe) and prior_key != key
            ]
        if cached is not None:
            return cached
        current_frame = frame if frame is not None else self._frame(asset, timeframe)
        prior_signals: pd.DataFrame | None = None
        prior_row_count: int | None = None
        for prior_key, signals, metadata in sorted(
            prior_entries, key=lambda item: item[0][2], reverse=True
        ):
            if metadata is None:
                continue
            prior_rows, prior_last_open_time = metadata
            if (
                prior_rows <= len(current_frame)
                and prior_rows > 0
                and int(current_frame["open_time"].iat[prior_rows - 1])
                == prior_last_open_time
            ):
                prior_signals = signals
                prior_row_count = prior_rows
                break
        signals = (
            incremental_strict_signals(
                current_frame, prior_signals, asset, timeframe, prior_row_count
            )
            if prior_signals is not None and prior_row_count is not None
            else detect_strict_signals(current_frame, asset, timeframe)
        )
        if source.stat().st_mtime_ns != source_mtime_ns:
            return self._signals(asset, timeframe)
        with self.lock:
            self.signal_cache = {
                existing_key: existing_signals
                for existing_key, existing_signals in self.signal_cache.items()
                if existing_key[:2] != (asset, timeframe)
            }
            self.signal_cache_metadata = {
                existing_key: metadata
                for existing_key, metadata in self.signal_cache_metadata.items()
                if existing_key[:2] != (asset, timeframe)
            }
            self.signal_cache[key] = signals
            self.signal_cache_metadata[key] = (
                len(current_frame),
                int(current_frame["open_time"].iat[-1]),
            )
        return signals

    def invalidate_source_caches(self) -> None:
        """Drop path outputs after a successful atomic refresh; frames advance incrementally."""
        with self.lock:
            self.scenario_cache.clear()

    def _v3_bundle(
        self, timeframe: str, elapsed_bars: int
    ) -> tuple[Any | None, str | None]:
        """Load the validated age expert for this pin without making V1 depend on it."""
        candidates: list[
            tuple[int, str, tuple[Path, Path, Path], dict[str, Any]]
        ] = []
        for directory in V3_LIVE_ARTIFACT_DIRS.get(timeframe, ()):
            root = self.root / "data" / directory
            paths = (
                root / "neural_path_v3_model.pt",
                root / "neural_path_v3_retrieval_index.npz",
                root / "neural_path_v3_summary.json",
            )
            if not all(path.exists() for path in paths):
                continue
            try:
                summary = json.loads(paths[2].read_text(encoding="utf-8"))
                gate = summary.get("deployment_gate", {})
                minimum = int(gate["min_elapsed_bars"])
                maximum = int(gate["max_elapsed_bars"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                continue
            if str(gate.get("timeframe")) != timeframe:
                continue
            if minimum <= elapsed_bars <= maximum:
                candidates.append((maximum - minimum, directory, paths, summary))
        if not candidates:
            return (
                None,
                f"No validated V3 {timeframe} age expert covers {elapsed_bars:,} bars; V1 remains active.",
            )
        _, cache_key, paths, _ = min(candidates, key=lambda item: (item[0], item[1]))
        stamp = tuple(path.stat().st_mtime_ns for path in paths)
        with self.lock:
            if self.v3_artifact_stamps.get(cache_key) == stamp:
                if cache_key in self.v3_bundle_cache:
                    return self.v3_bundle_cache[cache_key], None
                if cache_key in self.v3_load_errors:
                    return None, self.v3_load_errors[cache_key]
        try:
            from neural_path_v3_live import load_live_bundle

            bundle = load_live_bundle(*paths)
        except (Exception, SystemExit) as error:  # noqa: BLE001  # pragma: no cover
            message = (
                f"V3 artifacts could not be loaded: {type(error).__name__}: {error}"
            )
            with self.lock:
                self.v3_bundle_cache.pop(cache_key, None)
                self.v3_artifact_stamps[cache_key] = stamp
                self.v3_load_errors[cache_key] = message
            return None, message
        with self.lock:
            self.v3_bundle_cache[cache_key] = bundle
            self.v3_artifact_stamps[cache_key] = stamp
            self.v3_load_errors.pop(cache_key, None)
        return bundle, None

    def _apply_v3_normal_route(
        self,
        asset: str,
        timeframe: str,
        frame: pd.DataFrame,
        signal: pd.Series,
        as_of_index: int,
        snapshot_close_time_ms: int,
        target: float,
        direction_sign: int,
        current_state: dict[str, float],
        episodes: pd.DataFrame,
        path_states: pd.DataFrame,
        path_spans: dict[str, tuple[int, int]],
        projection: dict[str, Any],
    ) -> dict[str, Any]:
        """Replace only the normal route with V3's real-history choice, or keep V1."""
        elapsed_bars = as_of_index - int(signal["bar_index"])
        bundle, load_error = self._v3_bundle(timeframe, elapsed_bars)
        if bundle is None:
            return {"available": False, "active": False, "reason": load_error}
        try:
            from neural_path_v3_live import select_live_route

            observable_state = self._observable_state(
                asset, timeframe, int(signal["open_time"])
            )
            selection = select_live_route(
                bundle, frame, signal, as_of_index, observable_state
            )
            episode_id = str(selection["episode_id"])
            historical_asset = str(selection["historical_asset"])
            route_episodes = episodes
            route_path_states = path_states
            route_path_spans = path_spans
            if timeframe == "1m" and historical_asset != asset:
                (
                    route_episodes,
                    route_path_states,
                    _,
                    route_path_spans,
                ) = self._library_states(historical_asset, timeframe)
            span = route_path_spans.get(episode_id)
            if span is None:
                raise ValueError(
                    "selected V3 episode is absent from the runtime path cache"
                )
            event = route_episodes.loc[
                route_episodes["episode_id"].astype(str).eq(episode_id)
            ]
            if len(event) != 1:
                raise ValueError(
                    "selected V3 episode is not unique in the episode library"
                )
            if int(event["fill_close_time_ms"].iat[0]) > snapshot_close_time_ms:
                raise ValueError(
                    "selected V3 episode had not resolved by the live snapshot"
                )
            source = route_path_states.iloc[span[0] : span[1]]
            alignment_offset = int(selection["alignment_offset_bars"])
            aligned = source.loc[source["offset_bars"].eq(alignment_offset)]
            if len(aligned) != 1:
                raise ValueError(
                    "selected V3 alignment is absent from the runtime path cache"
                )
            row = aligned.iloc[0]
            historical_move = float(row["alignment_current_move_pct"])
            if historical_move <= 0:
                raise ValueError("selected V3 alignment is not away from the wick")
            remaining_bars = int(row["remaining_to_fill_bars"])
            historical_future = max(0.0, float(row["future_peak_move_pct"]))
            projected_future = max(
                0.0,
                historical_future
                * float(current_state["current_move_pct"])
                / historical_move,
            )

            cohort = projection["cohort"]
            cohort_duration = np.log1p(
                cohort["remaining_to_fill_bars"].to_numpy(dtype=float)
            )
            cohort_excursion = cohort["projected_future_max_away_move_pct"].to_numpy(
                dtype=float
            )
            duration_ranks = (
                pd.Series(cohort_duration).rank(pct=True, method="average").to_numpy()
            )
            excursion_ranks = (
                pd.Series(cohort_excursion).rank(pct=True, method="average").to_numpy()
            )
            cohort_joint = 0.55 * duration_ranks + 0.45 * excursion_ranks
            selected_joint = 0.55 * float(
                np.mean(cohort_duration <= np.log1p(remaining_bars))
            ) + 0.45 * float(np.mean(cohort_excursion <= projected_future))
            selected_percentile = float(np.mean(cohort_joint <= selected_joint))

            selector = str(selection["selector"])
            selector_description = (
                "sequence-embedding match"
                if selector == "neural_real_path"
                else "forecast-anchored match"
            )
            scenario: dict[str, Any] = {
                "name": "normal",
                "description": f"V3 {bundle.regime_label} {selector_description}: a real historical continuation selected from the learned live prefix; fast and extreme remain V1 references.",
                "selector": f"v3_{selector}",
                "episode_id": episode_id,
                "historical_asset": str(row["asset"]),
                "historical_timeframe": str(row["timeframe"]),
                "historical_direction": str(row["direction"]),
                "alignment_offset_bars": alignment_offset,
                "remaining_to_fill_bars": remaining_bars,
                "historical_alignment_current_move_pct": historical_move,
                "historical_alignment_peak_move_pct": float(
                    row["alignment_peak_move_pct"]
                ),
                "historical_alignment_drawdown_pct": float(
                    row["alignment_drawdown_pct"]
                ),
                "historical_future_max_away_move_pct": historical_future,
                "projected_future_max_away_move_pct": projected_future,
                "projected_additional_adverse_move_pct": max(
                    0.0, projected_future - float(current_state["current_move_pct"])
                ),
                "joint_risk_score": selected_joint,
                "joint_risk_score_percentile": selected_percentile,
                "matched_cohort_size": len(cohort),
                "matched_cohort_tail_at_or_above_fraction": float(
                    np.mean(cohort_joint >= selected_joint)
                ),
                "match_score": float(selection["hybrid_distance"]),
            }
            candles, scale = projected_candles(
                route_path_states,
                scenario,
                target,
                direction_sign,
                float(current_state["current_move_pct"]),
                snapshot_close_time_ms,
                int(signal["interval_minutes"]),
                route_path_spans,
            )
            scenario["normalization_scale"] = round(float(scale), 8)
            scenario["projected_candles"] = candles
            scenario.update(
                projected_path_metrics(
                    candles,
                    target,
                    direction_sign,
                    float(current_state["current_move_pct"]),
                )
            )
            scenario["projected_terminal_fill_candle_utc"] = utc_iso(
                candles[-1]["time"] * 1000
                + int(signal["interval_minutes"]) * 60_000
            )
            projection["scenarios"] = [
                scenario if item["name"] == "normal" else item
                for item in projection["scenarios"]
            ]
            return {
                "available": True,
                "active": True,
                "selector": selector,
                "regime_label": selection["regime_label"],
                "selected_episode_id": episode_id,
                "selected_alignment_offset_bars": alignment_offset,
                "neighbor_episode_count": selection["neighbor_episode_count"],
                "candidate_pool_size": selection["candidate_pool_size"],
                "conditional_ranges": {
                    "remaining_bars": selection["remaining_bars"],
                    "future_max_away_pct": selection["future_max_away_pct"],
                },
                "elapsed_bars": selection["elapsed_bars"],
                "supported_age_min_bars": selection["supported_age_min_bars"],
                "supported_age_max_bars": selection["supported_age_max_bars"],
                "artifact_generated_at_utc": selection["artifact_generated_at_utc"],
                "fit_rows": selection["fit_rows"],
                "fallback": "V1 normal route is retained automatically if V3 cannot load or the live age is unsupported.",
            }
        except Exception as error:  # noqa: BLE001  # pragma: no cover
            return {
                "available": True,
                "active": False,
                "reason": f"V3 fell back to V1: {type(error).__name__}: {error}",
            }

    def _apply_c2_numerical_routes(
        self,
        asset: str,
        timeframe: str,
        signal_time_ms: int,
        projection: dict[str, Any],
        risk: dict[str, Any],
        candidates_by_horizon: dict[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """Select real C2 continuations close to the frozen E0/A time-risk targets."""
        if timeframe != "5m":
            return {
                "available": False,
                "active": False,
                "reason": "Numerically aligned C2 routes are currently frozen for 5m only.",
            }
        artifact = risk.get("artifact", {})
        if artifact.get("forecast_source") != "forecast_v1":
            return {
                "available": False,
                "active": False,
                "reason": "The frozen 5m E0/A/C2 artifact is unavailable; V1 routes remain visible.",
            }
        try:
            frame = self._frame(asset, timeframe)
            signals = self._signals(asset, timeframe, frame)
            matches = signals.loc[signals["open_time"].eq(signal_time_ms)]
            if len(matches) != 1:
                raise ValueError("Pinned candle is not a unique strict wick signal")
            signal = matches.iloc[0]
            current_index = len(frame) - 1
            snapshot_close_time_ms = int(frame["close_time"].iat[current_index])
            target = float(signal["wick_target"])
            direction_sign = int(signal["direction_sign"])
            current_move_pct = float(projection["current_state"]["current_move_pct"])
            episodes, path_states, _, path_spans = self._library_states(asset, timeframe)
            event_by_id = {
                str(row.episode_id): row for row in episodes.itertuples(index=False)
            }
            route_specs = (
                (
                    "fast",
                    "1440m",
                    "p10",
                    "p50",
                    0.55,
                    0.35,
                    False,
                    "C2 real continuation closest to the frozen optimistic waiting-time and central adverse-risk target.",
                ),
                (
                    "normal",
                    "10080m",
                    "p50",
                    "p50",
                    0.55,
                    0.35,
                    False,
                    "C2 real continuation closest to the frozen central waiting-time and adverse-risk target.",
                ),
                (
                    "adverse",
                    "1440m",
                    "p50",
                    "p80",
                    0.10,
                    0.80,
                    True,
                    "A real C2 continuation that reaches the frozen 1-day p80 adverse level before eventually filling; this is the liquidation-risk path the central route can hide.",
                ),
                (
                    "extreme",
                    "43200m",
                    "p90",
                    "p90",
                    0.55,
                    0.35,
                    False,
                    "C2 real continuation closest to the frozen tail waiting-time and adverse-risk target; it is a stress reference, not a worst-case guarantee.",
                ),
            )
            selected_scenarios: list[dict[str, Any]] = []
            used_episodes: set[str] = set()
            selection_details: dict[str, Any] = {}
            normal_distribution: dict[str, Any] | None = None

            for (
                name,
                slug,
                wait_quantile,
                risk_quantile,
                duration_weight,
                adverse_weight,
                require_adverse_first,
                description,
            ) in route_specs:
                wait_target = float(
                    risk["remaining_time_minutes_if_filled_within_horizon"][slug][
                        wait_quantile
                    ]
                )
                adverse_target = float(
                    risk["additional_adverse_pct"][slug][risk_quantile]
                )
                raw_candidates = candidates_by_horizon.get(slug, [])
                options: list[dict[str, Any]] = []
                for candidate in raw_candidates:
                    episode_id = str(candidate["episode_id"])
                    if episode_id in used_episodes:
                        continue
                    event = event_by_id.get(episode_id)
                    span = path_spans.get(episode_id)
                    if event is None or span is None:
                        continue
                    if int(event.fill_close_time_ms) > snapshot_close_time_ms:
                        continue
                    requested_remaining_bars = int(
                        round(float(candidate["wait_minutes"]) / 5.0)
                    )
                    source = path_states.iloc[span[0] : span[1]]
                    aligned = source.loc[
                        source["remaining_to_fill_bars"].eq(requested_remaining_bars)
                    ]
                    if len(aligned) != 1:
                        continue
                    row = aligned.iloc[0]
                    historical_move = float(row["alignment_current_move_pct"])
                    if historical_move <= 0.0:
                        continue
                    remaining_bars = int(row["remaining_to_fill_bars"])
                    historical_future = max(0.0, float(row["future_peak_move_pct"]))
                    projected_future = max(
                        0.0,
                        historical_future * current_move_pct / historical_move,
                    )
                    entry_denominator = 100.0 + direction_sign * current_move_pct
                    if entry_denominator <= 0.0:
                        continue
                    projected_additional = (
                        max(0.0, projected_future - current_move_pct)
                        * 100.0
                        / entry_denominator
                    )
                    options.append(
                        {
                            "candidate": candidate,
                            "row": row,
                            "remaining_bars": remaining_bars,
                            "remaining_minutes": remaining_bars * 5.0,
                            "historical_move": historical_move,
                            "historical_future": historical_future,
                            "projected_future": projected_future,
                            "projected_additional": projected_additional,
                        }
                    )
                if len(options) < 3:
                    raise ValueError(
                        f"C2 {name} route has fewer than three drawable completed analogues"
                    )

                duration_values = np.log1p(
                    np.asarray([item["remaining_minutes"] for item in options], dtype=float)
                )
                adverse_values = np.asarray(
                    [item["projected_additional"] for item in options], dtype=float
                )
                distances = np.asarray(
                    [float(item["candidate"]["distance"]) for item in options],
                    dtype=float,
                )
                duration_error = np.abs(duration_values - np.log1p(wait_target))
                adverse_error = np.abs(adverse_values - adverse_target) / max(
                    adverse_target, 1.0
                )
                distance_penalty = distances / max(float(np.median(distances)), 1e-9)
                scores = (
                    duration_weight * duration_error
                    + adverse_weight * adverse_error
                    + 0.10 * distance_penalty
                )
                if require_adverse_first:
                    normalized_ohlc = (
                        "normalized_open_pct",
                        "normalized_high_pct",
                        "normalized_low_pct",
                        "normalized_close_pct",
                    )
                    definite_crossings = np.zeros(len(options), dtype=bool)
                    for option_index, option in enumerate(options):
                        episode_id = str(option["candidate"]["episode_id"])
                        span = path_spans[episode_id]
                        source = path_states.iloc[span[0] : span[1]]
                        future = source.loc[
                            source["offset_bars"].gt(
                                int(option["row"]["offset_bars"])
                            )
                        ]
                        normalized_envelope = future.loc[
                            :, list(normalized_ohlc)
                        ].max(axis=1).to_numpy(dtype=float)
                        projected_future_moves = (
                            normalized_envelope
                            * current_move_pct
                            / option["historical_move"]
                        )
                        projected_additional = np.maximum(
                            0.0,
                            projected_future_moves - current_move_pct,
                        )
                        projected_additional = (
                            projected_additional
                            * 100.0
                            / (100.0 + direction_sign * current_move_pct)
                        )
                        crossings = np.flatnonzero(
                            projected_additional >= adverse_target - 1e-9
                        )
                        first_crossing = (
                            int(crossings[0]) + 1 if len(crossings) else None
                        )
                        option["adverse_threshold_bars"] = first_crossing
                        option["adverse_threshold_before_fill"] = bool(
                            first_crossing is not None
                            and first_crossing < option["remaining_bars"]
                        )
                        definite_crossings[option_index] = option[
                            "adverse_threshold_before_fill"
                        ]
                    if not bool(np.any(definite_crossings)):
                        raise ValueError(
                            "C2 adverse-first route has no historical candidate reaching the 1-day p80 adverse level before its fill candle"
                        )
                    scores = scores + np.where(definite_crossings, 0.0, 100.0)
                chosen_index = int(np.argmin(scores))
                chosen = options[chosen_index]
                row = chosen["row"]
                candidate = chosen["candidate"]
                episode_id = str(candidate["episode_id"])
                used_episodes.add(episode_id)

                duration_ranks = (
                    pd.Series(duration_values).rank(pct=True, method="average").to_numpy()
                )
                adverse_ranks = (
                    pd.Series(adverse_values).rank(pct=True, method="average").to_numpy()
                )
                joint_scores = 0.55 * duration_ranks + 0.45 * adverse_ranks
                selected_joint = float(joint_scores[chosen_index])
                selected_percentile = float(np.mean(joint_scores <= selected_joint))
                scenario: dict[str, Any] = {
                    "name": name,
                    "description": description,
                    "selector": "c2_e0_a_numerical_aligned_real_path",
                    "episode_id": episode_id,
                    "historical_asset": str(row["asset"]),
                    "historical_timeframe": str(row["timeframe"]),
                    "historical_direction": str(row["direction"]),
                    "alignment_offset_bars": int(row["offset_bars"]),
                    "remaining_to_fill_bars": int(chosen["remaining_bars"]),
                    "historical_alignment_current_move_pct": float(
                        chosen["historical_move"]
                    ),
                    "historical_alignment_peak_move_pct": float(
                        row["alignment_peak_move_pct"]
                    ),
                    "historical_alignment_drawdown_pct": float(
                        row["alignment_drawdown_pct"]
                    ),
                    "historical_future_max_away_move_pct": float(
                        chosen["historical_future"]
                    ),
                    "projected_future_max_away_move_pct": float(
                        chosen["projected_future"]
                    ),
                    "projected_additional_adverse_move_pct": float(
                        chosen["projected_additional"]
                    ),
                    "joint_risk_score": selected_joint,
                    "joint_risk_score_percentile": selected_percentile,
                    "matched_cohort_size": len(options),
                    "matched_cohort_tail_at_or_above_fraction": float(
                        np.mean(joint_scores >= selected_joint)
                    ),
                    "match_score": float(candidate["distance"]),
                    "match_score_components": {
                        "c2_distance": float(candidate["distance"]),
                        "numerical_target_score": float(scores[chosen_index]),
                    },
                    "numerical_target": {
                        "horizon": slug,
                        "wait_quantile": wait_quantile,
                        "wait_minutes": wait_target,
                        "adverse_quantile": risk_quantile,
                        "additional_adverse_pct": adverse_target,
                    },
                    "c2_neighbor_rank": int(candidate["neighbor_rank"]),
                }
                candles, scale = projected_candles(
                    path_states,
                    scenario,
                    target,
                    direction_sign,
                    current_move_pct,
                    snapshot_close_time_ms,
                    int(signal["interval_minutes"]),
                    path_spans,
                )
                scenario["normalization_scale"] = round(float(scale), 8)
                scenario["projected_candles"] = candles
                scenario.update(
                    projected_path_metrics(
                        candles,
                        target,
                        direction_sign,
                        current_move_pct,
                    )
                )
                scenario["projected_terminal_fill_candle_utc"] = utc_iso(
                    candles[-1]["time"] * 1000
                    + int(signal["interval_minutes"]) * 60_000
                )
                if require_adverse_first:
                    entry_price = target * (
                        1.0 + direction_sign * current_move_pct / 100.0
                    )
                    adverse_threshold_price = entry_price * (
                        1.0 + direction_sign * adverse_target / 100.0
                    )
                    if direction_sign == 1:
                        threshold_hits = [
                            index
                            for index, candle in enumerate(candles)
                            if float(candle["high"]) >= adverse_threshold_price
                        ]
                    else:
                        threshold_hits = [
                            index
                            for index, candle in enumerate(candles)
                            if float(candle["low"]) <= adverse_threshold_price
                        ]
                    hit_index = threshold_hits[0] if threshold_hits else None
                    scenario["adverse_first_threshold_pct"] = adverse_target
                    scenario["projected_adverse_threshold_price"] = round(
                        float(adverse_threshold_price), 8
                    )
                    scenario["projected_adverse_threshold_bars"] = (
                        hit_index + 1 if hit_index is not None else None
                    )
                    scenario["projected_adverse_threshold_candle_utc"] = (
                        utc_iso(
                            candles[hit_index]["time"] * 1000
                            + int(signal["interval_minutes"]) * 60_000
                        )
                        if hit_index is not None
                        else None
                    )
                    scenario["projected_adverse_ordering"] = (
                        "adverse_before_fill"
                        if hit_index is not None and hit_index < len(candles) - 1
                        else (
                            "same_terminal_candle_ambiguous"
                            if hit_index is not None
                            else "p80_threshold_not_reached"
                        )
                    )
                selected_scenarios.append(scenario)
                selection_details[name] = {
                    "horizon": slug,
                    "candidate_count": len(options),
                    "selected_episode_id": episode_id,
                    "selected_neighbor_rank": int(candidate["neighbor_rank"]),
                    "target_wait_minutes": wait_target,
                    "selected_wait_minutes": float(chosen["remaining_minutes"]),
                    "target_additional_adverse_pct": adverse_target,
                    "selected_additional_adverse_pct": float(
                        scenario["projected_additional_adverse_move_pct"]
                    ),
                    "projected_adverse_threshold_bars": scenario.get(
                        "projected_adverse_threshold_bars"
                    ),
                    "projected_adverse_ordering": scenario.get(
                        "projected_adverse_ordering"
                    ),
                }
                if name == "normal":
                    normal_distribution = {
                        "remaining_time_to_fill_minutes": {
                            "p25": float(np.quantile(np.expm1(duration_values), 0.25)),
                            "p50": float(np.quantile(np.expm1(duration_values), 0.50)),
                            "p90": float(np.quantile(np.expm1(duration_values), 0.90)),
                        },
                        "projected_future_max_away_move_pct": {
                            "p50": float(
                                np.quantile(
                                    [item["projected_future"] for item in options], 0.50
                                )
                            ),
                            "p90": float(
                                np.quantile(
                                    [item["projected_future"] for item in options], 0.90
                                )
                            ),
                        },
                    }

            projection["scenarios"] = selected_scenarios
            if normal_distribution is not None:
                projection["cohort_distribution"] = normal_distribution
            projection["library"]["top_k_state_matched_episodes"] = int(
                selection_details["normal"]["candidate_count"]
            )
            return {
                "available": True,
                "active": True,
                "method": "C2 dynamic retrieval aligned to frozen E0/A numerical targets",
                "architecture_version": artifact.get("architecture_version"),
                "artifact_sha256": artifact.get("artifact_sha256"),
                "normal_candidate_count": selection_details["normal"][
                    "candidate_count"
                ],
                "selections": selection_details,
                "legacy_v3_active": False,
            }
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            return {
                "available": True,
                "active": False,
                "reason": f"Numerically aligned C2 routes fell back to V1: {type(error).__name__}: {error}",
            }

    def _prospective_bundle(
        self, timeframe: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        if timeframe not in {"1m", "5m"}:
            return None, "All-outcome risk is currently validated for 1m and 5m only."
        model_dir = self.prospective_artifact_root / timeframe
        manifest_path = active_manifest_path(model_dir)
        legacy_path = model_dir / "model.joblib"

        def file_stamp(path: Path) -> tuple[int, int]:
            if not path.exists():
                return (0, 0)
            stat = path.stat()
            return (stat.st_mtime_ns, stat.st_size)

        manifest: dict[str, Any] | None = None
        active_path: Path | None = None
        active_manifest_error: str | None = None
        if timeframe == "5m" and manifest_path.exists():
            try:
                manifest = load_active_manifest(model_dir)
                active_path = (model_dir / str(manifest["artifact_file"])).resolve()
                if active_path.parent != model_dir.resolve():
                    raise RuntimeError(
                        "Active forecast artifact escapes its timeframe model directory"
                    )
            except Exception as error:  # pragma: no cover - artifact boundary
                active_manifest_error = f"{type(error).__name__}: {error}"
                manifest = None
                active_path = None
        cache_key = (
            timeframe,
            *file_stamp(manifest_path),
            *(file_stamp(active_path) if active_path is not None else (0, 0)),
            *file_stamp(legacy_path),
        )
        with self.lock:
            if (
                self.prospective_model is not None
                and self.prospective_model_timeframe == timeframe
                and self.prospective_model_cache_key == cache_key
            ):
                return self.prospective_model, None
            if (
                self.prospective_load_error is not None
                and self.prospective_model_timeframe == timeframe
                and self.prospective_model_cache_key == cache_key
            ):
                return None, self.prospective_load_error

        model: dict[str, Any] | None = None
        active_failure = active_manifest_error
        if timeframe == "5m" and manifest is not None:
            try:
                verified_path, verified_manifest = resolve_active_artifact(model_dir)
                model = dict(load_architecture(verified_path))
                if model.get("architecture_id") != verified_manifest["architecture_version"]:
                    raise RuntimeError(
                        "Active forecast architecture version does not match its artifact"
                    )
                model["_artifact_path"] = str(verified_path)
                model["_artifact_sha256"] = verified_manifest["artifact_sha256"]
                model["_forecast_source"] = "forecast_v1"
                model["_active_manifest"] = verified_manifest
            except Exception as error:  # pragma: no cover - artifact boundary
                active_failure = f"{type(error).__name__}: {error}"
                model = None

        if model is None and legacy_path.exists():
            try:
                model = dict(load_prospective_bundle(legacy_path))
                model["_artifact_path"] = str(legacy_path)
                model["_artifact_sha256"] = sha256_file(legacy_path)
                model["_forecast_source"] = (
                    "legacy_fallback" if timeframe == "5m" else "legacy_primary"
                )
                if active_failure:
                    model["_active_forecast_error"] = active_failure
            except Exception as error:  # pragma: no cover - artifact boundary
                message = f"All-outcome model could not be loaded: {type(error).__name__}"
                with self.lock:
                    self.prospective_model = None
                    self.prospective_model_timeframe = timeframe
                    self.prospective_model_cache_key = cache_key
                    self.prospective_load_error = message
                return None, message

        if model is None:
            if active_failure:
                message = f"Frozen 5m forecast is unavailable: {active_failure}"
            else:
                message = f"Validated {timeframe} all-outcome model artifact is unavailable."
            with self.lock:
                self.prospective_model = None
                self.prospective_model_timeframe = timeframe
                self.prospective_model_cache_key = cache_key
                self.prospective_load_error = message
            return None, message
        with self.lock:
            # Keep only the currently used timeframe in RAM.  The 1m artifact
            # is large and should not coexist with the 5m model unnecessarily.
            self.prospective_model = model
            self.prospective_model_timeframe = timeframe
            self.prospective_model_cache_key = cache_key
            self.prospective_load_error = None
        return model, None

    def _prospective_risk(
        self, asset: str, timeframe: str, signal_time_ms: int
    ) -> dict[str, Any]:
        model, error = self._prospective_bundle(timeframe)
        if model is None:
            return {"available": False, "reason": error or "All-outcome model unavailable."}
        try:
            frame = self._frame(asset, timeframe)
            signals = self._signals(asset, timeframe, frame)
            matches = signals.loc[signals["open_time"].eq(signal_time_ms)]
            if len(matches) != 1:
                raise ValueError("Pinned candle is not a unique strict wick signal")
            signal = matches.iloc[0]
            signal_index = int(signal["bar_index"])
            current_index = len(frame) - 1
            direction_sign = int(signal["direction_sign"])
            later = frame.iloc[signal_index + 1 : current_index + 1]
            opposite = float(signal["opposite_extreme"])
            departed = (
                later["close"].to_numpy(dtype=float) >= opposite
                if direction_sign == 1
                else later["close"].to_numpy(dtype=float) <= opposite
            )
            offsets = np.flatnonzero(departed)
            if not len(offsets):
                raise ValueError("Pinned wick has no confirmed departure yet")
            departure_index = signal_index + 1 + int(offsets[0])
            state = state_for_live_path(
                frame,
                signal_index,
                current_index,
                direction_sign,
                float(signal["wick_target"]),
                int(signal["interval_minutes"]),
            )
            query, features = prospective_live_observation(
                frame,
                signal,
                asset,
                timeframe,
                current_index,
                departure_index,
                float(state["current_move_pct"]),
                float(state["peak_move_pct"]),
                float(state["drawdown_from_peak_pct"]),
            )
            if model.get("schema_version") == ARCHITECTURE_SCHEMA_VERSION:
                sequences = sequence_features_from_live(frame, query)
                prediction = predict_architecture(model, features, sequences, asset)
            else:
                prediction = predict_prospective_bundle(model, features)
        except (KeyError, TypeError, ValueError) as exception:
            return {
                "available": False,
                "reason": f"All-outcome estimate could not be constructed: {exception}",
            }
        display_horizons = {
            f"{minutes}m": prediction["fill_probability"].get(f"{minutes}m")
            for minutes in DISPLAY_HORIZONS_MINUTES
        }
        support = model.get("support", {})
        age_minutes = int(query["entry_age_minutes"])
        forecast_source = str(model.get("_forecast_source", "unknown"))
        artifact_hash = str(model.get("_artifact_sha256", ""))
        is_frozen_v1 = (
            model.get("schema_version") == ARCHITECTURE_SCHEMA_VERSION
            and forecast_source == "forecast_v1"
        )
        response = {
            "available": True,
            "name": "All-outcome risk from current close",
            "timeframe": timeframe,
            "entry_age_minutes": age_minutes,
            "entry_price": round(float(frame["close"].iat[current_index]), 8),
            "fill_probability": display_horizons,
            "additional_adverse_pct": {
                f"{minutes}m": prediction["additional_adverse_pct"].get(f"{minutes}m")
                for minutes in DISPLAY_HORIZONS_MINUTES
            },
            "remaining_time_minutes_if_filled_within_horizon": {
                f"{minutes}m": prediction[
                    "remaining_time_minutes_if_filled_within_horizon"
                ].get(f"{minutes}m")
                for minutes in DISPLAY_HORIZONS_MINUTES
            },
            "competing_outcomes": {
                f"{minutes}m_vs_{threshold:g}pct": prediction[
                    "competing_outcomes"
                ].get(f"{minutes}m_vs_{threshold:g}pct")
                for minutes in DISPLAY_HORIZONS_MINUTES
                for threshold in DISPLAY_THRESHOLDS_PCT
            },
            "historical_support": prediction.get("historical_support", {}),
            "_historical_route_candidates": prediction.get(
                "historical_route_candidates", {}
            ),
            "ownership": prediction.get(
                "ownership",
                {
                    "fill_probability": "A supervised model",
                    "adverse_p50": "A supervised model",
                    "adverse_p80_p90": "A calibrated tail",
                    "waiting_time": "A supervised model",
                    "historical_support": "unavailable",
                    "routes": "separate historical illustration layer",
                },
            ),
            "support": {
                "status": (
                    "inside_trained_age_range"
                    if int(support.get("entry_age_minutes_min", 0))
                    <= age_minutes
                    <= int(support.get("entry_age_minutes_max", 0))
                    else "outside_trained_age_range"
                ),
                "entry_age_minutes_min": support.get("entry_age_minutes_min"),
                "entry_age_minutes_max": support.get("entry_age_minutes_max"),
            },
            "artifact": {
                "architecture_id": model.get("architecture_id"),
                "architecture_version": model.get("architecture_id"),
                "artifact_sha256": artifact_hash,
                "forecast_source": forecast_source,
                "active_frozen": model.get("_active_manifest", {}).get("frozen"),
                "schema_version": model.get("schema_version"),
                "frozen": bool(model.get("frozen", False)),
                "prospective_validation_status": model.get(
                    "prospective_validation_status"
                ),
                "generated_at_utc": model.get("generated_at_utc"),
                "training_rows": model.get("training_rows"),
                "training_filled_rows": model.get("training_filled_rows"),
                "training_episodes": model.get("training_episodes"),
                "training_label_cutoff_utc": model.get("training_label_cutoff_utc"),
                "baseline_version": model.get("baseline_version"),
            },
            "holdout": {
                "model": model.get("holdout_metrics"),
                "baseline": model.get("baseline_holdout_metrics"),
            },
            "warning": (
                "Frozen experimental architecture under prospective validation. "
                "Probabilities and adverse percentages are estimates, not certainty or trade advice."
                if is_frozen_v1
                else "Probabilities and adverse percentages are estimates from unresolved and resolved historical entries, not certainty or trade advice."
            ),
        }
        if timeframe == "5m" and artifact_hash:
            try:
                architecture_version = str(
                    model.get("architecture_id")
                    or f"legacy_5m_{artifact_hash[:12]}"
                )
                inserted = record_forecast(
                    self.prospective_ledger_path,
                    architecture_version=architecture_version,
                    artifact_hash=artifact_hash,
                    forecast_source=forecast_source,
                    asset=asset,
                    timeframe=timeframe,
                    signal_open_time_ms=signal_time_ms,
                    observation_close_time_ms=int(frame["close_time"].iat[current_index]),
                    direction=str(signal["direction"]),
                    wick_target=float(signal["wick_target"]),
                    entry_price=float(frame["close"].iat[current_index]),
                    payload={
                        "fill_probability": response["fill_probability"],
                        "additional_adverse_pct": response["additional_adverse_pct"],
                        "remaining_time_minutes_if_filled_within_horizon": response[
                            "remaining_time_minutes_if_filled_within_horizon"
                        ],
                        "historical_support": response["historical_support"],
                        "ownership": response["ownership"],
                    },
                )
                response["prospective_log"] = {
                    "status": "recorded" if inserted else "already_recorded",
                    "forecast_source": forecast_source,
                }
            except (OSError, ValueError, sqlite3.Error) as error:
                response["prospective_log"] = {
                    "status": "unavailable",
                    "reason": type(error).__name__,
                }
        return response

    def _v2_bundle(self) -> tuple[dict[str, Any] | None, str | None]:
        """Load the standalone V2a artifact once, without making V1 depend on it."""
        if not self.v2_model_path.exists():
            return (
                None,
                "V2a model artifact is not present. Rebuild the V2 calibration artifacts first.",
            )
        model_mtime_ns = self.v2_model_path.stat().st_mtime_ns
        with self.lock:
            if self.v2_model is not None and self.v2_model_mtime_ns == model_mtime_ns:
                return self.v2_model, None
            if (
                self.v2_load_error is not None
                and self.v2_model_mtime_ns == model_mtime_ns
            ):
                return None, self.v2_load_error
        try:
            model = load_model_bundle(self.v2_model_path)
        except Exception as error:  # pragma: no cover - defensive artifact boundary
            message = f"V2a artifact could not be loaded: {type(error).__name__}"
            with self.lock:
                self.v2_model = None
                self.v2_model_mtime_ns = model_mtime_ns
                self.v2_load_error = message
            return None, message
        with self.lock:
            self.v2_model = model
            self.v2_model_mtime_ns = model_mtime_ns
            self.v2_load_error = None
        return model, None

    def _v2_summary_payload(self) -> dict[str, Any]:
        if not self.v2_summary_path.exists():
            return {}
        summary_mtime_ns = self.v2_summary_path.stat().st_mtime_ns
        with self.lock:
            if (
                self.v2_summary is not None
                and self.v2_summary_mtime_ns == summary_mtime_ns
            ):
                return self.v2_summary
        try:
            summary = json.loads(self.v2_summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        with self.lock:
            self.v2_summary = summary
            self.v2_summary_mtime_ns = summary_mtime_ns
        return summary

    def _observable_state(
        self, asset: str, timeframe: str, signal_time_ms: int
    ) -> dict[str, Any]:
        """Build sequence-model inputs strictly from the live candle snapshot."""
        frame = self._frame(asset, timeframe)
        signals = self._signals(asset, timeframe, frame)
        matches = signals.loc[signals["open_time"].eq(signal_time_ms)]
        if len(matches) != 1:
            raise ValueError(
                f"Pinned signal is not a unique strict {timeframe} wick in the source snapshot"
            )
        signal = matches.iloc[0]
        signal_index = int(signal["bar_index"])
        current_index = len(frame) - 1
        if signal_index >= current_index:
            raise ValueError(
                f"Sequence matching needs at least one completed {timeframe} candle after the pinned signal"
            )
        target = float(signal["wick_target"])
        direction_sign = int(signal["direction_sign"])
        observed = frame.iloc[signal_index : current_index + 1]
        normalized_open = np.asarray(
            normalized_price(
                observed["open"].to_numpy(dtype=float), target, direction_sign
            ),
            dtype=float,
        )
        normalized_high = np.asarray(
            normalized_price(
                observed["high"].to_numpy(dtype=float), target, direction_sign
            ),
            dtype=float,
        )
        normalized_low = np.asarray(
            normalized_price(
                observed["low"].to_numpy(dtype=float), target, direction_sign
            ),
            dtype=float,
        )
        normalized_close = np.asarray(
            normalized_price(
                observed["close"].to_numpy(dtype=float), target, direction_sign
            ),
            dtype=float,
        )
        current_state = state_for_live_path(
            frame,
            signal_index,
            current_index,
            direction_sign,
            target,
            interval_minutes=int(signal["interval_minutes"]),
        )
        close_changes = np.diff(normalized_close, prepend=normalized_close[0])
        volume_ratio = max(float(frame["volume"].iat[current_index]), 0.0) / max(
            float(frame["volume"].iat[signal_index]), 1e-12
        )
        state: dict[str, Any] = {
            "asset": asset,
            "direction": str(signal["direction"]),
            **{
                key: float(value)
                for key, value in current_state.items()
                if key != "elapsed_minutes"
            },
            "current_bar_range_pct": float(
                abs(normalized_high[-1] - normalized_low[-1])
            ),
            "current_bar_body_pct": float(
                abs(normalized_close[-1] - normalized_open[-1])
            ),
            "recent_bar_range_mean_3_pct": float(
                np.mean(np.abs(normalized_high[-3:] - normalized_low[-3:]))
            ),
            "recent_abs_close_change_mean_3_pct": float(
                np.mean(np.abs(close_changes[-3:]))
            ),
            "volume_ratio_to_signal": float(volume_ratio),
        }
        for feature in FEATURE_COLUMNS:
            state[feature] = float(signal[feature])
        return state

    def _v2_observable_state(self, asset: str, signal_time_ms: int) -> dict[str, Any]:
        """Compatibility wrapper for the standalone 5m V2 risk estimator."""
        return self._observable_state(asset, "5m", signal_time_ms)

    def _v2_risk(
        self, asset: str, timeframe: str, signal_time_ms: int
    ) -> dict[str, Any]:
        """Return optional V2a risk quantiles; V1 paths remain available if this is unavailable."""
        if timeframe != "5m":
            return {
                "available": False,
                "reason": "V2a is currently trained and calibrated only on 5m snapshots. V1 routes still use the selected timeframe.",
            }
        model, model_error = self._v2_bundle()
        if model is None:
            return {
                "available": False,
                "reason": model_error or "V2a model artifact is unavailable.",
            }
        try:
            observable_state = self._v2_observable_state(asset, signal_time_ms)
            prediction = predict_from_observable_state(model, observable_state)
        except (KeyError, TypeError, ValueError) as error:
            return {
                "available": False,
                "reason": f"V2a could not construct a valid observable state: {error}",
            }

        summary = self._v2_summary_payload()
        calibration = summary.get("calibration", {})
        snapshot_offsets = [
            value
            for raw in summary.get("data", {}).get("snapshot_offsets_bars", [])
            if (value := _optional_nonnegative_int(raw)) is not None and value > 0
        ]
        max_snapshot_offset = max(snapshot_offsets) if snapshot_offsets else None
        age_support = snapshot_age_support(
            int(observable_state["elapsed_bars"]), snapshot_offsets
        )
        remaining = calibration.get("remaining_bars", {})
        future_away = calibration.get("future_max_away_pct", {})
        future_p10 = next(
            (
                row.get("observed_coverage")
                for row in future_away.get("coverage_by_quantile", [])
                if float(row.get("quantile", -1)) == 0.10
            ),
            None,
        )
        return {
            "available": True,
            "name": "V2a calibrated conditional ranges",
            "estimator": model.get("estimator", "local conditional quantiles"),
            "prediction": prediction,
            "observable_state": {
                key: round(float(value), 8)
                for key, value in observable_state.items()
                if key not in {"asset", "direction"}
            },
            "calibration": {
                "holdout_snapshot_count": future_away.get("case_count"),
                "remaining_p10_p90_observed_coverage": remaining.get(
                    "p10_to_p90_interval_observed_coverage"
                ),
                "future_away_p10_p90_observed_coverage": future_away.get(
                    "p10_to_p90_interval_observed_coverage"
                ),
                "future_away_p10_observed_coverage": future_p10,
            },
            "state_coverage": {
                "elapsed_bars": int(observable_state["elapsed_bars"]),
                "training_snapshot_offsets_bars": snapshot_offsets,
                "max_training_snapshot_elapsed_bars": max_snapshot_offset,
                "elapsed_within_training_snapshot_range": (
                    None
                    if not snapshot_offsets
                    else min(snapshot_offsets)
                    <= int(observable_state["elapsed_bars"])
                    <= max_snapshot_offset
                ),
            },
            "age_support": age_support,
            "artifact": v2_artifact_metadata(summary),
            "warning": "Conditional clean-fill diagnostic only. Future move-away P10 under-covered in the chronological holdout, so it is never a sizing floor or risk limit.",
        }

    def available_unfilled_signals(self, asset: str, timeframe: str) -> dict[str, Any]:
        validate_asset_timeframe(asset, timeframe)
        frame = self._frame(asset, timeframe)
        signals = self._signals(asset, timeframe, frame)
        items: list[dict[str, Any]] = []
        # The tail is enough for a practical pin picker and prevents a costly
        # full-history scan every time the browser asks for candidates.
        for signal in signals.tail(400).iloc[::-1].itertuples(index=False):
            index = int(signal.bar_index)
            if index >= len(frame) - 1:
                continue
            target = float(signal.wick_target)
            direction_sign = int(signal.direction_sign)
            later = frame.iloc[index + 1 :]
            filled = (
                (later["low"] <= target).any()
                if direction_sign == 1
                else (later["high"] >= target).any()
            )
            if filled:
                continue
            departed = has_confirmed_departure(
                later, direction_sign, float(signal.opposite_extreme)
            )
            if not departed:
                continue
            state = state_for_live_path(
                frame,
                index,
                len(frame) - 1,
                direction_sign,
                target,
                int(signal.interval_minutes),
            )
            if state["current_move_pct"] <= 0:
                continue
            items.append(
                {
                    "signal_time_utc": utc_iso(int(signal.open_time)),
                    "direction": str(signal.direction),
                    "wick_target": target,
                    "current_move_pct": round(float(state["current_move_pct"]), 4),
                    "elapsed_minutes": int(state["elapsed_minutes"]),
                }
            )
            if len(items) >= 80:
                break
        return {
            "asset": asset,
            "timeframe": timeframe,
            "as_of_open_time_utc": utc_iso(int(frame["open_time"].iat[-1])),
            "signals": items,
        }

    def _v1_payload(
        self,
        asset: str,
        timeframe: str,
        signal_time_ms: int,
        matching_mode: str = "adaptive",
    ) -> dict[str, Any]:
        """Run the shared V1 selector directly against the immutable cached library."""
        started = time.perf_counter()
        frame = self._frame(asset, timeframe)
        frame_loaded = time.perf_counter()
        matching_signal = frame.loc[frame["open_time"].eq(signal_time_ms)]
        if matching_signal.empty:
            raise ValueError(
                f"Signal candle does not exist in {timeframe} source: {utc_iso(signal_time_ms)}"
            )
        signals = self._signals(asset, timeframe, frame)
        signals_loaded = time.perf_counter()
        signal_rows = signals.loc[signals["open_time"].eq(signal_time_ms)]
        if len(signal_rows) != 1:
            raise ValueError(
                "Pinned candle is not a strict wick signal with enough preceding context"
            )
        signal = signal_rows.iloc[0]
        signal_index = int(signal["bar_index"])
        as_of_index = len(frame) - 1
        if as_of_index <= signal_index:
            raise ValueError("As-of candle must be later than the pinned signal")
        direction_sign = int(signal["direction_sign"])
        target = float(signal["wick_target"])
        later = frame.iloc[signal_index + 1 : as_of_index + 1]
        filled = (
            (later["low"] <= target).any()
            if direction_sign == 1
            else (later["high"] >= target).any()
        )
        if filled:
            raise ValueError(
                "Pinned wick has already been fully touched by the supplied as-of candle"
            )
        if not has_confirmed_departure(
            later, direction_sign, float(signal["opposite_extreme"])
        ):
            raise ValueError(
                "Pinned wick has not made the confirmed close at or beyond the opposite signal extreme required for a conditional path projection"
            )
        interval_minutes = int(signal["interval_minutes"])
        current_state = state_for_live_path(
            frame, signal_index, as_of_index, direction_sign, target, interval_minutes
        )
        if current_state["current_move_pct"] <= 0:
            raise ValueError(
                "Current close is not on the away-from-wick side required for conditional projection"
            )

        episodes, path_states, native_state_index, path_spans = self._library_states(
            asset, timeframe
        )
        library_loaded = time.perf_counter()
        snapshot_close_time_ms = (
            int(frame["open_time"].iat[as_of_index]) + interval_minutes * 60_000
        )
        scenario_matching_mode = (
            "adaptive" if matching_mode == "legacy_v3" else matching_mode
        )
        projection = project_at(
            episodes,
            path_states,
            signal,
            current_state,
            snapshot_close_time_ms,
            target,
            direction_sign,
            snapshot_close_time_ms,
            interval_minutes,
            80,
            path_states=path_states,
            native_state_index=native_state_index,
            episode_row_spans=path_spans,
            matching_mode=scenario_matching_mode,
            archetype_pool_size=ARCHETYPE_POOL_SIZE,
            archetype_blend_weight=ARCHETYPE_BLEND_WEIGHT,
        )
        projection_built = time.perf_counter()
        if projection["insufficient_matches"]:
            raise ValueError(
                "Fewer than 12 comparable historical states; scenario selection would be too unstable"
            )

        if matching_mode == "legacy_v3":
            v3_route = self._apply_v3_normal_route(
                asset,
                timeframe,
                frame,
                signal,
                as_of_index,
                snapshot_close_time_ms,
                target,
                direction_sign,
                current_state,
                episodes,
                path_states,
                path_spans,
                projection,
            )
        elif matching_mode in {"blended", "archetype"}:
            v3_route = {
                "available": True,
                "active": False,
                "reason": "V3 is disabled in candle-shape experiments so all three routes come from the selected V1 matching cohort.",
            }
        else:
            v3_route = {
                "available": True,
                "active": False,
                "reason": "Legacy V3 is disabled in Adaptive mode; 5m routes are selected after the frozen E0/A/C2 numerical forecast is available.",
            }

        eligible = projection["eligible_events"]
        trajectory_eligible = projection["trajectory_events"]
        matched = projection["matched_states"]
        cohort = projection["cohort"]
        top_matches = []
        for row in matched.head(min(10, len(matched))).itertuples(index=False):
            row_values = row._asdict()
            top_matches.append(
                {
                    "episode_id": str(row.episode_id),
                    "asset": str(row.asset),
                    "timeframe": str(row.timeframe),
                    "direction": str(row.direction),
                    "alignment_offset_bars": int(row.offset_bars),
                    "remaining_to_fill_bars": int(row.remaining_to_fill_bars),
                    "historical_future_max_away_move_pct": float(
                        row.historical_future_max_away_move_pct
                    ),
                    "projected_future_max_away_move_pct": float(
                        row.projected_future_max_away_move_pct
                    ),
                    "projected_additional_adverse_move_pct": float(
                        row.projected_additional_adverse_move_pct
                    ),
                    "match_score": float(row.match_score),
                    "match_score_components": {
                        column: float(row_values[column])
                        for column in MATCH_SCORE_COMPONENT_COLUMNS
                        if column in row_values
                    },
                }
            )
        remaining_minutes = (
            cohort["remaining_to_fill_bars"].to_numpy(dtype=float) * interval_minutes
        )
        projected_future_move = cohort["projected_future_max_away_move_pct"].to_numpy(
            dtype=float
        )
        historical_future_move = cohort["historical_future_max_away_move_pct"].to_numpy(
            dtype=float
        )
        tail_start_index = max(0, as_of_index - 480 + 1)
        signal_context_start_index = max(0, signal_index - 96)
        actual_start_index = min(tail_start_index, signal_context_start_index)
        return {
            "schema_version": "1.1.0",
            "method": "state-conditioned empirical historical trajectory scenarios",
            "conditionality": "Every projected path is a rescaled real historical episode that eventually fully fills its wick; this is not an unconditional fill probability or a trade recommendation.",
            "projection_semantics": {
                "candidate_availability": "An analogue is eligible only when its terminal fill candle closed at or before the pinned observation snapshot closed.",
                "departure": "The pinned signal and analogue alignment must have a close at or beyond the opposite signal extreme; analogue alignment offsets before that departure are excluded.",
                "future_excursion_window": "Future move-away is the direction-normalized high/low candle envelope after the snapshot through and including the terminal fill candle.",
                "additional_adverse_denominator": "Additional adverse percentages are measured from the observable snapshot close, matching the numerical risk model; peak move-away remains measured from the wick target.",
                "intrabar_note": "OHLC cannot establish whether a terminal fill-candle extreme happened before or after the wick touch; this is a consistent candle-envelope measurement.",
            },
            "matching": {
                "mode": matching_mode,
                "baseline_version": (
                    SOFT_MATCHER_BASELINE_VERSION
                    if matching_mode == "blended"
                    else None
                ),
                "label": (
                    "Hard candle archetype"
                    if matching_mode == "archetype"
                    else (
                        "Soft archetype blend"
                        if matching_mode == "blended"
                        else (
                            "Legacy V3 comparison"
                            if matching_mode == "legacy_v3"
                            else "Adaptive live path"
                        )
                    )
                ),
                "direction_policy": (
                    "upper/lower mirrored"
                    if matching_mode in {"blended", "archetype"}
                    else "category-aware"
                ),
                "signal_configuration_pool_size": (
                    min(ARCHETYPE_POOL_SIZE, len(trajectory_eligible))
                    if matching_mode == "archetype"
                    else (
                        len(trajectory_eligible)
                        if matching_mode == "blended"
                        else None
                    )
                ),
                "archetype_blend_weight": (
                    ARCHETYPE_BLEND_WEIGHT
                    if matching_mode == "blended"
                    else None
                ),
                "archetype_component_scales": (
                    ARCHETYPE_COMPONENT_SCALES
                    if matching_mode in {"blended", "archetype"}
                    else None
                ),
                "pinned_archetype": (
                    archetype_profile(signal)
                    if matching_mode in {"blended", "archetype"}
                    else None
                ),
                "score_component_summary": projection.get(
                    "score_component_summary", {}
                ),
            },
            "library": {
                "directory": str(self._library_dir_for_timeframe(asset, timeframe)),
                "availability_cutoff_close_utc": utc_iso(snapshot_close_time_ms),
                "eligible_completed_episodes_before_snapshot": int(len(eligible)),
                "same_timeframe_trajectory_candidates_before_snapshot": int(
                    len(trajectory_eligible)
                ),
                "top_k_state_matched_episodes": int(len(cohort)),
                "matching_backend": native_state_index.backend,
                "matching_features": FEATURE_WEIGHTS,
                "matching_state_weights": STATE_WEIGHTS,
                "category_penalties": CATEGORY_PENALTIES,
            },
            "pinned_signal": {
                "asset": asset,
                "timeframe": timeframe,
                "signal_open_time_utc": utc_iso(signal_time_ms),
                "direction": str(signal["direction"]),
                "direction_sign": direction_sign,
                "wick_target": target,
                "signal_open": float(signal["open"]),
                "signal_high": float(signal["high"]),
                "signal_low": float(signal["low"]),
                "signal_close": float(signal["close"]),
                "signal_features": {
                    field: float(signal[field]) for field in FEATURE_COLUMNS
                },
            },
            "current_state": {
                **{key: round(float(value), 8) for key, value in current_state.items()},
                "as_of_open_time_utc": utc_iso(
                    int(frame["open_time"].iat[as_of_index])
                ),
                "as_of_close_time_utc": utc_iso(snapshot_close_time_ms),
                "as_of_close": float(frame["close"].iat[as_of_index]),
            },
            "cohort_distribution": {
                "remaining_time_to_fill_minutes": {
                    "p25": float(np.quantile(remaining_minutes, 0.25)),
                    "p50": float(np.quantile(remaining_minutes, 0.50)),
                    "p90": float(np.quantile(remaining_minutes, 0.90)),
                },
                "projected_future_max_away_move_pct": {
                    "p50": float(np.quantile(projected_future_move, 0.50)),
                    "p90": float(np.quantile(projected_future_move, 0.90)),
                },
                "historical_future_max_away_move_pct": {
                    "p50": float(np.quantile(historical_future_move, 0.50)),
                    "p90": float(np.quantile(historical_future_move, 0.90)),
                },
            },
            "actual_window": {
                "start_open_time_utc": utc_iso(
                    int(frame["open_time"].iat[actual_start_index])
                ),
                "end_open_time_utc": utc_iso(int(frame["open_time"].iat[as_of_index])),
                "signal_context_bars": 96,
                "signal_candle_index_in_window": int(signal_index - actual_start_index),
            },
            "actual_candles": actual_candles(frame, actual_start_index, as_of_index),
            "scenarios": projection["scenarios"],
            "v3_route": v3_route,
            "performance": {
                "frame_lookup_ms": round((frame_loaded - started) * 1000, 3),
                "signal_lookup_ms": round((signals_loaded - frame_loaded) * 1000, 3),
                "library_lookup_ms": round((library_loaded - signals_loaded) * 1000, 3),
                "path_projection_ms": round(
                    (projection_built - library_loaded) * 1000, 3
                ),
            },
            "top_matches": top_matches,
            "limitations": [
                "Scenario paths are conditional on eventual fill and are not unconditional price forecasts.",
                "Direct projected candles use only the pinned timeframe; mixing historical 15m bars into a 5m candle path would invent intrabar detail and distort time.",
                "The current feature weights are deliberately transparent starting values; they must be learned or tuned only inside chronological replay folds.",
                "The scenarios preserve real joint historical trajectories but still need path-coverage walk-forward validation before risk use.",
                "The terminal fill-candle risk uses a complete OHLC envelope; finer data is required to order an intrabar wick touch and extreme exactly.",
            ],
        }

    @staticmethod
    def _with_visual_aggregation(
        payload: dict[str, Any], source_timeframe: str, display_timeframe: str
    ) -> dict[str, Any]:
        """Return a display-sized copy without changing the cached raw projection."""
        _, display_minutes = display_timeframe_minutes(
            display_timeframe, source_timeframe
        )
        source_minutes = int(source_timeframe.removesuffix("m"))
        response = {
            **payload,
            "visualization": {
                "source_timeframe": source_timeframe,
                "display_timeframe": display_timeframe,
                "display_minutes": display_minutes,
                "server_aggregated": display_minutes > source_minutes,
            },
        }
        if display_minutes <= source_minutes:
            return response

        actual = payload["actual_candles"]
        display_actual, _ = aggregate_visual_candles(actual, [], display_minutes)
        display_scenarios: list[dict[str, Any]] = []
        for scenario in payload["scenarios"]:
            visual_candles, has_transition = aggregate_visual_candles(
                actual, scenario["projected_candles"], display_minutes
            )
            display_scenario = {
                key: value
                for key, value in scenario.items()
                if key != "projected_candles"
            }
            display_scenario["visual_candles"] = visual_candles
            display_scenario["visual_has_transition"] = has_transition
            display_scenarios.append(display_scenario)
        response["actual_candles"] = display_actual
        response["scenarios"] = display_scenarios
        return response

    def _display_response(
        self, payload: dict[str, Any], source_timeframe: str, display_timeframe: str
    ) -> dict[str, Any]:
        started = time.perf_counter()
        response = self._with_visual_aggregation(
            payload, source_timeframe, display_timeframe
        )
        response["performance"] = {
            **response["performance"],
            "visualization_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        return response

    def scenarios(
        self,
        asset: str,
        timeframe: str,
        signal_time: str,
        display_timeframe: str | None = None,
        matching_mode: str = "adaptive",
    ) -> dict[str, Any]:
        validate_asset_timeframe(asset, timeframe)
        matching_mode = validate_matching_mode(matching_mode)
        display_timeframe, _ = display_timeframe_minutes(display_timeframe, timeframe)
        signal_time_ms = parse_utc(signal_time)
        stamp = self._stamp(asset, timeframe)
        key = (asset, timeframe, signal_time, matching_mode, stamp)
        with self.lock:
            cached = self.scenario_cache.get(key)
        if cached is not None:
            return self._display_response(cached, timeframe, display_timeframe)
        started = time.perf_counter()
        payload = self._v1_payload(
            asset, timeframe, signal_time_ms, matching_mode=matching_mode
        )
        v1_built = time.perf_counter()
        payload["v2_risk"] = self._v2_risk(asset, timeframe, signal_time_ms)
        payload["performance"]["v2_risk_ms"] = round(
            (time.perf_counter() - v1_built) * 1000, 3
        )
        prospective_started = time.perf_counter()
        prospective_risk = self._prospective_risk(asset, timeframe, signal_time_ms)
        route_candidates = prospective_risk.pop("_historical_route_candidates", {})
        payload["prospective_risk"] = prospective_risk
        if matching_mode == "adaptive":
            payload["route_engine"] = self._apply_c2_numerical_routes(
                asset,
                timeframe,
                signal_time_ms,
                payload,
                prospective_risk,
                route_candidates,
            )
        else:
            payload["route_engine"] = {
                "available": True,
                "active": False,
                "reason": "The frozen C2 + E0/A route engine is used only by Adaptive mode.",
            }
        payload["performance"]["prospective_risk_ms"] = round(
            (time.perf_counter() - prospective_started) * 1000, 3
        )
        payload["performance"]["total_uncached_ms"] = round(
            (time.perf_counter() - started) * 1000, 3
        )
        with self.lock:
            self.scenario_cache[key] = payload
        return self._display_response(payload, timeframe, display_timeframe)


ENGINE = Engine(ROOT)


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class LiveRefreshCoordinator:
    """Keep local completed-candle sources current while the dashboard runs."""

    def __init__(
        self,
        engine: Engine,
        interval_seconds: int,
        enabled: bool,
        refresh_close_guard_seconds: int = 5,
    ) -> None:
        if interval_seconds < 60:
            raise ValueError("refresh interval must be at least 60 seconds")
        if (
            refresh_close_guard_seconds < 0
            or refresh_close_guard_seconds >= interval_seconds
        ):
            raise ValueError(
                "refresh close guard must be non-negative and shorter than the interval"
            )
        self.engine = engine
        self.interval_seconds = interval_seconds
        self.enabled = enabled
        self.refresh_close_guard_seconds = refresh_close_guard_seconds
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.state: dict[str, Any] = {
            "enabled": enabled,
            "interval_seconds": interval_seconds,
            "refresh_close_guard_seconds": refresh_close_guard_seconds,
            "in_progress": False,
            "last_attempt_utc": None,
            "last_success_utc": None,
            "last_error": None,
            "next_refresh_utc": None,
            "last_results": {},
        }

    def _seconds_until_next_boundary(self) -> float:
        now = time.time()
        next_boundary = (
            int((now - self.refresh_close_guard_seconds) // self.interval_seconds) + 1
        ) * self.interval_seconds + self.refresh_close_guard_seconds
        return max(1.0, next_boundary - now)

    def _set_next_refresh(self) -> None:
        if not self.enabled:
            return
        next_timestamp = time.time() + self._seconds_until_next_boundary()
        next_iso = (
            datetime.fromtimestamp(next_timestamp, tz=timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z")
        )
        with self.lock:
            self.state["next_refresh_utc"] = next_iso

    def start(self) -> None:
        if not self.enabled or self.thread is not None:
            return
        self.thread = threading.Thread(
            target=self._run, name="conditional-wick-live-refresh", daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)

    def _run(self) -> None:
        # Refresh immediately to catch up a server restarted between five-minute
        # boundaries, then align every later attempt to the next boundary.
        while not self.stop_event.is_set():
            self.refresh_once()
            self._set_next_refresh()
            if self.stop_event.wait(self._seconds_until_next_boundary()):
                break

    def refresh_once(self) -> bool:
        """Refresh each configured source once. A failed source keeps its old CSV."""
        if not self.enabled:
            return False
        with self.lock:
            if self.state["in_progress"]:
                return False
            self.state["in_progress"] = True
            self.state["last_attempt_utc"] = _utc_now_iso()
            self.state["last_error"] = None

        results: dict[str, dict[str, Any]] = {}
        changed = False
        errors: list[str] = []
        try:
            try:
                shared_server_time_ms = exchange_server_time_ms()
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                for _asset, _interval, source_key in LIVE_SOURCE_SPECS:
                    results[source_key] = {
                        "ok": False,
                        "error": f"Binance server time unavailable: {message}",
                    }
                errors.append(f"Binance server time: {message}")
            else:
                for asset, interval, source_key in LIVE_SOURCE_SPECS:
                    try:
                        result: RefreshResult = refresh_source(
                            asset,
                            self.engine._source_path(asset, interval),
                            interval=interval,
                            server_time_ms=shared_server_time_ms,
                        )
                        result_payload = result.as_payload()
                        results[source_key] = {"ok": True, **result_payload}
                        if result.changed:
                            self.engine.apply_source_append(asset, interval, result)
                        changed = changed or result.changed
                    except Exception as error:  # Keep the last verified source intact and expose the exact asset status.
                        message = f"{type(error).__name__}: {error}"
                        results[source_key] = {"ok": False, "error": message}
                        errors.append(f"{source_key}: {message}")
            if changed:
                self.engine.invalidate_source_caches()
        finally:
            finished_at = _utc_now_iso()
            with self.lock:
                self.state["in_progress"] = False
                self.state["last_results"] = results
                self.state["last_error"] = "; ".join(errors) if errors else None
                if not errors:
                    self.state["last_success_utc"] = finished_at
        return changed

    def status(self) -> dict[str, Any]:
        with self.lock:
            payload = dict(self.state)
            results = dict(self.state["last_results"])
        sources: dict[str, dict[str, Any]] = {}
        version_parts: list[str] = []
        for asset, interval, source_key in LIVE_SOURCE_SPECS:
            source = self.engine._source_path(asset, interval)
            try:
                snapshot = source_snapshot(source)
                source_payload = snapshot.as_payload()
                sources[source_key] = {
                    "available": True,
                    "last_closed_candle_open_time_utc": source_payload[
                        "last_closed_candle_open_time_utc"
                    ],
                    "source_mtime_ns": snapshot.mtime_ns,
                    "last_rows_added": results.get(source_key, {}).get("rows_added"),
                    "last_refresh_result": results.get(source_key),
                }
                version_parts.append(f"{source_key}:{snapshot.mtime_ns}")
            except Exception as error:  # pragma: no cover - defensive status boundary
                sources[source_key] = {
                    "available": False,
                    "error": f"{type(error).__name__}: {error}",
                }
                version_parts.append(f"{source_key}:unavailable")
        payload["data_version"] = "|".join(version_parts)
        payload["sources"] = sources
        return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "ConditionalWickDashboard/1.0"

    def _json(
        self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK
    ) -> None:
        encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def _html(self) -> None:
        encoded = HTML.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        try:
            if parsed.path == "/":
                self._html()
            elif parsed.path == "/api/health":
                refresher = getattr(self.server, "refresh_coordinator", None)
                self._json(
                    {
                        "ok": True,
                        "service": self.server_version,
                        "default": DEFAULT_SIGNAL,
                        "refresh": refresher.status()
                        if refresher is not None
                        else {"enabled": False},
                    }
                )
            elif parsed.path == "/api/refresh-status":
                refresher = getattr(self.server, "refresh_coordinator", None)
                if refresher is None:
                    raise ValueError("Live refresh coordinator is not configured")
                self._json(refresher.status())
            elif parsed.path == "/api/signals":
                asset = query.get("asset", [DEFAULT_SIGNAL["asset"]])[0]
                timeframe = query.get("timeframe", [DEFAULT_SIGNAL["timeframe"]])[0]
                validate_asset_timeframe(asset, timeframe)
                self._json(ENGINE.available_unfilled_signals(asset, timeframe))
            elif parsed.path == "/api/scenarios":
                asset = query.get("asset", [DEFAULT_SIGNAL["asset"]])[0]
                timeframe = query.get("timeframe", [DEFAULT_SIGNAL["timeframe"]])[0]
                signal_time = query.get("signal_time", [DEFAULT_SIGNAL["signal_time"]])[
                    0
                ]
                display_timeframe = query.get("display_timeframe", [timeframe])[0]
                matching_mode = query.get("matching_mode", ["adaptive"])[0]
                self._json(
                    ENGINE.scenarios(
                        asset,
                        timeframe,
                        signal_time,
                        display_timeframe,
                        matching_mode,
                    )
                )
            elif parsed.path == "/favicon.ico":
                self.send_response(HTTPStatus.NO_CONTENT)
                self.end_headers()
            else:
                self._json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
        except (ValueError, FileNotFoundError) as error:
            self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception as error:  # pragma: no cover - defensive HTTP boundary
            self._json(
                {"error": f"Internal dashboard error: {type(error).__name__}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8793)
    parser.add_argument(
        "--refresh-seconds",
        type=int,
        default=60,
        help="Completed-candle refresh cadence while the service runs (default: 60).",
    )
    parser.add_argument(
        "--disable-auto-refresh",
        action="store_true",
        help="Serve the existing local source without contacting Binance; useful for deterministic replay.",
    )
    parser.add_argument(
        "--refresh-close-guard-seconds",
        type=int,
        default=5,
        help="Wait this many seconds after each refresh boundary before polling Binance (default: 5).",
    )
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    refresher = LiveRefreshCoordinator(
        ENGINE,
        args.refresh_seconds,
        enabled=not args.disable_auto_refresh,
        refresh_close_guard_seconds=args.refresh_close_guard_seconds,
    )
    server.refresh_coordinator = refresher
    refresher.start()
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}/",
                "default_signal": DEFAULT_SIGNAL,
                "refresh_seconds": args.refresh_seconds,
                "refresh_close_guard_seconds": args.refresh_close_guard_seconds,
                "auto_refresh": not args.disable_auto_refresh,
            }
        ),
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        refresher.stop()
        server.server_close()


if __name__ == "__main__":
    main()
