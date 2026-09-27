# Live Conditional Wick Dashboard

The local dashboard turns the V1 conditional path engine into a pin-and-project tool.

## Open it

It is running separately from the existing generic Rust forecast dashboard:

- Wick-path dashboard: `http://127.0.0.1:8793/`
- Existing generic Rust dashboard: `http://127.0.0.2:8792/` (unchanged)

To start it again after a reboot or shutdown:

```powershell
cd F:\explore\candle_projection_algo
.venv\Scripts\python.exe .\serve_conditional_wick_dashboard.py --host 127.0.0.1 --port 8793
```

## What the controls do

1. Select `ETHUSDT`, `BTCUSDT`, `SOLUSDT`, `UNIUSDT`, or `NEARUSDT`, then `1m`, `5m`, or `15m`. Each asset has an isolated `1m` V1 experiment.
2. Pick a recent unfilled strict wick from the selector, or paste a UTC ISO timestamp.
3. Choose `Adaptive live path`, the opt-in `Soft archetype blend`, or `Hard candle archetype`.
4. Click **Project paths**.

The server rejects a timestamp unless it is a strict wick signal, remains unfilled as of the local data snapshot, and is on the away-from-wick side needed for a conditional path projection.

The browser receives three paths:

- **Fast** — a real comparable episode near the lower joint duration/projected-excursion score percentile.
- **Normal** — a real V3 learned continuation inside a validated 1m/5m age regime; otherwise the V1 cohort centre.
- **Extreme** — a real comparable episode near the upper joint duration/projected-excursion score percentile.

The candles in a displayed route are one rescaled historical OHLC episode. The actual-price window includes 96 candles before the pinned signal and every observed candle through the snapshot, so the signal candle and its move-away leg remain inspectable. The service does not splice candles, fabricate intrabar 5-minute detail from a 15-minute path, or present the result as a fill probability.

The primary **Risk from current price** card uses frozen architecture `forecast_architecture_v1_5m_2026-09-27` on every 5m pair: E0 owns fill odds and adverse p50, A owns waiting time and adverse p80/p90, and C2 supplies a separate Historical Support disclosure. One-minute views retain the previous all-outcome model until the same rolling evaluation is run on 1m. If a numerical artifact is unavailable, the older clean-fill V2a card remains a 5m-only fallback. Neither numerical layer changes the historical candles drawn by Fast/Normal/Extreme.

`Soft archetype blend` keeps every eligible path. Live distance, peak, retracement, and age remain primary; signal-feature similarity is blended from 70% adaptive robust distance and 30% direction-mirrored candle configuration, with a softened same-asset preference. `Hard candle archetype` retains only the 240 closest configurations and removes asset and direction penalties. Both experiments disable V3 so it cannot replace their Normal route with a broader learned match.

Both modes are deliberately labelled experimental. The hard gate was materially worse than adaptive matching across 215 paired snapshots. A smaller soft-blend pilot improved mean path error by 1.8%, but its uncertainty range still crosses zero. See [CANDLE_ARCHETYPE_EXPERIMENT.md](CANDLE_ARCHETYPE_EXPERIMENT.md) and [SOFT_ARCHETYPE_BLEND_PILOT.md](SOFT_ARCHETYPE_BLEND_PILOT.md).

V3 is a route selector, not the range card. It replaces only the Normal route after loading a validated local age expert. Fast and Extreme remain V1. The current live gates are 1–480 and 481–2,880 bars on 5m, plus 1–2,400 and 2,401–14,400 bars on 1m. Fresh experts use forecast-anchored retrieval; the mature 1m expert uses the learned sequence embedding directly. Pins outside those ranges, all 15m pins, missing artifacts, and any V3 runtime error automatically retain V1. The Observed state panel shows whether the current Normal route came from V3 or V1.

## Data and live refresh

While the dashboard service is running, it immediately catches the local Binance USD-M Futures sources up and then checks Binance every minute, five seconds after each UTC minute boundary by default. This includes the five 5m sources and all five 1m sources. A selected 1m source can append a fresh completed candle on every cycle; a 5m source advances only after its next five-minute candle has closed. The small post-close guard keeps each source on completed exchange bars. It asks Binance only for candles after the last local completed bar. It will append a complete, contiguous response atomically; incomplete candles, duplicate timestamps, and gaps are rejected without changing the prior source.

The header shows the last completed source candle and a live countdown to the next refresh. The browser polls the service status every 15 seconds. When the selected source generation changes, it reloads the current unfilled-signal list and re-runs the currently pinned route. Updates in another asset or timeframe do not cause an unnecessary route calculation. If fresh price action fills that pin, the server reports it rather than silently replacing the candle you chose.

```powershell
cd F:\explore\candle_projection_algo
.venv\Scripts\python.exe .\serve_conditional_wick_dashboard.py --host 127.0.0.1 --port 8793 --refresh-seconds 60
```

Use `--disable-auto-refresh` only for deterministic replay against a fixed local snapshot. `refresh_futures_klines.py` is also available for a one-shot catch-up. Its ignored `data/*.refresh.json` sidecar records each source-refresh outcome.

The initial sources are five-year snapshots, and the updater only appends new completed bars; it does not discard older history. The V1 episode library is intentionally not rebuilt every minute: live refresh immediately recalculates the pin's observed state, eligibility, actual candles, and selected route only when its selected source gains a completed candle, while newly resolved historical episodes join the library on its next explicit rebuild. This avoids replacing a displayed route with a partially rebuilt historical cohort.

V2a is also a static local artifact. The dashboard can reload a manually
rebuilt model and calibration summary, but the one-minute source refresh does
not rebuild the episode library or retrain V2a. Its card therefore shows the
artifact generation time, its training-label cutoff, and whether the pin age
matches an exact sampled snapshot age, falls between sampled ages, or lies
outside the sampled range.

The all-outcome 1m/5m numerical artifacts are also static. By default they live under `%LOCALAPPDATA%\candle_projection_algo\prospective_entry_models\<timeframe>\model.joblib`. Frozen 5m V1 is selected through `5m\active.json`, which pins its immutable versioned filename and SHA-256; the dashboard verifies both before loading it. Override the root with `CANDLE_PROJECTION_MODEL_DIR`. The manual maintenance command rebuilds the inputs and trains a new V1 candidate, but cannot replace or repoint the active frozen artifact. Frozen-v1 snapshots are deduplicated into `%LOCALAPPDATA%\candle_projection_algo\prospective_validation\forecasts.sqlite` with architecture version, artifact hash and actual forecast source. The `forecast_v1_evaluation` view excludes legacy-fallback rows from later prospective scoring.

## Manual learning update

After enough new wicks have resolved, run:

```powershell
cd F:\explore\candle_projection_algo
.venv\Scripts\python.exe .\wick_update.py update --dry-run
.venv\Scripts\python.exe .\wick_update.py update
```

This is a substantial CPU, RAM and disk-I/O workload and is best run overnight. It updates all five assets at 1m/5m, completed-route libraries, native caches, all-outcome observations and the gated numerical risk models. It does not retrain V3 neural age experts or use the GPU. Libraries are built in staging before each validated swap; if a later stage fails, already completed stages remain updated and rerunning the command safely converges the workflow. See [MODEL_UPDATE_WORKFLOW.md](MODEL_UPDATE_WORKFLOW.md) for the complete operating contract and run-manifest locations.

For every asset at 1m, the selected route remains fully native one-minute data. When a wider `Display candles` value is selected, the service aggregates only the chart payload before sending it to the browser; it does not alter matching, route selection, risk metrics, or the terminal wick touch. This prevents a long extreme route from transferring or rendering tens of thousands of native bars just to aggregate them in the browser. Historical alignment snapshots remain exact through four hours after a signal, sampled every five minutes through one day, and every fifteen minutes thereafter.

Build the ignored binary runtime cache once for each dense 1m asset library:

```powershell
python .\build_native_runtime_cache.py --timeframe 1m --asset ETHUSDT
```

The service fingerprint-checks each cache and falls back safely if it no longer matches its asset library. Once a 1m library is warm, each atomic one-minute source append updates the in-memory raw frame and recomputes from the last cached source generation plus the required detector lookback, rather than reparsing five years of CSV data. Scenario results remain invalidated on a fresh candle, so the displayed state stays current.

## Projection semantics

For a current pin, analogue episodes are eligible when their terminal fill
candle closed at or before the current snapshot closes. Analogue alignment
states before their own confirmed departure are excluded. Scenario selection
uses each analogue's rescaled projected-coordinate future risk, not its
unscaled historical percentage.

The future-risk window is the OHLC envelope after the current snapshot through
the terminal fill candle. The route cards separate peak distance from the wick
target from additional adverse movement from the current close. Because the
source is OHLC, a terminal candle's touch-versus-extreme order remains an
intrabar ambiguity rather than a claimed exact sequence.

## Risk boundary

The evaluation population is conditional on strict wick signals that had a clean departure and eventually filled inside the 180-day library cap. It is not an unconditional fill-rate model or a trading system. See [CONDITIONAL_PATH_ENGINE.md](CONDITIONAL_PATH_ENGINE.md), [METHOD_SELECTION.md](METHOD_SELECTION.md), and [PATH_REPLAY_VALIDATION.md](PATH_REPLAY_VALIDATION.md) for definitions and validation limits.
