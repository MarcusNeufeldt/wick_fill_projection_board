# Conditional Wick Path Engine

Local research and dashboard tooling for studying strict ETHUSDT, BTCUSDT, SOLUSDT, UNIUSDT, and NEARUSDT wick candles that move away from their wick target and later cleanly fill it. The live research scope is pooled across all five assets at 1m and 5m. Existing 15m paths remain available to V1, but 15m neural training is deliberately out of scope.

The main dashboard lets you pin an unfilled wick and inspect three rescaled, real historical candle trajectories:

- **Fast** — lower joint time/excursion risk.
- **Normal** — an age-gated V3 learned match to a real historical continuation on validated 1m/5m regimes; otherwise the V1 cohort centre.
- **Extreme** — upper-tail joint time/excursion risk.

Each route is conditional on the historical episode eventually filling its wick. It is a scenario and risk-sizing research tool, not an unconditional fill-probability model or trading system.

## Run the dashboard

Python 3.11+ is recommended. Use the project ML environment to enable the optional V3 Normal-route selector; without compatible V3 artifacts or PyTorch, the service safely uses V1.

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-ml.txt
.venv\Scripts\python.exe .\serve_conditional_wick_dashboard.py --host 127.0.0.1 --port 8793 --refresh-seconds 60
```

Open `http://127.0.0.1:8793/`.

The **Matcher** control keeps `Adaptive live path` as the production default. `Soft archetype blend` is frozen as research baseline `soft-state-v1.0.0`: it retains the complete eligible path pool and live state match, treats upper and lower wicks as mirrors, and gives the pinned candle configuration a 30% share of signal-feature similarity. Its API response records the baseline version and additive score contributions. `Hard candle archetype` instead restricts matching to the 240 closest signal-candle configurations. The pooled 5m library spans all five assets; each 1m library remains asset-specific. Both experiments disable the V3 Normal-route override so all three displayed routes come from one consistent V1 cohort.

The first chronological archetype replay did **not** beat the adaptive default. Across 215 paired snapshots from 110 distinct September-like episodes, its mean checkpoint path error was 2.848 percentage points versus 1.496 for adaptive matching. The damage was concentrated in older signals, where a hard candle-shape gate discarded useful long-path analogues. The control remains available for inspection, but the result is evidence against promoting this hard-gated version. See [CANDLE_ARCHETYPE_EXPERIMENT.md](CANDLE_ARCHETYPE_EXPERIMENT.md).

A 32-case pilot across 24 distinct episodes found the soft 30% blend 1.8% better on mean checkpoint path error, with a wide episode-clustered interval from -6.5% to +9.5%. That is promising enough for visual experimentation but not evidence to replace Adaptive. See [SOFT_ARCHETYPE_BLEND_PILOT.md](SOFT_ARCHETYPE_BLEND_PILOT.md).

The running service catches up the local ETHUSDT, BTCUSDT, SOLUSDT, UNIUSDT, and NEARUSDT 5m and 1m sources immediately, then checks Binance every minute for newly completed candles. A selected 1m source can therefore advance every minute; 5m sources advance when their next five-minute candle has closed. The dashboard shows the selected source timestamp and next-refresh countdown, and recalculates the pinned route only when that selected source changes. Episode libraries and numerical model artifacts remain static until explicitly rebuilt; source refresh alone does not retrain them. The promoted all-outcome risk card supports both 1m and 5m. See [LIVE_DASHBOARD.md](LIVE_DASHBOARD.md) for the live-update boundary and one-shot recovery command.

## Manually absorb newly resolved wicks

Run the maintenance CLI whenever enough new fills have accumulated, such as once a week:

```powershell
.venv\Scripts\python.exe .\wick_update.py update --dry-run
.venv\Scripts\python.exe .\wick_update.py update
```

The command refreshes all five Binance 1m/5m sources, rebuilds the pooled 5m and five isolated 1m completed-route libraries, regenerates their native caches, rebuilds the all-outcome observation dataset, and retrains both numerical risk models. Candidate risk models replace the installed artifacts only when their chronological promotion gate passes; otherwise the existing model remains installed. Every real run writes an ignored JSON manifest under `data/wick_update_runs/` with episode deltas, model decisions, timings, and failures. Each library uses a staging directory, so an interruption during its build leaves that library's last complete version installed; already completed earlier stages remain updated.

The experimental V3 neural age experts are deliberately outside this maintenance command because each expert needs its own chronological route-quality decision. Newly completed episodes still become immediately available to the historical V1/soft-state route engine, while the existing validated V3 experts remain unchanged until separately re-researched and promoted.

This is an overnight-style CPU, RAM and disk-I/O job; the default command does not use the GPU. See [MODEL_UPDATE_WORKFLOW.md](MODEL_UPDATE_WORKFLOW.md) for exact layer boundaries, resource expectations, interruption recovery and promotion semantics.

The live dashboard requires locally generated five-year candle data and the conditional path library. They are deliberately not committed to Git; use the downloader and library-builder scripts to recreate them:

```powershell
python .\download_futures_klines.py --help
python .\build_conditional_path_library.py --help
python .\build_sol_one_minute_path_library.py --asset ETHUSDT
python .\build_native_runtime_cache.py --timeframe 1m --asset ETHUSDT
```

## Layout

`build_sol_one_minute_path_library.py` builds an isolated 1m episode library for any configured asset without changing the five-asset 5m/15m library or its V2 calibration. V3 training pools those five isolated 1m libraries without materializing their multi-gigabyte path tables.

- `refresh_futures_klines.py` - atomic incremental completed-candle updater.
- `wick_update.py` - manual all-assets 1m/5m data, route-library, cache and gated risk-model update.

- `serve_conditional_wick_dashboard.py` — local pin-and-project service.
- `build_conditional_path_scenarios.py` — Fast / Normal / Extreme real-path selector.
- `build_conditional_path_library.py` — completed strict-wick episode library builder.
- `train_conditional_wick_v2.py` — optional conditional risk-quantile layer.
- `train_neural_path_v3.py` and `neural_path_v3_live.py` — GPU-capable pooled 1m/5m training plus optional age-gated Normal-route selectors.
- `prospective_entry_outcomes.py`, `prospective_entry_model.py`, and `train_prospective_entry_model.py` — all-outcome entry labels plus the promoted 1m/5m numerical risk and timing layer.
- `forecast_architecture_v1.py` and `train_forecast_architecture_v1.py` — frozen 5m production composition: E0 fill/adverse p50, A waiting/tail risk, and C2 historical support.
- `forecast_artifact_manifest.py` — immutable active-version pointer and SHA-256 verification for frozen forecast artifacts.
- `prospective_forecast_ledger.py` — append-only prospective forecast ledger with architecture version, artifact hash and actual forecast source, deduplicated by model version, signal and observation candle.
- `evaluate_behavioral_fingerprint.py` — research-only state/context/sequence fingerprint comparison; see `BEHAVIORAL_FINGERPRINT_POC.md`.
- `optimize_behavioral_fingerprint.py` — validation-only selection of horizon-specific fingerprint weights followed by one-time holdout scoring.
- `evaluate_rolling_forecast_systems.py` and `dynamic_neighborhood.py` — leakage-safe rolling comparison of supervised, dynamic-retrieval and validation-stacked forecasts, including effective-N historical-support diagnostics; see `ROLLING_FORECAST_EVALUATION.md`.
- `evaluate_conditional_path_replay.py` and `verify_conditional_path_engine.py` — replay and integrity checks.
- `CONDITIONAL_PATH_ENGINE.md`, `LIVE_DASHBOARD.md`, `V2_CALIBRATION.md`, and `NEURAL_PATH_V3.md` — method and operating notes.

The prospective-entry research layer is built by `prospective_entry_outcomes.py` and evaluated by `evaluate_prospective_entry_baseline.py`. It retains unresolved histories, measures adverse movement from the observable entry close, preserves intrabar ambiguity, and evaluates multi-day ages chronologically. See `PROSPECTIVE_ENTRY_RESEARCH.md`.

The live 5m dashboard now prefers frozen architecture `forecast_architecture_v1_5m_2026-09-27` for all five pairs through a hash-verified `active.json` manifest. One-minute views retain the previous prospective model until an equivalent 1m rolling evaluation is complete. Update runs may train versioned candidates but cannot repoint the frozen V1 during prospective collection. See `FORECAST_ARCHITECTURE_V1.md` for field ownership, artifact paths, fallback behavior and prospective logging.

The candle-type experiment is implemented in `candle_archetype.py` and replayed by `evaluate_candle_archetype_replay.py`; its generated research notes are `CANDLE_ARCHETYPE_EXPERIMENT.md` and `SOFT_ARCHETYPE_BLEND_PILOT.md`.

## Recorded empirical observations

The current wick-touch audit, including the strict and near-strict definitions, the fully observed 180-day results, and the July 2026 lower-wick examples is recorded in [WICK_TOUCH_OBSERVATIONS.md](WICK_TOUCH_OBSERVATIONS.md).

## Version-control policy

Source, documentation, dependency manifests, and the Rust source tree are tracked. Downloaded candles, generated path libraries, trained models, chart exports, and Rust build output are ignored because they are large and reproducible.

## Validation

With the local data artifacts available:

```powershell
.venv\Scripts\python.exe -m py_compile .\serve_conditional_wick_dashboard.py .\refresh_futures_klines.py .\wick_update.py .\build_conditional_path_scenarios.py .\candle_archetype.py .\evaluate_candle_archetype_replay.py .\prospective_entry_outcomes.py .\prospective_entry_model.py .\train_prospective_entry_model.py .\train_conditional_wick_v2.py .\train_neural_path_v3.py .\neural_path_v3_live.py
python .\verify_conditional_path_engine.py
python .\verify_live_refresh_service.py
.venv\Scripts\python.exe -m unittest discover -v
```
