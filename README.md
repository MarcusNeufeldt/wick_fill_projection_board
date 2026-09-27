# Conditional Wick Path Engine

Local research and dashboard tooling for studying strict ETHUSDT, BTCUSDT, SOLUSDT, UNIUSDT, and NEARUSDT wick candles that move away from their wick target and later cleanly fill it. The live research scope is pooled across all five assets at 1m and 5m. Existing 15m paths remain available to V1, but 15m neural training is deliberately out of scope.

The main dashboard lets you pin an unfilled wick and inspect four rescaled, real historical candle trajectories in Adaptive 5m:

- **Fast** — a C2 historical continuation aligned to the frozen optimistic waiting-time target and central adverse-risk target.
- **Normal** — a C2 historical continuation aligned to the frozen median waiting-time and adverse-risk targets.
- **Adverse first** — a C2 continuation that reaches the 1-day p80 adverse level before later filling the wick.
- **Extreme** — a C2 historical continuation aligned to the frozen long-wait and adverse-risk tail targets.

Each route is conditional on the historical episode eventually filling its wick. It is a scenario and risk-sizing research tool, not an unconditional fill-probability model or trading system.

## Run the dashboard

Python 3.11+ is recommended. Adaptive 5m routes use the frozen E0/A/C2 artifact. V3 remains available only as an explicit legacy comparison and requires its compatible local artifacts and PyTorch.

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-ml.txt
.venv\Scripts\python.exe .\serve_conditional_wick_dashboard.py --host 127.0.0.1 --port 8793 --refresh-seconds 60
```

Open `http://127.0.0.1:8793/`.

## Choosing a matcher and route

The matcher and the displayed route answer two different questions:

1. The **matcher** decides which historical wick episodes are comparable to the pinned setup.
2. **Fast / Normal / Adverse first / Extreme** select representative real continuations from that matched population.

### Matcher choices

| Matcher | What it emphasizes | Recommended use |
| --- | --- | --- |
| **Adaptive live path** | On 5m, C2 retrieves historically comparable observable states and E0/A time-risk estimates choose four real continuations; on 1m it currently retains the V1 state matcher | **Default for normal use** |
| **Legacy V3 comparison** | The former age-gated TCN route selector replaces only Normal while Fast and Extreme remain V1 | Explicit research comparison only |
| **Soft archetype blend** | The complete adaptive population with 70% adaptive signal-feature distance and 30% direction-mirrored candle-shape distance | Candle-geometry sensitivity experiment |
| **Hard candle archetype** | Only the 240 closest signal-candle configurations, with upper and lower wicks treated as mirrors | Research/debugging only; do not use as the primary forecast |

The pooled 5m library spans all five assets; each 1m library remains asset-specific. V3 is never inserted silently into Adaptive. Soft and Hard modes keep all three routes inside their selected V1 cohort.

### Route choices

In **Adaptive 5m**, C2 first supplies up to 96 distinct historical episodes for each numerical horizon. The route engine then chooses real continuations close to the frozen E0/A targets:

- **Fast** — 1-day waiting-time p10 plus 1-day adverse p50.
- **Normal** — 7-day waiting-time p50 plus 7-day adverse p50.
- **Adverse first** — a definite pre-fill crossing of the 1-day adverse p80 target, selected primarily by adverse-size compatibility rather than total waiting time.
- **Extreme** — 30-day waiting-time p90 plus 30-day adverse p90; use it as a stress reference, not a guaranteed worst case.

Duration compatibility, projected adverse compatibility and C2 fingerprint distance are scored together. Adverse-first candidates must cross p80 before the terminal fill candle; a same-candle crossing is rejected because OHLC cannot establish its ordering. Route adverse percentages are measured from the observable current entry price, matching the E0/A risk coordinates. The selected candles remain one rescaled historical suffix; E0/A do not generate candles. If the frozen artifact or enough drawable C2 episodes are unavailable, Adaptive fails closed to the V1 routes and reports why. One-minute Adaptive still uses V1 routes until an equivalent frozen 1m architecture is validated. Choose **Legacy V3 comparison** only when you deliberately want to inspect the old neural Normal-route selector.

For ordinary use, select **Adaptive live path** and treat **Normal** as the central illustration. Read **Adverse first** before sizing or placing a stop, and read Extreme as the combined long-wait/tail stress reference. The separate **Risk from current price** card shows fill odds, adverse p50/p80/p90, waiting-time ranges, 2%/5% target-first contests and Historical Support. Even though the 5m illustrations are now numerically aligned, they are still individual completed historical examples, not route probabilities or calibrated path envelopes.

The first chronological archetype replay did **not** beat the adaptive default. Across 215 paired snapshots from 110 distinct September-like episodes, its mean checkpoint path error was 2.848 percentage points versus 1.496 for adaptive matching. The damage was concentrated in older signals, where a hard candle-shape gate discarded useful long-path analogues. The control remains available for inspection, but the result is evidence against promoting this hard-gated version. See [CANDLE_ARCHETYPE_EXPERIMENT.md](CANDLE_ARCHETYPE_EXPERIMENT.md).

A 32-case pilot across 24 distinct episodes found the soft 30% blend 1.8% better on mean checkpoint path error, with a wide episode-clustered interval from -6.5% to +9.5%. That is promising enough for visual experimentation but not evidence to replace Adaptive. See [SOFT_ARCHETYPE_BLEND_PILOT.md](SOFT_ARCHETYPE_BLEND_PILOT.md).

The running service catches up the local ETHUSDT, BTCUSDT, SOLUSDT, UNIUSDT, and NEARUSDT 5m and 1m sources immediately, then checks Binance every minute for newly completed candles. A selected 1m source can therefore advance every minute; 5m sources advance when their next five-minute candle has closed. The dashboard shows the selected source timestamp and next-refresh countdown, and recalculates the pinned route only when that selected source changes. Episode libraries and numerical model artifacts remain static until explicitly rebuilt; source refresh alone does not retrain them. The promoted all-outcome risk card supports both 1m and 5m. See [LIVE_DASHBOARD.md](LIVE_DASHBOARD.md) for the live-update boundary and one-shot recovery command.

## Manually absorb newly resolved wicks

Run the maintenance CLI whenever enough new fills have accumulated, such as once a week:

```powershell
.venv\Scripts\python.exe .\wick_update.py update --dry-run
.venv\Scripts\python.exe .\wick_update.py update
```

The command refreshes all five Binance 1m/5m sources, rebuilds the pooled 5m and five isolated 1m completed-route libraries, regenerates their native caches, rebuilds the all-outcome observation dataset, and retrains both numerical risk models. It compiles one tracked Rust kernel and uses it for the expensive 1m/5m clean-route tracing, normalized path writing and forward outcome-label scans. Python independently verifies exact signal identity and retains the authoritative episode features, library summaries, dataset schemas and Parquet boundary. The compatibility-only derived 15m routes still use Python. A Rust contract or parity failure stops the manual update. Candidate risk models replace the installed artifacts only when their chronological promotion gate passes; otherwise the existing model remains installed. Every real run writes an ignored JSON manifest under `data/wick_update_runs/` with episode deltas, model decisions, timings, and failures. Each library uses a staging directory, so an interruption during its build leaves that library's last complete version installed; already completed earlier stages remain updated.

The experimental V3 neural age experts are deliberately outside this maintenance command because V3 is now a legacy comparison. Newly completed episodes become available to V1 and, after rebuilding the frozen/candidate forecast artifact, to C2 route retrieval.

This remains an overnight-style CPU, RAM and disk-I/O job because route-library rebuilding and training still dominate; the Rust label scan shortens one stage but does not make the entire update instant. The default command does not use the GPU. See [MODEL_UPDATE_WORKFLOW.md](MODEL_UPDATE_WORKFLOW.md) for exact layer boundaries, resource expectations, interruption recovery and promotion semantics.

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
- `forecast_architecture_v1.py` and `train_forecast_architecture_v1.py` — frozen 5m production composition: E0 fill/adverse p50, A waiting/tail risk, and C2 historical support plus candidate routes.
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
