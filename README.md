# Conditional Wick Path Engine

Local research and dashboard tooling for studying strict ETHUSDT/BTCUSDT wick candles that move away from their wick target and later cleanly fill it.

The main dashboard lets you pin an unfilled wick and inspect three rescaled, real historical candle trajectories:

- **Fast** — lower joint time/excursion risk.
- **Normal** — the centre of the matched historical cohort.
- **Extreme** — upper-tail joint time/excursion risk.

Each route is conditional on the historical episode eventually filling its wick. It is a scenario and risk-sizing research tool, not an unconditional fill-probability model or trading system.

## Run the dashboard

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python .\serve_conditional_wick_dashboard.py --host 127.0.0.1 --port 8793 --refresh-seconds 300
```

Open `http://127.0.0.1:8793/`.

The running service catches up the local ETHUSDT/BTCUSDT sources immediately, then appends newly completed Binance Futures 5-minute candles every five minutes. The dashboard shows the source timestamp and next-refresh countdown, and recalculates the pinned route when fresh data arrives. See [LIVE_DASHBOARD.md](LIVE_DASHBOARD.md) for the live-update boundary and one-shot recovery command.

The live dashboard requires locally generated five-year candle data and the conditional path library. They are deliberately not committed to Git; use the downloader and library-builder scripts to recreate them:

```powershell
python .\download_futures_klines.py --help
python .\build_conditional_path_library.py --help
```

## Layout

- `refresh_futures_klines.py` - atomic incremental completed-candle updater.

- `serve_conditional_wick_dashboard.py` — local pin-and-project service.
- `build_conditional_path_scenarios.py` — Fast / Normal / Extreme real-path selector.
- `build_conditional_path_library.py` — completed strict-wick episode library builder.
- `train_conditional_wick_v2.py` — optional conditional risk-quantile layer.
- `evaluate_conditional_path_replay.py` and `verify_conditional_path_engine.py` — replay and integrity checks.
- `CONDITIONAL_PATH_ENGINE.md`, `LIVE_DASHBOARD.md`, and `V2_CALIBRATION.md` — method and operating notes.

## Recorded empirical observations

The current wick-touch audit, including the strict and near-strict definitions, the fully observed 180-day results, and the July 2026 lower-wick examples is recorded in [WICK_TOUCH_OBSERVATIONS.md](WICK_TOUCH_OBSERVATIONS.md).

## Version-control policy

Source, documentation, dependency manifests, and the Rust source tree are tracked. Downloaded candles, generated path libraries, trained models, chart exports, and Rust build output are ignored because they are large and reproducible.

## Validation

With the local data artifacts available:

```powershell
python -m py_compile .\serve_conditional_wick_dashboard.py .\refresh_futures_klines.py .\build_conditional_path_scenarios.py .\train_conditional_wick_v2.py
python .\verify_conditional_path_engine.py
python .\verify_live_refresh.py
```
