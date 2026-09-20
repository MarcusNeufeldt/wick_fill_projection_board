# Live Conditional Wick Dashboard

The local dashboard turns the V1 conditional path engine into a pin-and-project tool.

## Open it

It is running separately from the existing generic Rust forecast dashboard:

- Wick-path dashboard: `http://127.0.0.1:8793/`
- Existing generic Rust dashboard: `http://127.0.0.2:8792/` (unchanged)

To start it again after a reboot or shutdown:

```powershell
cd F:\explore\candle_projection_algo
python .\serve_conditional_wick_dashboard.py --host 127.0.0.1 --port 8793
```

## What the controls do

1. Select `ETHUSDT` or `BTCUSDT` and `5m` or `15m`.
2. Pick a recent unfilled strict wick from the selector, or paste a UTC ISO timestamp.
3. Click **Project paths**.

The server rejects a timestamp unless it is a strict wick signal, remains unfilled as of the local data snapshot, and is on the away-from-wick side needed for a conditional path projection.

The browser receives three paths:

- **Fast** — a real comparable episode near the lower joint time/excursion risk percentile.
- **Normal** — a real comparable episode near the centre of the matched cohort.
- **Extreme** — a real comparable episode near the high joint time/excursion risk percentile.

The candles in a displayed route are one rescaled historical OHLC episode. The actual-price window includes 96 candles before the pinned signal and every observed candle through the snapshot, so the signal candle and its move-away leg remain inspectable. The service does not splice candles, fabricate intrabar 5-minute detail from a 15-minute path, or present the result as a fill probability.

## Data and live refresh

While the dashboard service is running, it immediately catches the local Binance USD-M Futures source up and then refreshes every five minutes, five seconds after each UTC candle boundary by default. The small post-close guard keeps both assets on the same completed exchange bar. It asks Binance only for candles after the last local completed bar. It will append a complete, contiguous response atomically; incomplete candles, duplicate timestamps, and gaps are rejected without changing the prior source.

The header shows the last completed source candle and a live countdown to the next refresh. The browser polls the service status every 15 seconds. When the source generation changes, it reloads the current unfilled-signal list and re-runs the currently pinned route. If fresh price action fills that pin, the server reports it rather than silently replacing the candle you chose.

```powershell
cd F:\explore\candle_projection_algo
python .\serve_conditional_wick_dashboard.py --host 127.0.0.1 --port 8793 --refresh-seconds 300
```

Use `--disable-auto-refresh` only for deterministic replay against a fixed local snapshot. `refresh_futures_klines.py` is also available for a one-shot catch-up. Its ignored `data/*.refresh.json` sidecar records each source-refresh outcome.

The initial sources are five-year snapshots, and the updater only appends new completed bars; it does not discard older history. The V1 episode library is intentionally not rebuilt every five minutes: live refresh immediately recalculates the pin's observed state, eligibility, actual candles, and selected route from the new source, while newly resolved historical episodes join the library on its next explicit rebuild. This avoids replacing a displayed route with a partially rebuilt historical cohort.

The first uncached pin normally takes roughly 20 seconds because it runs the transparent historical-path selector. Reopening a previously requested pin is cached only until the selected source or episode library changes.

## Risk boundary

The evaluation population is conditional on strict wick signals that had a clean departure and eventually filled inside the 180-day library cap. It is not an unconditional fill-rate model or a trading system. See [CONDITIONAL_PATH_ENGINE.md](CONDITIONAL_PATH_ENGINE.md), [METHOD_SELECTION.md](METHOD_SELECTION.md), and [PATH_REPLAY_VALIDATION.md](PATH_REPLAY_VALIDATION.md) for definitions and validation limits.
