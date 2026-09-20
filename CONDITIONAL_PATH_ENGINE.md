# Conditional Wick-Fill Path Engine

## Product decision

The product is **not** a binary “will this wick fill?” classifier and it is not a single attractive projected line.

When a user pins a wick candle, the product should answer this practical question:

> Given this candle and the price action since it formed, what have comparable historical paths looked like before they revisited this wick target?

The dashboard should produce three conditional scenario paths toward the pinned wick:

| UI scenario | Historical meaning | Risk use |
| --- | --- | --- |
| Fast | A real, similar episode with a short time-to-fill and comparatively low adverse excursion | Lower-duration / lower-excursion reference, not a promise |
| Normal | The medoid of the comparable historical path population | Central reference path |
| Extreme | A real, similar high-risk episode selected by joint duration and adverse-excursion score | Stress reference for time and drawdown planning |

Scenarios must preserve the **joint** path from actual historical episodes. The engine must not fabricate a path by combining unrelated “90th percentile duration” and “90th percentile drawdown” values.

## What is conditioned on

Each historical episode is direction-normalised so that positive distance means price is moving away from the wick target. The live state is then matched against historical states using:

- candle shape: body, dominant wick, opposite wick, range and volume context;
- market and timeframe: BTC/ETH and 5m/15m, with explicit source labels rather than blind pooling;
- elapsed bars since the signal;
- current normalised distance from the target wick;
- peak distance away from the wick so far;
- drawdown from that peak;
- short-horizon volatility and return context.

Before the engine will project a live path, price must also have made the
same confirmed departure used by the historical definition: a later close
beyond the signal's opposite extreme. A merely unfilled wick is not enough.

The path selection must update after every newly closed candle. A path that matched at signal time may cease to be relevant after a large live move-away.

## Fill assumption and guardrail

The scenario display may be conditional on eventual full fill, because that is the question under study. The data layer must nevertheless retain early fills, no-departure cases, censored cases, and very long waits. A finite historical sample cannot prove a mathematical 100% eventual-fill rule.

The displayed screen must state when it is showing a **conditional fill scenario** rather than an unconditional probability forecast.

## Data expansion now

The immediate training library is:

- Binance USD-M perpetual `BTCUSDT` and `ETHUSDT`;
- five years of completed 5-minute OHLCV candles;
- 15-minute bars derived from the aligned 5-minute source;
- strict upper- and lower-wick signal episodes;
- normalized post-signal OHLC paths through the first full wick touch.

Five years is a meaningful improvement because it adds more market regimes and rare high-displacement paths. It does not make all samples independent: BTC/ETH and 5m/15m episodes can share the same market move.

For V1 chart candles, a pinned 5m signal is matched to real 5m historical paths and a pinned 15m signal to real 15m paths. A 15m historical bar must not be stretched into three invented 5m candles. Cross-timeframe observations remain useful for separately normalised risk/quantile models, but direct visual scenarios preserve the pinned timeframe.

## Model ladder

### V1 — state-conditioned empirical trajectories (build now)

Use robust-scaled nearest historical episodes plus actual normalised OHLC path resampling. Produce fast, normal, and extreme paths, a cohort count, time-to-fill percentiles, and adverse-excursion percentiles.

This is not primitive if it is state-conditioned, leak-safe, empirically calibrated, and transparent. It is the correct benchmark for this data volume.

### V2 — probabilistic path components

Add separate, testable components for:

- time-to-fill survival / quantile estimates;
- maximum adverse excursion quantiles;
- conformal path envelopes with target coverage;
- regime-aware weighting or gradient-boosted quantile models.

These components can improve the cohort baseline without inventing unrealistic candles.

### V3 — neural sequence experiments (only after V1/V2 benchmarks)

Evaluate temporal transformers, diffusion-style trajectory generators, or state-space sequence models only against the V1/V2 walk-forward benchmark. They are candidates, not automatic upgrades. With only a few years of sparse wick episodes and OHLCV data, a large neural model can memorize regimes, generate plausible-looking but miscalibrated paths, and lose to a calibrated empirical model.

Rust is useful for low-latency serving and heavy matching. It is not itself a modelling technique. Python is appropriate for research and validation; the proven engine can later be served in Rust or another runtime.

## Current V1 implementation

The first interactive implementation is the separate local service described
in [LIVE_DASHBOARD.md](LIVE_DASHBOARD.md). It accepts a BTC/ETH, timeframe,
and pinned strict-wick timestamp, validates that the wick is still unfilled
and has departed, then renders the three historical candle paths on a
TradingView Lightweight Charts view.

The server uses a verified five-year local snapshot and caches completed
requests. It deliberately does **not** silently fetch exchange data or claim
that a browser page is continuously live. An explicit incremental source-data
refresh and library rebuild are the next operational layer.

## Dashboard contract

For a pinned signal the dashboard must show:

- actual candles up to the latest closed bar;
- wick target and current state relative to it;
- fast, normal, and extreme projected trajectories, each marked as a conditional historical scenario;
- time-to-fill P25/P50/P90 and maximum adverse-excursion P50/P90;
- cohort size, asset/timeframe mix, and the exact matching rule;
- a live update when the next bar closes;
- no order execution, account access, leverage recommendation, or claim of certainty.

The existing `rust_jev_projection` dashboard is an adjacent generic close-return forecaster with conformal bands. It is preserved as a separate experiment; it is not yet the wick-conditioned engine specified here.

Until the explicit refresh layer exists, the live-update requirement is
intentionally a pending contract item rather than a claim about the current
browser service.

## Validation contract

The path engine is evaluated with chronological replay, not a random split:

1. At a past signal/state time T, construct its cohort from episodes completed before T only.
2. Generate the three scenarios and interval statistics using only that historical library.
3. Reveal the actual subsequent path.
4. Score time-to-fill quantile calibration, adverse-excursion coverage, path-envelope coverage, and a baseline comparison.
5. Keep the newest six months untouched until design choices are frozen, then score it once as a final holdout.

The correct success criterion is calibrated risk/time coverage, not a visually convincing future candle line.

## Immediate build milestones

1. Download and validate five-year BTCUSDT and ETHUSDT 5-minute source data without replacing the existing three-year study.
2. Generate a multi-asset, direction-normalised completed-episode library with state snapshots and normalized OHLC paths.
3. Build a reusable three-scenario selector and JSON contract for a pinned signal/state.
4. Render the three scenarios in a local TradingView Lightweight Charts prototype.
5. Add chronological replay tests for scenario coverage before calling the dashboard predictive.
6. Add the separate local pin-and-project service, then add an auditable incremental refresh loop and V2 calibrated risk outputs.
