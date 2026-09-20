# ETHUSDT Wick-Touch Observations

This note records a dated empirical observation. It does not change the live signal definition or claim that a wick touch is a trade win.

## Snapshot and definitions

- Source: Binance USD-M Futures `ETHUSDT`, completed five-minute candles.
- Snapshot: source through the candle opened `2026-09-20T11:35:00Z`.
- A raw exact touch means a *later* five-minute low reaches or falls below a lower-wick low, or a later high reaches or exceeds an upper-wick high. The signal candle itself is not counted as its own fill.
- This is deliberately different from the V1 clean path: V1 also requires a confirmed departure beyond the opposite signal extreme before the later touch. See [MODEL_TRANSPARENCY.md](MODEL_TRANSPARENCY.md).

Two candle-shape tiers were compared:

| Tier | Body as % of range | Dominant wick as % | Opposite wick as % | Status |
| --- | ---: | ---: | ---: | --- |
| Strict | at most 5% | at least 75% | at most 20% | Current engine rule |
| Near-strict | at most 8% | at least 70% | at most 30% | Audit tier only; not yet a live model setting |

The near-strict tier was selected to include the visually similar July candles discussed below. It is not a frozen threshold choice and must be treated as a candidate pending a final untouched holdout.

## Current six-month as-of-today result

The table below uses signals from the preceding 180 days and asks whether their exact wick was touched by the snapshot date. Recent signals have less time to resolve, so this is an as-of-today descriptive rate, not an equally matured forecast sample.

| Tier | Signals | Later exact touches | Not touched by snapshot | Touch rate |
| --- | ---: | ---: | ---: | ---: |
| Strict | 921 | 914 | 7 | 99.240% |
| Near-strict | 1,984 | 1,967 | 17 | 99.143% |

## Fully observed 180-day test

To remove the recent-signal censoring issue, each signal is labelled only from the next 180 days of completed candles. Signals without a full 180-day future window are excluded. The final eligible signal is at `2026-03-24T11:35:00Z`.

| Tier | Fully observed signals | Touched within 180 days | Not touched within 180 days | Touch rate |
| --- | ---: | ---: | ---: | ---: |
| Strict | 7,321 | 7,291 | 30 | 99.590% |
| Near-strict | 16,749 | 16,687 | 62 | 99.630% |

The result was also checked across nine non-overlapping six-month cohorts from September 2021 through March 2026. The lowest cohort touch rate was 99.150% for strict and 99.263% for near-strict; the highest was 100.000% and 99.951%, respectively.

These observations are strong evidence of historical exact-touch recurrence under these definitions. They are not independent-trade probabilities: nearby five-minute signals can share one market move, and the table does not capture time to touch, maximum move-away, fees, funding, execution, or liquidation risk.

## July 2026 lower-wick audit

The following candles were inspected using Berlin / CEST candle-open times. Each exact low remained untouched at the snapshot. Four were rejected only because they narrowly missed a strict hard cutoff; all five satisfy the near-strict audit tier.

| Berlin candle open | Wick low | Strict tier | Why not strict, if rejected | Later minimum low by snapshot |
| --- | ---: | --- | --- | ---: |
| 26 Jun 05:00 | 1,510.87 | No | body 6.64%; lower wick 74.40% | 1,517.09 |
| 01 Jul 16:45 | 1,590.35 | No | body 6.87%; lower wick 74.76% | 1,596.14 |
| 02 Jul 04:10 | 1,613.59 | No | lower wick 72.60%; opposite wick 25.80% | 1,613.99 |
| 02 Jul 13:55 | 1,643.52 | Yes | - | 1,646.89 |
| 14 Jul 02:05 | 1,772.52 | No | body 6.78% | 1,773.55 |

At the snapshot ETH closed at 2,570.75. Reaching these lows from that close would require roughly a 31.05% to 41.23% downside move. This is why high eventual-touch recurrence cannot be used as a sizing rule.

## Modelling consequence

The path library should continue to train only on clean, resolved fill paths when the question is: "what route did price take to a fill?" Right-censored and unresolved signals must remain in the outcome dataset for a future resolution, time-to-fill, and tail-risk study; deleting them would make the path visuals cleaner but would destroy evidence about non-resolution and waiting-time risk.

The recommended next research sequence is:

1. Keep strict and near-strict as separate cohorts rather than replacing the current strict rule.
2. Freeze the tier choices before scoring an untouched final holdout.
3. Build a wick-zone map that groups nearby active targets and distinguishes strict from near-strict evidence.
4. Backtest the competing-target question: when multiple active zones exist, which one is touched first and what move-away occurs before it?
