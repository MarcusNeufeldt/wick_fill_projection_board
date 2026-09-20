# Multi-Asset Wick-Fill Panel and Walk-Forward Baseline

## What this adds

The original ETH 5m study remains a useful single-market analogue library. This second layer turns the idea into a more disciplined research panel:

- Markets: `BTCUSDT` and `ETHUSDT` Binance USD-M perpetuals.
- Timeframes: native 5m and 15m bars derived from the aligned 5m source.
- Shared source window: `2023-09-20T07:00:00Z` through `2026-09-20T05:35:00Z` (end exclusive).
- Strict signal rows: 13,022 total.

| Market | 5m signals | 15m signals |
| --- | ---: | ---: |
| BTCUSDT | 5,477 | 1,350 |
| ETHUSDT | 4,906 | 1,289 |

The extra rows improve coverage. They are not 13,022 independent bets: BTC and ETH are correlated, and 5m/15m observations overlap.

For perspective, full wick touches are common in the mature 30-day panel, while the stricter departure-first sequence is much less automatic:

| Market | Timeframe | Raw full touch within 30d | Clean departure then fill within 30d |
| --- | --- | ---: | ---: |
| BTCUSDT | 5m | 99.3% | 52.2% |
| BTCUSDT | 15m | 98.9% | 52.1% |
| ETHUSDT | 5m | 99.0% | 55.3% |
| ETHUSDT | 15m | 98.0% | 54.6% |

## Exact signal definition

A signal is a closed candle with either a lower or upper dominant wick:

- body <= 5% of the total candle range;
- dominant wick >= 75% of the range;
- opposite wick <= 20% of the range.

For each signal, all features are known at the signal candle close:

- body, dominant-wick, and opposite-wick percentages;
- range as a percentage of price;
- range relative to the previous 20-candle median;
- volume relative to the previous 20-candle mean;
- the preceding one-hour return, direction-normalised so upper/lower events can share a common representation.

## Labels: what is and is not being predicted

Each signal receives separate 1-day, 7-day, and 30-day labels.

- **Fill:** a later candle touches the wick extreme.
- **Departure:** a later close crosses the opposite extreme of the signal candle.
- **Clean departure then fill:** departure happens in an earlier candle than the first full wick touch.

The walk-forward model scores the third, stricter label because it matches the setup under study: price moves away first and only later comes back to fully fill the wick. If departure and fill appear in the same OHLC candle, their intrabar order is unknowable, so the event is conservatively not called clean.

Rows at the end of the data set are blank for labels whose entire future horizon is unavailable. They never enter training or scoring for that horizon.

## The first walk-forward test

The baseline is deliberately modest and transparent: a pooled, robust-scaled k-nearest-neighbour model (`k=75`). It pools normalised observations across BTC/ETH and 5m/15m, while adding small mismatch penalties for market, timeframe, and wick direction.

This run used that one fixed baseline; it did not sweep many algorithms or hyperparameters and pick the prettiest result. A final untouched holdout is still necessary before treating even this modest result as stable.

The test is chronological—not random:

- 365-day initial training period;
- 23 sequential, non-overlapping 30-day test windows;
- for a horizon H, training ends before the test window by H, so no training label can complete during or after its own test window;
- 8,329 out-of-sample predictions for each horizon.

| Clean departure then fill | OOS cases | Observed rate | Model Brier | Fold-base Brier | Brier skill | ROC AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Within 1 day | 8,329 | 48.9% | 0.2482 | 0.2500 | 0.7% | 0.549 |
| Within 7 days | 8,329 | 52.4% | 0.2459 | 0.2495 | 1.5% | 0.563 |
| Within 30 days | 8,329 | 53.3% | 0.2444 | 0.2490 | 1.8% | 0.569 |

`Brier skill` compares the forecast with using only the historical base rate within each fold. Positive is better.

## Honest read of the result

The model is better than the fold base rate, but only slightly. This supports continuing the research; it does **not** yet support a claim that candle shape alone reliably predicts the whole future path.

In particular:

- Raw wick touches are very common in this strict population, especially by 30 days. That fact alone is not an edge.
- The more useful departure-then-fill sequence lands near a 50/50 base rate, and the current features produce weak separation.
- The existing chart remains a conditional, rescaled historical analogue path. It is not the probability model's forecast.

## Reproducible artifacts

- Panel data: `data/multi_asset_panel/panel_signals.csv`
- Panel summary and label rates: `data/multi_asset_panel/panel_summary.json`
- Walk-forward report: `data/multi_asset_panel/walk_forward/WALK_FORWARD_RESULTS.md`
- Machine-readable metrics: `data/multi_asset_panel/walk_forward/walk_forward_metrics.json`
- Per-event out-of-sample forecasts: `data/multi_asset_panel/walk_forward/walk_forward_predictions.csv`
- Per-fold diagnostics: `data/multi_asset_panel/walk_forward/walk_forward_fold_metrics.csv`

The scripts are:

```powershell
python .\download_futures_klines.py --symbol BTCUSDT --interval 5m
python .\build_wick_fill_panel.py
python .\evaluate_wick_fill_walk_forward.py
```

## Best next experiment

Do not jump straight to a more complex model. First reserve the newest six months as a final untouched holdout, tune the existing signal thresholds/features only on the earlier walk-forward folds, and then score that holdout once. If the modest lift survives, the next useful layer is a separate conditional path model for drawdown, time-to-fill, and adverse excursion—not a prettier single path.
