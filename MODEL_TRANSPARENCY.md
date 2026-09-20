# ETHUSDT Wick Fill Model Transparency

## What this model is

The current model is not a trained machine-learning model. It is a rule-based nearest-historical-analogue model. It finds past ETHUSDT wick events with a similar candle shape and a similar path state, then uses their later price action to describe a conditional scenario.

The word conditional matters. The historical paths used for the projection are all paths that eventually filled their wick after first moving away. The model therefore describes a possible path if the current wick fills. It does not estimate the unconditional probability that this wick will fill.

## Data scope

| Item | Current value |
| --- | --- |
| Market | Binance USD-M ETHUSDT perpetual futures |
| Base interval | 5 minutes |
| Source history | 20 Sep 2023 through 20 Sep 2026 in the local rolling three-year file |
| Historical-label cutoff | 16 Sep 2026 18:35 UTC, immediately before the current signal candle |
| Strict wick-shaped candidates | 4,891 |
| Clean departure-then-fill events | 2,748 |
| Lower / upper clean events | 1,456 / 1,292 |
| Filled before a provable departure | 2,137 |
| Right-censored at the cutoff | 6 |

The historical study uses only candles before the current signal as training history. Candles after the current signal are used only to calculate the present state and show the actual left-hand portion of the chart.

## Signal candle definition

A qualifying signal is a very small-body candle with one dominant wick. For every five-minute candle:

```text
body          = abs(close - open) / (high - low)
lower wick    = (min(open, close) - low) / (high - low)
upper wick    = (high - max(open, close)) / (high - low)
```

The current strict rule is:

| Rule | Threshold |
| --- | --- |
| Body | At most 5% of the full candle range |
| Dominant wick | At least 75% of the full candle range |
| Opposite wick | At most 20% of the full candle range |
| Direction | Both lower-wick and upper-wick signals are accepted |

The 16 Sep 2026 20:35 Berlin signal is a lower-wick signal:

| Current feature | Value | Rule result |
| --- | ---: | --- |
| Open / high / low / close | 2,399.00 / 2,404.21 / 2,366.67 / 2,398.90 | - |
| Body as share of range | 0.266% | Pass |
| Dominant lower wick | 85.855% | Pass |
| Opposite upper wick | 13.879% | Pass |
| Range versus prior 20-candle median | 3.724x | Context feature |
| Volume versus prior 20-candle mean | 3.099x | Context feature |
| Prior one-hour return | +0.690% | Context feature |

It is therefore geared toward the exact family of candles you described: an almost-zero body and one unusually long shadow. The system treats upper and lower variants as mirror images by normalizing distance away from the wick: positive always means price moved away from the wick, regardless of direction.

## What counts as a clean historical wick-fill path

For a lower wick, the price must first close above the signal high. For an upper wick, it must first close below the signal low. That is the confirmed move-away step.

After that departure, a full fill occurs at the first later candle that touches the signal wick extreme: the original low for lower wicks or original high for upper wicks. The search has no arbitrary time limit; it searches until the historical cutoff.

Five-minute OHLC candles cannot prove intrabar order. If one later candle both touches the wick extreme and meets the departure-close condition, it is conservatively classified as `filled_before_departure` and excluded from the clean signal-to-departure-to-fill paths. This avoids pretending that the departure happened before the fill when the data cannot establish that ordering.

## Current state being matched

At the latest data point used in the chart, ETH was 995 five-minute bars after the signal:

| State variable | Current value |
| --- | ---: |
| Latest close | 2,579.60 |
| Distance from the wick low | +8.997% |
| Largest move away from the wick low | +12.774% |
| Pullback from that peak | 3.777 percentage points |

The +12.774% peak is unusual in this three-year completed-event sample. Only 65 of 2,748 completed paths, or 2.37%, reached at least that far from their wick before filling. That is why a generic short-duration wick model would not be appropriate here.

## How the 40 analogues are ranked

The model scores every completed historical event at every pre-fill point in its path. It retains the best state alignment for each event, then keeps the 40 lowest-scoring events.

The score combines two pieces:

1. Signal anatomy and context, with manually chosen weights:

   - body share of range: 0.8
   - dominant wick share: 0.8
   - opposite wick share: 0.8
   - range relative to the preceding 20-candle median: 0.6
   - volume relative to the preceding 20-candle mean: 0.6
   - prior one-hour return: 0.4

2. In-progress path state, using log-distance terms:

   - current distance from the wick: 1.8
   - peak distance from the wick: 1.2
   - drawdown from the peak: 1.2
   - bars elapsed from the signal: 0.4

Signal-feature differences are normalized by a robust historical scale (median absolute deviation, with standard deviation only as a fallback). The final score is the state distance plus half the feature distance. The weights are hand-set design choices; they have not been learned from a backtest.

## What the current chart specifically shows

The first version of the chart showed a median candle path and an uncertainty envelope for all 40 selected histories. It did not necessarily reach the target during the rendered horizon, which made it visually unhelpful for the question you are studying.

The current TradingView Lightweight Charts page intentionally shows one continuous scenario instead:

| Item | Current value |
| --- | --- |
| Selected chart path | `lower_wick_1720485000000` |
| Historical signal | 9 Jul 2024 00:30 UTC |
| Rank by match score among the 40 | 9 |
| Selection rule for the visual | Remaining time closest to the 40-analogue median |
| Remaining path length | 4,979 five-minute bars, about 17.29 days |
| Terminal wick touch in the scenario | 7 Oct 2026 12:30 UTC |

The path is an actual historical post-alignment candle sequence. It is rescaled so the alignment close starts at the current ETH close and its final candle touches the current wick target at 2,366.67. It is not the top-score match, the median price path, or a probability-weighted forecast. It is a representative-duration visual scenario.

## Known limitations

- The chart is conditional on a fill. It has no valid answer yet to “what is the chance this wick fills?”
- The top-40 paths are selected from completed fills only. That selection is useful for a path-to-fill visual but causes selection bias if interpreted as a forecast.
- Three years gives 2,748 completed paths, but only 65 reached a move-away peak at least as large as the current 12.774% peak. The most relevant state is therefore relatively rare.
- Several signals can occur in the same market episode and share the same later move. Treating them as fully independent would exaggerate effective sample size.
- Five-minute OHLC data cannot resolve the order of high and low inside a bar. The extractor handles one obvious ambiguous case conservatively, but one-minute or tick data would reduce this uncertainty.
- The model currently has no funding, open interest, liquidation, order-book, BTC-regime, volatility-regime, or broader market-context features.
- The visual analogue projection itself still has no walk-forward path-accuracy test. A separate pooled BTC/ETH 5m/15m probability baseline now has a chronological walk-forward evaluation, but its lift over the base rate is modest and it is not a tradable result after fees, funding, and slippage.

## Best improvements in order

1. Build a separate fill-probability and time-to-fill study from all 4,891 strict signals, including clean fills, early fills, censored cases, and fixed horizons such as 1 day, 7 days, 30 days, and 90 days.
2. Run a rolling walk-forward backtest. For each historical signal, rebuild the candidate pool using only information available at that time, generate the same projection, and score fill probability, time-to-fill error, and path calibration out of sample.
3. Expand ETH history beyond three years where reliable futures data is available. More history adds different market regimes and, most importantly, more rare high-displacement examples.
4. Add one-minute data for event chronology and add funding rate, open interest, liquidation, realized volatility, BTC return/regime, and ETH-BTC relative strength as separate contextual features.
5. Cluster overlapping signals into market episodes or use embargoed time splits so one future move does not appear as many effectively duplicate training examples.
6. Replace the single display path with a small set of calibrated scenarios: fast, median, and long-duration paths, each labeled with its historical frequency and only after walk-forward validation.

## Reproducibility

The study is reproducible from the scripts and data in this folder:

- `download_ethusdt_5m.py` downloads and refreshes the rolling ETHUSDT five-minute source data.
- `extract_filled_wick_paths.py` defines the signal and builds the historical event/path study.
- `build_wick_projection.py` selects the 40 state-matched completed-fill analogues for the current signal.
- `build_wick_fill_scenario.py` chooses the median-duration visual analogue and creates `data/current_wick_fill_scenario.json`.
- `build_wick_fill_chart_html.py` creates the TradingView chart at `wick-fill-scenario-chart.html`.

The interactive chart now includes a transparency section with the current signal thresholds, population counts, current-state rarity, match rule, and the first ten scored analogue candidates.

## Cross-asset panel update

`PANEL_AND_WALK_FORWARD.md` documents the new BTCUSDT + ETHUSDT, 5m + 15m labelled panel and its chronology-only walk-forward test. It deliberately scores the stricter departure-then-fill event rather than treating a raw wick touch as a prediction. The generated results and per-event forecasts live under `data/multi_asset_panel/walk_forward/`.
