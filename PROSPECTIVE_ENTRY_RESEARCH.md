# Prospective-entry risk and timing research

## Product question

The research target is an observable entry into an active wick setup:

- How much farther can price move adversely from the entry close?
- How long can the target remain unfilled?
- Over an explicit horizon, does the wick target trade first, does a chosen adverse threshold trade first, do neither trade, or do both trade in the same OHLC candle with unknowable ordering?

This target is separate from whether a projected historical candle route looks convincing.

## Frozen baseline

The current soft candle/state matcher is frozen as `soft-state-v1.0.0`:

- 70% adaptive signal-feature similarity and 30% direction-mirrored candle-archetype similarity inside the signal-feature term.
- Live-state distance remains the main term: current distance, peak distance, retracement from peak, and age.
- Upper and lower wicks are pooled as mirrored configurations.
- Real completed historical suffixes remain illustration-only.

The live response now includes the baseline version, per-route score components, and cohort mean/median/p90 component contributions. This permits chronological audits without inferring why a route matched from its appearance.

## All-outcome dataset

`prospective_entry_outcomes.py` builds a dataset alongside, not instead of, the completed-route library.

It retains every strict signal with its observed status. Entry observations exist only when all of the following are true at the entry candle close:

1. The signal has made the confirmed departure close.
2. The wick target has not already traded.
3. The entry candle itself and its close are available.

The default entry ages are 1 hour, 10 hours, 1 day, 3 days, 7 days, 14 days, 30 days, 60 days and 90 days. Default evaluation horizons run from 1 hour through 30 days. This deliberately includes the older swing setups that the first V3 live gate did not support.

The first five-asset 5-minute build contains:

- 46,166 strict signals.
- 19,458 observable post-departure, still-unfilled entry snapshots.
- 23,245 clean fills within the 180-day tracing window.
- 161 signals that were unfilled or right-censored at the tracing boundary; they remain in the signal population rather than disappearing from it.

The dataset partitions are under `data/prospective_entry_outcomes_v1/` and are intentionally ignored by Git.

The expanded materialization also contains all five 1-minute sources: 526,432 strict signals and 59,253 observable post-departure entry snapshots. Together with the refreshed 5-minute partitions, the current all-outcome store contains 572,599 signals and 78,712 observable entries. The chronological promotion result below is still specifically the completed 5-minute evaluation; the 1-minute population has been prepared, not silently treated as validated by the 5-minute result.

## Risk semantics

Adverse percentages are measured from the prospective entry close:

- Lower-wick target: later highs above the entry are adverse.
- Upper-wick target: later lows below the entry are adverse.

For a target-touch candle, OHLC cannot establish whether the adverse extreme occurred before or after the target touch. The dataset therefore stores:

- A lower bound excluding the target-touch candle.
- An upper bound including the target-touch candle.
- `ambiguous_intrabar` when target and adverse-threshold first hits occur in the same candle.

A numerical risk prediction is scored with zero error when it falls inside the observed lower/upper interval. It is not silently compared with a made-up intrabar ordering.

## Observable sequence inputs

Every observation retains raw-candle indices for:

- The eight hours before the signal.
- The recent 21 hours 20 minutes ending at the entry.
- The complete signal-to-entry sequence.

This matches the existing V3 numerical sequence contract. Candle geometry remains explicit in the static inputs; the sequence encoder does not replace body/wick/range fields.

## Chronological evaluation

`evaluate_prospective_entry_baseline.py` separates signals chronologically into fit, validation and final holdout periods. The 70/30 weight and direction pooling are not selected from the September ETH example.

- Blend weights and pooled-versus-separated direction handling are compared only on validation.
- The validation winner is frozen before the final holdout.
- A historical candidate is usable only after its longest required horizon has fully elapsed before the query entry.
- Reports are split by wick direction, entry age and entry distance from the target.
- Score-component contributions are recorded.

The promotion gate requires simultaneous held-out improvement in adverse-risk error and waiting-time error, with no regression in fill or competing-outcome Brier score. Route variety or visual plausibility cannot pass the gate.

## First five-asset 5-minute result

The first run used 11,607 fit observations, 4,088 validation observations and 3,763 final-period observations. It sampled 1,000 validation and 2,000 holdout entries; 1,999 holdout entries had sufficient past-only support. Each configuration was scored across seven horizons and six adverse thresholds.

Validation selected a 15% shape-weight, direction-pooled challenger. On the untouched holdout:

| Matcher | Adverse interval MAE | Fill Brier | Waiting-time MAE | Competing-outcome Brier |
| --- | ---: | ---: | ---: | ---: |
| Frozen 30% pooled baseline | 2.4330 pp | 0.11377 | 1,349.7 min | 0.07811 |
| Validation-selected 15% pooled | 2.4323 pp | 0.11374 | 1,350.5 min | 0.07811 |
| 30% direction-separated control | 2.4262 pp | 0.11356 | 1,361.5 min | 0.07815 |
| 0% shape, direction-pooled control | 2.4296 pp | 0.11367 | 1,349.8 min | 0.07805 |

The 15% challenger did not pass the promotion gate because its held-out waiting-time error was slightly worse. Direction separation improved adverse error but materially worsened waiting time and slightly worsened competing-outcome calibration. The current 30% pooled matcher therefore remains the versioned baseline; this run does not validate replacing it.

The subgroup report also shows why multi-day coverage must remain explicit. Baseline adverse error rose from 1.46 percentage points at a one-hour entry age to roughly 3.5-4.7 points across the 7-90 day ages. The oldest buckets are sparse: only 55, 39 and 29 supported holdout observations at 30, 60 and 90 days. These are reported rather than hidden, but they are not enough to claim a solved long-age model.

Upper-wick cases were harder than lower-wick cases in this holdout: 2.66 versus 2.20 percentage points of adverse interval error, and about 1,430 versus 1,274 minutes of waiting-time MAE. Entry distance also mattered strongly: adverse error was 1.16 points below 1% from target and 4-5 points beyond 7%.

The generated detailed report is `data/prospective_entry_baseline_evaluation_5m.json`.

## First five-asset 1-minute result

The separate 1-minute run used 35,318 fit observations, 12,245 validation observations and 11,690 final-period observations. It sampled 500 validation entries and 1,000 holdout entries. Retrieval remained same-asset, matching the isolated live 1-minute libraries; wick directions were still tested pooled versus separated.

Validation independently selected the existing 30% direction-pooled baseline. On holdout it beat both fixed controls on every reported metric:

| Matcher | Adverse interval MAE | Fill Brier | Waiting-time MAE | Competing-outcome Brier |
| --- | ---: | ---: | ---: | ---: |
| Frozen 30% pooled baseline | **4.1830 pp** | **0.11238** | **2,089.5 min** | **0.08122** |
| 30% direction-separated control | 4.1866 pp | 0.11364 | 2,124.4 min | 0.08168 |
| 0% shape, direction-pooled control | 4.1891 pp | 0.11261 | 2,093.2 min | 0.08140 |

This supports retaining both the 30% shape term and direction pooling for the current 1-minute baseline. It is still not a claim that the baseline is accurate enough for trading decisions: adverse interval error remains about 4.18 percentage points overall and rises above six points for entries 15-30% from their target. Multi-day timing errors remain measured in several days.

The generated detailed report is `data/prospective_entry_baseline_evaluation_1m.json`.

## V3 label integrity

The neural V3 evaluator already retains raw future distances separately from its floored and clipped training representation. Comparisons use those raw labels. The all-outcome evaluator does not reconstruct ground truth from encoded targets.

## Promoted numerical entry model

`train_prospective_entry_model.py` trains one pooled five-asset model per timeframe. Its 53 observable inputs retain explicit signal-candle geometry and add current state, the eight hours before the signal, recent returns/volatility/range/volume, and signal-to-entry movement. It never receives a future candle. Separate horizon models estimate additional adverse movement and remaining time; probability models estimate fill and target-first/adverse-first outcomes.

During this work, the baseline loader was corrected to reset partition indices after concatenating assets. Previously, equal parquet row numbers from different assets could be treated as duplicates during stratified sampling. The promotion comparison below recomputes the frozen matcher and the challenger on exactly the same corrected holdout rows.

| Timeframe | Holdout entries | Measure | Frozen matcher | Numerical model | Improvement |
| --- | ---: | --- | ---: | ---: | ---: |
| 5m | 1,999 | Adverse interval MAE | 2.5058 pp | 2.4401 pp | 2.6% |
| 5m | 1,999 | Waiting-time MAE | 1,373.1 min | 1,351.2 min | 1.6% |
| 5m | 1,999 | Fill Brier | 0.11315 | 0.10980 | 3.0% |
| 5m | 1,999 | Competing-outcome Brier | 0.07816 | 0.07585 | 3.0% |
| 1m | 1,000 | Adverse interval MAE | 4.1830 pp | 4.0344 pp | 3.6% |
| 1m | 1,000 | Waiting-time MAE | 2,089.5 min | 2,000.0 min | 4.3% |
| 1m | 1,000 | Fill Brier | 0.11238 | 0.10730 | 4.5% |
| 1m | 1,000 | Competing-outcome Brier | 0.08122 | 0.07744 | 4.7% |

Both timeframes pass the predeclared gate: risk and timing improve, with no regression in either probability score. The dashboard therefore exposes this model as a separate **Risk from current price** card. Fast/Normal/Extreme remain real historical route illustrations and are not replaced by generated candles.

Generated reports remain ignored under `data/prospective_entry_models/<timeframe>/training_report.json`. Large model artifacts default to `%LOCALAPPDATA%\candle_projection_algo\prospective_entry_models`; set `CANDLE_PROJECTION_MODEL_DIR` to override that location.

## Periodic data and model refresh

`wick_update.py update` is the supported manual path for absorbing newly resolved wicks. It refreshes all five 1m/5m sources, rebuilds completed-route and all-outcome evidence, and retrains both numerical entry models through the same chronological promotion gate described above. It deliberately does not retrain the experimental V3 route experts. Operational details, interruption semantics and generated evidence are documented in [MODEL_UPDATE_WORKFLOW.md](MODEL_UPDATE_WORKFLOW.md).
