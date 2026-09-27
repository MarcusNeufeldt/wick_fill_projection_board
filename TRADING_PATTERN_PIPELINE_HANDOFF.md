# Reusing the Event-to-Outcome Trading Research Pipeline

## Purpose

This repository began as a wick-fill projection experiment, but much of it is
now a reusable **trading-event research pipeline**.

Use this document when Marcus brings a new chart pattern or market event and
wants to study, backtest, model, retrieve historical analogues, or expose it in
the live dashboard.

The general research question is:

> When an objectively detectable event occurs at time A, what observable state
> are we in, what can happen before resolution B, how long can it take, and what
> did genuinely comparable historical cases do?

The goal is not to force every pattern into the wick-fill definition. Reuse the
pipeline, while replacing the event-specific assumptions.

## Prompt for the next agent

Marcus can start a future task with this:

```text
I want to study a new trading pattern using the existing candle projection
pipeline. Read TRADING_PATTERN_PIPELINE_HANDOFF.md first, then inspect the
current repository and README.

Pattern idea:
[Describe the pattern and provide exact example asset/timeframe/timestamps.]

What I believe marks the signal:
[Describe what is visible at the signal moment.]

What I want to predict or trade:
[Target, invalidation, adverse move, waiting time, or path behavior.]

First turn this into an objective PatternSpec that can be detected without
future information. Show me any ambiguous choices. Then build the smallest
chronological POC using the existing data, evaluation, retrieval, and outcome
machinery. Do not promote it into the live dashboard merely because its routes
look convincing.
```

## The reusable pipeline

```text
Raw completed candles
        |
        v
Objective event detector
        |
        v
Observable snapshots at prospective entry times
        |
        +--------------------------+
        |                          |
        v                          v
All-outcome dataset        Completed-route library
(resolved + unresolved)    (real historical paths)
        |                          |
        v                          v
Numerical forecasts        Historical retrieval
(probability/risk/time)     (routes/support)
        |                          |
        +-------------+------------+
                      v
          Chronological evaluation
                      |
                      v
          Frozen candidate contract
                      |
                      v
       Live calculation + prospective log
```

This structure supports more than target fills. A resolution can be a retest,
reversal, breakout target, stop threshold, expiry, competing first-touch event,
or another precisely defined outcome.

## What is already reusable

The following parts should normally be reused:

- Binance Futures candle download and incremental refresh.
- Five configured assets and native 1m/5m sources.
- Completed-candle and timestamp-integrity checks.
- Chronological fit, validation, and holdout discipline.
- Resolved, unresolved, and right-censored outcome handling.
- Prospective entry snapshots rather than signal-only hindsight.
- Pre-signal, signal-to-current, and current-state features.
- Direction normalization when the new event has a genuine mirrored form.
- Historical-neighbour retrieval and Historical Support/OOD diagnostics.
- Real historical continuations as illustrations.
- Numerical probability, adverse-risk, and waiting-time outputs.
- Intrabar ambiguity handling for OHLC data.
- Versioned model artifacts, promotion gates, and prospective logging.
- Rust acceleration with Python parity checks.
- Manual refresh/update manifests and atomic artifact promotion.
- The live dashboard, refresh coordinator, and chart renderer.

## What is wick-specific and must not be assumed

The current implementation contains choices that belong specifically to the
wick experiment:

- A strict tiny-body/dominant-wick signal detector.
- Upper and lower wicks treated as mirrored directions.
- A wick endpoint as the price target.
- Confirmed departure beyond the signal candle's opposite extreme.
- Full wick touch as the terminal fill event.
- Distance measured relative to the wick target.
- Entry-relative adverse movement defined opposite the target direction.
- Fast, Normal, Adverse first, and Extreme route meanings.
- Route rescaling toward a known target price.

A new pattern may use different direction logic, several targets, no fixed
target, a time expiry, or multiple competing outcomes. Define those explicitly.

## Required PatternSpec

Before running a large build, write a small versioned specification for the new
pattern. At minimum it must answer:

| Field | Required decision |
| --- | --- |
| Pattern name/version | Stable identifier for datasets, reports, and artifacts |
| Signal detector | Exact candle/sequence rules visible at signal close |
| Mirrored directions | Whether bullish/bearish cases can validly be normalized together |
| Pre-signal context | How much earlier price/volume action is observable and relevant |
| Confirmation/departure | Whether another completed candle is required before entry eligibility |
| Prospective entry | Exact observable entry time and entry price |
| Primary target | Exact success/resolution event, if one exists |
| Adverse event | Entry-relative thresholds or invalidation event |
| Competing events | Which first-touch outcomes must be distinguished |
| Expiry | When the setup stops being actionable |
| Censoring | How an unfinished history is represented when data ends |
| Intrabar ambiguity | What to do when target and adverse event occur in the same OHLC candle |
| Horizons | Explicit forecast windows, including multi-day cases when relevant |
| Assets/timeframes | Initial POC scope and later scaling scope |
| Route coordinates | How historical paths are normalized and, if valid, rescaled |
| Product outputs | Which numbers, routes, support disclosures, and warnings are useful |

The detector must produce identical results when run again on the same candle
history. Terms such as "strong candle," "clean breakout," or "looks trapped"
must be translated into measurable rules before backtesting.

## Implementation order

### 1. Inspect the current source of truth

Read these files before changing the pipeline:

- `README.md` - current product behavior and operating commands.
- `PROSPECTIVE_ENTRY_RESEARCH.md` - all-outcome semantics and entry-relative risk.
- `FORECAST_ARCHITECTURE_V1.md` - ownership of numbers, routes, and support.
- `ROLLING_FORECAST_EVALUATION.md` - chronological evaluation findings.
- `MODEL_UPDATE_WORKFLOW.md` - refresh, training, promotion, and resource boundaries.
- `CONDITIONAL_PATH_ENGINE.md` - historical-route principles and leakage rules.

Treat checked-in reports as development evidence. Verify current code, artifacts,
and live service state when the conclusion depends on them.

### 2. Build a detector-only audit

Implement the signal definition and produce:

- Signal counts by asset, timeframe, direction, and calendar period.
- Example timestamps for visual inspection.
- Feature distributions and rare/extreme cases.
- A check that changing future candles cannot change a past signal.

Do not start model tuning until Marcus agrees that the detector captures the
intended chart pattern.

### 3. Build an all-outcome dataset

Retain every valid signal, not only attractive completed examples.

For each prospective entry snapshot, store:

- Only features observable at that candle close.
- Resolution/target status over explicit horizons.
- Additional adverse movement from the entry price.
- Waiting time when the event resolves.
- Unresolved and right-censored status.
- Competing-event results.
- Lower/upper bounds when OHLC cannot establish intrabar order.
- Raw future labels separately from any clipped or transformed training target.

This dataset owns numerical probability, risk, and time evaluation.

### 4. Build a completed-route library

For events with a meaningful A-to-B path, retain real normalized OHLCV
continuations from the observable alignment state through resolution.

The route library is an illustration/evidence layer. It must not silently delete
unresolved cases from the numerical outcome population.

### 5. Establish simple baselines first

Compare at least:

- A prevalence or age/distance/regime prior.
- A simple supervised numerical model.
- Handcrafted historical retrieval.
- Sequence-aware or learned retrieval only when the simpler baseline justifies it.

Do not add a neural model solely because sequence data exists. A new model must
improve the relevant held-out numerical metrics.

### 6. Evaluate chronologically

At each historical query time, behave as though the system is truly standing at
that date:

- Fit scalers and models only on earlier permitted history.
- Admit a historical label only after its outcome horizon has matured.
- Keep future candles out of every query feature and retrieval prefix.
- Group repeated snapshots by signal episode.
- Treat overlapping assets and market periods as dependent observations.
- Select hyperparameters on validation, never on the final test period.
- Report individual time folds as well as aggregate results.
- Break results down by asset, direction, age, distance, volatility regime, and
  Historical Support.

For classification, report Brier score and calibration. For risk/time, report
absolute or quantile loss plus interval coverage and width. For routes, score
defined path checkpoints or events, not visual attractiveness.

### 7. Separate product ownership

Keep these concepts distinct:

```text
Numerical model
    = probability, adverse risk, and waiting-time estimates

Historical retrieval
    = real precedent and illustrative routes

Historical Support
    = how familiar the observable state is; not a probability
```

If a displayed historical route looks excellent but the numerical forecasts do
not improve, the research has not passed its promotion gate.

### 8. Freeze before prospective validation

Once an architecture has been selected:

- Give it an immutable version and artifact hash.
- Record its training-label cutoff.
- Prevent routine update commands from silently repointing the frozen artifact.
- Log architecture version, hash, and forecast source with every live prediction.
- Exclude fallbacks from the frozen architecture's prospective score.
- Accumulate genuinely future outcomes before changing the architecture again.

## Current implementation map

Use the following files as working examples, not universal interfaces:

| Responsibility | Current implementation |
| --- | --- |
| Asset/timeframe configuration | `conditional_wick_assets.py` |
| Historical download | `download_futures_klines.py` |
| Incremental source refresh | `refresh_futures_klines.py` |
| Wick detector and pooled path library | `build_conditional_path_library.py` |
| Isolated 1m route libraries | `build_sol_one_minute_path_library.py` |
| Runtime retrieval cache | `build_native_runtime_cache.py` |
| Live/state matching and scenarios | `build_conditional_path_scenarios.py`, `contextual_entry_matcher.py` |
| All-outcome dataset | `prospective_entry_outcomes.py` |
| Observable feature contract | `prospective_entry_model.py` |
| Numerical training/promotion evaluation | `train_prospective_entry_model.py` |
| Rolling research comparison | `evaluate_rolling_forecast_systems.py` |
| Frozen numerical/retrieval contract | `forecast_architecture_v1.py`, `train_forecast_architecture_v1.py` |
| Immutable artifact manifest | `forecast_artifact_manifest.py` |
| Prospective forecast ledger | `prospective_forecast_ledger.py` |
| Dashboard/runtime | `serve_conditional_wick_dashboard.py` |
| Manual maintenance pipeline | `wick_update.py` |
| Rust scanning/path/outcome kernel | `experiments/rust_training_throughput/src/main.rs` |
| Native Rust matcher | `native_wick_matcher/src/lib.rs` |
| Live contract verification | `verify_live_refresh_service.py` |

## Recommended second-pattern strategy

Do **not** begin by rewriting the repository into a generic framework. That risks
abstracting wick-specific assumptions into the wrong universal interface.

Instead:

1. Freeze the existing wick behavior with its tests.
2. Implement one new pattern as a narrow POC on one asset/timeframe.
3. Reuse candle ingestion, chronological splitting, metrics, and artifact rules.
4. Compare its detector/outcome needs with the wick implementation.
5. Extract only the interfaces that are genuinely shared.
6. Scale to all assets/timeframes after the POC passes chronological evaluation.
7. Add the new pattern to the dashboard only after its numerical contract is
   clear and its runtime cost is acceptable.

A likely interface after the second pattern is proven is:

```python
class TradingEventSpec:
    version: str

    def detect_signals(self, candles): ...
    def eligible_entries(self, candles, signal): ...
    def observable_features(self, candles, signal, entry): ...
    def normalized_prefix(self, candles, signal, entry): ...
    def evaluate_outcomes(self, candles, signal, entry, horizons): ...
    def completed_route(self, candles, signal, entry, outcome): ...
```

This is a design direction, not an instruction to refactor before the second
pattern reveals the real shared boundary.

## Acceptance checklist for a new pattern

A POC is ready for serious research only when:

- [ ] The signal rule is objective and versioned.
- [ ] At least several examples were visually checked against Marcus's intent.
- [ ] Detection uses no future candles.
- [ ] Entry timing and entry price are explicit.
- [ ] Target, adverse event, expiry, and censoring are explicit.
- [ ] Same-candle event ordering is bounded or marked ambiguous.
- [ ] Completed and unresolved examples are both retained.
- [ ] Upper/lower or bullish/bearish pooling is validated rather than assumed.
- [ ] Pre-signal context is present when relevant.
- [ ] Raw labels are preserved for evaluation.
- [ ] Candidate availability respects label maturation.
- [ ] Evaluation is chronological and grouped by episode/time period.
- [ ] A simple baseline exists.
- [ ] Numerical forecasts and illustrative routes are scored separately.
- [ ] Historical Support is disclosed separately from probability.
- [ ] Subgroup results include old/multi-day events when the strategy uses them.
- [ ] The newest evaluation period was not repeatedly tuned against.
- [ ] Generated data/models remain ignored by Git.
- [ ] Rust output has exact Python parity before becoming authoritative.
- [ ] Live promotion is explicit, versioned, and reversible.
- [ ] Prospective predictions can be attributed to an exact artifact hash.

## Expected first deliverable from the next agent

Before launching a long build, the next agent should return:

1. A concise `PatternSpec` with unresolved decisions called out.
2. The exact existing modules that can be reused unchanged.
3. The minimal new detector/outcome/retrieval modules required.
4. A one-asset/timeframe POC plan with expected runtime and disk use.
5. Leakage and ambiguity tests.
6. Chronological evaluation metrics and promotion gate.
7. A clear statement of what will remain illustration-only.

The first success criterion is not a beautiful projected chart. It is a
reproducible event definition and honest out-of-sample evidence about its
probability, risk, timing, and historical precedent.
