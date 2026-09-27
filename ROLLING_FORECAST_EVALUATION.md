# Rolling forecast evaluation

## Why this exists

The original behavioral-fingerprint holdout has been inspected and is permanently burned for model selection. `evaluate_rolling_forecast_systems.py` creates reusable expanding chronological folds so later experiments are selected on validation windows and measured on untouched test windows.

This harness evaluates numerical risk and timing predictions. Historical candle continuations remain an illustration layer and are not allowed to substitute for fill-probability, adverse-movement, or waiting-time accuracy.

## Controlled systems

| System | Forecast mechanism | Inputs |
| --- | --- | --- |
| A | Existing ExtraTrees fill classifier plus HGB adverse/wait regressors | Current 53 observable features |
| B | The same learners | The 53 features plus 40 mirrored pre-signal/recent/signal-to-entry sequence coordinates |
| C | Horizon-specific historical nearest-neighbour retrieval | Validation-selected handcrafted state/context/sequence fingerprint |
| C2 | Dynamic historical retrieval | C fingerprint plus validation-selected `k`, weighting and radius; up to 96 distinct episodes |
| E0 | Prediction-only numerical stack | Validation-trained fixed stack over A/B predictions |
| E | Support-aware numerical stack | E0 inputs plus observable retrieval-support variables |
| D | Existing V3 TCN embedding and retrieve/rerank | Pending fold-local training integration |

The eventual product may use the best numerical system for probabilities and risk while showing real routes from the best retrieval system. That is a product composition, not another predictive competitor unless their numerical forecasts are explicitly blended and evaluated.

## Fold contract

Each fold has an expanding history, a validation signal window, and a later non-overlapping test signal window.

- Every training or retrieval observation must have completed its full 30-day label window before the first query close.
- Tree hyperparameters and retrieval fingerprints are selected on validation only.
- Retrieval normalization is fitted on that fold's eligible historical population only.
- Test queries are identical across every system.
- Fill is scored with Brier loss, adverse movement with midpoint adverse MAE, and waiting time with conditional-on-fill MAE.
- Metrics and paired bootstrap uncertainty are balanced/resampled by signal episode rather than treating repeated snapshots as independent.
- Results remain visible per fold and in aggregate, with asset, wick direction, age, departure distance, volatility, and retrieval-support breakdowns.

The existing global V3 artifact is not accepted for rolling folds. For D, the encoder, normalization, retrieval population, reranker, and any calibration must be trained using only information available before the fold cutoff. The code contains a provenance guard that fails closed when this cannot be proven.

## Commands

Fast integration smoke (not research evidence):

```powershell
.venv\Scripts\python.exe .\evaluate_rolling_forecast_systems.py `
  --smoke --folds 1 --initial-train-fraction 0.70 `
  --validation-queries 50 --test-queries 75 --neighbors 8 `
  --bootstrap-samples 100 `
  --output data\rolling_forecast_comparison_smoke_5m.json
```

Full rolling comparison:

```powershell
.venv\Scripts\python.exe .\evaluate_rolling_forecast_systems.py `
  --folds 5 --initial-train-fraction 0.40 `
  --validation-queries 750 --test-queries 1000 --neighbors 32 `
  --bootstrap-samples 1000 `
  --output data\rolling_forecast_comparison_5m.json
```

The full V3-in-every-fold run is intentionally an overnight job. Do not reuse a globally trained artifact merely to make the table complete.

## Current status

The one-fold smoke completed successfully across all enabled systems. It proves the plumbing, target alignment, fold gating, aggregate report, subgroup report, and paired episode bootstrap execute end to end. Its tiny query sample and reduced model/config set are not evidence that one system wins.

The full five-fold comparison evaluated 4,766 test snapshots from 2,660 distinct signal episodes. Each fold used 750 validation snapshots; four test folds used 1,000 snapshots and the final fully matured window supplied 766.

Episode-balanced improvement versus A is below. Positive values are improvements; negative values are regressions.

| System | Horizon | Fill Brier | Adverse midpoint MAE | Conditional wait MAE |
| --- | ---: | ---: | ---: | ---: |
| B: supervised + sequence | 1 day | +0.49% | -0.26% | -0.05% |
| B: supervised + sequence | 7 days | +0.26% | -0.07% | -0.14% |
| B: supervised + sequence | 30 days | +1.11% | -0.64% | +0.07% |
| C: horizon retrieval | 1 day | +0.37% | +1.00% | -7.51% |
| C: horizon retrieval | 7 days | -0.35% | +1.25% | -4.44% |
| C: horizon retrieval | 30 days | -3.36% | +1.95% | -6.51% |

For B, the 1-day and 30-day fill-Brier paired bootstrap intervals were entirely favorable. Its other changes were negligible or unfavorable. The sequence fingerprint therefore contains incremental fill-probability information, but appending its raw coordinates does not improve the current adverse-risk or waiting-time regressors.

For C, the 30-day adverse improvement was the clearest result: its paired challenger-minus-baseline interval was -0.0769 to -0.0175 percentage points. Conversely, retrieval waiting-time error was conclusively worse at all three horizons. Retrieval should not replace the supervised numerical wait model based on this evidence.

Validation chose different retrieval fingerprints across horizons and folds, including `state_only` for two 30-day folds. This confirms that one universal similarity definition is not stable across market periods.

Support quality was strongly ordered. In the retrieval system, high-support versus low-support absolute results were:

| Horizon | Fill Brier high / low | Adverse MAE high / low | Wait MAE high / low |
| --- | ---: | ---: | ---: |
| 1 day | 0.111 / 0.140 | 1.02% / 1.94% | 170m / 234m |
| 7 days | 0.050 / 0.094 | 1.71% / 2.98% | 533m / 1,100m |
| 30 days | 0.033 / 0.052 | 3.04% / 4.00% | 1,152m / 2,397m |

These support bands are descriptive and may be confounded by age, distance, asset, or volatility. They are not calibrated confidence levels yet. They do justify the next controlled experiment: validation-selected distance weighting/dynamic neighbourhoods and an explicit support/OOD gate.

## Dynamic neighbourhood result

C2 tested 112 validation-only combinations per selected horizon fingerprint:

- `k` in 8, 16, 24, 32, 48, 64 and 96;
- uniform, inverse-distance, inverse-square and exponential weighting;
- unrestricted radius or validation-derived 25th/50th/75th percentile maximum distance.

Every selected policy used an unrestricted radius. Eleven of fifteen fold/horizon selections used inverse-square weighting, and most selected 48–96 raw neighbours. The evidence therefore favors broad retrieval with sharply declining weights, not a hard distance cutoff. Effective neighbour count and low-support flags remain exposed even when a scoreable nearest precedent is retained.

C2 improvement versus fixed C:

| Horizon | Fill Brier | Adverse midpoint MAE | Conditional wait MAE |
| --- | ---: | ---: | ---: |
| 1 day | +1.66% | +2.14% | +2.08% |
| 7 days | +1.83% | +1.85% | +0.17% |
| 30 days | +0.68% | +0.95% | -0.42% |

The 1-day fill/risk/wait and 7-day fill/risk paired intervals were entirely favorable. The 30-day fill and all 7/30-day waiting changes were uncertain.

C2 versus A improved adverse risk by 3.12%, 3.08% and 2.89% at 1/7/30 days. It improved fill Brier by 2.02% at one day and 1.49% at seven days, but regressed 30-day fill by 2.65%. It still regressed waiting time by 4.26–6.95%, so retrieval does not replace A for time-to-fill.

## What support actually means

Support buckets are calibrated from each fold's validation distance distribution, then frozen for test. The supervised A model itself becomes much less accurate as support falls:

| Horizon | Metric | Very high support | Very low support |
| --- | --- | ---: | ---: |
| 1 day | Fill Brier | 0.112 | 0.127 |
| 1 day | Adverse MAE | 0.98% | 2.45% |
| 1 day | Wait MAE | 160m | 243m |
| 7 days | Fill Brier | 0.054 | 0.109 |
| 7 days | Adverse MAE | 1.74% | 3.80% |
| 7 days | Wait MAE | 473m | 1,167m |
| 30 days | Fill Brier | 0.030 | 0.078 |
| 30 days | Adverse MAE | 2.92% | 5.82% |
| 30 days | Wait MAE | 1,169m | 3,140m |

Retrieval did not consistently beat A in high-support buckets. This is Case A from the research plan: support is primarily a general market-state difficulty/OOD indicator, not a reliable architecture router. The UI term should remain **Historical support**, never an invented probability-like confidence percentage.

## Numerical stacking and support ablation

E0 fits a fixed, small meta-model on each fold's validation period using only A/B fill, adverse and wait predictions. E uses the identical model plus observable support variables. Both weight rows inversely by snapshots per signal episode.

E0 versus A:

| Horizon | Fill Brier | Adverse midpoint MAE | Conditional wait MAE |
| --- | ---: | ---: | ---: |
| 1 day | +4.14% | +2.20% | -2.12% |
| 7 days | +11.23% | +3.58% | -3.01% |
| 30 days | +11.31% | +3.60% | -5.88% |

All E0 fill and adverse paired intervals were favorable. Fill calibration ECE improved from 0.145 to 0.062 at one day, 0.092 to 0.045 at seven days and 0.062 to 0.029 at 30 days. Waiting time worsened and stays with A.

Adding support variables to E0 produced no reliable risk or wait gain. It changed fill Brier by +0.22% at one day, -1.32% at seven days and **-5.30% at 30 days**; the 30-day regression interval was entirely unfavorable. Support therefore remains a separate difficulty diagnostic rather than a numerical model input.

## Current architecture indicated by the evidence

```text
Numerical fill probability     -> E0 validation-trained prediction stack
Numerical adverse risk         -> E0 validation-trained prediction stack
Numerical waiting time         -> A existing supervised model
Historical illustrated routes -> C2 candidates aligned to E0/A time-risk targets
Historical support            -> separate OOD/difficulty disclosure
```

This composition is now installed as frozen 5m architecture `forecast_architecture_v1_5m_2026-09-27` for prospective validation. Adaptive route candles select real completed C2 continuations close to the frozen E0/A time-risk targets; they remain illustrations and are evaluated separately from the numerical forecasts. Fold-local V3 is lower priority and the dashboard exposes it only as a legacy comparison.

## Interpretation boundary

The rolling folds are leakage-safe internally, but the same five test periods were inspected after A/B/C and then reused while designing C2 and the E/E0 ablation. They are therefore development evidence, not a permanently untouched final holdout for the complete architecture. No more parameter or architecture selection should be performed against these test periods. Promotion requires either newly accumulated future data or a newly reserved outer period that has not influenced design decisions.
