# Behavioral Fingerprint Retrieval POC

## Idea

Represent every observable wick situation with a similarity-preserving numerical fingerprint:

```text
pre-signal context + signal candle + observed path so far
                              |
                              v
                 behavioral fingerprint
                              |
                              v
          nearest historical observable snapshots
                              |
                              v
       their real continuations, risks and waiting times
```

This is not a cryptographic hash. SHA-style hashes intentionally make nearby inputs unrelated. The desired object is a vector or learned embedding in which comparable situations remain close.

Historical candles after an observation are never included in its fingerprint. They remain attached outcomes: the real continuation, target-fill time and additional adverse movement.

## POC question

Does progressively richer observable similarity improve held-out outcome estimates?

The first experiment uses all five assets at 5m and compares:

1. **State and geometry** — signal body/wick configuration, range/volume context, age, current and peak target distance, retracement, departure age and current candle geometry.
2. **Context summary** — the first block plus pre-signal and recent return, volatility, range and volume summaries over several clock windows.
3. **Sequence shape** — both prior blocks plus mirrored price trajectories from the pre-signal window, recent window and the entire signal-to-entry prefix.

Upper and lower wicks are mirrored into the same coordinates. Each feature block is robustly scaled on historical candidates and normalized so the 40-value sequence block cannot win merely by having more dimensions.

## Leakage and dependence controls

- Fingerprints end at the prospective entry close.
- Changing any later candle must not change the fingerprint.
- Historical candidates must have completed their entire 30-day outcome window before the first holdout entry.
- Chronological splits are made by signal time.
- Retrieval retains at most one snapshot from each historical signal episode.
- Future outcomes select no features and do not affect similarity.

The POC is a challenger, not a dashboard change. It does not modify route libraries, installed models or live selectors.

## Measurements

The same retrieved neighbours estimate outcomes at 1 day, 7 days and 30 days:

- Fill-probability Brier score.
- Median additional-adverse-movement absolute error.
- Median waiting-time absolute error among entries that fill inside the horizon.

Lower values are better. Improvements are reported relative to State and geometry. A visually compelling route is not sufficient for promotion.

## Run

```powershell
.venv\Scripts\python.exe .\evaluate_behavioral_fingerprint.py --max-queries 1500 --neighbors 32
```

The ignored detailed report is written to `data/behavioral_fingerprint_poc_5m.json`.

## Promotion boundary

Even a positive POC would not immediately replace Adaptive, V3 or the numerical risk card. The next gate would require broader chronological folds, explicit age/distance/direction subgroups, stability across assets and exact production-route reconstruction. A similarity percentile should be calibrated from held-out distances before the dashboard presents it as a user-facing support score.

## Recorded result

The first bounded run completed on 27 September 2026. It used:

- 19,459 total five-minute observation snapshots.
- 15,379 label-mature historical candidates from 8,496 distinct signal episodes.
- 1,500 chronological holdout queries from 874 distinct signal episodes.
- 32 distinct historical episodes per query.

Every candidate's entire 30-day outcome window ended before the first holdout entry close. The table below gives episode-balanced error reduction relative to the controlled State and geometry fingerprint; positive values are improvements.

| Fingerprint | Horizon | Fill Brier | Adverse P50 MAE | Wait P50 MAE |
| --- | ---: | ---: | ---: | ---: |
| Context summary | 1 day | +1.76% | +2.48% | +1.13% |
| Context summary | 7 days | +3.07% | +1.29% | +4.13% |
| Context summary | 30 days | -1.64% | +1.15% | +2.40% |
| Sequence shape | 1 day | **+5.56%** | **+3.34%** | +2.03% |
| Sequence shape | 7 days | **+4.55%** | +0.21% | **+4.15%** |
| Sequence shape | 30 days | **-5.30%** | **-2.47%** | **+5.61%** |

The strongest episode-bootstrap evidence was:

- Sequence shape improved 1-day fill Brier and adverse error; both 95% delta intervals were entirely below zero.
- Sequence shape improved 7-day fill Brier and waiting-time error; both intervals were entirely below zero.
- Sequence shape improved 30-day waiting-time error by 5.61%, with a paired bootstrap delta interval of -194.6 to -34.1 minutes.
- The same 30-day fingerprint was probably worse for fill Brier and did not improve adverse error. Its probability of beating State and geometry on fill Brier was only 3%.

The result supports the central idea that observed path shape contains useful information. It rejects the stronger idea that one equally weighted fingerprint should select analogues for every forecast horizon. A promising next experiment is a validation-selected, horizon-specific blend: retain more sequence weight for short-horizon fill and timing questions, while allowing state/context to dominate longer-horizon fill and adverse-risk estimates.

This is not a comparison with the exact deployed soft-state selector and does not score reconstructed Fast/Normal/Extreme candle routes. It is a controlled nearest-neighbour outcome experiment. No dashboard method or installed model was promoted.

## Validation-selected horizon fingerprints

A second experiment tested 21 block-weight and sequence-view configurations without choosing them on the final holdout. It used 11,336 label-mature historical observations and 1,000 validation queries for selection, followed by the same 15,379 historical observations and 1,500 later holdout queries for the one-time final comparison.

For each horizon, validation minimized the mean episode-balanced error ratio across fill Brier, adverse P50 error and waiting-time P50 error. No selected configuration was allowed to worsen any individual validation metric by more than 1%.

Validation selected:

- **1 day:** state + context + half-weight recent sequence + half-weight full signal-to-entry sequence.
- **7 days:** state + context + half-weight full signal-to-entry sequence.
- **30 days:** state and geometry only. Validation rejected every richer fingerprint under the no-regression rule.

The frozen selections produced these episode-balanced results on the later holdout:

| Horizon | Fill Brier | Adverse P50 MAE | Wait P50 MAE | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 1 day | **+4.02%** | **+3.49%** | +3.28% | All three improved; fill and adverse bootstrap intervals were entirely favorable |
| 7 days | **+4.14%** | -0.52% | +1.86% | Fill improvement was statistically strong; adverse remained effectively flat |
| 30 days | 0.00% | 0.00% | 0.00% | Validation retained the baseline and avoided the first POC's long-horizon regressions |

The 1-day wait improvement was favored in 95.8% of episode bootstraps, although its 95% interval narrowly crossed zero. The 7-day fill improvement was favored in 98.3% of bootstraps, with a paired Brier-score delta interval from -0.00536 to -0.00023.

This is stronger evidence than simply taking the best result from the first holdout table. It confirms the main finding: recent and signal-to-entry path shape adds useful short/medium-horizon information, but long-horizon fill and adverse risk need a different representation.

The optimization report is `data/behavioral_fingerprint_optimized_5m.json`, generated by:

```powershell
.venv\Scripts\python.exe .\optimize_behavioral_fingerprint.py
```

The holdout has now been inspected and must not become a tuning set. Further gains should come from rolling chronological folds, a newly accumulated future period, alternative long-horizon regime features, and eventually an exact comparison with the production selector—not repeated optimization against these same 1,500 queries.

The follow-up rolling harness is documented in `ROLLING_FORECAST_EVALUATION.md`. It compares the current 53-feature supervised model, the same model plus the sequence fingerprint, and horizon-specific retrieval on identical fold queries and targets. The existing V3 neural artifact is explicitly excluded until it can be retrained inside every fold.
