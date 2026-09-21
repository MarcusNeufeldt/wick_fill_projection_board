# V1 Conditional Path Replay Validation

Generated: `2026-09-20T16:00:19.610493Z`.

This is not a binary fill-probability test. It replays later clean-fill episodes at fixed, observable 5-minute offsets and asks whether the three historical path scenarios cover the eventual remaining time and maximum move away from the wick.

Leakage guard: a candidate analogue is eligible only when its own terminal fill candle has closed at or before the replay snapshot. Both target and analogue states must be after confirmed departure. The evaluation population is therefore conditional on strict signals that later made a clean departure and fully filled within the 180-day library cap.

## Aggregate

- Cases: **96**
- Duration-envelope coverage: **0.8125**
- Excursion-envelope coverage: **0.6979**
- Joint coverage: **0.625**
- Median normal duration log-error: **1.2239**
- Median normal excursion absolute error: **1.0001 percentage points**

## By snapshot offset

| Offset | Cases | Duration coverage | Excursion coverage | Joint coverage |
| ---: | ---: | ---: | ---: | ---: |
| 12 bars | 24 | 0.7917 | 0.6667 | 0.625 |
| 60 bars | 24 | 0.875 | 0.7083 | 0.6667 |
| 240 bars | 24 | 0.7917 | 0.6667 | 0.5833 |
| 960 bars | 24 | 0.7917 | 0.75 | 0.625 |

## Interpretation

The three displayed paths are deliberately real historical episodes, not calibrated confidence intervals. Coverage is a usefulness diagnostic for risk sizing, not a probability guarantee. The next model gate is to compare this V1 replay against survival/quantile/conformal envelopes on the same frozen replay protocol before introducing a neural trajectory generator.

Detailed rows: `replay_validation_5m_cases.csv`. Machine-readable summary: `replay_validation_5m.json`.
