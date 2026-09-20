# V1 Conditional Path Replay Validation

Generated: `2026-09-20T08:30:49.382181Z`.

This is not a binary fill-probability test. It replays later clean-fill episodes at fixed, observable 5-minute offsets and asks whether the three historical path scenarios cover the eventual remaining time and maximum move away from the wick.

Leakage guard: a candidate analogue is eligible only after its own fill candle has closed before the replay snapshot. The evaluation population is therefore conditional on strict signals that later made a clean departure and fully filled within the 180-day library cap.

## Aggregate

- Cases: **24**
- Duration-envelope coverage: **0.75**
- Excursion-envelope coverage: **0.7917**
- Joint coverage: **0.7083**
- Median normal duration log-error: **1.0777**
- Median normal excursion absolute error: **1.3018 percentage points**

## By snapshot offset

| Offset | Cases | Duration coverage | Excursion coverage | Joint coverage |
| ---: | ---: | ---: | ---: | ---: |
| 60 bars | 8 | 0.875 | 0.75 | 0.75 |
| 240 bars | 8 | 0.625 | 0.875 | 0.625 |
| 960 bars | 8 | 0.75 | 0.75 | 0.75 |

## Interpretation

The three displayed paths are deliberately real historical episodes, not calibrated confidence intervals. Coverage is a usefulness diagnostic for risk sizing, not a probability guarantee. The next model gate is to compare this V1 replay against survival/quantile/conformal envelopes on the same frozen replay protocol before introducing a neural trajectory generator.

Detailed rows: `replay_validation_5m_cases.csv`. Machine-readable summary: `replay_validation_5m.json`.
