# Pinned Candle Archetype Experiment

Reference: `ETHUSDT 5m 2026-09-16T18:35:00Z`. Upper and lower wicks are direction-mirrored.

## Reference candle configuration

- Body: **0.266%** of total candle range
- Dominant wick: **85.855%** of range
- Opposite wick: **13.879%** of range
- Total range: **1.565%** of close and **3.724x** the prior 20-candle median range
- Volume: **3.099x** the prior 20-candle mean
- Direction-aligned prior one-hour return: **0.997%**

Target-like cases have root-mean-square normalized archetype distance <= 2.0. One distance unit is 0.75 percentage points of body share, 3 points of either wick share, a 1.5x range factor, a 2x volume factor, or 1 percentage point of aligned prior return.

## How many close historical candles existed before the pin?

Among **23,188** five-asset 5m clean-fill episodes resolved before the reference candle:

- **1,747** matched the body and mirrored wick shares within 0.5 / 2 / 2 percentage points: BTC 141, ETH 141, SOL 329, UNI 560, and NEAR 576.
- **9** also had absolute candle range within 25% of the reference.
- **0** also matched the relative-volatility regime within 25%.
- **0** matched the full range, volume, and prior-return configuration.

The exact full configuration therefore has no historical clone; the matcher must use graded similarity rather than pretend identical examples exist.

Target-like signals from the prior 12 months with clean fills available in the current library: **383**.
Paired replay cases: **215** across **110** distinct target episodes.

## Result

- Adaptive mean checkpoint path MAE: **1.495506 percentage points**
- Archetype mean checkpoint path MAE: **2.848149 percentage points**
- Archetype mean-error change: **90.45% worse**
- Episode-clustered 95% range for that error increase: **64.43% to 115.61% worse**
- Archetype lower-error case fraction: **0.227907**

## By signal age

| Snapshot age | Adaptive path MAE | Archetype path MAE | Adaptive cases | Archetype cases |
| ---: | ---: | ---: | ---: | ---: |
| 12 bars (1.0h) | 0.790114 | 0.826306 | 48 | 48 |
| 60 bars (5.0h) | 1.142844 | 1.374757 | 48 | 48 |
| 240 bars (20.0h) | 1.568239 | 2.847307 | 48 | 48 |
| 960 bars (80.0h) | 2.048507 | 3.960448 | 31 | 31 |
| 1440 bars (120.0h) | 1.863469 | 4.378432 | 24 | 24 |
| 2880 bars (240.0h) | 2.596291 | 7.028067 | 12 | 12 |
| 8640 bars (720.0h) | 3.523473 | 14.459295 | 4 | 4 |

This is a conditional clean-fill path experiment, not an eventual-fill probability test. The dashboard keeps this matcher opt-in until the chronological evidence is clearly useful.

Detailed rows: `candle_archetype_replay_5m_cases.csv`. Machine summary: `candle_archetype_replay_5m.json`.
