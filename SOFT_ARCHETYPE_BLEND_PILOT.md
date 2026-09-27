# Soft Archetype Blend Pilot

Reference: `ETHUSDT 5m 2026-09-16T18:35:00Z`. Upper and lower wicks are direction-mirrored.

The soft matcher keeps the complete eligible historical pool. It blends 70% adaptive signal-feature distance with 30% scaled candle-archetype distance, preserves live age/distance/peak/retracement matching, and retains a softened same-asset preference.

## Reference candle configuration

- Body: **0.266%** of total candle range
- Dominant wick: **85.855%** of range
- Opposite wick: **13.879%** of range
- Total range: **1.565%** of close and **3.724x** the prior 20-candle median range
- Volume: **3.099x** the prior 20-candle mean
- Direction-aligned prior one-hour return: **0.997%**

Target-like cases have root-mean-square normalized archetype distance <= 2.0. One distance unit is 0.75 percentage points of body share, 3 points of either wick share, a 1.5x range factor, a 2x volume factor, or 1 percentage point of aligned prior return.

## How many close historical candles existed before the pin?

Among **23188** five-asset 5m clean-fill episodes resolved before the reference candle:

- **1747** matched the body and mirrored wick shares within 0.5 / 2 / 2 percentage points.
- **9** also had absolute candle range within 25% of the reference.
- **0** also matched the relative-volatility regime within 25%.
- **0** matched the full range, volume, and prior-return configuration.

The exact full configuration therefore has no historical clone; the matcher must use graded similarity rather than pretend identical examples exist.

Target-like signals from the prior 12 months with clean fills available in the current library: **383**.
Paired replay cases: **32** across **24** distinct target episodes.

## Result

- Adaptive mean checkpoint path MAE: **1.486031 percentage points**
- Soft 30% blend mean checkpoint path MAE: **1.458769 percentage points**
- Soft-blend relative improvement: **1.83%**
- Soft-blend episode-clustered 95% interval: **-6.5% to 9.47%**
- Soft blend lower-error case fraction: **0.28125**
- Hard archetype mean checkpoint path MAE: **2.999272 percentage points**
- Hard-archetype relative improvement: **-101.83%**
- Hard-archetype episode-clustered 95% interval: **-157.05% to -52.0%**

## By signal age

| Snapshot age | Adaptive path MAE | Soft blend path MAE | Hard archetype path MAE | Paired cases |
| ---: | ---: | ---: | ---: | ---: |
| 12 bars (1.0h) | 0.652007 | 0.71364 | 0.696792 | 8 |
| 240 bars (20.0h) | 1.32238 | 1.049926 | 1.770851 | 8 |
| 1440 bars (120.0h) | 1.946091 | 2.123364 | 3.432077 | 8 |
| 2880 bars (240.0h) | 2.023648 | 1.948146 | 6.097368 | 8 |

This is a conditional clean-fill path experiment, not an eventual-fill probability test. The dashboard keeps this matcher opt-in until the chronological evidence is clearly useful.

Detailed rows: `soft_blend_pilot_cases.csv`. Machine summary: `soft_blend_pilot.json`.
