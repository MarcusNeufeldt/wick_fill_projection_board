# Frozen forecast architecture v1

`forecast_architecture_v1_5m_2026-09-27` is the live experimental numerical
forecast for every supported 5-minute pair. It is frozen while genuinely new
predictions and outcomes accumulate.

## Ownership

| Dashboard output | Owner |
| --- | --- |
| Fill probability at 1/7/30 days | E0 prediction-only numerical stack |
| Additional adverse p50 | E0 prediction-only numerical stack |
| Additional adverse p80/p90 | A calibrated tail |
| Waiting-time p10/p50/p90 | A supervised model |
| Historical support | C2 dynamic weighted retrieval |
| Projected candles | C2 real-historical candidates aligned to E0/A time-risk targets |

Historical support is a familiarity/difficulty disclosure, not a probability
or confidence percentage. It is not fed back into E0.

## Route contract

Adaptive 5m uses the same frozen artifact to retrieve distinct C2 historical
episodes and then selects four drawable, completed historical suffixes:

- Fast targets 1-day waiting-time p10 and adverse p50.
- Normal targets 7-day waiting-time p50 and adverse p50.
- Adverse first targets 1-day adverse p80 and requires a definite threshold
  crossing before the terminal fill candle.
- Extreme targets 30-day waiting-time p90 and adverse p90.

The selector balances duration compatibility, projected adverse compatibility
and C2 fingerprint distance. Adverse first deliberately gives most of its
selection weight to adverse-size compatibility; Extreme continues to represent
combined long-wait and p90-tail stress. Route adverse percentages use the
observable entry close as their denominator, matching the numerical model.
E0/A provide numerical targets; they do not
generate candles. Every displayed route remains one rescaled real historical
continuation, and is therefore an illustration rather than a route probability
or calibrated full-path interval. If the frozen artifact or enough drawable C2
episodes are unavailable, Adaptive retains the V1 routes and exposes the
fallback reason. `Legacy V3 comparison` is opt-in and never silently replaces
Adaptive Normal. The C2-aligned route contract is currently 5m-only; 1m
Adaptive retains V1 until an equivalent 1m architecture is validated.

## Runtime

The runtime pointer and immutable artifact are:

```text
%LOCALAPPDATA%\candle_projection_algo\prospective_entry_models\5m\active.json
%LOCALAPPDATA%\candle_projection_algo\prospective_entry_models\5m\forecast_architecture_v1_5m_2026-09-27.joblib
```

`active.json` pins the architecture version, SHA-256, frozen flag and training
label cutoff. The dashboard verifies the hash before loading the model and also
checks that the embedded architecture ID matches the manifest. A changed or
corrupt artifact therefore cannot silently continue under the frozen V1 name.

If it cannot be loaded, the dashboard automatically falls back to the previous
`5m/model.joblib` and marks the result `legacy_fallback`. One-minute views
continue using their existing artifact until the rolling experiment is repeated
on 1-minute data.

Every distinct live 5-minute forecast snapshot is recorded once in:

```text
%LOCALAPPDATA%\candle_projection_algo\prospective_validation\forecasts.sqlite
```

Each row records `architecture_version`, `artifact_hash`, and `forecast_source`.
The primary key includes architecture version, pair, pinned signal and current
candle close, so dashboard refreshes cannot duplicate a prediction. Prospective
V1 scoring reads the `forecast_v1_evaluation` view, which excludes fallback rows.

## Rebuild

The complete manual updater rebuilds the data and trains a newly versioned V1
candidate after the existing 1m/5m numerical models:

```powershell
.venv\Scripts\python.exe wick_update.py update
```

While prospective validation is collecting, the updater saves that artifact
under `5m\candidates\` and is forbidden from changing `active.json`. It never
overwrites or silently repoints the frozen V1 artifact.

For a 5-minute-only candidate rebuild from the current local outcomes:

```powershell
$env:CANDLE_PROJECTION_TRAIN_JOBS='1'
$candidate = "forecast_architecture_v1_5m_candidate_$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))"
$output = Join-Path $env:LOCALAPPDATA "candle_projection_algo\prospective_entry_models\5m\candidates\$candidate.joblib"
.venv\Scripts\python.exe train_forecast_architecture_v1.py --architecture-id $candidate --output $output
```

Setting one training worker is recommended on machines with limited free RAM.
The trainer itself refuses to overwrite the artifact named by a frozen active
manifest. Activating any candidate is a separate, explicit future decision.

## Current artifact

- Generated: 2026-09-27 06:51:30 UTC
- Mature observations: 19,156
- Distinct episodes: 10,666
- Assets: ETHUSDT, BTCUSDT, SOLUSDT, UNIUSDT, NEARUSDT
- Training labels resolved through: 2026-09-22 10:39:59.999 UTC
- Status: frozen experimental architecture; prospective validation collecting
- SHA-256: `4092d958e15d9bfe98f02fa4206b46c06b14f7a7afe4316bdac18110140b0059`

The rolling five-fold results remain development evidence. They are documented
in `ROLLING_FORECAST_EVALUATION.md` and are not presented as an untouched final
holdout.
