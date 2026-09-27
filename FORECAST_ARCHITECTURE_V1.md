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
| Projected candles | Separate real-historical illustration layer |

Historical support is a familiarity/difficulty disclosure, not a probability
or confidence percentage. It is not fed back into E0.

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
