# V2 Conditional Wick Risk Calibration

**Status:** diagnostic prototype only. V1 real historical candle trajectories remain unchanged.

## Holdout result

- Cases: 1200 chronological holdout snapshots from 2046 disjoint episodes.
- Remaining-bars: 81.8% interval coverage (nominal 80%); p50 median absolute error 16.36.
- Future max move-away: 85.9% interval coverage (nominal 80%); p50 median absolute error 0.2396.
- Lower-tail check: future-move p10 covered 4.4% versus 10% nominal; do not use that lower bound as a calibrated risk limit yet.
- Evaluation window starts 2025-09-20T05:00:00Z; deterministic caps used 18000 train and 1200 holdout snapshots.

## Leakage guard

- Train labels had resolved by 2025-09-19T03:25:00Z; the earliest scored holdout snapshot closed at 2025-09-20T06:35:00Z (1630.0 minutes later).
- Embargo: 288 five-minute bars; same-episode overlap: 0.
- The split is chronological by episode signal time; there is no random row split.

## Scope and target

- Population: only 5m ETHUSDT/BTCUSDT strict-wick episodes that eventually made the library's clean fill. This is **not** an unconditional fill probability.
- Snapshot features use signal-time fields plus completed candles up to the snapshot close. Labels are remaining bars and maximum direction-normalized move away from the wick through the fill (including the observed snapshot candle, matching V1 replay semantics).

## Limitations

- Conditional calibration on historical clean fills is not predictive-edge evidence and excludes no-fill, invalidated, and censored signals.
- The local empirical quantiles are a transparent baseline; calibration can drift by regime, asset, direction, and sparse long-duration states.
- V2 has no live selector, UI, or position-sizing integration. Treat the JSON/CSV as a risk diagnostic beside V1 paths.

## Later live-display interface

- `python train_conditional_wick_v2.py --print-input-schema` prints the required observable-state JSON contract.
- `python train_conditional_wick_v2.py --predict-json F:\explore\candle_projection_algo\data\conditional_path_library_5y\v2_models\conditional_wick_v2_example_observable_state.json --model-path F:\explore\candle_projection_algo\data\conditional_path_library_5y\v2_models\conditional_wick_v2_model.pkl` emits one quantile response without retraining.
- Python callers can use `load_model_bundle(path)` and `predict_from_observable_state(model, state)` from this script; no server wiring is included.

- Machine-readable summary: `F:\explore\candle_projection_algo\data\conditional_path_library_5y\v2_models\conditional_wick_v2_summary.json`
- Holdout predictions: `F:\explore\candle_projection_algo\data\conditional_path_library_5y\v2_models\conditional_wick_v2_holdout_predictions.csv`
- Reusable local model bundle: `F:\explore\candle_projection_algo\data\conditional_path_library_5y\v2_models\conditional_wick_v2_model.pkl`
