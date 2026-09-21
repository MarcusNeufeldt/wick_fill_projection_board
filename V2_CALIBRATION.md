# V2 Conditional Wick Risk Calibration

**Status:** diagnostic prototype only. V1 real historical candle trajectories remain unchanged.

## Holdout result

- Cases: 1200 chronological holdout snapshots from 5021 disjoint episodes.
- Remaining-bars: 84.2% interval coverage (nominal 80%); p50 median absolute error 15.
- Future max move-away: 85.5% interval coverage (nominal 80%); p50 median absolute error 0.3651.
- Lower-tail check: future-move p10 covered 5.8% versus 10% nominal; do not use that lower bound as a calibrated risk limit yet.
- Evaluation window starts 2025-09-20T16:20:00Z; deterministic caps used 18000 train and 1200 holdout snapshots.

## Leakage guard

- Train labels had resolved by 2025-09-19T14:55:00Z; the earliest scored holdout snapshot closed at 2025-09-20T16:50:00Z (1555.0 minutes later).
- Embargo: 288 five-minute bars; same-episode overlap: 0.
- The split is chronological by episode signal time; there is no random row split.

## Scope and target

- Population: only 5m ETHUSDT/BTCUSDT/SOLUSDT/UNIUSDT/NEARUSDT strict-wick episodes that eventually made the library's clean fill. This is **not** an unconditional fill probability.
- Snapshot features use signal-time fields plus completed candles up to the snapshot close. Labels are subsequent bars to the fill and the maximum direction-normalized candle-envelope move away after the snapshot through the terminal fill candle, matching V1 displayed-path semantics.

## Limitations

- Conditional calibration on historical clean fills is not predictive-edge evidence and excludes no-fill, invalidated, and censored signals.
- The local empirical quantiles are a transparent baseline; calibration can drift by regime, asset, direction, and sparse long-duration states.
- The local dashboard may display this optional static-artifact diagnostic beside V1 routes; it never retrains V2, alters V1 route selection, or supplies a position-sizing rule.

## Later live-display interface

- `python train_conditional_wick_v2.py --print-input-schema` prints the required observable-state JSON contract.
- `python train_conditional_wick_v2.py --predict-json F:\explore\candle_projection_algo\data\conditional_path_library_5y_5assets\v2_models\conditional_wick_v2_example_observable_state.json --model-path F:\explore\candle_projection_algo\data\conditional_path_library_5y_5assets\v2_models\conditional_wick_v2_model.pkl` emits one quantile response without retraining.
- Python callers can use `load_model_bundle(path)` and `predict_from_observable_state(model, state)` from this script; dashboard use is optional and artifact-backed rather than live retraining.

Machine-readable summary: `F:\explore\candle_projection_algo\data\conditional_path_library_5y_5assets\v2_models\conditional_wick_v2_summary.json`
Holdout predictions: `F:\explore\candle_projection_algo\data\conditional_path_library_5y_5assets\v2_models\conditional_wick_v2_holdout_predictions.csv`
Reusable local model bundle: `F:\explore\candle_projection_algo\data\conditional_path_library_5y_5assets\v2_models\conditional_wick_v2_model.pkl`
