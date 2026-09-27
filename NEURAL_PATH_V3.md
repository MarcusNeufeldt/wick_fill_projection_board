# Neural historical-path retrieval (V3 experimental live selector)

This experiment tests whether a compact temporal convolutional network can learn which observed wick-path patterns are predictive while keeping the dashboard's projection layer grounded in real historical continuations.

The dashboard can use it for the **Normal** route inside validated 1m and 5m age regimes. Fast and Extreme remain V1 routes, and any missing artifact, unsupported age, or inference error automatically keeps the V1 Normal route. No 15m neural expert is trained or deployed.

## Inputs available at each snapshot

- 8 hours of pre-signal context, resampled to 96 steps so 1m and 5m models see the same wall-clock history.
- The latest 21 hours and 20 minutes of candles ending at the snapshot, resampled to 256 steps.
- A 128-step resampling of the complete signal-to-snapshot trajectory.
- Existing signal shape, distance, peak, retracement, volatility/volume, asset, and direction fields.

All price channels are direction-normalized relative to the pinned wick. Explicit scale remains in the static inputs so normalization does not make a 0.5% move indistinguishable from an 8% move.

## Supervision and retrieval

The 64-value embedding is trained against:

- remaining bars to fill;
- future maximum move away from the wick;
- distance from the target after 5, 15, 30, 60, 120, 240, 480, and 960 minutes; and
- target-directed versus adverse movement after 30, 60, and 240 minutes.

The experiment compares scalar, latent-neural, forecast-anchored, and hybrid retrieval. Forecast-anchored retrieval uses only the live prefix to predict a future signature, then looks for already-resolved historical continuations with a compatible realized signature. Hybrid retrieval takes the union of broad scalar and learned candidate pools, scores all candidate alignments, and only then reduces to one alignment per historical episode. The returned route is a real historical medoid, not a generated candle path.

## Leakage controls

- Outer and inner splits are chronological and episode-disjoint.
- Every fit label must have resolved before the earliest validation snapshot.
- Every outer-training label must have resolved before the earliest final-holdout snapshot.
- Scaling, neural weights, tree models, and embeddings are fit on permitted history only.
- Multiple snapshots from one episode receive inverse-frequency loss weights.

The first version remains conditional on completed clean fills because the current library does not export unresolved trajectories in an equivalent form. It is not an unconditional fill-probability model.

## Environment

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -r requirements-ml.txt
uv pip install --python .venv\Scripts\python.exe torch==2.13.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

The current workstation's RTX 3060 Ti is detected by the CUDA build. The script uses it automatically.

## Commands

Fast pipeline check:

```powershell
.venv\Scripts\python.exe train_neural_path_v3.py --smoke
```

Expanded corrected experiment:

```powershell
.venv\Scripts\python.exe train_neural_path_v3.py --device cuda --train-cap 40000 --validation-cap 6000 --holdout-cap 5000 --epochs 12 --batch-size 256 --snapshot-offsets 1,3,6,12,24,60,120,240,480,960,1440,2880 --retrieval-modes scalar,forecast_hybrid
```

Generated models, retrieval indexes, predictions, and reports stay under `data/neural_path_v3/` and are excluded from Git.

## Corrected expanded chronological result

Run date: 2026-09-21. The expanded model used 40,000 fit snapshots from 14,225 episodes, 6,000 validation snapshots from 2,409 later episodes, and 5,000 final-year holdout snapshots from 3,334 episodes. The best TCN checkpoint was epoch 6 on the RTX 3060 Ti.

| Method | Remaining-time p50 MAE | Excursion p50 MAE | Fixed-clock path MAE |
| --- | ---: | ---: | ---: |
| V1-style scalar real-path retrieval | 20.27 bars | 0.4006 pp | 0.4384 pp |
| Forecast-anchored + scalar real-path retrieval | 22.19 bars | 0.3675 pp | **0.4375 pp** |
| Neural direct diagnostic | 17.27 bars | 0.3366 pp | 0.4627 pp |
| Richer-feature quantile trees | **17.10 bars** | **0.3351 pp** | not a path selector |

The earlier 7.3% result survived the raw-label scoring correction but did not survive sample expansion as a snapshot-weighted claim. Across 5,000 holdout snapshots, aggregate improvement was only **0.2%**. When each wick episode receives equal weight, improvement was **6.3%**; the episode-clustered 95% interval for the absolute error delta was -0.0261 to -0.0161 percentage points. Episode-balanced improvement was positive on BTC (6.2%), ETH (9.6%), NEAR (6.4%), SOL (5.4%), and UNI (4.8%), and in both the first (6.8%) and later (5.1%) halves of the holdout year.

Evaluation now scores against separately retained raw future distances. The expanded holdout contained 18 denominator-floor rows and 19 clipped representation cells across 12 rows; neither floor nor clipping modifies evaluation truth.

## Live deployment gate

Performance depended strongly on snapshot age:

| Elapsed age | Episode-balanced path improvement |
| --- | ---: |
| 1–24 bars | +6.8% |
| 25–120 bars | +5.2% |
| 121–480 bars | -1.3% |
| 481–960 bars | -32.8% |
| 961–1,440 bars | -79.3% |
| 1,441–2,880 bars | -171.7% |

That original single-model result motivated separate age experts instead of one global gate. The current experimental live matrix is:

| Timeframe | Elapsed bars | Clock age | Selector | Holdout evidence |
| --- | ---: | ---: | --- | --- |
| 5m | 1–480 | 5 minutes–40 hours | Forecast-hybrid real path | Positive episode-balanced path improvement in all three promoted fresh bands |
| 5m | 481–2,880 | 40 hours–10 days | Forecast-hybrid real path | +2.8% to +4.3% episode-balanced improvement across the three mature bands |
| 1m | 1–2,400 | 1 minute–40 hours | Forecast-hybrid real path | 8,000 holdout states / 4,818 episodes; 4.1% row-weighted and 7.7% episode-balanced improvement |
| 1m | 2,401–14,400 | 40 hours–10 days | Neural-embedding real path | 6,000 holdout states / 832 episodes; 1.4% row-weighted and 1.0% episode-balanced improvement |

The 1m mature neural selector improved all three predefined age bands and four of five assets; SOL was 0.75% worse. Its episode bootstrap favored the neural selector in 98.8% of resamples. The alternative mature forecast-hybrid selector was 5.7% worse overall and was not deployed.

Pins older than ten days, every 15m pin, missing artifacts, and inference failures retain V1. These gates were selected from the recorded chronological holdouts and remain experimental; future promotion still requires later data or walk-forward confirmation.

The selected V3 cohort still produces intervals that are much too narrow: 28.6% remaining-time coverage and 46.9% excursion coverage for a nominal 80% range. They are retained as diagnostics in the API but are not presented as calibrated risk limits. The V2 card remains the separate time/risk layer.

The dashboard continues to draw one real historical continuation. The network selects evidence; it does not generate candles.
