# Neural historical-path retrieval (V3 research challenger)

This experiment tests whether a compact temporal convolutional network can learn which observed wick-path patterns are predictive while keeping the dashboard's projection layer grounded in real historical continuations.

It is deliberately isolated from the live dashboard. Promotion requires an out-of-sample win over the scalar matcher and richer-feature tree baseline.

## Inputs available at each snapshot

- 8 hours of pre-signal 5m context.
- The latest 21 hours and 20 minutes of 5m candles ending at the snapshot.
- A 128-step resampling of the complete signal-to-snapshot trajectory.
- Existing signal shape, distance, peak, retracement, volatility/volume, asset, and direction fields.

All price channels are direction-normalized relative to the pinned wick. Explicit scale remains in the static inputs so normalization does not make a 0.5% move indistinguishable from an 8% move.

## Supervision and retrieval

The 64-value embedding is trained against:

- remaining bars to fill;
- future maximum move away from the wick;
- distance from the target after 1, 3, 6, 12, 24, 48, 96, and 192 bars; and
- target-directed versus adverse movement after 6, 12, and 48 bars.

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

Full first experiment:

```powershell
.venv\Scripts\python.exe train_neural_path_v3.py
```

Generated models, retrieval indexes, predictions, and reports stay under `data/neural_path_v3/` and are excluded from Git.

## First chronological holdout result

Run date: 2026-09-21. The model used 18,000 fit snapshots from 11,226 episodes, 2,000 validation snapshots from 1,562 later episodes, and an untouched final-year holdout of 1,200 snapshots from 1,153 episodes. The best TCN checkpoint was epoch 7 on the RTX 3060 Ti.

| Method | Remaining-time p50 MAE | Excursion p50 MAE | Fixed-clock path MAE |
| --- | ---: | ---: | ---: |
| V1-style scalar real-path retrieval | 18.16 bars | 0.3756 pp | 0.4159 pp |
| Latent-neural + scalar real-path retrieval | 16.97 bars | 0.3669 pp | 0.4102 pp |
| Forecast-anchored + scalar real-path retrieval | 18.00 bars | 0.2968 pp | **0.3854 pp** |
| Neural direct diagnostic | 15.02 bars | 0.2979 pp | 0.4072 pp |
| Richer-feature quantile trees | **13.71 bars** | 0.3019 pp | not a path selector |

The forecast-anchored real route reduced aggregate path error by 7.3%. Episode-balanced improvement was 8.1%; the episode-clustered 95% interval for the absolute error delta was -0.0405 to -0.0242 percentage points, entirely favoring the challenger. Path error improved separately on BTC (6.5%), ETH (8.9%), NEAR (7.7%), SOL (5.0%), and UNI (7.9%).

The same selected cohort produced intervals that were much too narrow: only 32.8% remaining-time coverage and 52.7% excursion coverage for a nominal 80% range. Therefore forecast-anchored retrieval is a route-selection result only. Time and risk ranges must remain separate and receive their own chronological calibration before dashboard integration.

This is one chronological holdout, not final promotion evidence. The next gate is repeated walk-forward folds followed by frozen route-family definitions.
