# Method Selection: From Historical Analogues to Calibrated Path Risk

## Short answer

The existing model is a transparent, state-matched historical-analogue method. It is not a neural network or a learned generative trajectory model.

That does **not** mean it is obsolete or the wrong starting point. With a few thousand completed wick episodes and only OHLCV inputs, a large neural model can easily learn market-regime memorisation and generate visually plausible but badly calibrated paths. A calibrated empirical baseline is the benchmark that any more advanced method must beat.

Rust is an implementation/runtime choice, not a forecasting method. Use Python for research, evaluation, and model selection; move the proven matching/serving path into Rust only when latency or throughput demands it.

## Decision for this project

Build and validate a state-conditioned empirical trajectory engine first. It selects real, joint historical paths and exposes fast / normal / extreme scenarios rather than inventing candles.

Then compare more advanced components one at a time against the same walk-forward replay:

| Stage | Method | Why it belongs here | Gate before adoption |
| --- | --- | --- | --- |
| V1 | Robust state-conditioned analogue cohort and trajectory medoids | Transparent, uses real paths, works with limited event counts | Calibrated replay coverage for time and excursion |
| V2a | Discrete-time survival / hazard model for time-to-fill | Naturally keeps censored, early-fill, and unresolved observations | Better time-to-fill quantile calibration than V1 |
| V2b | Regularised boosted quantile models for adverse excursion and remaining duration | Gives P25/P50/P90 risk estimates without a synthetic path generator | Improves pinball loss and interval coverage out of sample |
| V2c | Sequential conformal calibration | Adjusts empirical intervals when base estimates are miscalibrated | Measured regime-by-regime coverage, not a blanket guarantee |
| V3 | Transformer or diffusion trajectory model | Diverse sampled paths may help after richer features/data arrive | Blinded win over V1/V2 on final holdout, not visual appeal |

## Techniques worth using now

### State-conditioned empirical trajectories

This is the engine being built now. It uses real completed paths, direction normalisation, and a live state match: elapsed bars, distance from target, peak adverse move, drawdown, signal anatomy, market, timeframe, and context features.

The three scenarios must be actual joint episodes. If slow duration and large adverse move occur in different histories, the dashboard should expose two separate risk facts rather than make up a single fictional “worst case” path.

### Survival and quantile components

The right next learned models are small-data-friendly, separately testable components:

- a discrete-time survival/hazard estimate for remaining time to fill;
- quantile estimates for future maximum adverse excursion and duration;
- calibration around these values.

This is more useful for risk sizing than a black-box point forecast because it provides a distribution of waiting time and adverse move.

### Conformal calibration, used honestly

Conformalized Quantile Regression gives a useful general calibration pattern, but classic assumptions do not automatically hold for nonstationary crypto time series. Sequential Predictive Conformal Inference is directly relevant because it studies non-exchangeable time series. We will measure realised coverage by time and regime rather than claiming a universal finite-sample guarantee.

## Techniques to defer

### Transformers and diffusion models

Temporal Fusion Transformers are a sensible multi-horizon quantile benchmark once we have external features such as funding, open interest, liquidation, and regime variables. Diffusion-style forecasters such as TimeGrad can generate diverse trajectories.

Neither is automatically better for this problem. Plausible sampled paths are not necessarily calibrated paths, and five years of sparse wick episodes is still a small specialised data set for a high-capacity neural generator. They should be evaluated as V3 experiments, with frozen V1/V2 baselines and an untouched final holdout.

### Matrix Profile

Matrix Profile is useful as a fast, explainable subsequence-similarity index for initial pattern retrieval. It can improve matching speed or candidate discovery; it does not by itself predict the target path.

## Validation standard

For every historical replay time T:

1. Build the candidate library only from episodes resolved before T.
2. Apply an embargo so overlapping market episodes cannot leak into both the library and evaluation target.
3. Generate fast / normal / extreme scenarios and risk summaries.
4. Reveal the future path.
5. Score time-to-fill quantile calibration, pinball loss, maximum-adverse-excursion coverage, path-envelope coverage, and conditional CRPS where appropriate.

Freeze choices, then run the newest six months once as the final untouched holdout.

## Primary sources

- Yeh et al., [Matrix Profile I](https://ieeexplore.ieee.org/document/7837992/)
- Lee et al., [DeepHit: Deep Learning for Survival Analysis With Competing Risks](https://ojs.aaai.org/index.php/AAAI/article/view/11842)
- Romano et al., [Conformalized Quantile Regression](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html)
- Xu and Xie, [Sequential Predictive Conformal Inference for Time Series](https://proceedings.mlr.press/v202/xu23r.html)
- Lim et al., [Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://doi.org/10.1016/j.ijforecast.2021.03.012)
- Rasul et al., [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting (TimeGrad)](https://proceedings.mlr.press/v139/rasul21a.html)
