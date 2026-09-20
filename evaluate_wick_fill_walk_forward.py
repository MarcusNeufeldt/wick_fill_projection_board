#!/usr/bin/env python3
"""Chronologically evaluate a small pooled k-nearest-neighbour wick-fill baseline.

This deliberately avoids random train/test splits.  Each fold trains only on
signals whose full future label horizon finished before the fold begins.  The
prediction target is the conservative ``clean_departure_then_fill`` outcome:
a later close first leaves the opposite side of the signal candle, then a
later candle fully touches the dominant wick extreme.  A departure/fill in the
same OHLC bar is not counted as clean because intrabar ordering is unknown.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


HORIZON_MINUTES = {"1d": 24 * 60, "7d": 7 * 24 * 60, "30d": 30 * 24 * 60}
CONTINUOUS_FEATURES = [
    ("body_pct_of_range", 1.0),
    ("dominant_wick_pct_of_range", 1.0),
    ("opposite_wick_pct_of_range", 0.8),
    ("range_pct_of_close", 0.7),
    ("log_range_vs_prior_20_median", 0.7),
    ("log_volume_vs_prior_20_mean", 0.5),
    ("aligned_prior_1h_return_pct", 0.5),
]
CATEGORY_PENALTIES = {"asset": 0.35, "timeframe": 0.25, "direction": 0.15}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp") as handle:
        temporary = Path(handle.name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def utc_iso(timestamp_ms: int) -> str:
    return pd.Timestamp(timestamp_ms, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc_iso(value: str) -> int:
    return int(pd.Timestamp(value).timestamp() * 1000)


def prepare_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path)
    required = {
        "asset",
        "timeframe",
        "direction",
        "open_time",
        "signal_open_time_utc",
        "body_pct_of_range",
        "dominant_wick_pct_of_range",
        "opposite_wick_pct_of_range",
        "range_pct_of_close",
        "range_vs_prior_20_median",
        "volume_vs_prior_20_mean",
        "aligned_prior_1h_return_pct",
    }
    for horizon in HORIZON_MINUTES:
        required.add(f"clean_departure_then_fill_{horizon}")
    missing = required.difference(panel.columns)
    if missing:
        raise RuntimeError(f"Panel is missing columns: {sorted(missing)}")
    panel = panel.copy()
    panel["open_time"] = pd.to_numeric(panel["open_time"], errors="raise").astype("int64")
    for column in ["range_vs_prior_20_median", "volume_vs_prior_20_mean"]:
        panel[column] = pd.to_numeric(panel[column], errors="raise")
        if (panel[column] <= 0).any():
            raise RuntimeError(f"{column} contains a non-positive value")
    panel["log_range_vs_prior_20_median"] = np.log(panel["range_vs_prior_20_median"])
    panel["log_volume_vs_prior_20_mean"] = np.log(panel["volume_vs_prior_20_mean"])
    for name, _weight in CONTINUOUS_FEATURES:
        panel[name] = pd.to_numeric(panel[name], errors="raise")
    for horizon in HORIZON_MINUTES:
        column = f"clean_departure_then_fill_{horizon}"
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
        values = set(panel[column].dropna().astype(int).unique())
        if not values.issubset({0, 1}):
            raise RuntimeError(f"{column} is not binary")
    return panel.sort_values("open_time", kind="stable").reset_index(drop=True)


def robust_scale(train_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.median(train_values, axis=0)
    mad = np.median(np.abs(train_values - center), axis=0) * 1.4826
    fallback = np.std(train_values, axis=0, ddof=0)
    scale = np.where(mad > 1e-12, mad, fallback)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return center, scale


def knn_predict(train: pd.DataFrame, test: pd.DataFrame, label_column: str, k: int, batch_size: int = 96) -> np.ndarray:
    feature_columns = [name for name, _weight in CONTINUOUS_FEATURES]
    weights = np.asarray([weight for _name, weight in CONTINUOUS_FEATURES], dtype=float)
    train_values = train[feature_columns].to_numpy(dtype=float)
    test_values = test[feature_columns].to_numpy(dtype=float)
    center, scale = robust_scale(train_values)
    train_scaled = (train_values - center) / scale * weights
    test_scaled = (test_values - center) / scale * weights
    labels = train[label_column].to_numpy(dtype=float)
    categories = {
        column: (train[column].astype(str).to_numpy(), test[column].astype(str).to_numpy(), penalty)
        for column, penalty in CATEGORY_PENALTIES.items()
    }
    neighbor_count = min(k, len(train))
    predictions = np.empty(len(test), dtype=float)
    for start in range(0, len(test), batch_size):
        stop = min(start + batch_size, len(test))
        distances = np.sum((test_scaled[start:stop, None, :] - train_scaled[None, :, :]) ** 2, axis=2)
        for train_values_category, test_values_category, penalty in categories.values():
            distances += (test_values_category[start:stop, None] != train_values_category[None, :]) * penalty**2
        nearest = np.argpartition(distances, neighbor_count - 1, axis=1)[:, :neighbor_count]
        nearest_distances = np.take_along_axis(distances, nearest, axis=1)
        nearest_labels = labels[nearest]
        inverse_distance = 1.0 / (np.sqrt(nearest_distances) + 0.20)
        predictions[start:stop] = np.sum(inverse_distance * nearest_labels, axis=1) / np.sum(inverse_distance, axis=1)
    return predictions


def roc_auc(y_true: np.ndarray, prediction: np.ndarray) -> float | None:
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(prediction, kind="mergesort")
    sorted_prediction = prediction[order]
    ranks = np.empty(len(prediction), dtype=float)
    start = 0
    while start < len(prediction):
        stop = start + 1
        while stop < len(prediction) and sorted_prediction[stop] == sorted_prediction[start]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    positive_rank_sum = float(ranks[y_true == 1].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def calibration_bins(y_true: np.ndarray, prediction: np.ndarray, bins: int = 5) -> list[dict[str, Any]]:
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    output: list[dict[str, Any]] = []
    for index in range(bins):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        mask = (prediction >= lower) & ((prediction < upper) if index < bins - 1 else (prediction <= upper))
        if not mask.any():
            continue
        output.append(
            {
                "range": f"{lower:.1f}-{upper:.1f}",
                "count": int(mask.sum()),
                "mean_prediction": float(prediction[mask].mean()),
                "observed_rate": float(y_true[mask].mean()),
            }
        )
    return output


def metrics_for_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    y_true = frame["observed"].to_numpy(dtype=int)
    prediction = frame["prediction"].to_numpy(dtype=float)
    baseline = frame["baseline_prediction"].to_numpy(dtype=float)
    brier = float(np.mean((prediction - y_true) ** 2))
    baseline_brier = float(np.mean((baseline - y_true) ** 2))
    return {
        "n_predictions": int(len(frame)),
        "positive_rate": float(y_true.mean()),
        "mean_prediction": float(prediction.mean()),
        "brier_score": brier,
        "baseline_brier_score": baseline_brier,
        "brier_skill_vs_fold_base_rate": None if baseline_brier == 0 else float(1.0 - brier / baseline_brier),
        "roc_auc": roc_auc(y_true, prediction),
        "calibration": calibration_bins(y_true, prediction),
    }


def make_folds(data_start_ms: int, data_end_exclusive_ms: int, initial_train_days: int, test_window_days: int, test_step_days: int) -> list[tuple[int, int]]:
    maximum_horizon_ms = max(HORIZON_MINUTES.values()) * 60_000
    first_test_start = data_start_ms + initial_train_days * 24 * 60 * 60 * 1000
    last_test_end_exclusive = data_end_exclusive_ms - maximum_horizon_ms
    window_ms = test_window_days * 24 * 60 * 60 * 1000
    step_ms = test_step_days * 24 * 60 * 60 * 1000
    folds: list[tuple[int, int]] = []
    test_start = first_test_start
    while test_start + window_ms <= last_test_end_exclusive:
        folds.append((test_start, test_start + window_ms))
        test_start += step_ms
    return folds


def build_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Multi-Asset Wick-Fill Walk-Forward Results",
        "",
        f"Generated from `{summary['input_panel']}`.",
        "",
        "This is an out-of-sample evaluation of a deliberately simple pooled k-nearest-neighbour baseline. It is research evidence, not a trade recommendation or a claim that every wick will fill.",
        "",
        "## Protocol",
        "",
        f"- Panel: BTCUSDT and ETHUSDT perpetuals, 5m plus 15m (the 15m bars are derived from aligned 5m bars).",
        f"- Test folds: {summary['config']['fold_count']} chronological {summary['config']['test_window_days']}d windows after an initial {summary['config']['initial_train_days']}d warm-up.",
        "- For a horizon H, training ends before the test window by H. Thus no label can finish inside or after its own test fold.",
        "- Target: a clean departure first, then a later full dominant-wick touch. Same-bar departure/fill is not treated as clean.",
        "- The model pools assets/timeframes only after percent/ratio normalisation, while applying small mismatch penalties for asset, timeframe, and wick direction.",
        "",
        "## Aggregate out-of-sample score",
        "",
        "| Horizon | OOS cases | Positive rate | Brier | Fold-base Brier | Skill | ROC AUC |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for result in summary["targets"]:
        metrics = result["metrics"]
        skill = metrics["brier_skill_vs_fold_base_rate"]
        auc = metrics["roc_auc"]
        lines.append(
            "| {horizon} | {n:,} | {rate:.1%} | {brier:.4f} | {base:.4f} | {skill} | {auc} |".format(
                horizon=result["horizon"],
                n=metrics["n_predictions"],
                rate=metrics["positive_rate"],
                brier=metrics["brier_score"],
                base=metrics["baseline_brier_score"],
                skill="n/a" if skill is None else f"{skill:.1%}",
                auc="n/a" if auc is None else f"{auc:.3f}",
            )
        )
    lines.extend(
        [
            "",
            "Brier skill compares the kNN forecast with simply using the training-window base rate. Positive is better; negative means the simple base-rate forecast won.",
            "",
            "## Interpretation guardrails",
            "",
            "- A high raw wick-touch rate is not automatically predictive value; this evaluation intentionally scores the stricter departure-then-fill sequence.",
            "- BTC and ETH are correlated, and 5m/15m samples overlap. The panel expands pattern coverage but does not create 13,000 independent market experiments.",
            "- This baseline estimates an event probability only. It does not yet forecast the full candle path, drawdown, execution costs, or the probability of a liquidation before a fill.",
            "",
            "## Calibration",
            "",
        ]
    )
    for result in summary["targets"]:
        lines.append(f"### {result['horizon']}")
        lines.append("")
        lines.append("| Predicted bin | Cases | Mean forecast | Observed rate |")
        lines.append("| --- | ---: | ---: | ---: |")
        for item in result["metrics"]["calibration"]:
            lines.append(
                f"| {item['range']} | {item['count']:,} | {item['mean_prediction']:.1%} | {item['observed_rate']:.1%} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Files",
            "",
            f"- Predictions: `{summary['output_files']['predictions_csv']}`",
            f"- Fold diagnostics: `{summary['output_files']['fold_metrics_csv']}`",
            f"- Machine-readable metrics: `{summary['output_files']['metrics_json']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=root / "data" / "multi_asset_panel" / "panel_signals.csv")
    parser.add_argument("--panel-summary", type=Path, default=root / "data" / "multi_asset_panel" / "panel_summary.json")
    parser.add_argument("--out-dir", type=Path, default=root / "data" / "multi_asset_panel" / "walk_forward")
    parser.add_argument("--k", type=int, default=75)
    parser.add_argument("--min-train", type=int, default=400)
    parser.add_argument("--initial-train-days", type=int, default=365)
    parser.add_argument("--test-window-days", type=int, default=30)
    parser.add_argument("--test-step-days", type=int, default=30)
    args = parser.parse_args()
    if args.k < 1 or args.min_train < 1:
        parser.error("--k and --min-train must be positive")

    panel_path = args.panel.resolve()
    summary_path = args.panel_summary.resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing panel summary: {summary_path}")
    with summary_path.open(encoding="utf-8") as handle:
        panel_summary = json.load(handle)
    data_start_ms = parse_utc_iso(panel_summary["common_source_window"]["start_open_time_utc"])
    data_end_exclusive_ms = parse_utc_iso(panel_summary["common_source_window"]["end_exclusive_utc"])
    panel = prepare_panel(panel_path)
    folds = make_folds(
        data_start_ms,
        data_end_exclusive_ms,
        args.initial_train_days,
        args.test_window_days,
        args.test_step_days,
    )
    if not folds:
        raise RuntimeError("No mature walk-forward folds were available")
    print(json.dumps({"stage": "loaded_panel", "rows": int(len(panel)), "fold_count": len(folds)}), flush=True)

    prediction_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    for horizon_name, horizon_minutes in HORIZON_MINUTES.items():
        label_column = f"clean_departure_then_fill_{horizon_name}"
        horizon_ms = horizon_minutes * 60_000
        print(json.dumps({"stage": "evaluate_horizon", "horizon": horizon_name}), flush=True)
        horizon_predictions: list[pd.DataFrame] = []
        for fold_index, (test_start_ms, test_end_ms) in enumerate(folds, start=1):
            train_cutoff_ms = test_start_ms - horizon_ms
            train = panel.loc[(panel["open_time"] < train_cutoff_ms) & panel[label_column].notna()].copy()
            test = panel.loc[
                (panel["open_time"] >= test_start_ms)
                & (panel["open_time"] < test_end_ms)
                & panel[label_column].notna()
            ].copy()
            if len(train) < args.min_train or test.empty:
                fold_rows.append(
                    {
                        "horizon": horizon_name,
                        "fold": fold_index,
                        "test_start_utc": utc_iso(test_start_ms),
                        "test_end_exclusive_utc": utc_iso(test_end_ms),
                        "train_cutoff_utc": utc_iso(train_cutoff_ms),
                        "n_train": int(len(train)),
                        "n_test": int(len(test)),
                        "status": "skipped_insufficient_training_or_test",
                    }
                )
                continue
            predictions = knn_predict(train, test, label_column, args.k)
            baseline_rate = float(train[label_column].mean())
            output = test[["asset", "timeframe", "direction", "open_time", "signal_open_time_utc"]].copy()
            output["horizon"] = horizon_name
            output["fold"] = fold_index
            output["test_start_utc"] = utc_iso(test_start_ms)
            output["train_cutoff_utc"] = utc_iso(train_cutoff_ms)
            output["observed"] = test[label_column].astype(int).to_numpy()
            output["prediction"] = predictions
            output["baseline_prediction"] = baseline_rate
            output["n_train"] = len(train)
            horizon_predictions.append(output)
            fold_metrics = metrics_for_predictions(output)
            fold_rows.append(
                {
                    "horizon": horizon_name,
                    "fold": fold_index,
                    "test_start_utc": utc_iso(test_start_ms),
                    "test_end_exclusive_utc": utc_iso(test_end_ms),
                    "train_cutoff_utc": utc_iso(train_cutoff_ms),
                    "n_train": int(len(train)),
                    "n_test": int(len(test)),
                    "status": "scored",
                    "positive_rate": fold_metrics["positive_rate"],
                    "brier_score": fold_metrics["brier_score"],
                    "baseline_brier_score": fold_metrics["baseline_brier_score"],
                    "brier_skill_vs_fold_base_rate": fold_metrics["brier_skill_vs_fold_base_rate"],
                    "roc_auc": fold_metrics["roc_auc"],
                }
            )
        if not horizon_predictions:
            raise RuntimeError(f"No scoreable folds for {horizon_name}")
        merged = pd.concat(horizon_predictions, ignore_index=True)
        prediction_parts.append(merged)
        stratum_metrics: list[dict[str, Any]] = []
        for (asset, timeframe), group in merged.groupby(["asset", "timeframe"], sort=True):
            result = metrics_for_predictions(group)
            result.update({"asset": asset, "timeframe": timeframe})
            stratum_metrics.append(result)
        targets.append(
            {
                "horizon": horizon_name,
                "label_column": label_column,
                "metrics": metrics_for_predictions(merged),
                "strata": stratum_metrics,
            }
        )

    output_dir = args.out_dir.resolve()
    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_metrics_frame = pd.DataFrame.from_records(fold_rows)
    predictions_path = output_dir / "walk_forward_predictions.csv"
    fold_metrics_path = output_dir / "walk_forward_fold_metrics.csv"
    metrics_path = output_dir / "walk_forward_metrics.json"
    markdown_path = output_dir / "WALK_FORWARD_RESULTS.md"
    output_files = {
        "predictions_csv": str(predictions_path),
        "fold_metrics_csv": str(fold_metrics_path),
        "metrics_json": str(metrics_path),
        "markdown_report": str(markdown_path),
    }
    summary = {
        "schema_version": "1.0.0",
        "input_panel": str(panel_path),
        "input_panel_summary": str(summary_path),
        "common_source_window": panel_summary["common_source_window"],
        "config": {
            "model": "pooled robust-scaled k-nearest-neighbour probability baseline",
            "target": "clean departure then full wick fill",
            "k": args.k,
            "min_train": args.min_train,
            "initial_train_days": args.initial_train_days,
            "test_window_days": args.test_window_days,
            "test_step_days": args.test_step_days,
            "fold_count": len(folds),
            "continuous_features": [{"name": name, "weight": weight} for name, weight in CONTINUOUS_FEATURES],
            "categorical_mismatch_penalties": CATEGORY_PENALTIES,
            "no_lookahead_rule": "For horizon H, training rows must have a signal open time earlier than test_start minus H.",
        },
        "targets": targets,
        "output_files": output_files,
    }
    atomic_write_csv(predictions_path, predictions)
    atomic_write_csv(fold_metrics_path, fold_metrics_frame)
    atomic_write_json(metrics_path, summary)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=markdown_path.parent, suffix=".tmp") as handle:
        handle.write(build_markdown(summary))
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, markdown_path)
    print(
        json.dumps(
            {
                "metrics_json": str(metrics_path),
                "markdown_report": str(markdown_path),
                "predictions": int(len(predictions)),
                "aggregate": [
                    {
                        "horizon": target["horizon"],
                        "n_predictions": target["metrics"]["n_predictions"],
                        "brier_skill": target["metrics"]["brier_skill_vs_fold_base_rate"],
                        "roc_auc": target["metrics"]["roc_auc"],
                    }
                    for target in targets
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
