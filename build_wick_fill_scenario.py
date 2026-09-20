"""Create one continuous, wick-fill-conditioned historical analogue scenario.

The scenario uses the selected historical episode whose remaining time-to-fill is
closest to the median among the 40 state-matched filled-wick analogues.  Its
post-alignment candle path is rescaled from the current wick target and current
price, so the displayed path starts at the actual latest close and ends at a
full touch of the current wick.
"""

from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
PROJECTION_INPUT = DATA / "current_wick_projection.json"
SOURCE_CANDLES = DATA / "ETHUSDT_5m_3y.csv"
EVENT_CANDLES = DATA / "filled_wick_study" / "filled_event_candles.csv.gz"
EVENTS = DATA / "filled_wick_study" / "filled_events.csv"
STUDY_SUMMARY = DATA / "filled_wick_study" / "summary.json"
OUTPUT = DATA / "current_wick_fill_scenario.json"


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def as_candle(time_seconds: int, open_: float, high: float, low: float, close: float) -> list[float | int]:
    """Compact [time, open, high, low, close] record for the local chart."""
    return [int(time_seconds), round(float(open_), 2), round(float(high), 2), round(float(low), 2), round(float(close), 2)]


def read_matching_event(event_id: str) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(EVENT_CANDLES, compression="gzip", chunksize=250_000):
        event_rows = chunk.loc[chunk["event_id"].eq(event_id)]
        if not event_rows.empty:
            chunks.append(event_rows)
    if not chunks:
        raise RuntimeError(f"Could not find event path for {event_id}")
    return pd.concat(chunks, ignore_index=True).sort_values("offset_candles")


def main() -> None:
    with PROJECTION_INPUT.open(encoding="utf-8") as handle:
        projection = json.load(handle)
    with STUDY_SUMMARY.open(encoding="utf-8") as handle:
        study_summary = json.load(handle)

    matches = projection["analogue_selection"]["matches"]
    median_remaining = sorted(float(match["remaining_to_fill_bars"]) for match in matches)[len(matches) // 2 - 1 : len(matches) // 2 + 1]
    median_remaining_bars = sum(median_remaining) / len(median_remaining)
    selected = min(
        matches,
        key=lambda match: abs(float(match["remaining_to_fill_bars"]) - median_remaining_bars),
    )
    selected_rank = next(index for index, match in enumerate(matches, start=1) if match["event_id"] == selected["event_id"])

    signal = projection["signal"]
    current = projection["current"]
    target = float(signal["wick_extreme_price"])
    current_close = float(current["close"])
    current_direction_sign = 1 if signal["direction"] == "lower_wick" else -1
    alignment_offset = int(selected["alignment_offset_bars"])
    historical_path = read_matching_event(str(selected["event_id"]))
    aligned = historical_path.loc[historical_path["offset_candles"].eq(alignment_offset)]
    if len(aligned) != 1:
        raise RuntimeError("The selected analogue does not have exactly one alignment candle")
    alignment_close_move = float(aligned.iloc[0]["normalized_close_pct"])
    if alignment_close_move <= 0:
        raise RuntimeError("The selected analogue alignment is not away from its wick")
    scale = float(current["current_move_pct"]) / alignment_close_move

    signal_ms = int(parse_utc(signal["open_time_utc"]).timestamp() * 1000)
    current_ms = int(parse_utc(current["open_time_utc"]).timestamp() * 1000)
    source = pd.read_csv(SOURCE_CANDLES, usecols=["open_time", "open", "high", "low", "close"])
    actual_source = source.loc[(source["open_time"] >= signal_ms) & (source["open_time"] <= current_ms)]
    if actual_source.empty:
        raise RuntimeError("Could not find actual candles from the signal to the latest close")
    actual = [
        as_candle(row.open_time // 1000, row.open, row.high, row.low, row.close)
        for row in actual_source.itertuples(index=False)
    ]
    if actual[-1][4] != round(current_close, 2):
        raise RuntimeError("The source candle close does not agree with the current projection state")

    def project_price(normalized_price_pct: float) -> float:
        return target * (1 + current_direction_sign * scale * float(normalized_price_pct) / 100)

    future_source = historical_path.loc[historical_path["offset_candles"].gt(alignment_offset)]
    if len(future_source) != int(selected["remaining_to_fill_bars"]):
        raise RuntimeError("Selected analogue remaining-bar count does not match its stored path")

    projection_start = parse_utc(current["open_time_utc"]) + timedelta(minutes=5)
    scenario: list[list[float | int]] = []
    for ordinal, row in enumerate(future_source.itertuples(index=False)):
        values = {
            "open": project_price(row.normalized_open_pct),
            "high": project_price(row.normalized_high_pct),
            "low": project_price(row.normalized_low_pct),
            "close": project_price(row.normalized_close_pct),
        }
        open_ = values["open"]
        close = values["close"]
        high = max(values.values())
        low = min(values.values())
        if ordinal == 0:
            # The first projected candle begins exactly from the actual latest close.
            open_ = current_close
            high = max(high, open_, close)
            low = min(low, open_, close)
        time_seconds = int((projection_start + timedelta(minutes=5 * ordinal)).timestamp())
        scenario.append(as_candle(time_seconds, open_, high, low, close))

    if scenario[0][0] != actual[-1][0] + 300:
        raise RuntimeError("Scenario must begin on the candle immediately after the actual series")
    if current_direction_sign == 1 and min(candle[3] for candle in scenario) > target + 0.01:
        raise RuntimeError("Lower-wick scenario did not touch the target")
    if current_direction_sign == -1 and max(candle[2] for candle in scenario) < target - 0.01:
        raise RuntimeError("Upper-wick scenario did not touch the target")

    terminal_open = datetime.fromtimestamp(scenario[-1][0], tz=timezone.utc)
    scenario_max_move_pct = max(
        current_direction_sign * (float(candle[2]) - target) / target * 100
        if current_direction_sign == 1
        else current_direction_sign * (float(candle[3]) - target) / target * 100
        for candle in scenario
    )
    completed_events = pd.read_csv(
        EVENTS,
        usecols=["direction", "max_away_move_pct_from_wick_extreme"],
    )
    current_peak = float(current["peak_move_pct"])
    reaches_current_peak = int(
        completed_events["max_away_move_pct_from_wick_extreme"].ge(current_peak).sum()
    )
    current_peak_percentile = float(
        completed_events["max_away_move_pct_from_wick_extreme"].lt(current_peak).mean() * 100
    )
    match_rows = [{"rank": rank, **match} for rank, match in enumerate(matches, start=1)]
    thresholds = study_summary["definition"]
    chart_data = {
        "symbol": "ETHUSDT perpetual",
        "interval_minutes": 5,
        "source": "Binance USD-M Futures",
        "source_data_ends_utc": projection["source_data_ends_utc"],
        "signal": {
            "open_time_utc": signal["open_time_utc"],
            "direction": signal["direction"],
            "wick_target": round(target, 2),
        },
        "current": {
            "open_time_utc": current["open_time_utc"],
            "close": round(current_close, 2),
            "move_from_wick_pct": round(float(current["current_move_pct"]), 3),
            "peak_move_from_wick_pct": round(float(current["peak_move_pct"]), 3),
        },
        "scenario": {
            "method": "median-duration state-matched filled-wick analogue",
            "selected_event_id": selected["event_id"],
            "selected_match_rank": selected_rank,
            "historical_signal_utc": selected["historical_signal_utc"],
            "historical_direction": selected["historical_direction"],
            "state_match_score": round(float(selected["score"]), 4),
            "alignment_offset_bars": alignment_offset,
            "remaining_bars": int(selected["remaining_to_fill_bars"]),
            "remaining_days": round(float(selected["remaining_to_fill_bars"]) * 5 / 1440, 2),
            "terminal_fill_candle_utc": iso_utc(terminal_open + timedelta(minutes=5)),
            "normalization_scale": round(scale, 6),
            "post_alignment_max_move_from_wick_pct": round(scenario_max_move_pct, 3),
        },
        "transparency": {
            "model_type": "rule-based nearest-historical-analogue model; no machine-learning parameters are fitted",
            "data_scope": {
                "history_start_utc": iso_utc(datetime.fromtimestamp(int(source["open_time"].min()) / 1000, tz=timezone.utc)),
                "history_end_utc_exclusive": study_summary["analysis_cutoff_utc_exclusive"],
                "interval_minutes": 5,
                "strict_shape_candidates": int(study_summary["candidate_signals"]),
                "clean_completed_departure_then_fill_events": int(study_summary["completed_filled_events"]),
                "filled_before_clear_departure": int(study_summary["outcomes"]["filled_before_departure"]),
                "right_censored_at_cutoff": int(study_summary["outcomes"]["right_censored"]),
                "lower_wick_completed_events": int(completed_events["direction"].eq("lower_wick").sum()),
                "upper_wick_completed_events": int(completed_events["direction"].eq("upper_wick").sum()),
            },
            "signal_definition": {
                "body_max_pct_of_range": thresholds["body_max_pct_of_range"],
                "dominant_wick_min_pct_of_range": thresholds["dominant_wick_min_pct_of_range"],
                "opposite_wick_max_pct_of_range": thresholds["opposite_wick_max_pct_of_range"],
                "departure_rule": thresholds["departure"],
                "fill_rule": thresholds["full_fill"],
                "same_bar_handling": thresholds["same_bar_ambiguity"],
            },
            "current_signal_features": {
                key: round(float(value), 6) for key, value in projection["signal_features"].items()
            },
            "current_state": {
                "elapsed_bars": int(current["elapsed_bars"]),
                "current_move_pct_from_wick": round(float(current["current_move_pct"]), 6),
                "peak_move_pct_from_wick": round(current_peak, 6),
                "drawdown_pct_from_peak": round(float(current["drawdown_from_peak_pct"]), 6),
                "completed_events_reaching_current_peak": reaches_current_peak,
                "current_peak_percentile_among_completed_events": round(current_peak_percentile, 2),
            },
            "matching": {
                "top_k": len(matches),
                "signal_shape_and_context_features": [
                    {"name": "body_pct_of_range", "weight": 0.8},
                    {"name": "dominant_wick_pct_of_range", "weight": 0.8},
                    {"name": "opposite_wick_pct_of_range", "weight": 0.8},
                    {"name": "range_vs_prior_20_median", "weight": 0.6},
                    {"name": "volume_vs_prior_20_mean", "weight": 0.6},
                    {"name": "prior_1h_return_pct", "weight": 0.4},
                ],
                "in_progress_state_weights": {
                    "current_move_pct_from_wick": 1.8,
                    "peak_move_pct_from_wick": 1.2,
                    "drawdown_pct_from_peak": 1.2,
                    "elapsed_bars": 0.4,
                },
                "matches_ranked_by_score": match_rows,
            },
            "scenario_selection": {
                "rule": "among the 40 scored analogues, choose the event with remaining time closest to their median",
                "not_the_same_as": "highest-probability outcome or the closest score match",
                "selected_rank_by_score": selected_rank,
                "selected_remaining_bars": int(selected["remaining_to_fill_bars"]),
                "median_remaining_bars_across_40": median_remaining_bars,
            },
        },
        "actual_5m": actual,
        "scenario_5m": scenario,
    }
    with OUTPUT.open("w", encoding="utf-8") as handle:
        json.dump(chart_data, handle, separators=(",", ":"), ensure_ascii=True)
    print(f"Wrote {OUTPUT} with {len(actual)} actual and {len(scenario)} scenario candles")


if __name__ == "__main__":
    main()
