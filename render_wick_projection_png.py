"""Render a portable static image of the current wick-fill analogue projection."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "current_wick_projection.json"
OUTPUT = ROOT / "wick-fill-projection.png"


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def draw_candles(ax, candles, width_days: float, alpha: float = 1.0) -> None:
    """Draw OHLC candles without needing an external finance plotting package."""
    for candle in candles:
        when = mdates.date2num(timestamp(candle["open_time_utc"]))
        opn, high, low, close = (float(candle[key]) for key in ("open", "high", "low", "close"))
        up = close >= opn
        colour = "#48d597" if up else "#ff6b7a"
        ax.vlines(when, low, high, color=colour, linewidth=0.85, alpha=alpha, zorder=3)
        body_low = min(opn, close)
        body_height = max(abs(close - opn), 0.04)
        ax.add_patch(
            Rectangle(
                (when - width_days / 2, body_low),
                width_days,
                body_height,
                facecolor=colour,
                edgecolor=colour,
                alpha=alpha,
                linewidth=0.0,
                zorder=4,
            )
        )


def aggregate_projection(candles, projection_start: datetime, bucket_size: int = 12):
    """Convert the 5-minute projection to readable one-hour candles."""
    result = []
    for start in range(0, len(candles), bucket_size):
        group = candles[start : start + bucket_size]
        if not group:
            continue
        result.append(
            {
                "open_time_utc": (
                    projection_start + timedelta(minutes=float(group[0]["minutes_from_now"]))
                ).isoformat(),
                "open": group[0]["open"],
                "high": max(float(row["high"]) for row in group),
                "low": min(float(row["low"]) for row in group),
                "close": group[-1]["close"],
                "envelope_low_p10": min(float(row["envelope_low_p10"]) for row in group),
                "envelope_high_p90": max(float(row["envelope_high_p90"]) for row in group),
            }
        )
    return result


def dollars(value: float) -> str:
    return f"${value:,.2f}"


def main() -> None:
    with INPUT.open(encoding="utf-8") as handle:
        data = json.load(handle)

    signal = data["signal"]
    current = data["current"]
    selection = data["analogue_selection"]
    actual = data["actual_candles_30m"]
    projection_start = timestamp(data["current"]["open_time_utc"]) + timedelta(minutes=5)
    projection = aggregate_projection(data["projection_candles_5m"], projection_start)

    target = float(signal["wick_extreme_price"])
    current_price = float(current["close"])
    analogue_count = int(selection["count"])
    median_minutes = float(selection["remaining_to_fill_minutes"]["median"])
    signal_time = timestamp(signal["open_time_utc"])
    source_end = timestamp(data["source_data_ends_utc"])

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    fig = plt.figure(figsize=(16, 9), facecolor="#0b1220")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.16, 0.84], wspace=0.18)
    header = fig.add_subplot(grid[0, :])
    observed_ax = fig.add_subplot(grid[1, 0])
    projected_ax = fig.add_subplot(grid[1, 1])

    for ax in (header, observed_ax, projected_ax):
        ax.set_facecolor("#0f1b2d")
    header.axis("off")

    header.text(
        0.02,
        0.72,
        "ETHUSDT wick-fill study — static analogue projection",
        color="#f7fbff",
        fontsize=21,
        fontweight="bold",
        va="center",
    )
    header.text(
        0.02,
        0.30,
        "Signal: 16 Sep 2026, 20:35 Berlin  •  5-minute Binance USD-M perpetual data",
        color="#b7c4d7",
        fontsize=10.5,
        va="center",
    )
    header.text(
        0.98,
        0.72,
        f"NOW  {dollars(current_price)}",
        color="#7fc9ff",
        fontsize=15,
        fontweight="bold",
        ha="right",
        va="center",
    )
    header.text(
        0.98,
        0.30,
        f"WICK TARGET  {dollars(target)}",
        color="#ffc56b",
        fontsize=10.5,
        ha="right",
        va="center",
    )

    draw_candles(observed_ax, actual, width_days=0.014, alpha=0.94)
    observed_ax.axhline(target, color="#ffc56b", linestyle=(0, (5, 4)), linewidth=1.4, zorder=2)
    observed_ax.axhline(current_price, color="#7fc9ff", linestyle=(0, (2, 3)), linewidth=1.0, alpha=0.9, zorder=2)
    observed_ax.scatter(
        mdates.date2num(signal_time), target, s=36, color="#ffc56b", edgecolor="#0f1b2d", linewidth=0.8, zorder=7
    )
    observed_ax.annotate(
        f"Signal wick\n{dollars(target)}",
        (mdates.date2num(signal_time), target),
        xytext=(12, 15),
        textcoords="offset points",
        color="#ffd28a",
        fontsize=9,
        arrowprops={"arrowstyle": "-", "color": "#ffc56b", "lw": 0.9},
    )
    observed_ax.set_title("Observed price path since the signal", loc="left", color="#f7fbff", fontsize=13, pad=14)
    observed_ax.text(
        0.0,
        1.01,
        "30-minute candles • price moved away before returning part-way",
        transform=observed_ax.transAxes,
        color="#9fb0c7",
        fontsize=9.5,
    )
    observed_high = max(float(row["high"]) for row in actual)
    observed_ax.set_ylim(target - 13, observed_high + 15)

    xs = [mdates.date2num(timestamp(row["open_time_utc"])) for row in projection]
    lower = [float(row["envelope_low_p10"]) for row in projection]
    upper = [float(row["envelope_high_p90"]) for row in projection]
    median_close = [float(row["close"]) for row in projection]
    projected_ax.fill_between(xs, lower, upper, color="#5e8eea", alpha=0.20, linewidth=0.0, zorder=1)
    projected_ax.plot(xs, median_close, color="#9dc3ff", linewidth=1.15, alpha=0.92, zorder=2)
    draw_candles(projected_ax, projection, width_days=0.026, alpha=0.74)
    projected_ax.axhline(target, color="#ffc56b", linestyle=(0, (5, 4)), linewidth=1.4, zorder=5)
    projected_ax.axhline(current_price, color="#7fc9ff", linestyle=(0, (2, 3)), linewidth=1.0, alpha=0.9, zorder=2)
    projected_ax.scatter(xs[0], current_price, s=36, color="#7fc9ff", edgecolor="#0f1b2d", linewidth=0.8, zorder=7)
    projected_ax.set_title("Conditional historical analogue path from now", loc="left", color="#f7fbff", fontsize=13, pad=14)
    projected_ax.text(
        0.0,
        1.01,
        f"{analogue_count} similar episodes that ultimately filled • 1-hour median candles • shaded range = 10–90%",
        transform=projected_ax.transAxes,
        color="#9fb0c7",
        fontsize=9.5,
    )

    for ax in (observed_ax, projected_ax):
        ax.grid(True, axis="y", color="#314055", alpha=0.52, linewidth=0.7)
        ax.grid(False, axis="x")
        ax.tick_params(colors="#b7c4d7", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#314055")
        ax.yaxis.set_major_formatter(lambda value, _: f"${value:,.0f}")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%H:%M", tz=signal_time.tzinfo))
        ax.set_xlabel("Berlin time", color="#b7c4d7", fontsize=9.5, labelpad=10)

    projected_ax.annotate(
        f"Conditional median remaining-to-fill time: {median_minutes / 1440:.1f} days",
        (0.02, 0.06),
        xycoords="axes fraction",
        color="#d9e7fb",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "fc": "#162943", "ec": "#38587e", "alpha": 0.95},
    )
    projected_ax.text(
        0.02,
        0.015,
        "This is a path comparison conditioned on historical fills — not a forecast guarantee or a fill probability.",
        transform=projected_ax.transAxes,
        color="#b7c4d7",
        fontsize=8.4,
    )

    fig.text(
        0.01,
        0.012,
        f"Data through {source_end.strftime('%d %b %Y %H:%M UTC')}  |  Generated from local historical-event study",
        color="#74849b",
        fontsize=8.5,
    )
    fig.savefig(OUTPUT, dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(OUTPUT)


if __name__ == "__main__":
    main()
