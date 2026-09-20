"""Build a portable TradingView Lightweight Charts HTML scenario viewer."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "current_wick_fill_scenario.json"
OUTPUT = ROOT / "wick-fill-scenario-chart.html"


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ETHUSDT wick-fill scenario</title>
  <script src="https://unpkg.com/lightweight-charts@5.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root {
      color-scheme: dark;
      --bg: #131722;
      --surface: #1e222d;
      --grid: #2a2e39;
      --text: #d1d4dc;
      --muted: #787b86;
      --actual: #26a69a;
      --scenario: #6f8dff;
      --target: #f5a623;
      --danger: #ef5350;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .shell { max-width: 1560px; margin: 0 auto; padding: 22px 24px 14px; }
    header { display: flex; flex-wrap: wrap; gap: 16px 28px; align-items: end; justify-content: space-between; }
    h1 { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: .01em; }
    .subtitle { margin: 5px 0 0; color: var(--muted); }
    .stats { display: flex; flex-wrap: wrap; gap: 18px; color: var(--muted); }
    .stat strong { display: block; color: var(--text); font-size: 16px; font-variant-numeric: tabular-nums; font-weight: 600; }
    .stat span { font-size: 12px; }
    .toolbar { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 10px; margin: 20px 0 10px; }
    .periods { display: inline-flex; gap: 5px; }
    button {
      border: 1px solid #363a45;
      border-radius: 4px;
      padding: 6px 10px;
      background: transparent;
      color: var(--muted);
      cursor: pointer;
      font: inherit;
    }
    button[aria-pressed="true"] { color: var(--text); border-color: var(--scenario); background: rgba(111, 141, 255, .14); }
    .key { display: flex; flex-wrap: wrap; gap: 12px; color: var(--muted); font-size: 12px; }
    .key span { display: inline-flex; align-items: center; gap: 6px; }
    .swatch { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }
    .actual { background: var(--actual); }
    .scenario { background: var(--scenario); }
    .target { background: var(--target); }
    #chart-wrap { position: relative; height: min(72vh, 760px); min-height: 430px; border: 1px solid #2a2e39; }
    #chart { width: 100%; height: 100%; }
    #tooltip {
      position: absolute;
      top: 12px;
      left: 12px;
      display: none;
      min-width: 180px;
      padding: 8px 10px;
      background: rgba(30, 34, 45, .94);
      border: 1px solid #3b404e;
      border-radius: 4px;
      color: var(--text);
      font-size: 12px;
      pointer-events: none;
      font-variant-numeric: tabular-nums;
    }
    #tooltip .date { color: var(--muted); margin-bottom: 4px; }
    #tooltip .row { display: flex; justify-content: space-between; gap: 16px; }
    #tooltip .label { color: var(--muted); }
    .scenario-note { margin: 12px 0 0; color: var(--muted); font-size: 12px; }
    .transparency { margin-top: 20px; border-top: 1px solid #2a2e39; padding-top: 12px; }
    .transparency summary { cursor: pointer; color: var(--text); font-weight: 600; }
    .transparency > div { margin-top: 14px; }
    .transparency-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px 24px; }
    .transparency h2 { margin: 0 0 7px; font-size: 14px; font-weight: 600; }
    .transparency p { margin: 0; color: var(--muted); font-size: 12px; }
    .transparency table { width: 100%; border-collapse: collapse; color: var(--text); font-size: 12px; }
    .transparency th, .transparency td { padding: 6px 5px; border-bottom: 1px solid #2a2e39; text-align: left; vertical-align: top; }
    .transparency th { color: var(--muted); font-weight: 500; }
    .transparency td:last-child, .transparency th:last-child { text-align: right; }
    .pass { color: #26a69a; }
    .details-note { margin-top: 10px !important; }
    .matches { grid-column: 1 / -1; overflow-x: auto; }
    .matches table { min-width: 640px; }
    footer { display: flex; flex-wrap: wrap; gap: 8px 18px; justify-content: space-between; margin-top: 16px; color: var(--muted); font-size: 11px; }
    a { color: #8aa7ff; }
    #chart-error { display: none; margin: 18px 0; color: #ff8a80; }
    @media (max-width: 640px) {
      .shell { padding: 16px 12px 10px; }
      h1 { font-size: 18px; }
      #chart-wrap { min-height: 390px; height: 62vh; }
      .toolbar { align-items: flex-start; }
      .transparency-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div>
        <h1>ETHUSDT perpetual — wick-fill scenario</h1>
        <p class="subtitle" id="source-label"></p>
      </div>
      <div class="stats" aria-label="Scenario summary">
        <div class="stat"><strong id="now-price"></strong><span>latest close</span></div>
        <div class="stat"><strong id="target-price"></strong><span>wick-fill target</span></div>
        <div class="stat"><strong id="duration"></strong><span>scenario to target touch</span></div>
      </div>
    </header>

    <div class="toolbar">
      <div class="periods" role="group" aria-label="Candle interval">
        <button type="button" data-minutes="5" aria-pressed="false">5m</button>
        <button type="button" data-minutes="30" aria-pressed="false">30m</button>
        <button type="button" data-minutes="60" aria-pressed="true">1h</button>
      </div>
      <div class="key" aria-label="Chart legend">
        <span><i class="swatch actual"></i>actual candles</span>
        <span><i class="swatch scenario"></i>historical-analogue scenario</span>
        <span><i class="swatch target"></i>wick target</span>
      </div>
    </div>

    <div id="chart-wrap">
      <div id="chart" role="img" aria-label="ETHUSDT actual candles followed by a wick-fill-conditioned analogue scenario"></div>
      <div id="tooltip" aria-live="polite"></div>
    </div>
    <p class="scenario-note" id="scenario-note"></p>
    <details class="transparency" open>
      <summary>Transparency — exact signal rule, historical sample, matching, and this scenario</summary>
      <div class="transparency-grid">
        <section>
          <h2>Current signal qualifies as</h2>
          <table><tbody id="signal-definition"></tbody></table>
        </section>
        <section>
          <h2>Historical labels before this signal</h2>
          <table><tbody id="data-scope"></tbody></table>
        </section>
        <section>
          <h2>Current state versus completed events</h2>
          <table><tbody id="current-state"></tbody></table>
        </section>
        <section>
          <h2>How analogue ranking works</h2>
          <p id="matching-rule"></p>
          <p class="details-note" id="scenario-rule"></p>
        </section>
        <section class="matches">
          <h2>Top ten of the 40 scored historical analogues</h2>
          <table>
            <thead><tr><th>Rank</th><th>Direction</th><th>Historical signal</th><th>Score</th><th>Remaining to fill</th></tr></thead>
            <tbody id="match-rows"></tbody>
          </table>
        </section>
      </div>
    </details>
    <p id="chart-error" role="alert">The chart engine did not load. Open this file with an internet connection so it can load the TradingView chart library.</p>
    <footer>
      <span id="data-through"></span>
      <span>Chart engine: <a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">TradingView Lightweight Charts™</a></span>
    </footer>
  </main>

  <script id="scenario-data" type="application/json">__DATA__</script>
  <script>
    (() => {
      const data = JSON.parse(document.getElementById('scenario-data').textContent);
      const chartNode = document.getElementById('chart');
      const tooltip = document.getElementById('tooltip');
      const fmtMoney = (value) => `$${Number(value).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
      const berlinDate = (seconds, withTime = true) => new Intl.DateTimeFormat('en-GB', {
        timeZone: 'Europe/Berlin', month: 'short', day: '2-digit', year: 'numeric',
        ...(withTime ? { hour: '2-digit', minute: '2-digit', hour12: false } : {}),
      }).format(new Date(seconds * 1000));

      document.getElementById('source-label').textContent = `${data.source} • source interval: ${data.interval_minutes}m • signal: ${berlinDate(data.actual_5m[0][0])} Berlin`;
      document.getElementById('now-price').textContent = fmtMoney(data.current.close);
      document.getElementById('target-price').textContent = fmtMoney(data.signal.wick_target);
      document.getElementById('duration').textContent = `${data.scenario.remaining_days.toFixed(1)} days`;
      document.getElementById('data-through').textContent = `Actual data through ${berlinDate(data.actual_5m[data.actual_5m.length - 1][0])} Berlin • scenario terminal touch: ${berlinDate(data.scenario_5m[data.scenario_5m.length - 1][0] + 300)} Berlin`;
      document.getElementById('scenario-note').textContent = `Scenario uses the state-matched filled-wick episode closest to the 40-analogue median duration (${data.scenario.remaining_bars.toLocaleString()} five-minute candles). It is rescaled to begin at the latest ETH close and ends only when the ${fmtMoney(data.signal.wick_target)} wick is touched.`;

      const transparency = data.transparency;
      const signalFeatures = transparency.current_signal_features;
      const definition = transparency.signal_definition;
      const percent = (value, decimals = 2) => `${Number(value).toFixed(decimals)}%`;
      const valueRow = (label, value, extra = '') => `<tr><th>${label}</th><td>${value}</td>${extra ? `<td>${extra}</td>` : ''}</tr>`;
      document.getElementById('signal-definition').innerHTML = [
        `<tr><th>Direction</th><td>${data.signal.direction.replace('_', ' ')}</td><td class="pass">pass</td></tr>`,
        `<tr><th>Body of range</th><td>${percent(signalFeatures.body_pct_of_range, 3)} / max ${percent(definition.body_max_pct_of_range, 0)}</td><td class="pass">pass</td></tr>`,
        `<tr><th>Dominant wick</th><td>${percent(signalFeatures.dominant_wick_pct_of_range, 2)} / min ${percent(definition.dominant_wick_min_pct_of_range, 0)}</td><td class="pass">pass</td></tr>`,
        `<tr><th>Opposite wick</th><td>${percent(signalFeatures.opposite_wick_pct_of_range, 2)} / max ${percent(definition.opposite_wick_max_pct_of_range, 0)}</td><td class="pass">pass</td></tr>`,
        valueRow('Range versus prior 20', `${Number(signalFeatures.range_vs_prior_20_median).toFixed(2)}x`),
        valueRow('Volume versus prior 20', `${Number(signalFeatures.volume_vs_prior_20_mean).toFixed(2)}x`),
        valueRow('Prior one-hour return', percent(signalFeatures.prior_1h_return_pct, 2)),
      ].join('');
      const scope = transparency.data_scope;
      document.getElementById('data-scope').innerHTML = [
        valueRow('Study interval', `${berlinDate(Math.floor(new Date(scope.history_start_utc).getTime() / 1000), false)} to the signal, 5m ETHUSDT`),
        valueRow('Strict wick-shaped candles', scope.strict_shape_candidates.toLocaleString()),
        valueRow('Clean departure then fill paths', scope.clean_completed_departure_then_fill_events.toLocaleString()),
        valueRow('Lower / upper completed paths', `${scope.lower_wick_completed_events.toLocaleString()} / ${scope.upper_wick_completed_events.toLocaleString()}`),
        valueRow('Filled before clear departure', `${scope.filled_before_clear_departure.toLocaleString()} (not used as paths)`),
        valueRow('Right-censored at cutoff', scope.right_censored_at_cutoff.toLocaleString()),
      ].join('');
      const state = transparency.current_state;
      document.getElementById('current-state').innerHTML = [
        valueRow('Elapsed since signal', `${state.elapsed_bars.toLocaleString()} bars / ${(state.elapsed_bars * 5 / 60).toFixed(1)} h`),
        valueRow('Latest close from wick', percent(state.current_move_pct_from_wick, 3)),
        valueRow('Peak move from wick', percent(state.peak_move_pct_from_wick, 3)),
        valueRow('Drawdown from peak', percent(state.drawdown_pct_from_peak, 3)),
        valueRow('Completed paths at or beyond that peak', `${state.completed_events_reaching_current_peak.toLocaleString()} of ${scope.clean_completed_departure_then_fill_events.toLocaleString()}`),
        valueRow('Current peak percentile', `${state.current_peak_percentile_among_completed_events.toFixed(2)}th`),
      ].join('');
      const matchingFeatures = transparency.matching.signal_shape_and_context_features.map((feature) => feature.name.replaceAll('_', ' ')).join(', ');
      document.getElementById('matching-rule').textContent = `${transparency.model_type}. It ranks completed paths using signal anatomy and context (${matchingFeatures}) plus current move, peak, drawdown, and elapsed bars. The 40 lowest-distance historical states are retained.`;
      document.getElementById('scenario-rule').textContent = `This chart does not show the closest score match. It deliberately uses rank ${transparency.scenario_selection.selected_rank_by_score}, chosen because its ${transparency.scenario_selection.selected_remaining_bars.toLocaleString()} remaining bars are closest to the 40-match median of ${Math.round(transparency.scenario_selection.median_remaining_bars_across_40).toLocaleString()} bars. That makes it a representative duration scenario, not a probability forecast.`;
      document.getElementById('match-rows').innerHTML = transparency.matching.matches_ranked_by_score.slice(0, 10).map((match) => `<tr><td>${match.rank}</td><td>${match.historical_direction.replace('_', ' ')}</td><td>${berlinDate(Math.floor(new Date(match.historical_signal_utc).getTime() / 1000))}</td><td>${Number(match.score).toFixed(3)}</td><td>${(match.remaining_to_fill_bars * 5 / 1440).toFixed(1)} days</td></tr>`).join('');

      if (!window.LightweightCharts) {
        document.getElementById('chart-error').style.display = 'block';
        return;
      }

      const chart = LightweightCharts.createChart(chartNode, {
        width: chartNode.clientWidth,
        height: chartNode.clientHeight,
        layout: { background: { type: 'solid', color: '#131722' }, textColor: '#d1d4dc', fontSize: 12 },
        grid: { vertLines: { color: '#1e222d' }, horzLines: { color: '#2a2e39' } },
        rightPriceScale: { borderColor: '#2a2e39', scaleMargins: { top: 0.08, bottom: 0.08 } },
        timeScale: { borderColor: '#2a2e39', timeVisible: true, secondsVisible: false, rightOffset: 3, tickMarkFormatter: (time) => berlinDate(Number(time), false) },
        crosshair: { mode: LightweightCharts.CrosshairMode.Normal, vertLine: { labelBackgroundColor: '#363a45' }, horzLine: { labelBackgroundColor: '#363a45' } },
        handleScroll: true,
        handleScale: true,
      });

      const actualSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: '#26a69a', downColor: '#ef5350', wickUpColor: '#26a69a', wickDownColor: '#ef5350', borderVisible: false,
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
      });
      const scenarioSeries = chart.addSeries(LightweightCharts.CandlestickSeries, {
        upColor: '#6f8dff', downColor: '#d97a9b', wickUpColor: '#6f8dff', wickDownColor: '#d97a9b', borderVisible: false,
        priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
      });
      scenarioSeries.createPriceLine({
        price: data.signal.wick_target,
        color: '#f5a623', lineWidth: 2, lineStyle: LightweightCharts.LineStyle.Dashed,
        axisLabelVisible: true, title: 'wick fill',
      });
      actualSeries.createPriceLine({
        price: data.current.close,
        color: '#787b86', lineWidth: 1, lineStyle: LightweightCharts.LineStyle.Dotted,
        axisLabelVisible: true, title: 'now',
      });

      const asBars = (rows) => rows.map(([time, open, high, low, close]) => ({ time, open, high, low, close }));
      const aggregate = (rows, minutes) => {
        const groupSize = Math.max(1, minutes / data.interval_minutes);
        const grouped = [];
        for (let start = 0; start < rows.length; start += groupSize) {
          const group = rows.slice(start, start + groupSize);
          grouped.push([
            group[0][0], group[0][1], Math.max(...group.map((row) => row[2])),
            Math.min(...group.map((row) => row[3])), group[group.length - 1][4],
          ]);
        }
        return grouped;
      };

      const render = (minutes) => {
        actualSeries.setData(asBars(aggregate(data.actual_5m, minutes)));
        scenarioSeries.setData(asBars(aggregate(data.scenario_5m, minutes)));
        chart.timeScale().fitContent();
        document.querySelectorAll('[data-minutes]').forEach((button) => {
          button.setAttribute('aria-pressed', String(Number(button.dataset.minutes) === minutes));
        });
      };

      document.querySelectorAll('[data-minutes]').forEach((button) => {
        button.addEventListener('click', () => render(Number(button.dataset.minutes)));
      });

      chart.subscribeCrosshairMove((param) => {
        if (!param.time || !param.seriesData || param.seriesData.size === 0) {
          tooltip.style.display = 'none';
          return;
        }
        const actual = param.seriesData.get(actualSeries);
        const scenario = param.seriesData.get(scenarioSeries);
        const row = actual || scenario;
        const label = actual ? 'Actual' : 'Scenario';
        if (!row) {
          tooltip.style.display = 'none';
          return;
        }
        tooltip.innerHTML = `<div class="date">${berlinDate(Number(param.time))} Berlin · ${label}</div><div class="row"><span class="label">O</span><span>${fmtMoney(row.open)}</span></div><div class="row"><span class="label">H</span><span>${fmtMoney(row.high)}</span></div><div class="row"><span class="label">L</span><span>${fmtMoney(row.low)}</span></div><div class="row"><span class="label">C</span><span>${fmtMoney(row.close)}</span></div>`;
        tooltip.style.display = 'block';
      });

      new ResizeObserver((entries) => {
        const entry = entries[0];
        if (entry) chart.applyOptions({ width: Math.floor(entry.contentRect.width), height: Math.floor(entry.contentRect.height) });
      }).observe(chartNode);
      render(60);
    })();
  </script>
</body>
</html>
'''


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    compact_data = json.dumps(data, separators=(",", ":"), ensure_ascii=True).replace("</", "<\\/")
    OUTPUT.write_text(HTML.replace("__DATA__", compact_data), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
