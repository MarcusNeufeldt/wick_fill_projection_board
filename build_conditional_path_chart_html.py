"""Create a portable three-scenario TradingView Lightweight Charts viewer."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "data" / "current_conditional_path_scenarios.json"
OUTPUT = ROOT / "conditional-wick-path-dashboard.html"


HTML = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Conditional wick path console</title>
  <script src="https://unpkg.com/lightweight-charts@5.2.0/dist/lightweight-charts.standalone.production.js"></script>
  <style>
    :root {
      color-scheme: dark;
      --night: #10151d;
      --panel: #18212d;
      --panel-2: #202b38;
      --rule: #304052;
      --text: #e4e9ed;
      --muted: #96a6b6;
      --target: #eca846;
      --fast: #38b7a6;
      --normal: #8f93f7;
      --stress: #db7a54;
      --actual-up: #68c4b4;
      --actual-down: #d27571;
      --mono: "IBM Plex Mono", "Cascadia Mono", "SFMono-Regular", Consolas, monospace;
      --body: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--night); color: var(--text); font: 14px/1.42 var(--body); }
    main { max-width: 1640px; margin: 0 auto; padding: 24px clamp(14px, 3vw, 42px) 32px; }
    .eyebrow { color: var(--target); font: 11px/1 var(--mono); letter-spacing: .12em; text-transform: uppercase; }
    header { display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 24px; align-items: end; padding-bottom: 18px; border-bottom: 1px solid var(--rule); }
    h1 { margin: 8px 0 4px; font: 600 clamp(24px, 3vw, 38px)/1.08 var(--body); letter-spacing: -.04em; }
    .subhead { margin: 0; max-width: 720px; color: var(--muted); }
    .signal-stamp { font: 12px var(--mono); color: var(--muted); text-align: right; white-space: nowrap; }
    .signal-stamp b { display: block; margin-bottom: 3px; color: var(--text); font-weight: 500; }
    .state-strip { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1px; margin: 18px 0; background: var(--rule); border: 1px solid var(--rule); }
    .state { min-height: 72px; padding: 12px 14px; background: var(--panel); }
    .state label { display: block; color: var(--muted); font: 10px/1.2 var(--mono); letter-spacing: .08em; text-transform: uppercase; }
    .state strong { display: block; margin-top: 8px; font: 600 17px/1 var(--mono); font-variant-numeric: tabular-nums; }
    .route-heading { display: flex; justify-content: space-between; gap: 16px; align-items: end; margin: 24px 0 10px; }
    .route-heading h2 { margin: 0; font: 600 14px/1.2 var(--body); letter-spacing: .01em; }
    .route-heading p { margin: 0; color: var(--muted); font-size: 12px; }
    .route-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; }
    .route {
      appearance: none; position: relative; min-height: 144px; overflow: hidden; padding: 15px 16px 13px;
      border: 1px solid var(--rule); border-radius: 2px; background: var(--panel); color: var(--text); text-align: left; cursor: pointer;
      transition: border-color .16s ease, transform .16s ease, background .16s ease;
    }
    .route::before { content: ""; position: absolute; inset: 0 auto 0 0; width: 4px; background: var(--route-color); }
    .route:hover { transform: translateY(-2px); background: var(--panel-2); }
    .route[aria-pressed="true"] { border-color: var(--route-color); background: color-mix(in srgb, var(--route-color) 9%, var(--panel)); }
    .route-name { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; padding-left: 4px; }
    .route-name b { font: 600 15px/1 var(--body); }
    .route-name span { color: var(--route-color); font: 10px/1 var(--mono); letter-spacing: .12em; text-transform: uppercase; }
    .route-description { min-height: 37px; margin: 10px 0 13px; padding-left: 4px; color: var(--muted); font-size: 12px; }
    .route-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding-left: 4px; }
    .route-metrics label { display: block; color: var(--muted); font: 10px/1 var(--mono); text-transform: uppercase; }
    .route-metrics strong { display: block; margin-top: 5px; color: var(--text); font: 600 14px/1 var(--mono); }
    .route-rail { position: absolute; right: 14px; bottom: 13px; width: 44%; height: 2px; background: rgba(255,255,255,.12); }
    .route-rail::after { content: ""; position: absolute; left: 0; top: -2px; width: var(--rail); height: 6px; border-radius: 10px; background: var(--route-color); }
    .chart-shell { position: relative; min-height: 560px; margin-top: 12px; border: 1px solid var(--rule); background: #121a24; }
    #chart { width: 100%; height: min(72vh, 760px); min-height: 560px; }
    .chart-label { position: absolute; z-index: 3; left: 12px; top: 12px; padding: 7px 9px; border: 1px solid #334456; background: rgba(16,21,29,.88); color: var(--muted); font: 11px/1.25 var(--mono); pointer-events: none; }
    .chart-label b { display: block; margin-bottom: 2px; color: var(--text); font-weight: 600; }
    .chart-note { display: flex; gap: 10px; align-items: flex-start; margin: 12px 0 0; color: var(--muted); font-size: 12px; }
    .dot { flex: 0 0 auto; width: 8px; height: 8px; margin-top: 5px; border-radius: 50%; background: var(--target); }
    details { margin-top: 28px; border-top: 1px solid var(--rule); padding-top: 14px; }
    summary { cursor: pointer; color: var(--text); font-weight: 600; }
    .details-grid { display: grid; grid-template-columns: 1fr 1.3fr; gap: 24px; margin-top: 16px; }
    h3 { margin: 0 0 8px; color: var(--muted); font: 10px/1 var(--mono); letter-spacing: .1em; text-transform: uppercase; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 8px 0; border-bottom: 1px solid rgba(48,64,82,.8); text-align: left; }
    th { color: var(--muted); font-weight: 400; } td { font-family: var(--mono); text-align: right; }
    .method { margin: 0; color: var(--muted); font-size: 12px; }
    footer { margin-top: 18px; color: var(--muted); font-size: 11px; }
    a { color: #b3b6ff; }
    #error { display:none; color:#ffad9b; margin:12px 0; }
    @media (max-width: 860px) {
      header { grid-template-columns: 1fr; } .signal-stamp { text-align: left; } .state-strip { grid-template-columns: repeat(2, 1fr); }
      .route-grid { grid-template-columns: 1fr; } .details-grid { grid-template-columns: 1fr; } #chart, .chart-shell { min-height: 440px; height: 62vh; }
    }
    @media (prefers-reduced-motion: reduce) { .route { transition: none; } .route:hover { transform: none; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="eyebrow">Conditional wick path console</div>
        <h1 id="title">Wick target trajectories</h1>
        <p class="subhead">Choose a historical path regime. Each route is a rescaled, completed historical episode—not an unconditional forecast or a trade instruction.</p>
      </div>
      <div class="signal-stamp"><b id="signal-time"></b><span id="source-window"></span></div>
    </header>

    <section class="state-strip" aria-label="Pinned signal and current state">
      <div class="state"><label>Wick target</label><strong id="target"></strong></div>
      <div class="state"><label>Latest close</label><strong id="latest"></strong></div>
      <div class="state"><label>Current distance</label><strong id="current-distance"></strong></div>
      <div class="state"><label>Peak away move</label><strong id="peak-distance"></strong></div>
      <div class="state"><label>Elapsed</label><strong id="elapsed"></strong></div>
    </section>

    <section>
      <div class="route-heading"><h2>Choose a conditional route</h2><p id="cohort-note"></p></div>
      <div class="route-grid" id="route-grid"></div>
    </section>

    <section class="chart-shell">
      <div class="chart-label"><b id="selected-name"></b><span id="selected-detail"></span></div>
      <div id="chart" role="img" aria-label="Actual ETHUSDT candles followed by a selected historical conditional wick-fill trajectory"></div>
    </section>
    <p class="chart-note"><i class="dot"></i><span id="chart-note"></span></p>

    <details open>
      <summary>Method and risk context</summary>
      <div class="details-grid">
        <section><h3>Matched cohort after current state</h3><table><tbody id="cohort-table"></tbody></table></section>
        <section><h3>What this chart is—and is not</h3><p class="method" id="method"></p></section>
      </div>
    </details>
    <p id="error" role="alert">The TradingView chart library did not load. Reopen this file with an internet connection.</p>
    <footer>Chart engine: <a href="https://www.tradingview.com/" target="_blank" rel="noreferrer">TradingView Lightweight Charts™</a></footer>
  </main>

  <script id="scenario-data" type="application/json">__DATA__</script>
  <script>
    (() => {
      const data = JSON.parse(document.getElementById('scenario-data').textContent);
      const colors = { fast: '#38b7a6', normal: '#8f93f7', extreme: '#db7a54' };
      const labels = { fast: 'Fast', normal: 'Normal', extreme: 'Extreme' };
      const money = value => `$${Number(value).toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
      const pct = value => `${Number(value).toFixed(2)}%`;
      const duration = bars => {
        const minutes = Number(bars) * data.pinned_signal.timeframe.replace('m','');
        return minutes >= 1440 ? `${(minutes / 1440).toFixed(1)}d` : `${Math.round(minutes)}m`;
      };
      const berlin = seconds => new Intl.DateTimeFormat('en-GB', {timeZone:'Europe/Berlin', day:'2-digit', month:'short', year:'numeric', hour:'2-digit', minute:'2-digit', hour12:false}).format(new Date(seconds * 1000));
      const current = data.current_state;
      document.getElementById('title').textContent = `${data.pinned_signal.asset} ${data.pinned_signal.timeframe} / ${data.pinned_signal.direction.replace('_', ' ')}`;
      document.getElementById('signal-time').textContent = `Pinned: ${data.pinned_signal.signal_open_time_utc.replace('T',' ').replace('Z',' UTC')}`;
      document.getElementById('source-window').textContent = `${data.library.same_timeframe_trajectory_candidates_before_signal.toLocaleString()} resolved ${data.pinned_signal.timeframe} episodes available before this signal`;
      document.getElementById('target').textContent = money(data.pinned_signal.wick_target);
      document.getElementById('latest').textContent = money(current.as_of_close);
      document.getElementById('current-distance').textContent = pct(current.current_move_pct);
      document.getElementById('peak-distance').textContent = pct(current.peak_move_pct);
      document.getElementById('elapsed').textContent = `${(current.elapsed_minutes / 1440).toFixed(1)}d`;
      document.getElementById('cohort-note').textContent = `Top ${data.library.top_k_state_matched_episodes} closest states · P50 remaining ${Math.round(data.cohort_distribution.remaining_time_to_fill_minutes.p50 / 60)}h · P90 adverse move ${pct(data.cohort_distribution.future_max_away_move_pct.p90)}`;
      document.getElementById('cohort-table').innerHTML = [
        ['Resolved episodes before signal', data.library.eligible_completed_episodes_before_signal.toLocaleString()],
        ['Same-timeframe trajectory candidates', data.library.same_timeframe_trajectory_candidates_before_signal.toLocaleString()],
        ['State-matched cohort', data.library.top_k_state_matched_episodes.toLocaleString()],
        ['Remaining time P25 / P50 / P90', `${Math.round(data.cohort_distribution.remaining_time_to_fill_minutes.p25/60)}h / ${Math.round(data.cohort_distribution.remaining_time_to_fill_minutes.p50/60)}h / ${(data.cohort_distribution.remaining_time_to_fill_minutes.p90/1440).toFixed(1)}d`],
        ['Future move-away P50 / P90', `${pct(data.cohort_distribution.future_max_away_move_pct.p50)} / ${pct(data.cohort_distribution.future_max_away_move_pct.p90)}`],
      ].map(([key, value]) => `<tr><th>${key}</th><td>${value}</td></tr>`).join('');
      document.getElementById('method').textContent = data.conditionality + ' Direct scenario candles preserve the pinned timeframe: a 15m historical path is not stretched into invented 5m candles. The current weights and scenario selection remain research settings until chronological path-coverage validation is complete.';

      const routeGrid = document.getElementById('route-grid');
      const risk = data.scenarios.map(s => s.joint_risk_percentile);
      data.scenarios.forEach((scenario, index) => {
        const button = document.createElement('button');
        button.className = 'route'; button.type = 'button'; button.dataset.scenario = scenario.name;
        button.style.setProperty('--route-color', colors[scenario.name]);
        button.style.setProperty('--rail', `${Math.max(8, Math.min(100, scenario.joint_risk_percentile * 100))}%`);
        button.setAttribute('aria-pressed', index === 1 ? 'true' : 'false');
        button.innerHTML = `<div class="route-name"><b>${labels[scenario.name]}</b><span>risk p${Math.round(scenario.joint_risk_percentile * 100)}</span></div>
          <p class="route-description">${scenario.description}</p>
          <div class="route-metrics"><div><label>to wick touch</label><strong>${duration(scenario.remaining_to_fill_bars)}</strong></div><div><label>future move-away</label><strong>${pct(scenario.future_max_away_move_pct)}</strong></div></div><i class="route-rail"></i>`;
        button.addEventListener('click', () => selectScenario(scenario.name));
        routeGrid.appendChild(button);
      });

      if (!window.LightweightCharts) { document.getElementById('error').style.display = 'block'; return; }
      const chartNode = document.getElementById('chart');
      const chart = LightweightCharts.createChart(chartNode, {
        layout: { background: { color: '#121a24' }, textColor: '#aab8c5', fontFamily: getComputedStyle(document.body).fontFamily },
        grid: { vertLines: { color: 'rgba(48,64,82,.48)' }, horzLines: { color: 'rgba(48,64,82,.48)' } },
        rightPriceScale: { borderColor: '#304052' }, timeScale: { borderColor: '#304052', timeVisible: true, secondsVisible: false },
        crosshair: { vertLine: { color: '#60748a', labelBackgroundColor: '#263545' }, horzLine: { color: '#60748a', labelBackgroundColor: '#263545' } },
        handleScroll: true, handleScale: true,
      });
      const actual = chart.addSeries(LightweightCharts.CandlestickSeries, { upColor:'#68c4b4', downColor:'#d27571', borderVisible:false, wickUpColor:'#68c4b4', wickDownColor:'#d27571' });
      const projected = chart.addSeries(LightweightCharts.CandlestickSeries, { upColor:colors.normal, downColor:colors.normal, borderUpColor:colors.normal, borderDownColor:colors.normal, wickUpColor:colors.normal, wickDownColor:colors.normal });
      actual.setData(data.actual_candles);
      actual.createPriceLine({ price:data.pinned_signal.wick_target, color:'#eca846', lineWidth:2, lineStyle:LightweightCharts.LineStyle.Dashed, axisLabelVisible:true, title:'wick target' });
      let selected = null;
      function selectScenario(name) {
        selected = data.scenarios.find(item => item.name === name);
        document.querySelectorAll('.route').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.scenario === name)));
        const color = colors[name];
        projected.applyOptions({ upColor:color, downColor:color, borderUpColor:color, borderDownColor:color, wickUpColor:color, wickDownColor:color });
        projected.setData(selected.projected_candles);
        document.getElementById('selected-name').textContent = `${labels[name]} route — conditional historical trajectory`;
        document.getElementById('selected-detail').textContent = `${duration(selected.remaining_to_fill_bars)} remaining · historical ${selected.historical_asset} ${selected.historical_timeframe} · terminal touch ${berlin(selected.projected_candles[selected.projected_candles.length - 1].time + Number(data.pinned_signal.timeframe.replace('m','')) * 60)}`;
        document.getElementById('chart-note').textContent = `${labels[name]} is one rescaled historical episode selected from the matched cohort. Its future move-away before the wick touch was ${pct(selected.future_max_away_move_pct)} in its original normalised history. It is a scenario reference, not a probability-weighted forecast.`;
        chart.timeScale().fitContent();
      }
      const observer = new ResizeObserver(entries => { for (const entry of entries) chart.applyOptions({ width: entry.contentRect.width, height: entry.contentRect.height }); });
      observer.observe(chartNode);
      selectScenario('normal');
    })();
  </script>
</body>
</html>'''


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    OUTPUT.write_text(HTML.replace("__DATA__", payload), encoding="utf-8")
    print(json.dumps({"input": str(INPUT), "output": str(OUTPUT), "scenarios": [item["name"] for item in data["scenarios"]]}))


if __name__ == "__main__":
    main()
