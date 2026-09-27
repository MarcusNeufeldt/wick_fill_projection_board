use chrono::{DateTime, SecondsFormat, Utc};
use csv::WriterBuilder;
use flate2::Compression;
use flate2::write::GzEncoder;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const BODY_MAX_PCT: f64 = 0.05;
const DOMINANT_WICK_MIN_PCT: f64 = 0.75;
const OPPOSITE_WICK_MAX_PCT: f64 = 0.20;

#[derive(Clone, Copy)]
struct Candle {
    open_time: i64,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

#[derive(Clone, Copy)]
struct Signal {
    index: usize,
    direction_sign: i8,
    wick_target: f64,
    opposite_extreme: f64,
}

#[derive(Clone, Copy)]
struct TraceResult {
    status: &'static str,
    departure_index: Option<usize>,
    fill_index: Option<usize>,
}

struct Args {
    mode: String,
    input: PathBuf,
    output: Option<PathBuf>,
    path_output: Option<PathBuf>,
    path_format: String,
    asset: Option<String>,
    timeframe: String,
    maximum_fill_days: usize,
    maximum_rows: Option<usize>,
    minimum_open_time_ms: Option<i64>,
    maximum_open_time_exclusive_ms: Option<i64>,
    entry_ages_minutes: Vec<usize>,
    horizons_minutes: Vec<usize>,
    adverse_thresholds_pct: Vec<f64>,
}

fn parse_usize_list(value: &str, name: &str) -> Result<Vec<usize>, String> {
    let mut parsed = value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .map(|item| {
            item.trim()
                .parse::<usize>()
                .map_err(|_| format!("invalid {name}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    parsed.sort_unstable();
    parsed.dedup();
    if parsed.is_empty() || parsed[0] == 0 {
        return Err(format!("{name} must contain positive integers"));
    }
    Ok(parsed)
}

fn parse_f64_list(value: &str, name: &str) -> Result<Vec<f64>, String> {
    let mut parsed = value
        .split(',')
        .filter(|item| !item.trim().is_empty())
        .map(|item| {
            item.trim()
                .parse::<f64>()
                .map_err(|_| format!("invalid {name}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    parsed.sort_by(f64::total_cmp);
    parsed.dedup();
    if parsed.is_empty() || parsed[0] <= 0.0 {
        return Err(format!("{name} must contain positive numbers"));
    }
    Ok(parsed)
}

fn parse_args() -> Result<Args, String> {
    let mut mode = "scan".to_string();
    let mut input = None;
    let mut output = None;
    let mut path_output = None;
    let mut path_format = "compact".to_string();
    let mut asset = None;
    let mut timeframe = None;
    let mut maximum_fill_days = 180usize;
    let mut maximum_rows = None;
    let mut minimum_open_time_ms = None;
    let mut maximum_open_time_exclusive_ms = None;
    let mut entry_ages_minutes = parse_usize_list(
        "60,600,1440,4320,10080,20160,43200,86400,129600",
        "entry ages",
    )?;
    let mut horizons_minutes = parse_usize_list("60,240,1440,4320,10080,20160,43200", "horizons")?;
    let mut adverse_thresholds_pct = parse_f64_list("1,2,5,10,20,40", "adverse thresholds")?;
    let values: Vec<String> = env::args().skip(1).collect();
    let mut index = 0usize;
    while index < values.len() {
        let key = &values[index];
        index += 1;
        if index >= values.len() {
            return Err(format!("missing value for {key}"));
        }
        let value = &values[index];
        index += 1;
        match key.as_str() {
            "--mode" => mode = value.clone(),
            "--input" => input = Some(PathBuf::from(value)),
            "--output" => output = Some(PathBuf::from(value)),
            "--path-output" => path_output = Some(PathBuf::from(value)),
            "--path-format" => path_format = value.clone(),
            "--asset" => asset = Some(value.clone()),
            "--timeframe" => timeframe = Some(value.clone()),
            "--maximum-fill-days" => {
                maximum_fill_days = value
                    .parse::<usize>()
                    .map_err(|_| "invalid --maximum-fill-days".to_string())?
            }
            "--maximum-rows" => {
                maximum_rows = Some(
                    value
                        .parse::<usize>()
                        .map_err(|_| "invalid --maximum-rows".to_string())?,
                )
            }
            "--minimum-open-time-ms" => {
                minimum_open_time_ms = Some(
                    value
                        .parse::<i64>()
                        .map_err(|_| "invalid --minimum-open-time-ms".to_string())?,
                )
            }
            "--maximum-open-time-exclusive-ms" => {
                maximum_open_time_exclusive_ms = Some(
                    value
                        .parse::<i64>()
                        .map_err(|_| "invalid --maximum-open-time-exclusive-ms".to_string())?,
                )
            }
            "--entry-ages-minutes" => {
                entry_ages_minutes = parse_usize_list(value, "--entry-ages-minutes")?
            }
            "--horizons-minutes" => {
                horizons_minutes = parse_usize_list(value, "--horizons-minutes")?
            }
            "--adverse-thresholds-pct" => {
                adverse_thresholds_pct = parse_f64_list(value, "--adverse-thresholds-pct")?
            }
            _ => return Err(format!("unknown argument: {key}")),
        }
    }
    if mode != "scan" && mode != "outcomes" && mode != "export" && mode != "routes" {
        return Err("--mode must be scan, outcomes, export, or routes".to_string());
    }
    if mode == "export" && output.is_none() {
        return Err("--output is required in export mode".to_string());
    }
    if mode == "routes" && (output.is_none() || path_output.is_none() || asset.is_none()) {
        return Err("--output, --path-output, and --asset are required in routes mode".to_string());
    }
    if path_format != "compact" && path_format != "full" {
        return Err("--path-format must be compact or full".to_string());
    }
    let timeframe = timeframe.ok_or_else(|| "--timeframe is required".to_string())?;
    if timeframe != "1m" && timeframe != "5m" && timeframe != "15m" {
        return Err("--timeframe must be 1m, 5m, or 15m".to_string());
    }
    Ok(Args {
        mode,
        input: input.ok_or_else(|| "--input is required".to_string())?,
        output,
        path_output,
        path_format,
        asset,
        timeframe,
        maximum_fill_days,
        maximum_rows,
        minimum_open_time_ms,
        maximum_open_time_exclusive_ms,
        entry_ages_minutes,
        horizons_minutes,
        adverse_thresholds_pct,
    })
}

fn parse_f64(fields: &[&str], index: usize, name: &str) -> Result<f64, String> {
    fields
        .get(index)
        .ok_or_else(|| format!("missing {name}"))?
        .parse::<f64>()
        .map_err(|_| format!("invalid {name}"))
}

fn read_candles(
    path: &Path,
    interval_ms: i64,
    maximum_rows: Option<usize>,
    minimum_open_time_ms: Option<i64>,
    maximum_open_time_exclusive_ms: Option<i64>,
) -> Result<Vec<Candle>, String> {
    let file =
        File::open(path).map_err(|error| format!("cannot open {}: {error}", path.display()))?;
    let mut reader = BufReader::with_capacity(8 * 1024 * 1024, file);
    let mut header = String::new();
    reader
        .read_line(&mut header)
        .map_err(|error| format!("cannot read header: {error}"))?;
    let columns: Vec<&str> = header.trim_end().split(',').collect();
    let column = |name: &str| {
        columns
            .iter()
            .position(|value| *value == name)
            .ok_or_else(|| format!("missing column {name}"))
    };
    let open_time_index = column("open_time")?;
    let open_index = column("open")?;
    let high_index = column("high")?;
    let low_index = column("low")?;
    let close_index = column("close")?;
    let volume_index = column("volume")?;
    let mut candles = Vec::new();
    let mut line = String::new();
    while maximum_rows.is_none_or(|limit| candles.len() < limit) {
        line.clear();
        if reader
            .read_line(&mut line)
            .map_err(|error| format!("cannot read row: {error}"))?
            == 0
        {
            break;
        }
        let fields: Vec<&str> = line.trim_end().split(',').collect();
        let open_time = fields
            .get(open_time_index)
            .ok_or_else(|| "missing open_time".to_string())?
            .parse::<i64>()
            .map_err(|_| "invalid open_time".to_string())?;
        if minimum_open_time_ms.is_some_and(|minimum| open_time < minimum) {
            continue;
        }
        if maximum_open_time_exclusive_ms.is_some_and(|maximum| open_time >= maximum) {
            break;
        }
        candles.push(Candle {
            open_time,
            open: parse_f64(&fields, open_index, "open")?,
            high: parse_f64(&fields, high_index, "high")?,
            low: parse_f64(&fields, low_index, "low")?,
            close: parse_f64(&fields, close_index, "close")?,
            volume: parse_f64(&fields, volume_index, "volume")?,
        });
    }
    for pair in candles.windows(2) {
        if pair[1].open_time - pair[0].open_time != interval_ms {
            return Err(format!(
                "gap or unexpected interval at {}",
                pair[1].open_time
            ));
        }
    }
    Ok(candles)
}

fn prior_median_range(candles: &[Candle], index: usize) -> f64 {
    let mut values = [0.0f64; 20];
    for (position, candle) in candles[index - 20..index].iter().enumerate() {
        values[position] = candle.high - candle.low;
    }
    values.sort_by(f64::total_cmp);
    (values[9] + values[10]) / 2.0
}

fn detect_signals(candles: &[Candle], bars_per_hour: usize) -> Vec<Signal> {
    let mut signals = Vec::new();
    let start = 20usize.max(bars_per_hour);
    for index in start..candles.len() {
        let candle = candles[index];
        let range = candle.high - candle.low;
        if !(range > 0.0 && candle.close != 0.0) {
            continue;
        }
        let body = (candle.close - candle.open).abs() / range;
        let lower_wick = (candle.open.min(candle.close) - candle.low) / range;
        let upper_wick = (candle.high - candle.open.max(candle.close)) / range;
        let prior_range = prior_median_range(candles, index);
        let prior_volume = candles[index - 20..index]
            .iter()
            .map(|value| value.volume)
            .sum::<f64>()
            / 20.0;
        let prior_close = candles[index - bars_per_hour].close;
        let finite_features = (range / candle.close * 100.0).is_finite()
            && (range / prior_range).is_finite()
            && (candle.volume / prior_volume).is_finite()
            && (candle.close / prior_close - 1.0).is_finite();
        if !finite_features || body > BODY_MAX_PCT {
            continue;
        }
        if lower_wick >= DOMINANT_WICK_MIN_PCT && upper_wick <= OPPOSITE_WICK_MAX_PCT {
            signals.push(Signal {
                index,
                direction_sign: 1,
                wick_target: candle.low,
                opposite_extreme: candle.high,
            });
        } else if upper_wick >= DOMINANT_WICK_MIN_PCT && lower_wick <= OPPOSITE_WICK_MAX_PCT {
            signals.push(Signal {
                index,
                direction_sign: -1,
                wick_target: candle.high,
                opposite_extreme: candle.low,
            });
        }
    }
    signals
}

fn trace_signal(candles: &[Candle], signal: Signal, maximum_future_bars: usize) -> TraceResult {
    let end_index = (signal.index + maximum_future_bars).min(candles.len().saturating_sub(1));
    let fully_observed = signal.index + maximum_future_bars < candles.len();
    if end_index <= signal.index {
        return TraceResult {
            status: "right_censored",
            departure_index: None,
            fill_index: None,
        };
    }
    let mut departure_index = None;
    for (relative, candle) in candles[signal.index + 1..=end_index].iter().enumerate() {
        let index = signal.index + 1 + relative;
        let fill = if signal.direction_sign == 1 {
            candle.low <= signal.wick_target
        } else {
            candle.high >= signal.wick_target
        };
        let departure = if signal.direction_sign == 1 {
            candle.close >= signal.opposite_extreme
        } else {
            candle.close <= signal.opposite_extreme
        };
        if departure_index.is_none() {
            if fill {
                return TraceResult {
                    status: "filled_before_departure",
                    departure_index: None,
                    fill_index: None,
                };
            }
            if departure {
                departure_index = Some(index);
            }
        } else if fill {
            return TraceResult {
                status: "filled",
                departure_index,
                fill_index: Some(index),
            };
        }
    }
    if departure_index.is_some() {
        if fully_observed {
            TraceResult {
                status: "unfilled",
                departure_index,
                fill_index: None,
            }
        } else {
            TraceResult {
                status: "right_censored",
                departure_index,
                fill_index: None,
            }
        }
    } else if fully_observed {
        TraceResult {
            status: "no_departure",
            departure_index: None,
            fill_index: None,
        }
    } else {
        TraceResult {
            status: "right_censored",
            departure_index: None,
            fill_index: None,
        }
    }
}

#[derive(Default, Serialize)]
struct FirstAdverseAggregate {
    hit_count: usize,
    hit_bars_sum: u128,
}

#[derive(Default, Serialize)]
struct HorizonAggregate {
    fully_observed_count: usize,
    target_hit_count: usize,
    adverse_lower_sum_pct: f64,
    adverse_upper_sum_pct: f64,
    outcomes: BTreeMap<String, BTreeMap<String, usize>>,
}

#[derive(Default, Serialize)]
struct OutcomeAggregate {
    observation_count: usize,
    omitted_entry_counts: BTreeMap<String, usize>,
    observations_by_age: BTreeMap<String, usize>,
    entry_distance_sum_pct: f64,
    peak_distance_sum_pct: f64,
    drawdown_sum_pct: f64,
    target_touch_count: usize,
    target_touch_bars_sum: u128,
    first_adverse: BTreeMap<String, FirstAdverseAggregate>,
    horizons: BTreeMap<String, HorizonAggregate>,
}

#[derive(Serialize)]
struct TraceExport {
    signal_index: usize,
    status: &'static str,
    departure_index: Option<usize>,
    fill_index: Option<usize>,
}

#[derive(Serialize)]
struct CompactPathRow<'a> {
    episode_id: &'a str,
    asset: &'a str,
    timeframe: &'a str,
    direction: &'a str,
    direction_sign: i8,
    offset_bars: usize,
    normalized_open_pct: f64,
    normalized_high_pct: f64,
    normalized_low_pct: f64,
    normalized_close_pct: f64,
}

#[derive(Serialize)]
struct FullPathRow<'a> {
    episode_id: &'a str,
    asset: &'a str,
    timeframe: &'a str,
    direction: &'a str,
    direction_sign: i8,
    event_open_time_utc: &'a str,
    offset_bars: usize,
    phase: &'a str,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    normalized_open_pct: f64,
    normalized_high_pct: f64,
    normalized_low_pct: f64,
    normalized_close_pct: f64,
}

#[derive(Serialize)]
struct HorizonExport {
    horizon_minutes: usize,
    fully_observed: bool,
    target_hit: bool,
    adverse_lower_pct: f64,
    adverse_upper_pct: f64,
    outcomes: Vec<&'static str>,
}

#[derive(Serialize)]
struct ObservationExport {
    signal_index: usize,
    entry_age_minutes: usize,
    entry_index: usize,
    departure_index: usize,
    available_future_bars: usize,
    target_touch_bars_from_entry: Option<usize>,
    entry_distance_from_target_pct: f64,
    peak_distance_from_target_pct: f64,
    drawdown_from_peak_pct: f64,
    first_adverse_bars_from_entry: Vec<Option<usize>>,
    horizons: Vec<HorizonExport>,
}

#[derive(Default)]
struct OutcomeBatch {
    aggregate: OutcomeAggregate,
    observations: Vec<ObservationExport>,
}

impl OutcomeAggregate {
    fn merge(&mut self, other: OutcomeAggregate) {
        self.observation_count += other.observation_count;
        self.entry_distance_sum_pct += other.entry_distance_sum_pct;
        self.peak_distance_sum_pct += other.peak_distance_sum_pct;
        self.drawdown_sum_pct += other.drawdown_sum_pct;
        self.target_touch_count += other.target_touch_count;
        self.target_touch_bars_sum += other.target_touch_bars_sum;
        for (key, value) in other.omitted_entry_counts {
            *self.omitted_entry_counts.entry(key).or_default() += value;
        }
        for (key, value) in other.observations_by_age {
            *self.observations_by_age.entry(key).or_default() += value;
        }
        for (key, value) in other.first_adverse {
            let target = self.first_adverse.entry(key).or_default();
            target.hit_count += value.hit_count;
            target.hit_bars_sum += value.hit_bars_sum;
        }
        for (key, value) in other.horizons {
            let target = self.horizons.entry(key).or_default();
            target.fully_observed_count += value.fully_observed_count;
            target.target_hit_count += value.target_hit_count;
            target.adverse_lower_sum_pct += value.adverse_lower_sum_pct;
            target.adverse_upper_sum_pct += value.adverse_upper_sum_pct;
            for (threshold, outcomes) in value.outcomes {
                let target_outcomes = target.outcomes.entry(threshold).or_default();
                for (outcome, count) in outcomes {
                    *target_outcomes.entry(outcome).or_default() += count;
                }
            }
        }
    }
}

impl OutcomeBatch {
    fn merge(&mut self, other: OutcomeBatch) {
        self.aggregate.merge(other.aggregate);
        self.observations.extend(other.observations);
    }
}

fn utc_iso(timestamp_ms: i64) -> Result<String, String> {
    DateTime::<Utc>::from_timestamp_millis(timestamp_ms)
        .map(|value| value.to_rfc3339_opts(SecondsFormat::Secs, true))
        .ok_or_else(|| format!("invalid timestamp: {timestamp_ms}"))
}

fn normalized(value: f64, target: f64, direction_sign: i8) -> f64 {
    direction_sign as f64 * (value / target - 1.0) * 100.0
}

fn write_route_paths(
    path: &Path,
    candles: &[Candle],
    signals: &[Signal],
    traces: &[TraceResult],
    asset: &str,
    timeframe: &str,
    path_format: &str,
) -> Result<(usize, usize), String> {
    let file = File::create(path)
        .map_err(|error| format!("cannot create route path {}: {error}", path.display()))?;
    let encoder = GzEncoder::new(file, Compression::fast());
    let mut writer = WriterBuilder::new().has_headers(true).from_writer(encoder);
    let mut completed = 0usize;
    let mut rows_written = 0usize;
    for (signal, trace) in signals.iter().zip(traces) {
        if trace.status != "filled" {
            continue;
        }
        let departure_index = trace
            .departure_index
            .expect("filled route has a departure index");
        let fill_index = trace.fill_index.expect("filled route has a fill index");
        let direction = if signal.direction_sign == 1 {
            "lower_wick"
        } else {
            "upper_wick"
        };
        let episode_id = format!(
            "{asset}_{timeframe}_{direction}_{}",
            candles[signal.index].open_time
        );
        completed += 1;
        for (index, candle) in candles
            .iter()
            .copied()
            .enumerate()
            .take(fill_index + 1)
            .skip(signal.index)
        {
            let offset_bars = index - signal.index;
            if path_format == "compact" {
                writer
                    .serialize(CompactPathRow {
                        episode_id: &episode_id,
                        asset,
                        timeframe,
                        direction,
                        direction_sign: signal.direction_sign,
                        offset_bars,
                        normalized_open_pct: normalized(
                            candle.open,
                            signal.wick_target,
                            signal.direction_sign,
                        ),
                        normalized_high_pct: normalized(
                            candle.high,
                            signal.wick_target,
                            signal.direction_sign,
                        ),
                        normalized_low_pct: normalized(
                            candle.low,
                            signal.wick_target,
                            signal.direction_sign,
                        ),
                        normalized_close_pct: normalized(
                            candle.close,
                            signal.wick_target,
                            signal.direction_sign,
                        ),
                    })
                    .map_err(|error| format!("cannot write compact route row: {error}"))?;
            } else {
                let phase = if index == signal.index {
                    "signal"
                } else if index < departure_index {
                    "pre_departure"
                } else if index == departure_index {
                    "departure"
                } else if index < fill_index {
                    "return_path"
                } else {
                    "fill"
                };
                let event_open_time_utc = utc_iso(candle.open_time)?;
                writer
                    .serialize(FullPathRow {
                        episode_id: &episode_id,
                        asset,
                        timeframe,
                        direction,
                        direction_sign: signal.direction_sign,
                        event_open_time_utc: &event_open_time_utc,
                        offset_bars,
                        phase,
                        open: candle.open,
                        high: candle.high,
                        low: candle.low,
                        close: candle.close,
                        volume: candle.volume,
                        normalized_open_pct: normalized(
                            candle.open,
                            signal.wick_target,
                            signal.direction_sign,
                        ),
                        normalized_high_pct: normalized(
                            candle.high,
                            signal.wick_target,
                            signal.direction_sign,
                        ),
                        normalized_low_pct: normalized(
                            candle.low,
                            signal.wick_target,
                            signal.direction_sign,
                        ),
                        normalized_close_pct: normalized(
                            candle.close,
                            signal.wick_target,
                            signal.direction_sign,
                        ),
                    })
                    .map_err(|error| format!("cannot write full route row: {error}"))?;
            }
            rows_written += 1;
        }
    }
    writer
        .flush()
        .map_err(|error| format!("cannot flush route path {}: {error}", path.display()))?;
    let encoder = writer
        .into_inner()
        .map_err(|error| format!("cannot finish route CSV {}: {error}", path.display()))?;
    encoder
        .finish()
        .map_err(|error| format!("cannot finish route gzip {}: {error}", path.display()))?;
    Ok((completed, rows_written))
}

fn threshold_slug(value: f64) -> String {
    format!("{value}").replace('.', "p")
}

fn horizon_slug(value: usize) -> String {
    format!("{value}m")
}

fn increment(map: &mut BTreeMap<String, usize>, key: &str) {
    *map.entry(key.to_string()).or_default() += 1;
}

fn target_hit(candle: Candle, signal: Signal) -> bool {
    if signal.direction_sign == 1 {
        candle.low <= signal.wick_target
    } else {
        candle.high >= signal.wick_target
    }
}

fn adverse_from_entry(candle: Candle, signal: Signal, entry_price: f64) -> f64 {
    let value = if signal.direction_sign == 1 {
        (candle.high / entry_price - 1.0) * 100.0
    } else {
        (1.0 - candle.low / entry_price) * 100.0
    };
    value.max(0.0)
}

fn away_from_target(candle: Candle, signal: Signal) -> f64 {
    let value = if signal.direction_sign == 1 {
        (candle.high / signal.wick_target - 1.0) * 100.0
    } else {
        (1.0 - candle.low / signal.wick_target) * 100.0
    };
    value.max(0.0)
}

fn prefix_value(prefix: &[f64], count: usize) -> f64 {
    if count == 0 { 0.0 } else { prefix[count - 1] }
}

fn competing_outcome(
    target_bars: Option<usize>,
    adverse_bars: Option<usize>,
    horizon_bars: usize,
    available_future_bars: usize,
) -> &'static str {
    let target_within = target_bars.is_some_and(|value| value <= horizon_bars);
    let adverse_within = adverse_bars.is_some_and(|value| value <= horizon_bars);
    if target_within && adverse_within {
        if target_bars == adverse_bars {
            "ambiguous_intrabar"
        } else if target_bars < adverse_bars {
            "target_first"
        } else {
            "adverse_first"
        }
    } else if target_within {
        "target_first"
    } else if adverse_within {
        "adverse_first"
    } else if available_future_bars >= horizon_bars {
        "neither"
    } else {
        "right_censored"
    }
}

fn build_outcome_batch(
    candles: &[Candle],
    signals: &[Signal],
    traces: &[TraceResult],
    interval_minutes: usize,
    entry_ages_minutes: &[usize],
    horizons_minutes: &[usize],
    adverse_thresholds_pct: &[f64],
) -> OutcomeBatch {
    let mut aggregate = OutcomeAggregate::default();
    let mut observations = Vec::new();
    for threshold in adverse_thresholds_pct {
        aggregate
            .first_adverse
            .insert(threshold_slug(*threshold), FirstAdverseAggregate::default());
    }
    for horizon in horizons_minutes {
        aggregate
            .horizons
            .insert(horizon_slug(*horizon), HorizonAggregate::default());
    }
    let maximum_horizon_bars =
        horizons_minutes.iter().max().copied().unwrap_or_default() / interval_minutes;

    for (signal, trace) in signals.iter().zip(traces) {
        let mut peak_move = 0.0f64;
        let mut peak_cursor = signal.index;
        for age_minutes in entry_ages_minutes {
            let entry_index = signal.index + age_minutes / interval_minutes;
            if entry_index >= candles.len() {
                increment(&mut aggregate.omitted_entry_counts, "entry_not_observed");
                continue;
            }
            while peak_cursor <= entry_index {
                peak_move = peak_move.max(away_from_target(candles[peak_cursor], *signal));
                peak_cursor += 1;
            }
            if trace
                .departure_index
                .is_none_or(|departure| departure > entry_index)
            {
                increment(&mut aggregate.omitted_entry_counts, "not_departed_by_entry");
                continue;
            }
            if trace.fill_index.is_some_and(|fill| fill <= entry_index) {
                increment(&mut aggregate.omitted_entry_counts, "filled_by_entry");
                continue;
            }

            aggregate.observation_count += 1;
            *aggregate
                .observations_by_age
                .entry(format!("{age_minutes}m"))
                .or_default() += 1;
            let entry_price = candles[entry_index].close;
            let current_move =
                (signal.direction_sign as f64 * (entry_price / signal.wick_target - 1.0) * 100.0)
                    .max(0.0);
            aggregate.entry_distance_sum_pct += current_move;
            aggregate.peak_distance_sum_pct += peak_move;
            aggregate.drawdown_sum_pct += (peak_move - current_move).max(0.0);

            let available_future_bars =
                maximum_horizon_bars.min(candles.len().saturating_sub(entry_index + 1));
            let mut target_bars = None;
            let mut first_adverse = vec![None; adverse_thresholds_pct.len()];
            let mut adverse_prefix = Vec::with_capacity(available_future_bars);
            let mut adverse_max = 0.0f64;
            for offset in 1..=available_future_bars {
                let candle = candles[entry_index + offset];
                if target_bars.is_none() && target_hit(candle, *signal) {
                    target_bars = Some(offset);
                }
                let adverse = adverse_from_entry(candle, *signal, entry_price);
                adverse_max = adverse_max.max(adverse);
                adverse_prefix.push(adverse_max);
                for (threshold_index, threshold) in adverse_thresholds_pct.iter().enumerate() {
                    if first_adverse[threshold_index].is_none() && adverse >= *threshold {
                        first_adverse[threshold_index] = Some(offset);
                    }
                }
            }
            if let Some(target) = target_bars {
                aggregate.target_touch_count += 1;
                aggregate.target_touch_bars_sum += target as u128;
            }
            for (threshold_index, threshold) in adverse_thresholds_pct.iter().enumerate() {
                if let Some(bars) = first_adverse[threshold_index] {
                    let item = aggregate
                        .first_adverse
                        .get_mut(&threshold_slug(*threshold))
                        .expect("threshold aggregate exists");
                    item.hit_count += 1;
                    item.hit_bars_sum += bars as u128;
                }
            }

            let mut horizon_exports = Vec::with_capacity(horizons_minutes.len());
            for horizon_minutes in horizons_minutes {
                let horizon_bars = horizon_minutes / interval_minutes;
                let visible_bars = horizon_bars.min(available_future_bars);
                let target_within = target_bars.is_some_and(|value| value <= horizon_bars);
                let strict_cutoff = visible_bars.min(if target_within {
                    target_bars.expect("target exists") - 1
                } else {
                    visible_bars
                });
                let envelope_cutoff = visible_bars.min(if target_within {
                    target_bars.expect("target exists")
                } else {
                    visible_bars
                });
                let horizon = aggregate
                    .horizons
                    .get_mut(&horizon_slug(*horizon_minutes))
                    .expect("horizon aggregate exists");
                if available_future_bars >= horizon_bars {
                    horizon.fully_observed_count += 1;
                }
                if target_within {
                    horizon.target_hit_count += 1;
                }
                let adverse_lower_pct = prefix_value(&adverse_prefix, strict_cutoff);
                let adverse_upper_pct = prefix_value(&adverse_prefix, envelope_cutoff);
                horizon.adverse_lower_sum_pct += adverse_lower_pct;
                horizon.adverse_upper_sum_pct += adverse_upper_pct;
                let mut outcomes = Vec::with_capacity(adverse_thresholds_pct.len());
                for (threshold_index, threshold) in adverse_thresholds_pct.iter().enumerate() {
                    let outcome = competing_outcome(
                        target_bars,
                        first_adverse[threshold_index],
                        horizon_bars,
                        available_future_bars,
                    );
                    let counts = horizon
                        .outcomes
                        .entry(threshold_slug(*threshold))
                        .or_default();
                    *counts.entry(outcome.to_string()).or_default() += 1;
                    outcomes.push(outcome);
                }
                horizon_exports.push(HorizonExport {
                    horizon_minutes: *horizon_minutes,
                    fully_observed: available_future_bars >= horizon_bars,
                    target_hit: target_within,
                    adverse_lower_pct,
                    adverse_upper_pct,
                    outcomes,
                });
            }
            observations.push(ObservationExport {
                signal_index: signal.index,
                entry_age_minutes: *age_minutes,
                entry_index,
                departure_index: trace
                    .departure_index
                    .expect("eligible observation has a departure"),
                available_future_bars,
                target_touch_bars_from_entry: target_bars,
                entry_distance_from_target_pct: current_move,
                peak_distance_from_target_pct: peak_move,
                drawdown_from_peak_pct: (peak_move - current_move).max(0.0),
                first_adverse_bars_from_entry: first_adverse,
                horizons: horizon_exports,
            });
        }
    }
    OutcomeBatch {
        aggregate,
        observations,
    }
}

fn build_outcome_batch_parallel(
    candles: &[Candle],
    signals: &[Signal],
    traces: &[TraceResult],
    interval_minutes: usize,
    entry_ages_minutes: &[usize],
    horizons_minutes: &[usize],
    adverse_thresholds_pct: &[f64],
) -> OutcomeBatch {
    if signals.is_empty() {
        return build_outcome_batch(
            candles,
            signals,
            traces,
            interval_minutes,
            entry_ages_minutes,
            horizons_minutes,
            adverse_thresholds_pct,
        );
    }
    let workers = std::thread::available_parallelism()
        .map(|value| value.get())
        .unwrap_or(1)
        .min(signals.len().max(1));
    let chunk_size = signals.len().div_ceil(workers);
    let pieces = signals
        .par_chunks(chunk_size)
        .zip(traces.par_chunks(chunk_size))
        .map(|(signal_chunk, trace_chunk)| {
            build_outcome_batch(
                candles,
                signal_chunk,
                trace_chunk,
                interval_minutes,
                entry_ages_minutes,
                horizons_minutes,
                adverse_thresholds_pct,
            )
        })
        .collect::<Vec<_>>();
    let mut total = OutcomeBatch::default();
    for piece in pieces {
        total.merge(piece);
    }
    total
}

fn main() -> Result<(), String> {
    let args = parse_args()?;
    let interval_minutes = args
        .timeframe
        .strip_suffix('m')
        .ok_or_else(|| "invalid timeframe".to_string())?
        .parse::<usize>()
        .map_err(|_| "invalid timeframe".to_string())?;
    if args
        .entry_ages_minutes
        .iter()
        .chain(args.horizons_minutes.iter())
        .any(|value| value % interval_minutes != 0)
    {
        return Err("entry ages and horizons must be exact timeframe multiples".to_string());
    }
    if (args.mode == "outcomes" || args.mode == "export")
        && args.maximum_fill_days * 24 * 60
            < args
                .entry_ages_minutes
                .iter()
                .max()
                .copied()
                .unwrap_or_default()
                + args
                    .horizons_minutes
                    .iter()
                    .max()
                    .copied()
                    .unwrap_or_default()
    {
        return Err(
            "maximum follow-up is shorter than oldest age plus longest horizon".to_string(),
        );
    }
    let interval_ms = interval_minutes as i64 * 60_000;
    let total_start = Instant::now();
    let read_start = Instant::now();
    let candles = read_candles(
        &args.input,
        interval_ms,
        args.maximum_rows,
        args.minimum_open_time_ms,
        args.maximum_open_time_exclusive_ms,
    )?;
    let read_seconds = read_start.elapsed().as_secs_f64();
    let detect_start = Instant::now();
    let signals = detect_signals(&candles, 60 / interval_minutes);
    let detect_seconds = detect_start.elapsed().as_secs_f64();
    let trace_start = Instant::now();
    let maximum_future_bars = args.maximum_fill_days * 24 * 60 / interval_minutes;
    let mut statuses: BTreeMap<&str, usize> = BTreeMap::new();
    let mut traces = Vec::with_capacity(signals.len());
    for signal in &signals {
        let trace = trace_signal(&candles, *signal, maximum_future_bars);
        *statuses.entry(trace.status).or_default() += 1;
        traces.push(trace);
    }
    let trace_seconds = trace_start.elapsed().as_secs_f64();
    let lower = signals
        .iter()
        .filter(|value| value.direction_sign == 1)
        .count();
    let upper = signals.len() - lower;
    let checksum: u128 = signals.iter().map(|value| value.index as u128).sum();
    let mut payload = json!({
        "engine": "rust",
        "mode": args.mode,
        "rows": candles.len(),
        "strict_signals": signals.len(),
        "lower_signals": lower,
        "upper_signals": upper,
        "signal_index_checksum": checksum,
        "statuses": &statuses,
        "read_seconds": read_seconds,
        "detect_seconds": detect_seconds,
        "trace_seconds": trace_seconds,
    });
    if args.mode == "routes" {
        let route_start = Instant::now();
        let output = args.output.as_ref().expect("route output validated");
        let path_output = args
            .path_output
            .as_ref()
            .expect("route path output validated");
        let asset = args.asset.as_ref().expect("route asset validated");
        let (completed_episode_count, path_rows_written) = write_route_paths(
            path_output,
            &candles,
            &signals,
            &traces,
            asset,
            &args.timeframe,
            &args.path_format,
        )?;
        let trace_rows = signals
            .iter()
            .zip(&traces)
            .map(|(signal, trace)| TraceExport {
                signal_index: signal.index,
                status: trace.status,
                departure_index: trace.departure_index,
                fill_index: trace.fill_index,
            })
            .collect::<Vec<_>>();
        let route_payload = json!({
            "schema_version": "rust-route-kernel-v1",
            "asset": asset,
            "timeframe": args.timeframe,
            "source_rows": candles.len(),
            "source_start_open_time_ms": candles.first().map(|value| value.open_time),
            "source_end_open_time_ms": candles.last().map(|value| value.open_time),
            "strict_signals": signals.len(),
            "signal_index_checksum": checksum,
            "statuses": &statuses,
            "completed_episode_count": completed_episode_count,
            "path_rows_written": path_rows_written,
            "path_format": args.path_format,
            "traces": trace_rows,
        });
        let file = File::create(output)
            .map_err(|error| format!("cannot create {}: {error}", output.display()))?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, &route_payload)
            .map_err(|error| format!("cannot write {}: {error}", output.display()))?;
        writer
            .flush()
            .map_err(|error| format!("cannot flush {}: {error}", output.display()))?;
        payload["completed_episode_count"] = json!(completed_episode_count);
        payload["path_rows_written"] = json!(path_rows_written);
        payload["route_seconds"] = json!(route_start.elapsed().as_secs_f64());
    } else if args.mode == "outcomes" || args.mode == "export" {
        let outcome_start = Instant::now();
        let batch = build_outcome_batch_parallel(
            &candles,
            &signals,
            &traces,
            interval_minutes,
            &args.entry_ages_minutes,
            &args.horizons_minutes,
            &args.adverse_thresholds_pct,
        );
        if args.mode == "outcomes" {
            payload["outcomes"] = serde_json::to_value(batch.aggregate)
                .map_err(|error| format!("cannot serialize outcomes: {error}"))?;
        } else {
            let trace_rows = signals
                .iter()
                .zip(&traces)
                .map(|(signal, trace)| TraceExport {
                    signal_index: signal.index,
                    status: trace.status,
                    departure_index: trace.departure_index,
                    fill_index: trace.fill_index,
                })
                .collect::<Vec<_>>();
            let export_payload = json!({
                "schema_version": "rust-prospective-kernel-v1",
                "timeframe": args.timeframe,
                "interval_minutes": interval_minutes,
                "source_rows": candles.len(),
                "strict_signals": signals.len(),
                "signal_index_checksum": checksum,
                "entry_ages_minutes": args.entry_ages_minutes,
                "horizons_minutes": args.horizons_minutes,
                "adverse_thresholds_pct": args.adverse_thresholds_pct,
                "traces": trace_rows,
                "observations": batch.observations,
                "aggregate": batch.aggregate,
            });
            let output = args.output.as_ref().expect("export output validated");
            let file = File::create(output)
                .map_err(|error| format!("cannot create {}: {error}", output.display()))?;
            let mut writer = BufWriter::new(file);
            serde_json::to_writer(&mut writer, &export_payload)
                .map_err(|error| format!("cannot write {}: {error}", output.display()))?;
            writer
                .flush()
                .map_err(|error| format!("cannot flush {}: {error}", output.display()))?;
            payload["observations"] = json!(
                export_payload["observations"]
                    .as_array()
                    .map_or(0, Vec::len)
            );
            payload["export_path"] = json!(output);
        }
        payload["outcome_seconds"] = json!(outcome_start.elapsed().as_secs_f64());
    }
    payload["total_seconds"] = json!(total_start.elapsed().as_secs_f64());
    println!(
        "{}",
        serde_json::to_string(&payload)
            .map_err(|error| format!("cannot serialize result: {error}"))?
    );
    Ok(())
}
