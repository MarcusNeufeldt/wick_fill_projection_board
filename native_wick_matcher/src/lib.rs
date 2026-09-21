use std::cmp::Ordering;

fn log_distance(left: f64, right: f64, floor: f64) -> f64 {
    let safe_left = left.max(0.0) + floor;
    let safe_right = right.max(0.0) + floor;
    (safe_left / safe_right).ln().abs()
}

struct MatchInputs<'a> {
    episodes: &'a [u32],
    current: &'a [f64],
    peaks: &'a [f64],
    drawdowns: &'a [f64],
    offsets: &'a [i32],
    fills: &'a [i64],
    features: &'a [f64],
    categories: &'a [f64],
    snapshot_close_time_ms: i64,
    current_move: f64,
    peak_move: f64,
    drawdown: f64,
    elapsed_bars: f64,
    current_weight: f64,
    peak_weight: f64,
    drawdown_weight: f64,
    elapsed_weight: f64,
}

fn scan_range(inputs: &MatchInputs<'_>, start: usize, end: usize) -> (Vec<f64>, Vec<usize>) {
    let episode_count = inputs.fills.len();
    let mut best_scores = vec![f64::INFINITY; episode_count];
    let mut best_states = vec![usize::MAX; episode_count];
    for state_index in start..end {
        let episode = inputs.episodes[state_index] as usize;
        if episode >= episode_count || inputs.fills[episode] > inputs.snapshot_close_time_ms {
            continue;
        }
        let score = inputs.current_weight * log_distance(inputs.current[state_index], inputs.current_move, 0.30)
            + inputs.peak_weight * log_distance(inputs.peaks[state_index], inputs.peak_move, 0.30)
            + inputs.drawdown_weight * log_distance(inputs.drawdowns[state_index], inputs.drawdown, 0.30)
            + inputs.elapsed_weight * log_distance(inputs.offsets[state_index] as f64, inputs.elapsed_bars, 3.0)
            + 0.5 * inputs.features[episode]
            + inputs.categories[episode];
        if score.is_finite() && score < best_scores[episode] {
            best_scores[episode] = score;
            best_states[episode] = state_index;
        }
    }
    (best_scores, best_states)
}

/// Scan all already-valid historical states, keep each episode's best alignment,
/// then return the requested lowest-score episode states.  Python owns every
/// input and output buffer for the duration of this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wick_match_top_k(
    state_episode_codes: *const u32,
    state_current_move: *const f64,
    state_peak_move: *const f64,
    state_drawdown: *const f64,
    state_offsets: *const i32,
    state_count: usize,
    fill_close_times: *const i64,
    feature_distance: *const f64,
    category_distance: *const f64,
    episode_count: usize,
    snapshot_close_time_ms: i64,
    current_move: f64,
    peak_move: f64,
    drawdown: f64,
    elapsed_bars: f64,
    current_weight: f64,
    peak_weight: f64,
    drawdown_weight: f64,
    elapsed_weight: f64,
    requested_top_k: usize,
    output_episode_codes: *mut u32,
    output_state_positions: *mut u64,
    output_scores: *mut f64,
) -> usize {
    if state_episode_codes.is_null()
        || state_current_move.is_null()
        || state_peak_move.is_null()
        || state_drawdown.is_null()
        || state_offsets.is_null()
        || fill_close_times.is_null()
        || feature_distance.is_null()
        || category_distance.is_null()
        || output_episode_codes.is_null()
        || output_state_positions.is_null()
        || output_scores.is_null()
    {
        return 0;
    }

    let episodes = unsafe { std::slice::from_raw_parts(state_episode_codes, state_count) };
    let current = unsafe { std::slice::from_raw_parts(state_current_move, state_count) };
    let peaks = unsafe { std::slice::from_raw_parts(state_peak_move, state_count) };
    let drawdowns = unsafe { std::slice::from_raw_parts(state_drawdown, state_count) };
    let offsets = unsafe { std::slice::from_raw_parts(state_offsets, state_count) };
    let fills = unsafe { std::slice::from_raw_parts(fill_close_times, episode_count) };
    let features = unsafe { std::slice::from_raw_parts(feature_distance, episode_count) };
    let categories = unsafe { std::slice::from_raw_parts(category_distance, episode_count) };

    let inputs = MatchInputs {
        episodes,
        current,
        peaks,
        drawdowns,
        offsets,
        fills,
        features,
        categories,
        snapshot_close_time_ms,
        current_move,
        peak_move,
        drawdown,
        elapsed_bars,
        current_weight,
        peak_weight,
        drawdown_weight,
        elapsed_weight,
    };
    let worker_count = std::thread::available_parallelism()
        .map(|count| count.get())
        .unwrap_or(1)
        .min(6)
        .min(state_count.max(1));
    let chunk_size = state_count.div_ceil(worker_count);
    let mut partials: Vec<(Vec<f64>, Vec<usize>)> = Vec::with_capacity(worker_count);
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(worker_count);
        let inputs_ref = &inputs;
        for worker in 0..worker_count {
            let start = worker * chunk_size;
            let end = ((worker + 1) * chunk_size).min(state_count);
            if start < end {
                handles.push(scope.spawn(move || scan_range(inputs_ref, start, end)));
            }
        }
        for handle in handles {
            if let Ok(partial) = handle.join() {
                partials.push(partial);
            }
        }
    });
    if partials.is_empty() {
        return 0;
    }
    let mut best_scores = vec![f64::INFINITY; episode_count];
    let mut best_states = vec![usize::MAX; episode_count];
    for (scores, states) in partials {
        for episode in 0..episode_count {
            let candidate_score = scores[episode];
            let candidate_state = states[episode];
            if candidate_score < best_scores[episode]
                || (candidate_score == best_scores[episode] && candidate_state < best_states[episode])
            {
                best_scores[episode] = candidate_score;
                best_states[episode] = candidate_state;
            }
        }
    }

    let mut matched: Vec<(f64, usize, usize)> = best_scores
        .iter()
        .enumerate()
        .filter_map(|(episode, score)| {
            let state = best_states[episode];
            score.is_finite().then_some((*score, episode, state))
        })
        .collect();
    matched.sort_by(|left, right| {
        let comparison = left.0.total_cmp(&right.0);
        if comparison == Ordering::Equal {
            left.1.cmp(&right.1)
        } else {
            comparison
        }
    });
    let count = requested_top_k.min(matched.len());
    let result_episodes = unsafe { std::slice::from_raw_parts_mut(output_episode_codes, count) };
    let result_states = unsafe { std::slice::from_raw_parts_mut(output_state_positions, count) };
    let result_scores = unsafe { std::slice::from_raw_parts_mut(output_scores, count) };
    for (position, (score, episode, state)) in matched.into_iter().take(count).enumerate() {
        result_episodes[position] = episode as u32;
        result_states[position] = state as u64;
        result_scores[position] = score;
    }
    count
}
