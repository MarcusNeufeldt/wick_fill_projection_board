# Rust throughput and data-scaling POC

This experiment answers two separate questions without changing the frozen
dashboard architecture:

1. Is raw wick detection and outcome tracing slow enough that Rust materially
   improves research iteration time?
2. Does the existing supervised forecast still improve as more independent
   historical wick episodes are added?

The Rust program and the Python reference both implement the current strict
signal geometry and conservative departure/fill ordering from
`build_conditional_path_library.py`. The scanner can report benchmark
aggregates or export row-level traces and prospective labels. Python remains
the authority for feature geometry, record IDs, schema assembly and Parquet
output.

`python_outcome_scan.py` and `compare_outcomes.py` extend the parity contract
through the complete prospective-entry label kernel: entry-age eligibility,
target touches, adverse thresholds, intrabar ambiguity, censoring, and the
lower/upper adverse envelope at every horizon.

The learning-curve script trains the existing `moderate` all-outcome model on
nested, chronologically eligible history subsets. Every subset is evaluated on
the same later, fully matured query window. Smaller subsets retain the most
recent eligible history, so the experiment asks whether adding older episodes
helps the same forecast period rather than moving the test cutoff.

## Build the Rust scanner

```powershell
$env:PATH = "C:\Program Files\Rust stable MSVC 1.98\bin;$env:PATH"
cargo build --release --manifest-path experiments\rust_training_throughput\Cargo.toml
```

## Compare Python and Rust on identical data

```powershell
.venv\Scripts\python.exe experiments\rust_training_throughput\python_wick_scan.py `
  --input data\ETHUSDT_5m_5y.csv --timeframe 5m --maximum-fill-days 180

experiments\rust_training_throughput\target\release\wick-throughput-poc.exe `
  --input data\ETHUSDT_5m_5y.csv --timeframe 5m --maximum-fill-days 180
```

The following fields must match before a speed comparison is accepted:

- rows
- strict signal count and lower/upper split
- signal-index checksum
- every trace status count

## Production update bridge

`prospective_entry_outcomes.py --engine rust` runs the row-level Rust export,
then validates source-row count, configuration, exact strict-signal order and
signal-index checksum before accepting any result. It pins Rust to the number
of rows already loaded by Python so the live source refresher cannot create a
false mismatch by appending a candle during the scan.

The manual updater compiles the release binary and requests Rust explicitly:

```powershell
.venv\Scripts\python.exe .\wick_update.py update --dry-run
.venv\Scripts\python.exe .\wick_update.py update
```

The same binary also writes the established full 5m and compact 1m historical
path formats. Both library builders validate exact signal order/checksum and
retain Python episode features and summaries. An explicit Rust update fails
closed if the binary, contract or parity check fails. Direct outcome-dataset
builds default to `--engine auto`, which reports the problem and uses the
established Python implementation as a compatibility fallback. Neither path
changes the frozen forecast manifest.

## Run the chronological learning curve

```powershell
.venv\Scripts\python.exe experiments\rust_training_throughput\learning_curve.py `
  --fractions 0.2,0.4,0.6,0.8,1.0 --maximum-queries 1500
```

For the stronger multi-origin check:

```powershell
.venv\Scripts\python.exe experiments\rust_training_throughput\rolling_learning_curve.py `
  --folds 3 --fractions 0.2,0.4,0.6,0.8,1.0 --maximum-queries 750
```

This first curve deliberately measures the established A numerical model, not
the visual route selector. If the final increments still improve held-out fill,
adverse-risk, or target-versus-adverse metrics, the next experiment should run
the same curve through the full E0 stack. If it is flat, extra raw rows are not
the immediate constraint.

## Scope boundary

- No live model is replaced.
- No `active.json` manifest is changed.
- The manual updater may rebuild the production outcome dataset through the
  parity-checked bridge; this experiment never repoints the frozen forecast
  manifest.
- Route images remain illustrations; numerical risk metrics are evaluated
  independently.
