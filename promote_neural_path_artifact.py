"""Promote a validated neural-route artifact into one explicit live age regime."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ARTIFACT_FILES = (
    "neural_path_v3_model.pt",
    "neural_path_v3_retrieval_index.npz",
    "neural_path_v3_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--minimum-age-bars", required=True, type=int)
    parser.add_argument("--maximum-age-bars", required=True, type=int)
    parser.add_argument("--label", required=True)
    parser.add_argument("--selector", default="forecast_hybrid_real_path")
    parser.add_argument("--evidence", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_age_bars < 1 or args.maximum_age_bars < args.minimum_age_bars:
        raise ValueError("invalid live age regime")
    missing = [name for name in ARTIFACT_FILES if not (args.source / name).exists()]
    if missing:
        raise FileNotFoundError(f"source artifact is missing: {', '.join(missing)}")
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in ARTIFACT_FILES[:2]:
        shutil.copy2(args.source / name, args.destination / name)
    summary = json.loads(
        (args.source / "neural_path_v3_summary.json").read_text(encoding="utf-8")
    )
    summary["deployment_gate"] = {
        "status": "experimental live selector with automatic V1 fallback",
        "timeframe": args.timeframe,
        "regime_label": args.label,
        "selector": args.selector,
        "min_elapsed_bars": args.minimum_age_bars,
        "max_elapsed_bars": args.maximum_age_bars,
        "selection_note": args.evidence,
        "fallback": "V1 remains active whenever this age expert cannot load or the live age is unsupported.",
    }
    target = args.destination / "neural_path_v3_summary.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    print(
        json.dumps(
            {
                "source": str(args.source),
                "destination": str(args.destination),
                "deployment_gate": summary["deployment_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
