#!/usr/bin/env python3
"""Compare two benchmark artifact directories and write a question-level diff."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.pipeline_shared import compare_benchmark_dirs, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Diff two benchmark result directories")
    parser.add_argument("--base-dir", required=True, help="Baseline benchmark directory")
    parser.add_argument("--candidate-dir", required=True, help="Candidate benchmark directory")
    parser.add_argument("--output-dir", default=None, help="Where to write the diff artifacts")
    args = parser.parse_args()

    summary, diff_rows = compare_benchmark_dirs(args.base_dir, args.candidate_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.candidate_dir) / "diff_vs_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "diff_summary.json", summary)
    write_csv(output_dir / "question_diff.csv", diff_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
