#!/usr/bin/env python3
"""Compare two benchmark directories using persisted retrieval ranking artifacts."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.pipeline_shared import compare_ir_benchmark_dirs, write_csv, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval IR metrics between two benchmark runs")
    parser.add_argument("--base-dir", required=True, help="Baseline benchmark directory")
    parser.add_argument("--candidate-dir", required=True, help="Candidate benchmark directory")
    parser.add_argument("--output-dir", default=None, help="Where to write IR comparison artifacts")
    args = parser.parse_args()

    summary, question_rows = compare_ir_benchmark_dirs(args.base_dir, args.candidate_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.candidate_dir) / "ir_diff_vs_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "ir_diff_summary.json", summary)
    write_csv(output_dir / "ir_question_diff.csv", question_rows)
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
