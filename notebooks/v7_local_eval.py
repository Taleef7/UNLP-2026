"""
v7 local evaluation wrapper.

Runs the shared preset-backed benchmark path with the scored v7 baseline.
"""

import argparse
import json
from pathlib import Path

from pipeline_shared import run_pipeline_from_preset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v7 baseline through the shared local benchmark path")
    parser.add_argument("--questions", default=None, help="CSV question file to evaluate")
    parser.add_argument("--output-dir", default=None, help="Benchmark output directory")
    parser.add_argument("--n-questions", type=int, default=0, help="Optional question limit for smoke tests")
    parser.add_argument("--no-qwen3", action="store_true", help="Force BGE reranker instead of Qwen3-0.6B")
    parser.add_argument("--no-voting", action="store_true", help="Force single-pass answer generation")
    args = parser.parse_args()

    overrides = {}
    if args.no_qwen3:
        overrides.setdefault("rerank", {})["model_preference"] = ["bge"]
    if args.no_voting:
        overrides.setdefault("llm", {}).update({"base_passes": 1, "hard_passes": 1})

    output_dir = args.output_dir or str(Path("outputs") / "benchmarks" / "v7_baseline")
    result = run_pipeline_from_preset(
        preset_name="v7_baseline",
        questions_path=args.questions,
        output_dir=output_dir,
        env="local",
        n_questions=args.n_questions,
        overrides=overrides or None,
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=True))
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
