"""Kaggle submission wrapper for the scored v7 baseline."""

import json

from pipeline_shared import run_pipeline_from_preset


if __name__ == "__main__":
    result = run_pipeline_from_preset(
        preset_name="v7_baseline",
        output_dir="/kaggle/working",
        env="kaggle",
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=True))
