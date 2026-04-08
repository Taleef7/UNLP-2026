#!/usr/bin/env python3
"""Generate a standalone Kaggle submission script from the shared pipeline."""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.kaggle_bundle import render_standalone_kaggle_script


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone Kaggle submission script")
    parser.add_argument("--preset", required=True, help="Preset name from notebooks/pipeline_presets.json")
    parser.add_argument("--output", required=True, help="Output .py path for the standalone Kaggle script")
    args = parser.parse_args()

    output_path = render_standalone_kaggle_script(args.preset, Path(args.output))
    print(output_path)


if __name__ == "__main__":
    main()
