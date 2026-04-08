"""Build standalone Kaggle submission scripts from the shared pipeline."""

from __future__ import annotations

from pathlib import Path


NOTEBOOK_DIR = Path(__file__).resolve().parent
PIPELINE_SHARED_PATH = NOTEBOOK_DIR / "pipeline_shared.py"
PIPELINE_PRESETS_PATH = NOTEBOOK_DIR / "pipeline_presets.json"


def render_standalone_kaggle_script(
    preset_name: str,
    output_path: Path | str,
    source_path: Path | str = PIPELINE_SHARED_PATH,
    presets_path: Path | str = PIPELINE_PRESETS_PATH,
) -> Path:
    output_path = Path(output_path)
    source_text = Path(source_path).read_text(encoding="utf-8").rstrip() + "\n"
    presets_text = Path(presets_path).read_text(encoding="utf-8")
    trailer = f"""

# Standalone Kaggle bundle payload.
EMBEDDED_PIPELINE_PRESETS = json.loads({presets_text!r})


if __name__ == "__main__":
    result = run_pipeline_from_preset(
        preset_name={preset_name!r},
        output_dir="/kaggle/working",
        env="kaggle",
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=True))
"""
    output_path.write_text(source_text + trailer.lstrip("\n"), encoding="utf-8")
    return output_path
