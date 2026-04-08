#!/usr/bin/env python3
"""Compare the shared runtime against the rendered Kaggle bundle on the same input."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.kaggle_bundle import render_standalone_kaggle_script
from notebooks.pipeline_shared import load_csv, write_json


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare the shared pipeline against the Kaggle bundle")
    parser.add_argument("--preset", required=True, help="Preset name from notebooks/pipeline_presets.json")
    parser.add_argument("--questions", default=None, help="CSV file to score")
    parser.add_argument("--output-dir", required=True, help="Directory to store parity artifacts")
    parser.add_argument("--n-questions", type=int, default=0, help="Optional question limit for smoke runs")
    parser.add_argument("--runtime-budget-hours", type=float, default=None, help="Override runtime budget gate")
    parser.add_argument("--env", default="local", choices=["local", "kaggle"], help="Filesystem/runtime environment")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for reproducibility")
    parser.add_argument("--hardware-tag", default="unknown", help="Free-form hardware label for manifests")
    parser.add_argument(
        "--override-json",
        default=None,
        help="Inline JSON object merged into the preset for one-off candidate experiments",
    )
    return parser.parse_args(argv)


def _coerce_overrides(overrides: Optional[dict[str, Any] | str]) -> Optional[dict[str, Any]]:
    if overrides is None:
        return None
    if isinstance(overrides, str):
        return json.loads(overrides)
    return overrides


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["Question_ID"]: row for row in rows}


def compare_prediction_rows(base_rows: Iterable[dict[str, Any]], candidate_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    base_by_id = _row_index(base_rows)
    candidate_by_id = _row_index(candidate_rows)
    qids = sorted(set(base_by_id) & set(candidate_by_id), key=lambda value: int(value))

    mismatches = []
    for qid in qids:
        base = base_by_id[qid]
        candidate = candidate_by_id[qid]
        if base != candidate:
            mismatches.append(
                {
                    "Question_ID": qid,
                    "base": base,
                    "candidate": candidate,
                }
            )

    return {
        "n_questions": len(qids),
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
    }


def compare_run_manifests(base_manifest: dict[str, Any], candidate_manifest: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(base_manifest) | set(candidate_manifest))
    diff = {}
    mismatched_keys = []
    for key in keys:
        if base_manifest.get(key) != candidate_manifest.get(key):
            diff[key] = {"base": base_manifest.get(key), "candidate": candidate_manifest.get(key)}
            mismatched_keys.append(key)
    return {"mismatched_keys": mismatched_keys, "diff": diff}


def compare_parity_runs(base_dir: Path | str, candidate_dir: Path | str) -> dict[str, Any]:
    base_dir = Path(base_dir)
    candidate_dir = Path(candidate_dir)
    if (candidate_dir / "shared").exists() and (candidate_dir / "bundle").exists():
        base_dir = candidate_dir / "shared"
        candidate_dir = candidate_dir / "bundle"

    base_predictions = load_csv(base_dir / "predictions.csv")
    candidate_predictions = load_csv(candidate_dir / "predictions.csv")
    prediction_diff = compare_prediction_rows(base_predictions, candidate_predictions)

    base_manifest_path = base_dir / "run_manifest.json"
    candidate_manifest_path = candidate_dir / "run_manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text(encoding="utf-8")) if base_manifest_path.exists() else {}
    candidate_manifest = (
        json.loads(candidate_manifest_path.read_text(encoding="utf-8")) if candidate_manifest_path.exists() else {}
    )
    manifest_diff = compare_run_manifests(base_manifest, candidate_manifest)

    return {
        "base_dir": str(base_dir),
        "candidate_dir": str(candidate_dir),
        "prediction_diff": prediction_diff,
        "manifest_diff": manifest_diff,
        "predictions_match": prediction_diff["n_mismatches"] == 0,
        "manifests_match": len(manifest_diff["mismatched_keys"]) == 0,
    }


def _build_manifest(
    *,
    preset: str,
    questions_path: Optional[Path],
    output_dir: Path,
    env: str,
    seed: int,
    hardware_tag: str,
    runtime_budget_hours: float | None,
    overrides: Optional[dict[str, Any]],
    summary: dict[str, Any],
    role: str,
) -> dict[str, Any]:
    from scripts.benchmark_candidate import build_run_manifest

    return {
        **build_run_manifest(
            preset_name=preset,
            questions_path=questions_path,
            output_dir=output_dir,
            env=env,
            seed=seed,
            hardware_tag=hardware_tag,
            split=None,
            fold=None,
            repeat_index=0,
            repeat_count=1,
            parity_mode=role,
            runtime_budget_hours=runtime_budget_hours,
            overrides=overrides,
            result_summary=summary,
        ),
        "role": role,
    }


def _run_pipeline(
    module: Any,
    *,
    preset: str,
    questions_path: Optional[Path],
    output_dir: Path,
    env: str,
    n_questions: int,
    runtime_budget_hours: float | None,
    overrides: Optional[dict[str, Any]],
    seed: int,
    run_metadata: Optional[dict[str, Any]],
):
    seed_everything(seed)
    return module.run_pipeline_from_preset(
        preset_name=preset,
        questions_path=str(questions_path) if questions_path else None,
        output_dir=str(output_dir),
        env=env,
        n_questions=n_questions,
        runtime_budget_hours=runtime_budget_hours,
        overrides=overrides,
        run_metadata=run_metadata,
    )


def prepare_bundle_workspace(
    *,
    repo_root: Path,
    temp_root: Path,
    preset_name: str,
) -> tuple[Path, Path]:
    workspace_root = temp_root / "bundle_repo"
    notebooks_dir = workspace_root / "notebooks"
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    for name in ("data", "kaggle_datasets", "models", "local_packages"):
        source = repo_root / name
        target = workspace_root / name
        if target.exists() or target.is_symlink():
            continue
        if hasattr(target, "symlink_to"):
            target.symlink_to(source, target_is_directory=source.is_dir())
        else:
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)

    bundle_path = render_standalone_kaggle_script(preset_name, notebooks_dir / "standalone_kaggle.py")
    return workspace_root, bundle_path


def run_kaggle_parity(
    *,
    preset_name: str,
    questions_path: Optional[Path],
    output_dir: Path | str,
    env: str,
    n_questions: int,
    runtime_budget_hours: float | None,
    overrides: Optional[dict[str, Any] | str],
    seed: int,
    hardware_tag: str,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    overrides = _coerce_overrides(overrides)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        _, bundle_path = prepare_bundle_workspace(
            repo_root=REPO_ROOT,
            temp_root=tmpdir,
            preset_name=preset_name,
        )
        bundle_module = _load_module_from_path("standalone_kaggle_parity", bundle_path)

        shared_dir = output_dir / "shared"
        bundle_dir = output_dir / "bundle"
        shared_dir.mkdir(parents=True, exist_ok=True)
        bundle_dir.mkdir(parents=True, exist_ok=True)

        shared_result = _run_pipeline(
            __import__("notebooks.pipeline_shared", fromlist=["run_pipeline_from_preset"]),
            preset=preset_name,
            questions_path=questions_path,
            output_dir=shared_dir,
            env=env,
            n_questions=n_questions,
            runtime_budget_hours=runtime_budget_hours,
            overrides=overrides,
            seed=seed,
            run_metadata={
                "seed": seed,
                "hardware_tag": hardware_tag,
                "parity_role": "shared",
                "parity_mode": "compare",
            },
        )
        bundle_result = _run_pipeline(
            bundle_module,
            preset=preset_name,
            questions_path=questions_path,
            output_dir=bundle_dir,
            env=env,
            n_questions=n_questions,
            runtime_budget_hours=runtime_budget_hours,
            overrides=overrides,
            seed=seed,
            run_metadata={
                "seed": seed,
                "hardware_tag": hardware_tag,
                "parity_role": "bundle",
                "parity_mode": "compare",
            },
        )

        shared_manifest = shared_result.get("run_manifest") or _build_manifest(
            preset=preset_name,
            questions_path=questions_path,
            output_dir=shared_dir,
            env=env,
            seed=seed,
            hardware_tag=hardware_tag,
            runtime_budget_hours=runtime_budget_hours,
            overrides=overrides,
            summary=shared_result["summary"],
            role="shared",
        )
        bundle_manifest = bundle_result.get("run_manifest") or _build_manifest(
            preset=preset_name,
            questions_path=questions_path,
            output_dir=bundle_dir,
            env=env,
            seed=seed,
            hardware_tag=hardware_tag,
            runtime_budget_hours=runtime_budget_hours,
            overrides=overrides,
            summary=bundle_result["summary"],
            role="bundle",
        )
        if not shared_result.get("run_manifest"):
            write_json(shared_dir / "run_manifest.json", shared_manifest)
        if not bundle_result.get("run_manifest"):
            write_json(bundle_dir / "run_manifest.json", bundle_manifest)

        report = compare_parity_runs(shared_dir, bundle_dir)
        report.update(
            {
                "preset": preset_name,
                "questions_path": str(questions_path) if questions_path else None,
                "seed": seed,
                "hardware_tag": hardware_tag,
                "shared": {
                    "output_dir": str(shared_dir),
                    "summary": shared_result["summary"],
                    "manifest": shared_manifest,
                },
                "bundle": {
                    "output_dir": str(bundle_dir),
                    "summary": bundle_result["summary"],
                    "manifest": bundle_manifest,
                },
            }
        )
        return report


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    report = run_kaggle_parity(
        preset_name=args.preset,
        questions_path=Path(args.questions) if args.questions else None,
        output_dir=Path(args.output_dir),
        env=args.env,
        n_questions=args.n_questions,
        runtime_budget_hours=args.runtime_budget_hours,
        overrides=args.override_json,
        seed=args.seed,
        hardware_tag=args.hardware_tag,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
