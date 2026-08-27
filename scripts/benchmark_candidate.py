#!/usr/bin/env python3
"""Run preset-backed benchmarks and parity checks for the UNLP 2026 pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLITS_DIR = REPO_ROOT / "data" / "splits"
DEFAULT_BENCHMARK_ROOT = REPO_ROOT / "outputs" / "benchmarks"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.pipeline_shared import run_pipeline_from_preset, write_json


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UNLP 2026 local benchmark runner")
    parser.add_argument("--preset", required=True, help="Preset name from notebooks/pipeline_presets.json")
    parser.add_argument("--questions", default=None, help="CSV file to score")
    parser.add_argument("--split", default=None, help="Named split or split file stem under data/splits")
    parser.add_argument("--fold", default=None, help="Optional fold identifier used with --split")
    parser.add_argument("--output-dir", default=None, help="Benchmark artifact directory")
    parser.add_argument("--n-questions", type=int, default=0, help="Optional question limit for smoke runs")
    parser.add_argument("--repeat-count", type=int, default=1, help="Run the same benchmark multiple times")
    parser.add_argument("--seed", type=int, default=42, help="Seed used for local reproducibility")
    parser.add_argument("--hardware-tag", default="unknown", help="Free-form hardware label for manifests")
    parser.add_argument(
        "--parity-mode",
        choices=["off", "compare"],
        default="off",
        help="Run the local parity harness against the rendered Kaggle bundle",
    )
    parser.add_argument("--runtime-budget-hours", type=float, default=None, help="Override runtime budget gate")
    parser.add_argument("--env", default="local", choices=["local", "kaggle"], help="Filesystem/runtime environment")
    parser.add_argument(
        "--override-json",
        default=None,
        help="Inline JSON object merged into the preset for one-off candidate experiments",
    )
    return parser.parse_args(argv)


def _normalize_questions_path(questions: Path | str | None) -> Optional[Path]:
    if questions is None:
        return None
    return Path(questions)


def resolve_questions_path(
    questions: Path | str | None = None,
    split: str | None = None,
    fold: str | None = None,
    splits_dir: Path | str = DEFAULT_SPLITS_DIR,
) -> Optional[Path]:
    if questions is not None:
        return Path(questions)
    if split is None:
        return None

    splits_dir = Path(splits_dir)
    split_path = Path(split)
    if split_path.exists():
        return split_path
    if split_path.suffix == ".csv":
        candidate = splits_dir / split_path.name
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Split file not found: {candidate}")

    attempts = []
    if fold is not None:
        attempts.extend(
            [
                splits_dir / f"{split}_fold_{fold}.csv",
                splits_dir / f"{split}_fold{fold}.csv",
                splits_dir / split / f"fold_{fold}.csv",
                splits_dir / split / f"{fold}.csv",
            ]
        )
        if split in {"grouped_cv", "cv", "fold", "fold_val"} or str(split).endswith("_val"):
            attempts.append(splits_dir / f"fold_{fold}_val.csv")
        if split in {"grouped_cv_train", "cv_train", "fold_train"} or str(split).endswith("_train"):
            attempts.append(splits_dir / f"fold_{fold}_train.csv")
    attempts.append(splits_dir / f"{split}.csv")

    for candidate in attempts:
        if candidate.exists():
            return candidate

    attempted = ", ".join(str(path) for path in attempts)
    raise FileNotFoundError(f"Could not resolve split '{split}' with fold '{fold}'. Tried: {attempted}")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _stable_hash(*parts: Any) -> str:
    payload = _json_dumps(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
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


def _coerce_overrides(overrides: Optional[dict[str, Any] | str]) -> Optional[dict[str, Any]]:
    if overrides is None:
        return None
    if isinstance(overrides, str):
        return json.loads(overrides)
    return overrides


def build_output_dir(
    output_dir: Path | str | None,
    preset_name: str,
    split: str | None,
    fold: str | None,
    seed: int,
    repeat_index: int,
    repeat_count: int,
) -> Path:
    base = Path(output_dir) if output_dir else DEFAULT_BENCHMARK_ROOT / preset_name
    if repeat_count <= 1:
        return base

    run_name = [preset_name]
    if split:
        run_name.append(split)
    if fold is not None:
        run_name.append(f"fold-{fold}")
    run_name.append(f"seed-{seed}")
    run_name.append(f"repeat-{repeat_index + 1:02d}")
    return base / "__".join(run_name)


def build_run_manifest(
    *,
    preset_name: str,
    questions_path: Path | None,
    output_dir: Path,
    env: str,
    seed: int,
    hardware_tag: str,
    split: str | None,
    fold: str | None,
    repeat_index: int,
    repeat_count: int,
    parity_mode: str,
    runtime_budget_hours: float | None,
    overrides: Optional[dict[str, Any]],
    result_summary: dict[str, Any],
) -> dict[str, Any]:
    manifest = {
        "preset": preset_name,
        "questions_path": str(questions_path) if questions_path is not None else None,
        "output_dir": str(output_dir),
        "env": env,
        "seed": seed,
        "hardware_tag": hardware_tag,
        "split": split,
        "fold": fold,
        "repeat_index": repeat_index,
        "repeat_count": repeat_count,
        "parity_mode": parity_mode,
        "runtime_budget_hours": runtime_budget_hours,
        "override_json": overrides,
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "config_hash": _stable_hash(
            preset_name,
            str(questions_path),
            str(output_dir),
            env,
            seed,
            hardware_tag,
            split,
            fold,
            repeat_index,
            repeat_count,
            parity_mode,
            runtime_budget_hours,
            overrides,
        ),
        "summary": {
            "n_questions": result_summary.get("n_questions"),
            "reranker_name": result_summary.get("reranker_name"),
            "llm_n_ctx": result_summary.get("llm_n_ctx"),
            "timings": result_summary.get("timings"),
            "diagnostics": result_summary.get("diagnostics"),
        },
    }
    return manifest


def diff_prediction_rows(base_rows: Iterable[dict[str, Any]], candidate_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    base_by_id = {row["Question_ID"]: row for row in base_rows}
    candidate_by_id = {row["Question_ID"]: row for row in candidate_rows}
    qids = sorted(set(base_by_id) & set(candidate_by_id), key=lambda value: int(value))

    diff_rows = []
    counts = {"answer_changed": 0, "doc_changed": 0, "page_changed": 0}
    for qid in qids:
        base = base_by_id[qid]
        candidate = candidate_by_id[qid]
        answer_changed = base.get("Correct_Answer") != candidate.get("Correct_Answer")
        doc_changed = base.get("Doc_ID") != candidate.get("Doc_ID")
        page_changed = base.get("Page_Num") != candidate.get("Page_Num")
        counts["answer_changed"] += int(answer_changed)
        counts["doc_changed"] += int(doc_changed)
        counts["page_changed"] += int(page_changed)
        diff_rows.append(
            {
                "Question_ID": qid,
                "base_answer": base.get("Correct_Answer"),
                "candidate_answer": candidate.get("Correct_Answer"),
                "base_doc_id": base.get("Doc_ID"),
                "candidate_doc_id": candidate.get("Doc_ID"),
                "base_page_num": base.get("Page_Num"),
                "candidate_page_num": candidate.get("Page_Num"),
                "answer_changed": answer_changed,
                "doc_changed": doc_changed,
                "page_changed": page_changed,
            }
        )

    return {
        "n_questions": len(qids),
        **counts,
        "question_diffs": diff_rows,
    }


def _run_pipeline(
    *,
    preset_name: str,
    questions_path: Optional[Path],
    output_dir: Path,
    env: str,
    n_questions: int,
    runtime_budget_hours: float | None,
    overrides: Optional[dict[str, Any]],
    seed: int,
    run_metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    seed_everything(seed)
    result = run_pipeline_from_preset(
        preset_name=preset_name,
        questions_path=str(questions_path) if questions_path else None,
        output_dir=str(output_dir),
        env=env,
        n_questions=n_questions,
        runtime_budget_hours=runtime_budget_hours,
        overrides=overrides,
        run_metadata=run_metadata,
    )
    return result


def run_single_benchmark(
    *,
    preset_name: str,
    questions_path: Optional[Path],
    output_dir: Path,
    env: str,
    n_questions: int,
    runtime_budget_hours: float | None,
    overrides: Optional[dict[str, Any]],
    seed: int,
    hardware_tag: str,
    split: str | None,
    fold: str | None,
    repeat_index: int,
    repeat_count: int,
    parity_mode: str,
) -> dict[str, Any]:
    run_metadata = {
        "seed": seed,
        "hardware_tag": hardware_tag,
        "split": split,
        "fold": fold,
        "repeat_index": repeat_index,
        "repeat_count": repeat_count,
        "parity_mode": parity_mode,
    }
    result = _run_pipeline(
        preset_name=preset_name,
        questions_path=questions_path,
        output_dir=output_dir,
        env=env,
        n_questions=n_questions,
        runtime_budget_hours=runtime_budget_hours,
        overrides=overrides,
        seed=seed,
        run_metadata=run_metadata,
    )
    manifest = build_run_manifest(
        preset_name=preset_name,
        questions_path=questions_path,
        output_dir=output_dir,
        env=env,
        seed=seed,
        hardware_tag=hardware_tag,
        split=split,
        fold=fold,
        repeat_index=repeat_index,
        repeat_count=repeat_count,
        parity_mode=parity_mode,
        runtime_budget_hours=runtime_budget_hours,
        overrides=overrides,
        result_summary=result["summary"],
    )
    if result.get("run_manifest"):
        write_json(output_dir / "benchmark_manifest.json", manifest)
        return {"result": result, "manifest": result["run_manifest"], "benchmark_manifest": manifest}
    write_json(output_dir / "run_manifest.json", manifest)
    return {"result": result, "manifest": manifest, "benchmark_manifest": manifest}


def _load_parity_helpers():
    from scripts.check_kaggle_parity import compare_parity_runs, run_kaggle_parity

    return compare_parity_runs, run_kaggle_parity


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    overrides = _coerce_overrides(args.override_json)
    questions_path = resolve_questions_path(args.questions, args.split, args.fold)
    compare_parity_runs, run_kaggle_parity = _load_parity_helpers()

    repeat_summaries = []
    repeat_manifests = []
    for repeat_index in range(args.repeat_count):
        run_seed = args.seed + repeat_index
        run_output_dir = build_output_dir(
            args.output_dir,
            args.preset,
            args.split,
            args.fold,
            run_seed,
            repeat_index,
            args.repeat_count,
        )
        run_output_dir.mkdir(parents=True, exist_ok=True)
        bundle = run_single_benchmark(
            preset_name=args.preset,
            questions_path=questions_path,
            output_dir=run_output_dir,
            env=args.env,
            n_questions=args.n_questions,
            runtime_budget_hours=args.runtime_budget_hours,
            overrides=overrides,
            seed=run_seed,
            hardware_tag=args.hardware_tag,
            split=args.split,
            fold=args.fold,
            repeat_index=repeat_index,
            repeat_count=args.repeat_count,
            parity_mode=args.parity_mode,
        )
        repeat_summaries.append(bundle["result"]["summary"])
        repeat_manifests.append(bundle["manifest"])

        if args.parity_mode != "off":
            parity_output_dir = run_output_dir / "parity"
            parity_report = run_kaggle_parity(
                preset_name=args.preset,
                questions_path=questions_path,
                output_dir=parity_output_dir,
                env=args.env,
                n_questions=args.n_questions,
                runtime_budget_hours=args.runtime_budget_hours,
                overrides=overrides,
                seed=run_seed,
                hardware_tag=args.hardware_tag,
            )
            write_json(run_output_dir / "parity_report.json", parity_report)
            compare_parity_runs(run_output_dir, parity_output_dir)

    if len(repeat_summaries) == 1:
        output = {
            "summary": repeat_summaries[0],
            "manifest": repeat_manifests[0],
        }
    else:
        output = {
            "repeat_count": args.repeat_count,
            "summaries": repeat_summaries,
            "manifests": repeat_manifests,
        }
    print(json.dumps(output, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
