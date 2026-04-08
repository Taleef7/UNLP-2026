#!/usr/bin/env python3
"""Run paper-facing ablation follow-ups with clean artifacts, diffs, and notes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_candidate.py"
DIFF_SCRIPT = REPO_ROOT / "scripts" / "diff_benchmark_runs.py"
PAPER_RUNS_ROOT = REPO_ROOT / "outputs" / "paper_ablation_runs"
PAPER_RUNS_R2_ROOT = REPO_ROOT / "outputs" / "paper_ablation_runs_r2" / "2026-03-31_a30"

SPLIT_SPECS = {
    "fold0": {
        "questions": SPLITS_DIR / "fold_0_val.csv",
        "baseline_dir": REPO_ROOT / "outputs" / "benchmarks" / "v7_baseline_fold0",
    },
    "fold1": {
        "questions": SPLITS_DIR / "fold_1_val.csv",
        "baseline_dir": REPO_ROOT / "outputs" / "benchmarks" / "v7_baseline_fold1",
    },
    "lockbox": {
        "questions": SPLITS_DIR / "lockbox.csv",
        "baseline_dir": REPO_ROOT / "outputs" / "benchmarks" / "v7_baseline_lockbox",
    },
    "full_dev": {
        "questions": REPO_ROOT / "data" / "dev_questions.csv",
        "baseline_dir": REPO_ROOT / "outputs" / "benchmarks" / "v7_full_dev",
    },
}

GATE_SPLITS = ("fold0", "fold1", "lockbox")
FULL_LADDER_SPLITS = (*GATE_SPLITS, "full_dev")
MANDATORY_PRESET = "candidate_dense_margin_lock_v1"
V13_PRESET = "candidate_v5_refocus_v1"
FAIR_COMBO_PRESETS = (
    "candidate_v5refocus_reranker4b_fair",
    "candidate_dense_doc_lock_v3_reranker4b_fair",
)
TRIPLE_PRESET = "candidate_v5refocus_ddl_v3_reranker4b_fair"
FEASIBILITY_PRESET = "candidate_stronger_reranker_v3_8b_feasibility"


@dataclass(frozen=True)
class RunSpec:
    preset: str
    split_key: str
    output_dir: Path
    baseline_dir: Path
    questions_path: Path
    n_questions: int = 0
    override_json: Optional[dict[str, Any]] = None
    stage: str = ""
    purpose: str = ""


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UNLP 2026 paper-facing follow-up suites")
    parser.add_argument(
        "--stage",
        choices=[
            "mandatory_dense_margin",
            "fair_combo_gate",
            "8b_feasibility",
            "v13_full_dev",
            "fair_combo_matrix",
            "8b_teardown",
            "paper_round2",
            "all",
        ],
        default="paper_round2",
        help="Which paper follow-up stage to run",
    )
    parser.add_argument(
        "--hardware-tag",
        default="a30-paper-r2",
        help="Hardware label recorded in benchmark manifests",
    )
    parser.add_argument(
        "--output-root",
        default=str(PAPER_RUNS_R2_ROOT),
        help="Directory where paper run artifacts should be written",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned actions without executing them")
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a failed stage and record the failure in the suite report",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse finished artifact directories that already contain summary, diff, and notes.",
    )
    parser.add_argument(
        "--gate-mean-threshold",
        type=float,
        default=-0.001,
        help="Legacy threshold for the March 30 fair combo gate stage.",
    )
    parser.add_argument(
        "--gate-min-threshold",
        type=float,
        default=-0.01,
        help="Legacy minimum delta for the March 30 fair combo gate stage.",
    )
    return parser.parse_args(argv)


def build_run_spec(
    *,
    preset: str,
    split_key: str,
    output_root: Path,
    stage: str,
    purpose: str,
    n_questions: int = 0,
    override_json: Optional[dict[str, Any]] = None,
    suffix: str = "",
) -> RunSpec:
    split_spec = SPLIT_SPECS[split_key]
    run_name = f"{preset}_{split_key}{suffix}"
    return RunSpec(
        preset=preset,
        split_key=split_key,
        output_dir=output_root / run_name,
        baseline_dir=split_spec["baseline_dir"],
        questions_path=split_spec["questions"],
        n_questions=n_questions,
        override_json=override_json,
        stage=stage,
        purpose=purpose,
    )


def benchmark_command(spec: RunSpec, hardware_tag: str) -> list[str]:
    command = [
        sys.executable,
        str(BENCHMARK_SCRIPT),
        "--preset",
        spec.preset,
        "--questions",
        str(spec.questions_path),
        "--output-dir",
        str(spec.output_dir),
        "--hardware-tag",
        hardware_tag,
    ]
    if spec.n_questions:
        command.extend(["--n-questions", str(spec.n_questions)])
    if spec.override_json is not None:
        command.extend(["--override-json", json.dumps(spec.override_json, ensure_ascii=True, sort_keys=True)])
    return command


def diff_command(spec: RunSpec) -> list[str]:
    return [
        sys.executable,
        str(DIFF_SCRIPT),
        "--base-dir",
        str(spec.baseline_dir),
        "--candidate-dir",
        str(spec.output_dir),
    ]


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> subprocess.CompletedProcess[str]:
    if dry_run:
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps({"cmd": command}), stderr="")
    return subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def is_completed_run(spec: RunSpec) -> bool:
    return (
        (spec.output_dir / "summary.json").exists()
        and (spec.output_dir / "timings.json").exists()
        and (spec.output_dir / "run_manifest.json").exists()
        and (spec.output_dir / "benchmark_manifest.json").exists()
        and (spec.output_dir / "diff_vs_baseline" / "diff_summary.json").exists()
        and (spec.output_dir / "paper_notes.md").exists()
    )


def render_run_notes(spec: RunSpec, summary: dict[str, Any], diff_summary: Optional[dict[str, Any]]) -> str:
    lines = [
        f"# {spec.preset} on {spec.split_key}",
        "",
        f"- Stage: `{spec.stage}`",
        f"- Purpose: {spec.purpose}",
        f"- Questions: `{spec.questions_path}`",
        f"- Baseline: `{spec.baseline_dir}`",
        f"- Output dir: `{spec.output_dir}`",
        f"- Composite score: `{summary.get('composite_score')}`",
        f"- Answer accuracy: `{summary.get('answer_accuracy')}`",
        f"- Doc accuracy: `{summary.get('doc_accuracy')}`",
        f"- Page proximity: `{summary.get('page_proximity')}`",
        f"- Reranker: `{summary.get('reranker_name')}`",
        f"- LLM n_ctx: `{summary.get('llm_n_ctx')}`",
        "",
    ]
    if diff_summary:
        lines.extend(
            [
                "## Diff vs v7 Baseline",
                "",
                f"- Average score delta: `{diff_summary.get('avg_score_delta')}`",
                f"- Top-1 doc flip rate: `{diff_summary.get('top_1_doc_flip_rate')}`",
                f"- Page change rate: `{diff_summary.get('page_change_rate')}`",
                f"- Answer change rate: `{diff_summary.get('answer_change_rate')}`",
                f"- Per-domain deltas: `{json.dumps(diff_summary.get('per_domain_score_delta', {}), ensure_ascii=True, sort_keys=True)}`",
                f"- Counts: `{json.dumps(diff_summary.get('counts', {}), ensure_ascii=True, sort_keys=True)}`",
                "",
            ]
        )
    else:
        lines.extend(["## Diff vs v7 Baseline", "", "- No diff summary was produced for this run.", ""])

    diagnostics = summary.get("diagnostics") or {}
    gpu_snapshots = diagnostics.get("gpu_memory_snapshots") or []
    if gpu_snapshots:
        lines.extend(["## GPU Memory Snapshots", ""])
        for snapshot in gpu_snapshots:
            label = snapshot.get("label", "unknown")
            free_gib = snapshot.get("free_bytes")
            total_gib = snapshot.get("total_bytes")
            allocated = snapshot.get("allocated_bytes")
            reserved = snapshot.get("reserved_bytes")
            if isinstance(free_gib, int) and isinstance(total_gib, int):
                lines.append(
                    f"- `{label}`: free={free_gib / (1024 ** 3):.2f} GiB, total={total_gib / (1024 ** 3):.2f} GiB, allocated={allocated / (1024 ** 3):.2f} GiB, reserved={reserved / (1024 ** 3):.2f} GiB"
                )
            else:
                lines.append(f"- `{label}`: {json.dumps(snapshot, ensure_ascii=True, sort_keys=True)}")
        lines.append("")

    if spec.override_json is not None:
        lines.extend(
            [
                "## Override JSON",
                "",
                "```json",
                json.dumps(spec.override_json, indent=2, ensure_ascii=True, sort_keys=True),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def execute_run_spec(
    spec: RunSpec,
    *,
    hardware_tag: str,
    dry_run: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    if skip_existing and is_completed_run(spec):
        summary = load_json(spec.output_dir / "summary.json")
        diff_summary = load_json(spec.output_dir / "diff_vs_baseline" / "diff_summary.json")
        return {
            "preset": spec.preset,
            "split": spec.split_key,
            "output_dir": str(spec.output_dir),
            "status": "skipped_existing",
            "composite_score": summary.get("composite_score"),
            "avg_score_delta": diff_summary.get("avg_score_delta"),
            "answer_change_rate": diff_summary.get("answer_change_rate"),
        }

    benchmark = run_command(benchmark_command(spec, hardware_tag), cwd=REPO_ROOT, dry_run=dry_run)
    diff = run_command(diff_command(spec), cwd=REPO_ROOT, dry_run=dry_run)
    if dry_run:
        return {
            "preset": spec.preset,
            "split": spec.split_key,
            "output_dir": str(spec.output_dir),
            "benchmark_cmd": json.loads(benchmark.stdout)["cmd"],
            "diff_cmd": json.loads(diff.stdout)["cmd"],
        }

    summary = load_json(spec.output_dir / "summary.json")
    diff_summary = load_json(spec.output_dir / "diff_vs_baseline" / "diff_summary.json")
    write_text(spec.output_dir / "paper_notes.md", render_run_notes(spec, summary, diff_summary))
    return {
        "preset": spec.preset,
        "split": spec.split_key,
        "output_dir": str(spec.output_dir),
        "status": "completed",
        "summary_path": str(spec.output_dir / "summary.json"),
        "diff_summary_path": str(spec.output_dir / "diff_vs_baseline" / "diff_summary.json"),
        "composite_score": summary.get("composite_score"),
        "avg_score_delta": diff_summary.get("avg_score_delta"),
        "answer_change_rate": diff_summary.get("answer_change_rate"),
    }


def assess_gate_stability(
    gate_results: dict[str, dict[str, Any]],
    *,
    mean_threshold: float,
    min_threshold: float,
) -> dict[str, Any]:
    deltas = [float(result["avg_score_delta"]) for result in gate_results.values() if result.get("avg_score_delta") is not None]
    if len(deltas) != len(gate_results):
        return {
            "stable": False,
            "reason": "missing_diff_summary",
            "mean_avg_score_delta": None,
            "min_avg_score_delta": None,
        }
    mean_delta = sum(deltas) / len(deltas)
    min_delta = min(deltas)
    stable = mean_delta >= mean_threshold and min_delta >= min_threshold
    return {
        "stable": stable,
        "reason": "passed" if stable else "gate_threshold_failed",
        "mean_avg_score_delta": mean_delta,
        "min_avg_score_delta": min_delta,
        "thresholds": {"mean_threshold": mean_threshold, "min_threshold": min_threshold},
    }


def write_suite_report(output_root: Path, filename: str, payload: dict[str, Any], *, dry_run: bool) -> Optional[Path]:
    if dry_run:
        return None
    path = output_root / filename
    write_text(path, json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    return path


def run_stage_mandatory_dense_margin(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    results = []
    for split_key in FULL_LADDER_SPLITS:
        spec = build_run_spec(
            preset=MANDATORY_PRESET,
            split_key=split_key,
            output_root=output_root,
            stage="mandatory_dense_margin",
            purpose="Complete the dense-margin lock evidence across fold0, fold1, lockbox, and full_dev.",
        )
        results.append(
            execute_run_spec(spec, hardware_tag=args.hardware_tag, dry_run=args.dry_run, skip_existing=args.skip_existing)
        )
    report = {"stage": "mandatory_dense_margin", "results": results}
    write_suite_report(output_root, "mandatory_dense_margin_report.json", report, dry_run=args.dry_run)
    return report


def _run_smoke_then_splits(
    *,
    preset: str,
    args: argparse.Namespace,
    output_root: Path,
    stage: str,
    purpose: str,
    override_json: Optional[dict[str, Any]] = None,
    splits: tuple[str, ...] = FULL_LADDER_SPLITS,
) -> dict[str, Any]:
    smoke = execute_run_spec(
        build_run_spec(
            preset=preset,
            split_key="fold0",
            output_root=output_root,
            stage=stage,
            purpose=f"{purpose} Smoke run before the full split ladder.",
            n_questions=1,
            override_json=override_json,
            suffix="_smoke",
        ),
        hardware_tag=args.hardware_tag,
        dry_run=args.dry_run,
        skip_existing=args.skip_existing,
    )

    split_results = {}
    for split_key in splits:
        split_results[split_key] = execute_run_spec(
            build_run_spec(
                preset=preset,
                split_key=split_key,
                output_root=output_root,
                stage=stage,
                purpose=purpose,
                override_json=override_json,
            ),
            hardware_tag=args.hardware_tag,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )
    return {"smoke": smoke, "splits": split_results}


def run_stage_v13_full_dev(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    spec = build_run_spec(
        preset=V13_PRESET,
        split_key="full_dev",
        output_root=output_root,
        stage="v13_full_dev",
        purpose="Close the missing full-dev offline evidence for the exact v13 submission family.",
    )
    result = execute_run_spec(spec, hardware_tag=args.hardware_tag, dry_run=args.dry_run, skip_existing=args.skip_existing)
    report = {"stage": "v13_full_dev", "result": result}
    write_suite_report(output_root, "v13_full_dev_report.json", report, dry_run=args.dry_run)
    return report


def run_stage_fair_combo_matrix(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    families: dict[str, dict[str, Any]] = {}
    purposes = {
        "candidate_v5refocus_reranker4b_fair": "Run the clean prompt-plus-fair-4B matrix for the paper.",
        "candidate_dense_doc_lock_v3_reranker4b_fair": "Run the clean dense-doc-lock-plus-fair-4B matrix for the paper.",
        TRIPLE_PRESET: "Run the clean triple-combo matrix for the paper, including negative results if needed.",
    }
    for preset in (*FAIR_COMBO_PRESETS, TRIPLE_PRESET):
        families[preset] = _run_smoke_then_splits(
            preset=preset,
            args=args,
            output_root=output_root,
            stage="fair_combo_matrix",
            purpose=purposes[preset],
        )
    report = {"stage": "fair_combo_matrix", "families": families}
    write_suite_report(output_root, "fair_combo_matrix_report.json", report, dry_run=args.dry_run)
    return report


def run_stage_8b_teardown(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    override_json = {
        "runtime": {
            "record_gpu_memory_snapshots": True,
            "teardown_reranker_before_llm": True,
        }
    }
    run_name = f"{FEASIBILITY_PRESET}_teardown_fold0_smoke"
    spec = RunSpec(
        preset=FEASIBILITY_PRESET,
        split_key="fold0",
        output_dir=output_root / run_name,
        baseline_dir=SPLIT_SPECS["fold0"]["baseline_dir"],
        questions_path=SPLIT_SPECS["fold0"]["questions"],
        n_questions=1,
        override_json=override_json,
        stage="8b_teardown",
        purpose="Final qwen3_8b diagnostic with explicit reranker teardown before MamayLM load and GPU memory snapshots.",
    )
    try:
        result = execute_run_spec(spec, hardware_tag=args.hardware_tag, dry_run=args.dry_run, skip_existing=args.skip_existing)
        report = {"stage": "8b_teardown", "result": result}
    except subprocess.CalledProcessError as exc:
        failure = {
            "stage": "8b_teardown",
            "preset": spec.preset,
            "split": spec.split_key,
            "output_dir": str(spec.output_dir),
            "returncode": exc.returncode,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
        if not args.dry_run:
            notes = [
                f"# {spec.preset} teardown feasibility failure",
                "",
                "- Status: failed before a full summary could be written.",
                f"- Purpose: {spec.purpose}",
                "",
                "## stderr",
                "",
                "```text",
                exc.stderr or "",
                "```",
                "",
            ]
            write_text(spec.output_dir / "paper_notes.md", "\n".join(notes))
            write_suite_report(spec.output_dir, "feasibility_failure.json", failure, dry_run=False)
        report = failure
        if not args.continue_on_error:
            raise
    write_suite_report(output_root, "8b_teardown_report.json", report, dry_run=args.dry_run)
    return report


def run_stage_fair_combo_gate(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    gate_results: dict[str, dict[str, dict[str, Any]]] = {}
    full_results: dict[str, dict[str, Any]] = {}

    for preset in FAIR_COMBO_PRESETS:
        gate_results[preset] = {}
        for split_key in GATE_SPLITS:
            spec = build_run_spec(
                preset=preset,
                split_key=split_key,
                output_root=output_root,
                stage="fair_combo_gate",
                purpose="Legacy March 30 gate run retained for backwards compatibility.",
            )
            gate_results[preset][split_key] = execute_run_spec(
                spec,
                hardware_tag=args.hardware_tag,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )

    gate_assessments = {
        preset: assess_gate_stability(
            split_results,
            mean_threshold=args.gate_mean_threshold,
            min_threshold=args.gate_min_threshold,
        )
        for preset, split_results in gate_results.items()
    }

    eligible = all(assessment["stable"] for assessment in gate_assessments.values())
    if eligible:
        for preset in FAIR_COMBO_PRESETS:
            full_results[preset] = execute_run_spec(
                build_run_spec(
                    preset=preset,
                    split_key="full_dev",
                    output_root=output_root,
                    stage="fair_combo_gate",
                    purpose="Legacy gated promotion to full_dev.",
                ),
                hardware_tag=args.hardware_tag,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
        triple_runs = {}
        for split_key in FULL_LADDER_SPLITS:
            triple_runs[split_key] = execute_run_spec(
                build_run_spec(
                    preset=TRIPLE_PRESET,
                    split_key=split_key,
                    output_root=output_root,
                    stage="fair_combo_gate",
                    purpose="Legacy gated triple combo.",
                ),
                hardware_tag=args.hardware_tag,
                dry_run=args.dry_run,
                skip_existing=args.skip_existing,
            )
        full_results[TRIPLE_PRESET] = triple_runs

    report = {
        "stage": "fair_combo_gate",
        "gate_results": gate_results,
        "gate_assessments": gate_assessments,
        "eligible_for_triple": eligible,
        "full_results": full_results,
    }
    write_suite_report(output_root, "fair_combo_gate_report.json", report, dry_run=args.dry_run)
    return report


def run_stage_8b_feasibility(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    attempts = [
        {"label": "strict", "override_json": None, "n_questions": 1},
        {"label": "batch2_retry", "override_json": {"rerank": {"batch_size": 2}}, "n_questions": 1},
    ]
    results = []
    for attempt in attempts:
        run_name = f"{FEASIBILITY_PRESET}_{attempt['label']}_fold0_smoke"
        spec = RunSpec(
            preset=FEASIBILITY_PRESET,
            split_key="fold0",
            output_dir=output_root / run_name,
            baseline_dir=SPLIT_SPECS["fold0"]["baseline_dir"],
            questions_path=SPLIT_SPECS["fold0"]["questions"],
            n_questions=attempt["n_questions"],
            override_json=attempt["override_json"],
            stage="8b_feasibility",
            purpose="Legacy March 30 qwen3_8b feasibility check without reranker teardown.",
        )
        try:
            result = execute_run_spec(spec, hardware_tag=args.hardware_tag, dry_run=args.dry_run, skip_existing=args.skip_existing)
            result["attempt"] = attempt["label"]
            results.append(result)
            if not args.dry_run:
                break
        except subprocess.CalledProcessError as exc:
            failure = {
                "preset": spec.preset,
                "split": spec.split_key,
                "attempt": attempt["label"],
                "output_dir": str(spec.output_dir),
                "returncode": exc.returncode,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            }
            if not args.dry_run:
                write_text(
                    spec.output_dir / "paper_notes.md",
                    "\n".join(
                        [
                            f"# {spec.preset} {attempt['label']} feasibility failure",
                            "",
                            "- Status: failed before a full summary could be written.",
                            f"- Purpose: {spec.purpose}",
                            "",
                            "## stderr",
                            "",
                            "```text",
                            exc.stderr or "",
                            "```",
                            "",
                        ]
                    ),
                )
                write_suite_report(spec.output_dir, "feasibility_failure.json", failure, dry_run=False)
            results.append(failure)
            if not args.continue_on_error and attempt["label"] == attempts[-1]["label"]:
                raise
    report = {"stage": "8b_feasibility", "results": results}
    write_suite_report(output_root, "8b_feasibility_report.json", report, dry_run=args.dry_run)
    return report


def run_stage_paper_round2(args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    return {
        "stage": "paper_round2",
        "v13_full_dev": run_stage_v13_full_dev(args, output_root),
        "fair_combo_matrix": run_stage_fair_combo_matrix(args, output_root),
        "8b_teardown": run_stage_8b_teardown(args, output_root),
    }


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}
    stages = [args.stage] if args.stage != "all" else [
        "mandatory_dense_margin",
        "fair_combo_gate",
        "8b_feasibility",
        "v13_full_dev",
        "fair_combo_matrix",
        "8b_teardown",
    ]
    stage_handlers = {
        "mandatory_dense_margin": run_stage_mandatory_dense_margin,
        "fair_combo_gate": run_stage_fair_combo_gate,
        "8b_feasibility": run_stage_8b_feasibility,
        "v13_full_dev": run_stage_v13_full_dev,
        "fair_combo_matrix": run_stage_fair_combo_matrix,
        "8b_teardown": run_stage_8b_teardown,
        "paper_round2": run_stage_paper_round2,
    }

    for stage in stages:
        try:
            reports[stage] = stage_handlers[stage](args, output_root)
        except subprocess.CalledProcessError as exc:
            reports[stage] = {
                "stage": stage,
                "status": "failed",
                "returncode": exc.returncode,
                "stdout": exc.stdout,
                "stderr": exc.stderr,
            }
            if not args.continue_on_error:
                break

    print(json.dumps(reports, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
