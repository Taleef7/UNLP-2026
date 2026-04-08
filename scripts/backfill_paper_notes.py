#!/usr/bin/env python3
"""Backfill paper_notes.md for pre-orchestrator paper ablation runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_paper_followup_suite import PAPER_RUNS_R2_ROOT, PAPER_RUNS_ROOT, RunSpec, SPLIT_SPECS, render_run_notes


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _infer_split_key(run_dir: Path, benchmark_manifest: dict[str, Any]) -> Optional[str]:
    run_name = run_dir.name
    questions_path = str(benchmark_manifest.get("questions_path") or "")
    if "fold0" in run_name or questions_path.endswith("fold_0_val.csv"):
        return "fold0"
    if "fold1" in run_name or questions_path.endswith("fold_1_val.csv"):
        return "fold1"
    if "lockbox" in run_name or questions_path.endswith("lockbox.csv"):
        return "lockbox"
    if "full_dev" in run_name or run_name.endswith("_full") or questions_path.endswith("val.csv") or questions_path.endswith("dev_questions.csv"):
        return "full_dev"
    return None


def backfill_notes(root: Path = PAPER_RUNS_ROOT) -> list[dict[str, Any]]:
    written = []
    for run_dir in sorted(root.rglob("*")):
        if not run_dir.is_dir():
            continue
        summary = _read_json(run_dir / "summary.json")
        benchmark_manifest = _read_json(run_dir / "benchmark_manifest.json")
        diff_summary = _read_json(run_dir / "diff_vs_baseline" / "diff_summary.json")
        if summary is None or benchmark_manifest is None or diff_summary is None:
            continue

        split_key = _infer_split_key(run_dir, benchmark_manifest)
        if split_key is None:
            continue

        base_dir = Path(diff_summary.get("base_dir") or SPLIT_SPECS[split_key]["baseline_dir"])
        questions_path = Path(benchmark_manifest.get("questions_path") or SPLIT_SPECS[split_key]["questions"])
        spec = RunSpec(
            preset=benchmark_manifest.get("preset") or run_dir.name,
            split_key=split_key,
            output_dir=run_dir,
            baseline_dir=base_dir,
            questions_path=questions_path,
            override_json=benchmark_manifest.get("override_json"),
            stage="backfill",
            purpose="Backfilled paper notes for a paper ablation run that was launched before the orchestrator handled note generation.",
        )
        notes_path = run_dir / "paper_notes.md"
        notes_path.write_text(render_run_notes(spec, summary, diff_summary), encoding="utf-8")
        written.append({"run_dir": str(run_dir), "notes_path": str(notes_path)})
    return written


def main() -> None:
    written = backfill_notes(PAPER_RUNS_ROOT)
    if PAPER_RUNS_R2_ROOT.exists():
        written.extend(backfill_notes(PAPER_RUNS_R2_ROOT))
    payload = {"written": written}
    print(json.dumps(payload, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
