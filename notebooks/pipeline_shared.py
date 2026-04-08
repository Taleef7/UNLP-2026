from __future__ import annotations

import csv
import gc
import hashlib
import importlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def _resolve_runtime_roots() -> Tuple[Path, Path]:
    if "__file__" in globals():
        runtime_path = Path(__file__).resolve()
        return runtime_path.parents[1], runtime_path.parent

    cwd = Path.cwd().resolve()
    search_roots = [cwd, *cwd.parents]
    for candidate in search_roots:
        notebook_dir = candidate / "notebooks"
        if (notebook_dir / "pipeline_presets.json").exists():
            return candidate, notebook_dir
        if (candidate / "pipeline_presets.json").exists():
            return candidate.parent, candidate

    return cwd.parent, cwd


REPO_ROOT, NOTEBOOK_DIR = _resolve_runtime_roots()
PRESET_PATH = NOTEBOOK_DIR / "pipeline_presets.json"
LOCAL_PACKAGES = REPO_ROOT / "local_packages"
if str(LOCAL_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LOCAL_PACKAGES))

os.environ.setdefault("HF_HOME", str(REPO_ROOT / "cache" / "hf_cache"))
EMBEDDED_PIPELINE_PRESETS = None

ANSWER_CHOICES = ["A", "B", "C", "D", "E", "F"]
UA_TO_LATIN = {
    "А": "A",
    "В": "B",
    "С": "C",
    "Д": "D",
    "Е": "E",
    "Ф": "F",
    "а": "A",
    "в": "B",
    "с": "C",
    "д": "D",
    "е": "E",
    "ф": "F",
}


def stable_json_dumps(payload) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def compute_payload_hash(payload) -> str:
    return hashlib.sha1(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def compute_margin(items: List[Tuple[str, float]] | List[Tuple[str, int, float]]) -> Optional[float]:
    if len(items) < 2:
        return None
    return float(items[0][-1] - items[1][-1])


def should_lock_to_dense_top_doc(
    *,
    page_sel_cfg: dict,
    dense_candidates: List[Tuple[str, int, float]],
    doc_meta: Dict[str, dict],
) -> bool:
    if not page_sel_cfg.get("lock_to_dense_top_doc") or not dense_candidates:
        return False

    dense_top_doc = dense_candidates[0][0]

    min_pages = int(page_sel_cfg.get("lock_to_dense_top_doc_min_pages", 0) or 0)
    if min_pages > 0:
        dense_top_pages = int(doc_meta.get(dense_top_doc, {}).get("n_pages", 0) or 0)
        if dense_top_pages < min_pages:
            return False

    margin_threshold = page_sel_cfg.get("lock_dense_by_margin_threshold", None)
    if margin_threshold is not None:
        dense_margin = compute_margin(dense_candidates)
        if dense_margin is None or dense_margin <= float(margin_threshold):
            return False

    return True


def tokenize_sparse_text(text: str) -> List[str]:
    tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    return tokens or text.lower().split()


def reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, int, float]]],
    *,
    k: int = 60,
    weights: Optional[List[float]] = None,
) -> List[Tuple[str, int, float]]:
    fused_scores = defaultdict(float)
    weights = weights or [1.0] * len(ranked_lists)
    for ranked, weight in zip(ranked_lists, weights):
        for rank, (doc_id, page_num, _) in enumerate(ranked, start=1):
            fused_scores[(doc_id, page_num)] += float(weight) / float(k + rank)
    fused = [
        (doc_id, page_num, float(score))
        for (doc_id, page_num), score in fused_scores.items()
    ]
    fused.sort(key=lambda item: item[2], reverse=True)
    return fused


def detect_git_commit(repo_root: Path | str = REPO_ROOT) -> str:
    try:
        output = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return output or "unknown"
    except Exception:
        return "unknown"


def detect_hardware_tag(*, explicit_tag: Optional[str] = None, use_gpu: Optional[bool] = None) -> str:
    if explicit_tag:
        return explicit_tag
    if use_gpu is None:
        return "unknown"
    return "gpu" if use_gpu else "cpu"


def build_run_manifest(
    *,
    preset: dict,
    questions_path: Path | str,
    env: str,
    question_ids: List[str],
    run_metadata: Optional[dict],
    component_manifest: dict,
    extraction_manifest: dict,
    fallback_events: List[dict],
    use_gpu: Optional[bool],
) -> dict:
    metadata = deepcopy(run_metadata or {})
    return {
        "preset": preset.get("name"),
        "config_hash": compute_payload_hash(preset),
        "questions_path": str(questions_path),
        "question_count": len(question_ids),
        "question_ids_hash": compute_payload_hash(question_ids),
        "env": env,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": detect_git_commit(),
        "hardware_tag": detect_hardware_tag(
            explicit_tag=metadata.get("hardware_tag"),
            use_gpu=use_gpu,
        ),
        "run_metadata": metadata,
        "components": deepcopy(component_manifest),
        "extraction": deepcopy(extraction_manifest),
        "fallback_events": deepcopy(fallback_events),
    }


def extract_text_from_page(page, extraction_cfg: dict) -> str:
    mode = extraction_cfg.get("mode", "raw")
    text = ""
    if mode == "blocks":
        blocks = []
        for block in page.get_text("blocks"):
            if len(block) >= 5 and str(block[4]).strip():
                blocks.append(block)
        blocks.sort(key=lambda block: (round(block[1], 1), round(block[0], 1)))
        text = "\n".join(str(block[4]).strip() for block in blocks if str(block[4]).strip())
    elif mode == "words":
        words = []
        for word in page.get_text("words"):
            if len(word) >= 5 and str(word[4]).strip():
                words.append(word)
        words.sort(key=lambda word: (round(word[1], 1), round(word[0], 1)))
        text = " ".join(str(word[4]).strip() for word in words if str(word[4]).strip())
    elif mode == "blocks_plus_words":
        block_text = extract_text_from_page(page, {"mode": "blocks", "table_aware": False})
        word_text = extract_text_from_page(page, {"mode": "words", "table_aware": False})
        text = block_text if len(block_text) >= len(word_text) else word_text
    else:
        text = page.get_text("text")

    if extraction_cfg.get("table_aware"):
        table_snippets = []
        try:
            tables = page.find_tables()
            for table in getattr(tables, "tables", []):
                rows = table.extract() or []
                for row in rows:
                    cells = [str(cell).strip() for cell in row if cell is not None and str(cell).strip()]
                    if cells:
                        table_snippets.append(" | ".join(cells))
        except Exception:
            pass
        if table_snippets:
            text = "\n".join(filter(None, [text.strip(), "\n".join(table_snippets)]))

    return text.strip()


def load_csv(path: Path | str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_json(path: Path | str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path | str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path | str, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path | str, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def write_jsonl(path: Path | str, rows: List[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def deep_merge_dicts(base: dict, overrides: dict) -> dict:
    merged = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_pipeline_presets(path: Path | str = PRESET_PATH) -> Dict[str, dict]:
    path = Path(path)
    if EMBEDDED_PIPELINE_PRESETS is not None and (path == PRESET_PATH or not path.exists()):
        return deepcopy(EMBEDDED_PIPELINE_PRESETS)
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def first_existing_path(candidates: Iterable[Path | str]) -> Path:
    normalized = [Path(candidate) for candidate in candidates]
    for candidate in normalized:
        if candidate.exists():
            return candidate
    return normalized[0]


def rank_wheel_candidate(wheel_path: Path | str) -> Tuple[int, str]:
    wheel_name = Path(wheel_path).name
    current_cp = f"cp{sys.version_info.major}{sys.version_info.minor}"

    score = 0
    if f"-{current_cp}-{current_cp}-" in wheel_name:
        score += 100
    elif f"-{current_cp}-abi3-" in wheel_name:
        score += 95
    elif "-abi3-" in wheel_name:
        score += 90
    elif "-py3-none-any" in wheel_name or "-py3-none-" in wheel_name:
        score += 80

    if "manylinux" in wheel_name or "linux_x86_64" in wheel_name:
        score += 10
    elif wheel_name.endswith("any.whl"):
        score += 5

    return (score, wheel_name)


def install_best_matching_wheel(
    import_name: str,
    wheels_dir: Path | str,
    pattern: str,
    missing_message: str,
    *,
    use_find_links: bool = False,
    no_deps: bool = True,
):
    wheels_dir = Path(wheels_dir)
    wheel_candidates = sorted(
        wheels_dir.glob(pattern),
        key=rank_wheel_candidate,
        reverse=True,
    )
    if not wheel_candidates:
        raise ModuleNotFoundError(missing_message)

    install_errors = []
    for wheel_path in wheel_candidates:
        try:
            pip_cmd = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "--no-index",
            ]
            if use_find_links:
                pip_cmd.append(f"--find-links={wheels_dir}")
            if no_deps:
                pip_cmd.append("--no-deps")
            pip_cmd.append(str(wheel_path))
            subprocess.check_call(
                pip_cmd
            )
            importlib.invalidate_caches()
            return importlib.import_module(import_name)
        except subprocess.CalledProcessError as exc:
            install_errors.append((wheel_path.name, exc.returncode))
        except Exception as exc:
            install_errors.append((wheel_path.name, f"{type(exc).__name__}: {exc}"))

    attempted = ", ".join(f"{name} ({status})" for name, status in install_errors)
    raise ModuleNotFoundError(f"{missing_message}. Attempted wheels: {attempted}")


def ensure_fitz_available(wheels_dir: Optional[Path | str] = None):
    try:
        return importlib.import_module("fitz")
    except ModuleNotFoundError:
        if wheels_dir is None:
            raise

    return install_best_matching_wheel(
        import_name="fitz",
        wheels_dir=wheels_dir,
        pattern="pymupdf-*.whl",
        missing_message="No module named 'fitz' and no PyMuPDF wheel was found",
        no_deps=True,
    )


def ensure_llama_cpp_available(wheels_dir: Optional[Path | str] = None):
    try:
        return importlib.import_module("llama_cpp")
    except ModuleNotFoundError:
        if wheels_dir is None:
            raise

    return install_best_matching_wheel(
        import_name="llama_cpp",
        wheels_dir=wheels_dir,
        pattern="llama_cpp_python-*.whl",
        missing_message="No module named 'llama_cpp' and no llama_cpp_python wheel was found",
        use_find_links=True,
        no_deps=False,
    )


def resolve_preset(name: str, path: Path | str = PRESET_PATH, overrides: Optional[dict] = None) -> dict:
    presets = load_pipeline_presets(path)
    if name not in presets:
        raise KeyError(f"Unknown preset: {name}")

    resolved = deepcopy(presets[name])
    parent_name = resolved.pop("inherits", None)
    if parent_name:
        resolved = deep_merge_dicts(resolve_preset(parent_name, path), resolved)

    if "name" not in resolved:
        resolved["name"] = name
    if overrides:
        resolved = deep_merge_dicts(resolved, overrides)
    return resolved


def evaluate_predictions(predictions: List[dict], ground_truth: List[dict]) -> dict:
    gt_by_id = {row["Question_ID"]: row for row in ground_truth}
    results = []
    per_domain = defaultdict(list)

    for pred in predictions:
        qid = pred["Question_ID"]
        if qid not in gt_by_id:
            continue
        gt = gt_by_id[qid]
        n_pages = max(int(gt.get("N_Pages", 1) or 1), 1)
        answer_correct = 1.0 if pred["Correct_Answer"].strip() == gt["Correct_Answer"].strip() else 0.0
        doc_correct = 1.0 if pred["Doc_ID"].strip() == gt["Doc_ID"].strip() else 0.0
        if doc_correct:
            pred_page = int(pred["Page_Num"])
            true_page = int(gt["Page_Num"])
            page_proximity = max(0.0, 1.0 - abs(pred_page - true_page) / n_pages)
        else:
            page_proximity = 0.0

        score = 0.5 * answer_correct + 0.25 * doc_correct + 0.25 * page_proximity
        row = {
            "Question_ID": qid,
            "Domain": gt.get("Domain", "unknown"),
            "answer_correct": answer_correct,
            "doc_correct": doc_correct,
            "page_proximity": page_proximity,
            "score": score,
        }
        results.append(row)
        per_domain[row["Domain"]].append(row)

    n_questions = len(results)
    if n_questions == 0:
        return {
            "n_questions": 0,
            "answer_accuracy": 0.0,
            "doc_accuracy": 0.0,
            "page_proximity": 0.0,
            "composite_score": 0.0,
            "per_domain": {},
        }

    summary = {
        "n_questions": n_questions,
        "answer_accuracy": sum(row["answer_correct"] for row in results) / n_questions,
        "doc_accuracy": sum(row["doc_correct"] for row in results) / n_questions,
        "page_proximity": sum(row["page_proximity"] for row in results) / n_questions,
        "composite_score": sum(row["score"] for row in results) / n_questions,
        "per_domain": {},
    }
    for domain, rows in sorted(per_domain.items()):
        n_domain = len(rows)
        summary["per_domain"][domain] = {
            "n": n_domain,
            "answer_acc": sum(row["answer_correct"] for row in rows) / n_domain,
            "doc_acc": sum(row["doc_correct"] for row in rows) / n_domain,
            "page_prox": sum(row["page_proximity"] for row in rows) / n_domain,
            "composite": sum(row["score"] for row in rows) / n_domain,
        }
    return summary


def build_per_question_scores(predictions: List[dict], ground_truth: List[dict]) -> List[dict]:
    gt_by_id = {row["Question_ID"]: row for row in ground_truth}
    rows = []
    for pred in predictions:
        qid = pred["Question_ID"]
        if qid not in gt_by_id:
            continue
        gt = gt_by_id[qid]
        n_pages = max(int(gt.get("N_Pages", 1) or 1), 1)
        answer_correct = 1.0 if pred["Correct_Answer"].strip() == gt["Correct_Answer"].strip() else 0.0
        doc_correct = 1.0 if pred["Doc_ID"].strip() == gt["Doc_ID"].strip() else 0.0
        page_proximity = (
            max(0.0, 1.0 - abs(int(pred["Page_Num"]) - int(gt["Page_Num"])) / n_pages) if doc_correct else 0.0
        )
        rows.append(
            {
                "Question_ID": qid,
                "Domain": gt.get("Domain", "unknown"),
                "true_answer": gt["Correct_Answer"],
                "true_doc_id": gt["Doc_ID"],
                "true_page_num": gt["Page_Num"],
                "pred_answer": pred["Correct_Answer"],
                "pred_doc_id": pred["Doc_ID"],
                "pred_page_num": pred["Page_Num"],
                "answer_correct": answer_correct,
                "doc_correct": doc_correct,
                "page_proximity": page_proximity,
                "score": 0.5 * answer_correct + 0.25 * doc_correct + 0.25 * page_proximity,
            }
        )
    return rows


def should_skip_doc_rerank(candidates: List[Tuple[str, int, float]], rerank_cfg: dict) -> bool:
    if not candidates or len(candidates) <= 1:
        return True
    top_doc = candidates[0][0]
    same_doc_topn = sum(1 for doc_id, _, _ in candidates[: rerank_cfg.get("doc_skip_topn", 3)] if doc_id == top_doc)
    best_other_score = next((score for doc_id, _, score in candidates if doc_id != top_doc), None)
    if best_other_score is None:
        return True
    return same_doc_topn >= rerank_cfg.get("doc_skip_topn", 3) and (
        candidates[0][2] - best_other_score
    ) >= rerank_cfg.get("doc_skip_margin", 0.035)


def build_doc_candidates(
    candidates: List[Tuple[str, int, float]],
    page_text_index: Dict[Tuple[str, int], str],
    rerank_cfg: dict,
) -> List[dict]:
    if not candidates:
        return []
    by_doc = defaultdict(list)
    doc_order = []
    for doc_id, page_num, score in candidates:
        if doc_id not in by_doc:
            doc_order.append(doc_id)
        by_doc[doc_id].append((page_num, score))

    doc_candidates = []
    preview_chars = rerank_cfg.get("page_text_chars", 4000) // max(rerank_cfg.get("doc_preview_pages", 2), 1)
    for doc_id in doc_order[: rerank_cfg.get("doc_candidates_max", 6)]:
        pages_for_doc = by_doc[doc_id]
        preview_parts = []
        for page_num, _ in pages_for_doc[: rerank_cfg.get("doc_preview_pages", 2)]:
            text = page_text_index.get((doc_id, page_num), "").strip()
            if text:
                preview_parts.append(f"[Page {page_num}]\n{text[:preview_chars]}")
        doc_candidates.append(
            {
                "doc_id": doc_id,
                "score": float(pages_for_doc[0][1]),
                "preview_text": "\n\n".join(preview_parts) or "[No text]",
            }
        )
    return doc_candidates


def serialize_page_ranking(candidates: List[Tuple[str, int, float]], limit: Optional[int] = None) -> List[dict]:
    if limit is not None:
        candidates = candidates[:limit]
    return [
        {
            "doc_id": doc_id,
            "page_num": int(page_num),
            "score": float(score),
        }
        for doc_id, page_num, score in candidates
    ]


def serialize_doc_ranking(doc_ranking: List[Tuple[str, float]], limit: Optional[int] = None) -> List[dict]:
    if limit is not None:
        doc_ranking = doc_ranking[:limit]
    return [{"doc_id": doc_id, "score": float(score)} for doc_id, score in doc_ranking]


def _find_doc_rank(items: List[dict], true_doc_id: str) -> Optional[int]:
    for index, item in enumerate(items, start=1):
        if item["doc_id"] == true_doc_id:
            return index
    return None


def _find_page_rank(items: List[dict], true_doc_id: str, true_page_num: int) -> Optional[int]:
    for index, item in enumerate(items, start=1):
        if item["doc_id"] == true_doc_id and int(item["page_num"]) == true_page_num:
            return index
    return None


def _page_proximity(pred_page_num: int, true_page_num: int, n_pages: int) -> float:
    return max(0.0, 1.0 - abs(pred_page_num - true_page_num) / max(n_pages, 1))


def _compute_rank_metrics(ranks: List[Optional[int]], ks: Iterable[int]) -> dict:
    metrics = {
        "mrr": sum(0.0 if rank is None else 1.0 / rank for rank in ranks) / max(len(ranks), 1),
        "ndcg": sum(0.0 if rank is None else 1.0 / (1.0 if rank == 1 else __import__("math").log2(rank + 1)) for rank in ranks)
        / max(len(ranks), 1),
        "found_rate": sum(rank is not None for rank in ranks) / max(len(ranks), 1),
        "mean_rank_found": (
            sum(rank for rank in ranks if rank is not None) / max(sum(rank is not None for rank in ranks), 1)
        ),
        "median_rank_found": None,
    }
    found_ranks = sorted(rank for rank in ranks if rank is not None)
    if found_ranks:
        mid = len(found_ranks) // 2
        if len(found_ranks) % 2:
            metrics["median_rank_found"] = float(found_ranks[mid])
        else:
            metrics["median_rank_found"] = 0.5 * (found_ranks[mid - 1] + found_ranks[mid])
    for k in ks:
        metrics[f"recall_at_{k}"] = sum(rank is not None and rank <= k for rank in ranks) / max(len(ranks), 1)
    return metrics


def compute_ir_metrics(ranking_rows: List[dict], ks: Tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    stage_to_field = {
        "dense_doc": "dense_docs",
        "final_doc": "final_docs",
        "dense_page": "dense_pages",
        "final_page": "final_pages",
    }
    per_domain_rows = defaultdict(list)
    for row in ranking_rows:
        per_domain_rows[row.get("Domain", "unknown")].append(row)

    def summarize(rows: List[dict]) -> dict:
        summary = {"n_questions": len(rows), "stages": {}}
        for stage_name, field_name in stage_to_field.items():
            if stage_name.endswith("_doc"):
                ranks = [_find_doc_rank(row.get(field_name, []), row["true_doc_id"]) for row in rows]
                summary["stages"][stage_name] = _compute_rank_metrics(ranks, ks)
            else:
                ranks = [
                    _find_page_rank(row.get(field_name, []), row["true_doc_id"], int(row["true_page_num"]))
                    for row in rows
                ]
                stage_metrics = _compute_rank_metrics(ranks, ks)
                for k in ks:
                    prox_values = []
                    for row in rows:
                        best = 0.0
                        for item in row.get(field_name, [])[:k]:
                            if item["doc_id"] != row["true_doc_id"]:
                                continue
                            best = max(
                                best,
                                _page_proximity(int(item["page_num"]), int(row["true_page_num"]), int(row["n_pages"])),
                            )
                        prox_values.append(best)
                    stage_metrics[f"page_proximity_at_{k}"] = sum(prox_values) / max(len(prox_values), 1)
                summary["stages"][stage_name] = stage_metrics
        return summary

    overall = summarize(ranking_rows)
    overall["per_domain"] = {domain: summarize(rows) for domain, rows in sorted(per_domain_rows.items())}
    return overall


def choose_n_passes(
    doc_ranking: List[Tuple[str, float]],
    top_doc: Optional[str],
    runtime_cfg: dict,
    llm_cfg: dict,
    doc_pages_dict: Dict[str, List[int]],
    page_margin: Optional[float] = None,
) -> int:
    def _doc_page_count(doc_id: Optional[str]) -> int:
        if not doc_id:
            return 0
        pages = doc_pages_dict.get(doc_id, [])
        if isinstance(pages, int):
            return pages
        return len(pages)

    if runtime_cfg.get("fast_mode"):
        return 1

    base_passes = runtime_cfg.get("passes", llm_cfg.get("base_passes", 1))
    hard_passes = llm_cfg.get("hard_passes", base_passes)
    if llm_cfg.get("confidence_gate_enabled"):
        easy_passes = llm_cfg.get("easy_passes", base_passes)
        easy_doc_margin = llm_cfg.get("easy_doc_margin", 0.12)
        easy_page_margin = llm_cfg.get("easy_page_margin", 0.05)
        top_score = doc_ranking[0][1] if doc_ranking else None
        second_score = doc_ranking[1][1] if len(doc_ranking) > 1 else None
        is_long_doc = _doc_page_count(top_doc) >= llm_cfg.get("long_doc_hard_pages", 80)
        if (
            easy_passes < base_passes
            and top_score is not None
            and second_score is not None
            and (top_score - second_score) >= easy_doc_margin
            and (page_margin is not None and page_margin >= easy_page_margin)
            and not is_long_doc
        ):
            return easy_passes
    if base_passes == hard_passes:
        return base_passes
    if len(doc_ranking) < 2:
        if _doc_page_count(top_doc) >= llm_cfg.get("long_doc_hard_pages", 80):
            return hard_passes
        return base_passes

    top_score = doc_ranking[0][1]
    second_score = doc_ranking[1][1]
    if (top_score - second_score) < llm_cfg.get("vote_margin", 0.08):
        return hard_passes
    if _doc_page_count(top_doc) >= llm_cfg.get("long_doc_hard_pages", 80):
        return hard_passes
    return base_passes


def maybe_enable_fast_mode(
    runtime_cfg: dict,
    runtime_limits: dict,
    done: int,
    total: int,
    elapsed_total_seconds: float,
    elapsed_mcq_seconds: float,
) -> bool:
    if runtime_cfg.get("fast_mode"):
        return False
    if done < runtime_limits.get("fast_mode_warmup_q", 25):
        return False

    remaining_q = max(total - done, 0)
    projected_total = elapsed_total_seconds + (elapsed_mcq_seconds / max(done, 1)) * remaining_q
    budget_seconds = runtime_limits.get("time_budget_hours", 8.5) * 3600
    runtime_cfg["projected_total_seconds"] = projected_total
    if projected_total <= budget_seconds:
        return False

    runtime_cfg["fast_mode"] = True
    runtime_cfg["passes"] = 1
    runtime_cfg["top_k_context"] = min(
        runtime_cfg.get("top_k_context", 4), runtime_limits.get("fast_context_pages_cap", 3)
    )
    runtime_cfg["max_chars_per_page"] = min(
        runtime_cfg.get("max_chars_per_page", 3200), runtime_limits.get("fast_max_chars_per_page_cap", 2500)
    )
    if runtime_limits.get("fast_disable_page_stage2", True):
        runtime_cfg["enable_page_stage2"] = False
    runtime_cfg["fast_mode_triggered_at"] = done
    return True


def write_benchmark_artifacts(
    output_dir: Path | str,
    predictions: List[dict],
    per_question_rows: List[dict],
    summary: dict,
    timings: dict,
    ranking_rows: Optional[List[dict]] = None,
    run_manifest: Optional[dict] = None,
    parity_report: Optional[dict] = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "predictions.csv", predictions)
    write_csv(output_dir / "per_question.csv", per_question_rows)
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "timings.json", timings)
    if ranking_rows is not None:
        write_jsonl(output_dir / "ranking_details.jsonl", ranking_rows)
    if run_manifest is not None:
        write_json(output_dir / "run_manifest.json", run_manifest)
    if parity_report is not None:
        write_json(output_dir / "parity_report.json", parity_report)


def compare_benchmark_dirs(base_dir: Path | str, candidate_dir: Path | str) -> Tuple[dict, List[dict]]:
    base_dir = Path(base_dir)
    candidate_dir = Path(candidate_dir)
    base_rows = {row["Question_ID"]: row for row in load_csv(base_dir / "per_question.csv")}
    candidate_rows = {row["Question_ID"]: row for row in load_csv(candidate_dir / "per_question.csv")}

    diff_rows = []
    counters = Counter()
    delta_score = 0.0
    per_domain_delta = defaultdict(lambda: {"n": 0, "score_delta_sum": 0.0})

    for qid in sorted(set(base_rows) & set(candidate_rows), key=lambda value: int(value)):
        base = base_rows[qid]
        candidate = candidate_rows[qid]
        base_score = float(base["score"])
        candidate_score = float(candidate["score"])
        score_delta = candidate_score - base_score
        delta_score += score_delta
        domain = candidate.get("Domain", base.get("Domain", "unknown"))
        per_domain_delta[domain]["n"] += 1
        per_domain_delta[domain]["score_delta_sum"] += score_delta
        if score_delta > 0:
            counters["improved"] += 1
        elif score_delta < 0:
            counters["regressed"] += 1
        else:
            counters["unchanged"] += 1
        if base["pred_doc_id"] != candidate["pred_doc_id"]:
            counters["doc_changed"] += 1
        if base["pred_page_num"] != candidate["pred_page_num"]:
            counters["page_changed"] += 1
        if base["pred_answer"] != candidate["pred_answer"]:
            counters["answer_changed"] += 1

        diff_rows.append(
            {
                "Question_ID": qid,
                "Domain": candidate.get("Domain", base.get("Domain", "unknown")),
                "base_score": base["score"],
                "candidate_score": candidate["score"],
                "score_delta": f"{score_delta:.6f}",
                "base_answer": base["pred_answer"],
                "candidate_answer": candidate["pred_answer"],
                "base_doc_id": base["pred_doc_id"],
                "candidate_doc_id": candidate["pred_doc_id"],
                "base_page_num": base["pred_page_num"],
                "candidate_page_num": candidate["pred_page_num"],
                "doc_changed": str(base["pred_doc_id"] != candidate["pred_doc_id"]),
                "page_changed": str(base["pred_page_num"] != candidate["pred_page_num"]),
                "answer_changed": str(base["pred_answer"] != candidate["pred_answer"]),
            }
        )

    question_count = len(diff_rows)
    summary = {
        "base_dir": str(base_dir),
        "candidate_dir": str(candidate_dir),
        "n_questions": question_count,
        "avg_score_delta": delta_score / question_count if question_count else 0.0,
        "top_1_doc_flip_rate": counters["doc_changed"] / question_count if question_count else 0.0,
        "page_change_rate": counters["page_changed"] / question_count if question_count else 0.0,
        "answer_change_rate": counters["answer_changed"] / question_count if question_count else 0.0,
        "per_domain_score_delta": {
            domain: values["score_delta_sum"] / values["n"] for domain, values in sorted(per_domain_delta.items())
        },
        "counts": dict(counters),
    }
    return summary, diff_rows


def _subtract_metric_trees(base, candidate):
    if isinstance(base, dict) and isinstance(candidate, dict):
        keys = set(base) | set(candidate)
        return {key: _subtract_metric_trees(base.get(key), candidate.get(key)) for key in sorted(keys)}
    if isinstance(base, (int, float)) and isinstance(candidate, (int, float)):
        return candidate - base
    return None


def compare_ir_benchmark_dirs(base_dir: Path | str, candidate_dir: Path | str) -> Tuple[dict, List[dict]]:
    base_dir = Path(base_dir)
    candidate_dir = Path(candidate_dir)
    base_rankings = {row["Question_ID"]: row for row in load_jsonl(base_dir / "ranking_details.jsonl")}
    candidate_rankings = {row["Question_ID"]: row for row in load_jsonl(candidate_dir / "ranking_details.jsonl")}
    base_scores = {row["Question_ID"]: row for row in load_csv(base_dir / "per_question.csv")}
    candidate_scores = {row["Question_ID"]: row for row in load_csv(candidate_dir / "per_question.csv")}

    def _rank_or_zero(rank: Optional[int]) -> int:
        return rank if rank is not None else 0

    question_rows = []
    swing_counts = Counter()
    common_qids = sorted(set(base_rankings) & set(candidate_rankings), key=lambda value: int(value))
    for qid in common_qids:
        base = base_rankings[qid]
        candidate = candidate_rankings[qid]
        base_dense_doc_rank = _find_doc_rank(base.get("dense_docs", []), base["true_doc_id"])
        candidate_dense_doc_rank = _find_doc_rank(candidate.get("dense_docs", []), candidate["true_doc_id"])
        base_final_doc_rank = _find_doc_rank(base.get("final_docs", []), base["true_doc_id"])
        candidate_final_doc_rank = _find_doc_rank(candidate.get("final_docs", []), candidate["true_doc_id"])
        base_dense_page_rank = _find_page_rank(base.get("dense_pages", []), base["true_doc_id"], int(base["true_page_num"]))
        candidate_dense_page_rank = _find_page_rank(
            candidate.get("dense_pages", []), candidate["true_doc_id"], int(candidate["true_page_num"])
        )
        base_final_page_rank = _find_page_rank(base.get("final_pages", []), base["true_doc_id"], int(base["true_page_num"]))
        candidate_final_page_rank = _find_page_rank(
            candidate.get("final_pages", []), candidate["true_doc_id"], int(candidate["true_page_num"])
        )

        if (base_final_doc_rank == 1) != (candidate_final_doc_rank == 1):
            swing_counts["final_doc_top1_fixed" if candidate_final_doc_rank == 1 else "final_doc_top1_broken"] += 1
        if (base_final_page_rank == 1) != (candidate_final_page_rank == 1):
            swing_counts["final_page_top1_fixed" if candidate_final_page_rank == 1 else "final_page_top1_broken"] += 1
        if (base_dense_doc_rank == 1) != (candidate_dense_doc_rank == 1):
            swing_counts["dense_doc_top1_fixed" if candidate_dense_doc_rank == 1 else "dense_doc_top1_broken"] += 1
        if (base_dense_page_rank == 1) != (candidate_dense_page_rank == 1):
            swing_counts["dense_page_top1_fixed" if candidate_dense_page_rank == 1 else "dense_page_top1_broken"] += 1

        base_score_row = base_scores.get(qid, {})
        candidate_score_row = candidate_scores.get(qid, {})
        base_page_prox = float(base_score_row.get("page_proximity", 0.0) or 0.0)
        candidate_page_prox = float(candidate_score_row.get("page_proximity", 0.0) or 0.0)
        if base_score_row.get("doc_correct") == "1.0" and candidate_score_row.get("doc_correct") == "1.0":
            if candidate_page_prox > base_page_prox:
                swing_counts["final_page_proximity_improved_same_doc"] += 1
            elif candidate_page_prox < base_page_prox:
                swing_counts["final_page_proximity_regressed_same_doc"] += 1

        question_rows.append(
            {
                "Question_ID": qid,
                "Domain": candidate.get("Domain", base.get("Domain", "unknown")),
                "base_dense_doc_rank": str(_rank_or_zero(base_dense_doc_rank)),
                "candidate_dense_doc_rank": str(_rank_or_zero(candidate_dense_doc_rank)),
                "base_final_doc_rank": str(_rank_or_zero(base_final_doc_rank)),
                "candidate_final_doc_rank": str(_rank_or_zero(candidate_final_doc_rank)),
                "base_dense_page_rank": str(_rank_or_zero(base_dense_page_rank)),
                "candidate_dense_page_rank": str(_rank_or_zero(candidate_dense_page_rank)),
                "base_final_page_rank": str(_rank_or_zero(base_final_page_rank)),
                "candidate_final_page_rank": str(_rank_or_zero(candidate_final_page_rank)),
                "base_score": base_score_row.get("score", ""),
                "candidate_score": candidate_score_row.get("score", ""),
                "score_delta": (
                    f"{float(candidate_score_row.get('score', 0.0)) - float(base_score_row.get('score', 0.0)):.6f}"
                    if base_score_row and candidate_score_row
                    else ""
                ),
                "base_doc_correct": base_score_row.get("doc_correct", ""),
                "candidate_doc_correct": candidate_score_row.get("doc_correct", ""),
                "base_page_proximity": base_score_row.get("page_proximity", ""),
                "candidate_page_proximity": candidate_score_row.get("page_proximity", ""),
                "base_answer_correct": base_score_row.get("answer_correct", ""),
                "candidate_answer_correct": candidate_score_row.get("answer_correct", ""),
            }
        )

    base_ir = compute_ir_metrics(list(base_rankings.values()))
    candidate_ir = compute_ir_metrics(list(candidate_rankings.values()))
    summary = {
        "base_dir": str(base_dir),
        "candidate_dir": str(candidate_dir),
        "n_questions": len(common_qids),
        "base_ir": base_ir,
        "candidate_ir": candidate_ir,
        "ir_delta": _subtract_metric_trees(base_ir, candidate_ir),
        "swing_counts": dict(swing_counts),
    }
    return summary, question_rows


def build_runtime_paths(env: str = "local") -> dict:
    if env == "kaggle":
        return {
            "mamaylm_dir": first_existing_path(
                [
                    "/kaggle/input/mamaylm-gemma3-12b-gguf",
                    "/kaggle/input/datasets/taleeftamsal/mamaylm-gemma3-12b-gguf",
                ]
            ),
            "bge_m3_dir": first_existing_path(
                [
                    "/kaggle/input/bge-m3",
                    "/kaggle/input/datasets/taleeftamsal/bge-m3",
                ]
            ),
            "bge_reranker_dir": first_existing_path(
                [
                    "/kaggle/input/bge-reranker-v2-m3",
                    "/kaggle/input/datasets/taleeftamsal/bge-reranker-v2-m3",
                ]
            ),
            "qwen3_0_6b_dir": first_existing_path(
                [
                    "/kaggle/input/qwen3-reranker-0-6b",
                    "/kaggle/input/datasets/taleeftamsal/qwen3-reranker-0-6b",
                ]
            ),
            "qwen3_4b_dir": first_existing_path(
                [
                    "/kaggle/input/qwen3-reranker-4b",
                    "/kaggle/input/datasets/taleeftamsal/qwen3-reranker-4b",
                ]
            ),
            "qwen3_8b_dir": first_existing_path(
                [
                    "/kaggle/input/qwen3-reranker-8b",
                    "/kaggle/input/datasets/taleeftamsal/qwen3-reranker-8b",
                ]
            ),
            "wheels_dir": first_existing_path(
                [
                    "/kaggle/input/unlp2026-wheels",
                    "/kaggle/input/datasets/taleeftamsal/unlp2026-wheels",
                ]
            ),
            "pdf_dir": Path(
                "/kaggle/input/competitions/unlp-2026-shared-task-on-multi-domain-document-understanding/test"
            ),
            "questions_path": Path(
                "/kaggle/input/competitions/unlp-2026-shared-task-on-multi-domain-document-understanding/test.csv"
            ),
            "submission_path": Path("/kaggle/working/submission.csv"),
        }
    return {
        "mamaylm_dir": REPO_ROOT / "models" / "mamaylm",
        "lapalm_dir": REPO_ROOT / "models" / "lapalm",
        "bge_m3_dir": REPO_ROOT / "kaggle_datasets" / "bge-m3",
        "bge_reranker_dir": REPO_ROOT / "kaggle_datasets" / "bge-reranker-v2-m3",
        "qwen3_0_6b_dir": REPO_ROOT / "kaggle_datasets" / "qwen3-reranker-0-6b",
        "qwen3_4b_dir": REPO_ROOT / "kaggle_datasets" / "qwen3-reranker-4b",
        "qwen3_8b_dir": REPO_ROOT / "kaggle_datasets" / "qwen3-reranker-8b",
        "wheels_dir": REPO_ROOT / "kaggle_datasets" / "unlp2026-wheels",
        "pdf_dir": REPO_ROOT / "data" / "raw_pdfs",
        "questions_path": REPO_ROOT / "data" / "dev_questions.csv",
        "submission_path": REPO_ROOT / "outputs" / "submission.csv",
    }


def extract_answer(text: str) -> str:
    text = text.strip()
    for letter in ANSWER_CHOICES:
        if text.upper().startswith(letter):
            return letter
    for cyrillic, latin in UA_TO_LATIN.items():
        if text.startswith(cyrillic):
            return latin
    match = re.search(r"\b([A-F])\b", text.upper())
    if match:
        return match.group(1)
    for char in text.upper():
        if char in "ABCDEF":
            return char
    return "A"


def build_prompt(question_row: dict, context: str, prompt_variant: str) -> str:
    options = "\n".join(f"{choice}. {question_row[choice]}" for choice in ANSWER_CHOICES)
    if prompt_variant == "direct":
        return (
            f"{context}\n\n"
            f"Питання: {question_row['Question']}\n\n"
            f"Варіанти:\n{options}\n\n"
            f"Відповідь (лише буква):"
        )
    if prompt_variant == "v5_refocus":
        return (
            f"Документ:\n{context}\n\n"
            f"Питання: {question_row['Question']}\n\n"
            f"Варіанти:\n{options}\n\n"
            f"Питання: {question_row['Question']}\n"
            f"Правильний варіант:"
        )
    return (
        f"Документ:\n{context}\n\n"
        f"Питання: {question_row['Question']}\n\n"
        f"Варіанти:\n{options}\n\n"
        f"Знайди відповідний фрагмент тексту, потім вибери правильний варіант.\n"
        f"Правильний варіант:"
    )


class PipelineRunner:
    def __init__(
        self,
        preset: dict,
        questions_path: Optional[Path | str] = None,
        output_dir: Optional[Path | str] = None,
        env: str = "local",
        n_questions: int = 0,
        runtime_budget_hours: Optional[float] = None,
        run_metadata: Optional[dict] = None,
    ) -> None:
        self.preset = deepcopy(preset)
        self.env = env
        self.paths = build_runtime_paths(env)
        self.questions_path = Path(questions_path) if questions_path else self.paths["questions_path"]
        self.output_dir = Path(output_dir) if output_dir else None
        self.n_questions = n_questions
        self.run_metadata = deepcopy(run_metadata or {})
        self.page_text_index: Dict[Tuple[str, int], str] = {}
        self.doc_meta: Dict[str, dict] = {}
        self.questions: List[dict] = []
        self.pages: List[dict] = []
        self.page_embs_dict = {}
        self.doc_pages_dict = defaultdict(list)
        self.bge_results = {}
        self.dense_results = {}
        self.sparse_results = {}
        self.reranked_results = {}
        self.doc_ranked_results = {}
        self.doc_rerank_skip_by_qid = {}
        self.doc_rerank_changed_by_qid = {}
        self.doc_guard_changed_by_qid = {}
        self.segment_doc_cache = {}
        self.segment_build_stats = {"docs": 0, "pages": 0, "segments": 0}
        self.structure_chunk_cache = {}
        self.structure_chunk_stats = {"docs": 0, "pages": 0, "chunks": 0}
        self.page_summary_text_index = {}
        self.page_summary_embs_dict = {}
        self.timings = {"stages": {}}
        self.component_manifest = {
            "questions_path": str(self.questions_path),
            "mamaylm_dir": str(self.paths["mamaylm_dir"]),
            "bge_m3_dir": str(self.paths["bge_m3_dir"]),
            "bge_reranker_dir": str(self.paths["bge_reranker_dir"]),
            "qwen3_0_6b_dir": str(self.paths["qwen3_0_6b_dir"]),
            "qwen3_4b_dir": str(self.paths["qwen3_4b_dir"]),
            "qwen3_8b_dir": str(self.paths["qwen3_8b_dir"]),
            "wheels_dir": str(self.paths["wheels_dir"]),
        }
        self.extraction_manifest = {}
        self.fallback_events = []
        self.run_manifest = {}
        self.runtime_diagnostics = {
            "doc_rerank_skipped": 0,
            "doc_rerank_changed": 0,
            "doc_guard_changed": 0,
            "easy_pass_count": 0,
            "hard_pass_count": 0,
            "page_stage2_count": 0,
            "fast_mode_triggered_at": None,
            "projected_total_seconds": None,
            "gpu_memory_snapshots": [],
            "reranker_teardown_performed": False,
        }
        self.runtime_cfg = {}
        self.reranker_name = "none"
        if runtime_budget_hours is not None:
            self.preset["runtime"]["time_budget_hours"] = runtime_budget_hours

    def _contracts(self) -> dict:
        return self.preset.get("contracts", {})

    def _record_fallback(self, component: str, detail: str) -> None:
        self.fallback_events.append({"component": component, "detail": detail})

    def _maybe_fail_on_fallback(self, component: str, detail: str) -> None:
        self._record_fallback(component, detail)
        if self._contracts().get("fail_on_component_fallback"):
            raise RuntimeError(f"Component fallback blocked by preset contract: {component} ({detail})")

    def log(self, message: str) -> None:
        print(f"[{self.preset['name']}] {message}", flush=True)

    def _torch(self):
        import torch

        return torch

    def _setup_device(self) -> None:
        torch = self._torch()
        self.use_gpu = False
        if torch.cuda.is_available():
            try:
                _ = torch.zeros(1, device="cuda")
                self.use_gpu = True
            except Exception:
                self.use_gpu = False
        self.device = "cuda:0" if self.use_gpu else "cpu"
        self.dtype = torch.float16 if self.use_gpu else torch.float32

    def _should_record_gpu_memory(self) -> bool:
        return bool(self.preset.get("runtime", {}).get("record_gpu_memory_snapshots"))

    def _record_gpu_memory_snapshot(self, label: str) -> None:
        if not self._should_record_gpu_memory():
            return

        snapshot = {
            "label": label,
            "device": self.device,
            "use_gpu": bool(getattr(self, "use_gpu", False)),
        }
        if getattr(self, "use_gpu", False):
            try:
                torch = self._torch()
                torch.cuda.synchronize()
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                snapshot.update(
                    {
                        "free_bytes": int(free_bytes),
                        "total_bytes": int(total_bytes),
                        "allocated_bytes": int(torch.cuda.memory_allocated()),
                        "reserved_bytes": int(torch.cuda.memory_reserved()),
                    }
                )
            except Exception as exc:
                snapshot["error"] = f"{type(exc).__name__}: {exc}"
        self.runtime_diagnostics["gpu_memory_snapshots"].append(snapshot)

    def teardown_reranker(self) -> None:
        if self.reranker_model is None and self.reranker_tokenizer is None:
            return

        self.log("Tearing down reranker before LLM load")
        self._record_gpu_memory_snapshot("before_reranker_teardown")
        try:
            del self.reranker_model
        except Exception:
            pass
        try:
            del self.reranker_tokenizer
        except Exception:
            pass
        self.reranker_model = None
        self.reranker_tokenizer = None
        self.reranker_yes_token = None
        self.reranker_no_token = None
        self.has_reranker = False
        self.reranker_kind = None
        gc.collect()
        if getattr(self, "use_gpu", False):
            self._torch().cuda.empty_cache()
        self.runtime_diagnostics["reranker_teardown_performed"] = True
        self._record_gpu_memory_snapshot("after_reranker_teardown")

    def _find_gguf(self) -> str:
        llm_model = self.preset.get("llm_model", "mamaylm")
        if llm_model == "lapalm":
            search_dir = self.paths.get("lapalm_dir")
        else:
            search_dir = self.paths.get("mamaylm_dir")
        gguf_files = list(search_dir.glob("*.gguf")) if search_dir and search_dir.exists() else []
        required_name = self._contracts().get("require_gguf_filename")
        if required_name:
            for gguf_path in gguf_files:
                if gguf_path.name == required_name:
                    return str(gguf_path)
            raise FileNotFoundError(
                f"Required GGUF {required_name} was not found under {search_dir}"
            )
        if gguf_files:
            return str(gguf_files[0])
        raise FileNotFoundError(f"No GGUF found under {search_dir}")

    def load_questions(self) -> None:
        self.questions = load_csv(self.questions_path)
        if self.n_questions:
            self.questions = self.questions[: self.n_questions]
        self.log(f"Loaded {len(self.questions)} questions from {self.questions_path}")

    def extract_all_pages(self) -> None:
        try:
            importlib.import_module("fitz")
        except ModuleNotFoundError:
            if self.paths.get("wheels_dir"):
                self._record_fallback("fitz", "installed_from_local_wheel")
        fitz = ensure_fitz_available(self.paths.get("wheels_dir"))

        pages = []
        doc_meta = {}
        extraction_cfg = self.preset.get("extraction", {})
        bleed_chars = max(int(extraction_cfg.get("neighbor_bleed_chars", 200) or 0), 0)
        self.extraction_manifest = {
            "mode": extraction_cfg.get("mode", "raw"),
            "neighbor_bleed_chars": bleed_chars,
            "table_aware": bool(extraction_cfg.get("table_aware", False)),
        }
        self.log(f"Extracting text from PDFs under {self.paths['pdf_dir']}")
        for pdf_path in sorted(self.paths["pdf_dir"].rglob("*.pdf")):
            doc_id = pdf_path.name
            domain = pdf_path.parent.name
            doc = fitz.open(str(pdf_path))
            doc_meta[doc_id] = {"n_pages": len(doc), "domain": domain}
            raw_pages = [extract_text_from_page(page, extraction_cfg) for page in doc]
            doc.close()
            for index, text in enumerate(raw_pages):
                prefix = raw_pages[index - 1][-bleed_chars:] if bleed_chars > 0 and index > 0 else ""
                suffix = raw_pages[index + 1][:bleed_chars] if bleed_chars > 0 and index < len(raw_pages) - 1 else ""
                page_num = index + 1
                page_text = (prefix + text + suffix).strip()
                pages.append({"doc_id": doc_id, "page_num": page_num, "text": page_text})
                self.page_text_index[(doc_id, page_num)] = page_text
        self.pages = pages
        self.doc_meta = doc_meta
        self.log(f"Extracted {len(self.pages)} pages across {len(self.doc_meta)} documents")

    def encode_texts(self, texts: List[str], batch_size: int, max_length: int) -> "np.ndarray":
        import numpy as np
        import torch.nn.functional as F

        all_embs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = self.bge_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(self.device)
            with self._torch().no_grad():
                outputs = self.bge_model(**encoded)
                embs = F.normalize(outputs.last_hidden_state[:, 0], dim=-1)
            all_embs.append(embs.cpu().float().numpy())
        return np.vstack(all_embs).astype("float32")

    def load_bge(self) -> None:
        from transformers import AutoModel, AutoTokenizer

        self._setup_device()
        self.log(f"Loading BGE-M3 on {self.device}")
        self.bge_tokenizer = AutoTokenizer.from_pretrained(str(self.paths["bge_m3_dir"]))
        self.bge_model = AutoModel.from_pretrained(
            str(self.paths["bge_m3_dir"]), dtype=self.dtype
        ).to(self.device).eval()
        self.component_manifest["bge_model_name"] = "bge-m3"
        self.component_manifest["bge_model_path"] = str(self.paths["bge_m3_dir"])
        self.log("BGE-M3 loaded")

    def build_sparse_results(self, query_texts: List[str], top_k: int) -> dict:
        retrieval_cfg = self.preset["retrieval"]
        sparse_backend = retrieval_cfg.get("sparse_backend", "bm25")
        if sparse_backend != "bm25":
            raise ValueError(f"Unsupported sparse backend: {sparse_backend}")
        try:
            from rank_bm25 import BM25Okapi
        except ModuleNotFoundError:
            if self._contracts().get("require_bm25_backend"):
                raise RuntimeError("BM25 backend is required by preset contract but rank_bm25 is unavailable")
            self._maybe_fail_on_fallback("bm25", "rank_bm25_unavailable")
            return {}

        tokenized_pages = [tokenize_sparse_text(page["text"]) for page in self.pages]
        bm25 = BM25Okapi(tokenized_pages)
        sparse_results = {}
        for question_row, query_text in zip(self.questions, query_texts):
            scores = bm25.get_scores(tokenize_sparse_text(query_text))
            top_indices = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:top_k]
            sparse_results[question_row["Question_ID"]] = [
                (
                    self.pages[index]["doc_id"],
                    self.pages[index]["page_num"],
                    float(scores[index]),
                )
                for index in top_indices
            ]
        self.component_manifest["sparse_backend"] = sparse_backend
        return sparse_results

    def run_retrieval(self) -> None:
        import numpy as np

        retrieval_cfg = self.preset["retrieval"]
        self.load_bge()
        self.log("Embedding pages and questions for dense retrieval")
        page_texts = [page["text"] for page in self.pages]
        query_texts = [row["Question"] for row in self.questions]

        page_embs = self.encode_texts(
            page_texts,
            batch_size=retrieval_cfg.get("bge_batch_size", 8) if self.use_gpu else 16,
            max_length=retrieval_cfg.get("bge_max_length", 1536) if self.use_gpu else 1024,
        )
        query_embs = self.encode_texts(
            query_texts,
            batch_size=retrieval_cfg.get("query_batch_size", 32),
            max_length=retrieval_cfg.get("query_max_length", 256),
        )
        scores = query_embs @ page_embs.T

        dense_results = {}
        self.page_embs_dict = {
            (self.pages[index]["doc_id"], self.pages[index]["page_num"]): page_embs[index].copy()
            for index in range(len(self.pages))
        }
        for doc_id, page_num in self.page_embs_dict:
            self.doc_pages_dict[doc_id].append(page_num)
        for doc_id in self.doc_pages_dict:
            self.doc_pages_dict[doc_id].sort()

        index_mode = retrieval_cfg.get("index_mode", "page")
        top_k = retrieval_cfg.get("top_k", 20)
        if index_mode == "structure_chunks":
            seed_top_k = max(top_k, retrieval_cfg.get("structure_seed_top_k", 60))
            seed_indices = np.argsort(-scores, axis=1)[:, :seed_top_k]
            for query_index, question_row in enumerate(self.questions):
                query_emb = query_embs[query_index]
                base_candidates = [
                    (
                        self.pages[candidate_idx]["doc_id"],
                        self.pages[candidate_idx]["page_num"],
                        float(scores[query_index, candidate_idx]),
                    )
                    for candidate_idx in seed_indices[query_index]
                ]
                doc_order = []
                seen_docs = set()
                for doc_id, _, _ in base_candidates:
                    if doc_id not in seen_docs:
                        doc_order.append(doc_id)
                        seen_docs.add(doc_id)
                selected_docs = doc_order[: retrieval_cfg.get("structure_doc_topn", 3)]
                page_scores = {(doc_id, page_num): score for doc_id, page_num, score in base_candidates}
                _sc_page_threshold = retrieval_cfg.get("structure_doc_page_threshold", 0)
                _top_doc_pages = self.doc_meta.get(doc_order[0], {}).get("n_pages", 0) if doc_order else 0
                for doc_id in selected_docs:
                    if _sc_page_threshold > 0 and _top_doc_pages < _sc_page_threshold:
                        break  # Top-1 doc is short: skip ALL structure chunks for this question
                    structure_chunks = self.ensure_doc_structure_chunks(doc_id)
                    doc_page_scores = []
                    for key, emb_list in structure_chunks.items():
                        if not emb_list:
                            continue
                        chunk_score = max(float(np.dot(query_emb, emb)) for emb in emb_list)
                        doc_page_scores.append((key[0], key[1], chunk_score))
                    doc_page_scores.sort(key=lambda row: row[2], reverse=True)
                    for doc_page_id, page_num, chunk_score in doc_page_scores[: retrieval_cfg.get("structure_pages_per_doc", 6)]:
                        key = (doc_page_id, page_num)
                        page_scores[key] = max(page_scores.get(key, -2.0), chunk_score)
                ranked_pages = sorted(
                    [(doc_id, page_num, score) for (doc_id, page_num), score in page_scores.items()],
                    key=lambda row: row[2],
                    reverse=True,
                )
                dense_results[question_row["Question_ID"]] = ranked_pages[:top_k]
        else:
            top_indices = np.argsort(-scores, axis=1)[:, :top_k]
            for index, question_row in enumerate(self.questions):
                dense_results[question_row["Question_ID"]] = [
                    (
                        self.pages[candidate_idx]["doc_id"],
                        self.pages[candidate_idx]["page_num"],
                        float(scores[index, candidate_idx]),
                    )
                    for candidate_idx in top_indices[index]
                ]

        self.dense_results = dense_results
        self.sparse_results = {}
        if retrieval_cfg.get("hybrid_enabled"):
            sparse_top_k = max(top_k, int(retrieval_cfg.get("sparse_top_k", top_k)))
            sparse_results = self.build_sparse_results(query_texts, sparse_top_k)
            if sparse_results:
                self.sparse_results = sparse_results
                dense_weight = float(retrieval_cfg.get("hybrid_dense_weight", 1.0))
                sparse_weight = float(retrieval_cfg.get("hybrid_sparse_weight", 1.0))
                fusion_k = int(retrieval_cfg.get("hybrid_rrf_k", 60))
                fused_results = {}
                for question_row in self.questions:
                    qid = question_row["Question_ID"]
                    fused_results[qid] = reciprocal_rank_fusion(
                        [
                            dense_results.get(qid, []),
                            sparse_results.get(qid, []),
                        ],
                        k=fusion_k,
                        weights=[dense_weight, sparse_weight],
                    )[:top_k]
                self.bge_results = fused_results
                self.component_manifest["retrieval_stage"] = "hybrid_rrf"
                self.component_manifest["hybrid_rrf_k"] = fusion_k
                self.component_manifest["hybrid_dense_weight"] = dense_weight
                self.component_manifest["hybrid_sparse_weight"] = sparse_weight
            else:
                self.bge_results = dense_results
                self.component_manifest["retrieval_stage"] = "dense_only_fallback"
        else:
            self.bge_results = dense_results
            self.component_manifest["retrieval_stage"] = "dense_only"
        self.component_manifest["dense_index_mode"] = index_mode

        del page_embs, query_embs, scores
        if index_mode != "structure_chunks":
            del top_indices
        else:
            del seed_indices
        gc.collect()
        if self.use_gpu:
            self._torch().cuda.empty_cache()

        if self.preset["page_selection"].get("segment_mode") == "precompute":
            self.log("Precomputing segment embeddings for all pages")
            self.precompute_segments()
        self.log("Retrieval stage complete")

    def _segment_texts_for_page(self, text: str) -> List[str]:
        page_cfg = self.preset["page_selection"]
        segs = []
        window = page_cfg.get("segment_window", 900)
        stride = page_cfg.get("segment_stride", 512)
        for start in range(0, max(1, len(text) - window + 1), stride):
            segment = text[start : start + window].strip()
            if segment:
                segs.append(segment)
        if not segs:
            segs = [text[:window]]
        return segs

    def _is_heading_like(self, line: str, max_chars: int) -> bool:
        if not line or len(line) > max_chars:
            return False
        alpha_chars = [char for char in line if char.isalpha()]
        if not alpha_chars:
            return False
        uppercase_ratio = sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
        return uppercase_ratio >= 0.6 or line.endswith(":") or line[:1].isdigit()

    def _build_structure_chunks_for_page(self, page_num: int, text: str) -> List[str]:
        retrieval_cfg = self.preset["retrieval"]
        chunk_chars = retrieval_cfg.get("structure_chunk_chars", 700)
        overlap_chars = retrieval_cfg.get("structure_chunk_overlap", 120)
        heading_max_chars = retrieval_cfg.get("structure_heading_max_chars", 120)
        min_chunk_chars = retrieval_cfg.get("structure_chunk_min_chars", 180)

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        chunks = []
        current_lines = []
        current_chars = 0
        current_heading = None

        def flush() -> None:
            nonlocal current_lines, current_chars, current_heading
            if not current_lines:
                return
            body = "\n".join(current_lines).strip()
            if current_heading and current_heading not in body:
                chunk_text = f"[Сторінка {page_num}] {current_heading}\n{body}"
            else:
                chunk_text = f"[Сторінка {page_num}]\n{body}"
            chunks.append(chunk_text[: chunk_chars + 64])
            if overlap_chars > 0 and body:
                tail = body[-overlap_chars:].strip()
                current_lines = [tail] if tail else []
                current_chars = len(tail)
            else:
                current_lines = []
                current_chars = 0
            current_heading = None

        for line in lines:
            is_heading = self._is_heading_like(line, heading_max_chars)
            if is_heading and current_chars >= min_chunk_chars:
                flush()
                current_heading = line
                continue
            current_lines.append(line)
            current_chars += len(line) + 1
            if is_heading and current_heading is None:
                current_heading = line
            if current_chars >= chunk_chars:
                flush()

        flush()
        if chunks:
            return chunks
        return [f"[Сторінка {page_num}]\n{text[:chunk_chars]}"]

    def _build_page_summary(self, page_num: int, text: str) -> str:
        page_cfg = self.preset["page_selection"]
        summary_chars = page_cfg.get("summary_chars", 320)
        heading_max_chars = page_cfg.get("summary_heading_max_chars", 120)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        heading = next((line for line in lines if self._is_heading_like(line, heading_max_chars)), "")
        body_parts = []
        total_chars = 0
        for line in lines[:12]:
            if line == heading:
                continue
            body_parts.append(line)
            total_chars += len(line) + 1
            if total_chars >= summary_chars:
                break
        body = " ".join(body_parts)[:summary_chars].strip()
        if heading and body:
            return f"[Сторінка {page_num}] {heading} {body}"
        if body:
            return f"[Сторінка {page_num}] {body}"
        return f"[Сторінка {page_num}] {text[:summary_chars].strip()}"

    def ensure_doc_structure_chunks(self, doc_id: str) -> dict:
        if doc_id in self.structure_chunk_cache:
            return self.structure_chunk_cache[doc_id]

        page_nums = self.doc_pages_dict.get(doc_id, [])
        per_page = defaultdict(list)
        chunk_texts = []
        chunk_meta = []
        for page_num in page_nums:
            text = self.page_text_index.get((doc_id, page_num), "")
            if not text:
                continue
            for chunk_text in self._build_structure_chunks_for_page(page_num, text):
                chunk_texts.append(chunk_text)
                chunk_meta.append(page_num)
        if not chunk_texts:
            self.structure_chunk_cache[doc_id] = {}
            return self.structure_chunk_cache[doc_id]

        chunk_embs = self.encode_texts(
            chunk_texts,
            batch_size=self.preset["retrieval"].get("structure_bge_batch_size", 32 if self.use_gpu else 8),
            max_length=self.preset["retrieval"].get("structure_max_length", 512),
        )
        for index, page_num in enumerate(chunk_meta):
            per_page[(doc_id, page_num)].append(chunk_embs[index].copy())
        self.structure_chunk_cache[doc_id] = per_page
        self.structure_chunk_stats["docs"] += 1
        self.structure_chunk_stats["pages"] += len(per_page)
        self.structure_chunk_stats["chunks"] += len(chunk_texts)
        del chunk_embs, chunk_texts, chunk_meta
        gc.collect()
        if self.use_gpu:
            self._torch().cuda.empty_cache()
        return per_page

    def ensure_doc_page_summary_embeddings(self, doc_id: str) -> dict:
        page_nums = self.doc_pages_dict.get(doc_id, [])
        missing = [page_num for page_num in page_nums if (doc_id, page_num) not in self.page_summary_embs_dict]
        if missing:
            summaries = []
            keys = []
            for page_num in missing:
                text = self.page_text_index.get((doc_id, page_num), "")
                summary_text = self._build_page_summary(page_num, text)
                self.page_summary_text_index[(doc_id, page_num)] = summary_text
                summaries.append(summary_text)
                keys.append((doc_id, page_num))
            if summaries:
                summary_embs = self.encode_texts(
                    summaries,
                    batch_size=32 if self.use_gpu else 8,
                    max_length=self.preset["page_selection"].get("summary_max_length", 256),
                )
                for index, key in enumerate(keys):
                    self.page_summary_embs_dict[key] = summary_embs[index].copy()
                del summary_embs, summaries, keys
                gc.collect()
                if self.use_gpu:
                    self._torch().cuda.empty_cache()
        return {
            (doc_id, page_num): self.page_summary_embs_dict[(doc_id, page_num)]
            for page_num in page_nums
            if (doc_id, page_num) in self.page_summary_embs_dict
        }

    def precompute_segments(self) -> None:
        per_page = defaultdict(list)
        seg_texts_flat = []
        seg_meta_flat = []
        for page in self.pages:
            for segment in self._segment_texts_for_page(page["text"]):
                seg_texts_flat.append(segment)
                seg_meta_flat.append((page["doc_id"], page["page_num"]))
        if not seg_texts_flat:
            return
        seg_embs_flat = self.encode_texts(
            seg_texts_flat,
            batch_size=32 if self.use_gpu else 8,
            max_length=self.preset["page_selection"].get("segment_max_length", 256),
        )
        for index, (doc_id, page_num) in enumerate(seg_meta_flat):
            per_page[(doc_id, page_num)].append(seg_embs_flat[index].copy())
        for (doc_id, page_num), emb_list in per_page.items():
            if doc_id not in self.segment_doc_cache:
                self.segment_doc_cache[doc_id] = {}
            self.segment_doc_cache[doc_id][(doc_id, page_num)] = emb_list
        self.segment_build_stats["docs"] = len(self.segment_doc_cache)
        self.segment_build_stats["pages"] = len(per_page)
        self.segment_build_stats["segments"] = len(seg_texts_flat)
        del seg_embs_flat, seg_texts_flat, seg_meta_flat
        gc.collect()
        if self.use_gpu:
            self._torch().cuda.empty_cache()

    def ensure_doc_segments(self, doc_id: str) -> dict:
        page_cfg = self.preset["page_selection"]
        segment_mode = page_cfg.get("segment_mode", "disabled")
        if segment_mode == "disabled":
            return {}
        if segment_mode == "precompute":
            return self.segment_doc_cache.get(doc_id, {})
        if doc_id in self.segment_doc_cache:
            return self.segment_doc_cache[doc_id]

        page_nums = self.doc_pages_dict.get(doc_id, [])
        if len(page_nums) <= page_cfg.get("short_doc_max_pages", 12):
            self.segment_doc_cache[doc_id] = {}
            return self.segment_doc_cache[doc_id]

        seg_texts_flat = []
        seg_meta_flat = []
        for page_num in page_nums:
            text = self.page_text_index.get((doc_id, page_num), "")
            if not text:
                continue
            for segment in self._segment_texts_for_page(text):
                seg_texts_flat.append(segment)
                seg_meta_flat.append((doc_id, page_num))

        if not seg_texts_flat:
            self.segment_doc_cache[doc_id] = {}
            return self.segment_doc_cache[doc_id]

        seg_embs_flat = self.encode_texts(
            seg_texts_flat,
            batch_size=32 if self.use_gpu else 8,
            max_length=page_cfg.get("segment_max_length", 256),
        )
        per_page = defaultdict(list)
        for index, (_, page_num) in enumerate(seg_meta_flat):
            per_page[(doc_id, page_num)].append(seg_embs_flat[index].copy())
        self.segment_doc_cache[doc_id] = per_page
        self.segment_build_stats["docs"] += 1
        self.segment_build_stats["pages"] += len(page_nums)
        self.segment_build_stats["segments"] += len(seg_texts_flat)
        del seg_embs_flat, seg_texts_flat, seg_meta_flat
        gc.collect()
        if self.use_gpu:
            self._torch().cuda.empty_cache()
        return per_page

    def load_reranker(self) -> None:
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        rerank_cfg = self.preset["rerank"]
        required_reranker = self._contracts().get("require_reranker")
        model_dirs = {
            "qwen3_0_6b": self.paths["qwen3_0_6b_dir"],
            "qwen3_4b": self.paths["qwen3_4b_dir"],
            "qwen3_8b": self.paths["qwen3_8b_dir"],
            "bge": self.paths["bge_reranker_dir"],
        }
        self.has_reranker = False
        self.reranker_kind = None
        self.reranker_tokenizer = None
        self.reranker_model = None
        self.reranker_yes_token = None
        self.reranker_no_token = None
        self._record_gpu_memory_snapshot("before_reranker_load")

        for model_name in rerank_cfg.get("model_preference", []):
            model_dir = model_dirs.get(model_name)
            if model_dir and (model_dir / "config.json").exists():
                self.log(f"Loading reranker: {model_name}")
                if model_name.startswith("qwen3"):
                    self.reranker_tokenizer = AutoTokenizer.from_pretrained(str(model_dir), padding_side="left")
                    self.reranker_model = AutoModelForCausalLM.from_pretrained(
                        str(model_dir), dtype=self._torch().float16 if self.use_gpu else self._torch().float32
                    ).to(self.device).eval()
                    self.reranker_kind = "qwen3"
                    self.reranker_yes_token = self.reranker_tokenizer.convert_tokens_to_ids("yes")
                    self.reranker_no_token = self.reranker_tokenizer.convert_tokens_to_ids("no")
                else:
                    self.reranker_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
                    self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
                        str(model_dir), dtype=self.dtype
                    ).to(self.device).eval()
                    self.reranker_kind = "bge"
                self.reranker_name = model_name
                self.has_reranker = True
                self.component_manifest["reranker_name"] = model_name
                self.component_manifest["reranker_path"] = str(model_dir)
                self.log(f"Reranker ready: {self.reranker_name}")
                self._record_gpu_memory_snapshot("after_reranker_load")
                break
        if not self.has_reranker:
            if required_reranker:
                raise RuntimeError(f"Required reranker {required_reranker} is unavailable")
            self.log("No reranker available; using dense retrieval order")
        elif required_reranker and self.reranker_name != required_reranker:
            raise RuntimeError(
                f"Preset contract requires reranker {required_reranker}, but loaded {self.reranker_name}"
            )

    def format_qwen_pair(self, query: str, doc_text: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    'Judge whether the Document meets the requirements based on the Query and the '
                    'Instruct provided. Note that the answer can only be "yes" or "no".'
                ),
            },
            {
                "role": "user",
                "content": (
                    "<Instruct>: Given a Ukrainian question about a document, determine if the document passage "
                    f"contains the answer to the question.\n<Query>: {query}\n<Document>: {doc_text[:4000]}"
                ),
            },
        ]
        text = self.reranker_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text + "<think>\n\n</think>\n"

    def rerank_pages(self, question_text: str, candidates: List[Tuple[str, int, float]]) -> List[Tuple[str, int, float]]:
        if not candidates or not self.has_reranker:
            return candidates
        batch_size = self.preset["rerank"].get("batch_size", 4)
        max_length = self.preset["rerank"].get("max_length", 1024)
        if self.reranker_kind == "qwen3":
            pairs = [
                self.format_qwen_pair(question_text, self.page_text_index.get((doc_id, page_num), ""))
                for doc_id, page_num, _ in candidates
            ]
            all_scores = []
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                encoded = self.reranker_tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
                ).to(self.device)
                with self._torch().no_grad():
                    logits = self.reranker_model(**encoded, logits_to_keep=1).logits[:, -1, :]
                    yes_logits = logits[:, self.reranker_yes_token]
                    no_logits = logits[:, self.reranker_no_token]
                    probs = self._torch().softmax(self._torch().stack([no_logits, yes_logits], dim=1), dim=1)[:, 1]
                all_scores.extend(probs.cpu().float().tolist())
        else:
            pairs = [
                [question_text, self.page_text_index.get((doc_id, page_num), "")[: self.preset["rerank"].get("page_text_chars", 4000)]]
                for doc_id, page_num, _ in candidates
            ]
            all_scores = []
            for start in range(0, len(pairs), batch_size):
                encoded = self.reranker_tokenizer(
                    pairs[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                with self._torch().no_grad():
                    scores = self.reranker_model(**encoded).logits.view(-1).float().cpu().tolist()
                all_scores.extend(scores)

        ranked = sorted(zip(candidates, all_scores), key=lambda item: item[1], reverse=True)
        return [(doc_id, page_num, float(score)) for (doc_id, page_num, _), score in ranked]

    def rerank_docs(self, question_text: str, doc_candidates: List[dict]) -> List[Tuple[str, float]]:
        if not doc_candidates:
            return []
        if not self.has_reranker:
            return [(candidate["doc_id"], candidate["score"]) for candidate in doc_candidates]

        batch_size = self.preset["rerank"].get("batch_size", 4)
        max_length = self.preset["rerank"].get("max_length", 1024)
        if self.reranker_kind == "qwen3":
            pairs = [self.format_qwen_pair(question_text, candidate["preview_text"]) for candidate in doc_candidates]
            all_scores = []
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                encoded = self.reranker_tokenizer(
                    batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
                ).to(self.device)
                with self._torch().no_grad():
                    logits = self.reranker_model(**encoded, logits_to_keep=1).logits[:, -1, :]
                    yes_logits = logits[:, self.reranker_yes_token]
                    no_logits = logits[:, self.reranker_no_token]
                    probs = self._torch().softmax(self._torch().stack([no_logits, yes_logits], dim=1), dim=1)[:, 1]
                all_scores.extend(probs.cpu().float().tolist())
        else:
            pairs = [[question_text, candidate["preview_text"]] for candidate in doc_candidates]
            all_scores = []
            for start in range(0, len(pairs), batch_size):
                encoded = self.reranker_tokenizer(
                    pairs[start : start + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                ).to(self.device)
                with self._torch().no_grad():
                    scores = self.reranker_model(**encoded).logits.view(-1).float().cpu().tolist()
                all_scores.extend(scores)

        ranked = sorted(zip(doc_candidates, all_scores), key=lambda item: item[1], reverse=True)
        return [(candidate["doc_id"], float(score)) for candidate, score in ranked]

    def run_reranking(self) -> None:
        self.load_reranker()
        rerank_cfg = self.preset["rerank"]
        mode = rerank_cfg.get("mode", "page")
        self.log(f"Running {mode}-level reranking")
        progress_every = max(1, int(self.preset.get("runtime", {}).get("progress_every", 25)))
        rerank_start = time.time()
        if mode == "page":
            self.reranked_results = {}
            total = len(self.questions)
            for index, question_row in enumerate(self.questions, start=1):
                qid = question_row["Question_ID"]
                candidates = self.bge_results.get(qid, [])[: self.preset["retrieval"].get("top_k", 20)]
                self.reranked_results[qid] = self.rerank_pages(question_row["Question"], candidates)
                if index % progress_every == 0 or index == total:
                    elapsed = time.time() - rerank_start
                    self.log(f"Reranking progress: {index}/{total} questions in {elapsed:.1f}s")
        else:
            self.doc_ranked_results = {}
            total = len(self.questions)
            for index, question_row in enumerate(self.questions, start=1):
                qid = question_row["Question_ID"]
                candidates = self.bge_results.get(qid, [])[: self.preset["retrieval"].get("top_k", 20)]
                doc_candidates = build_doc_candidates(candidates, self.page_text_index, rerank_cfg)
                should_skip = should_skip_doc_rerank(candidates, rerank_cfg) or len(doc_candidates) <= 1
                self.doc_rerank_skip_by_qid[qid] = should_skip
                if should_skip:
                    ranked_docs = [(candidate["doc_id"], candidate["score"]) for candidate in doc_candidates]
                    self.runtime_diagnostics["doc_rerank_skipped"] += 1
                else:
                    ranked_docs = self.rerank_docs(question_row["Question"], doc_candidates)
                    if ranked_docs and doc_candidates and ranked_docs[0][0] != doc_candidates[0]["doc_id"]:
                        self.doc_rerank_changed_by_qid[qid] = True
                        self.runtime_diagnostics["doc_rerank_changed"] += 1
                self.doc_ranked_results[qid] = ranked_docs
                if index % progress_every == 0 or index == total:
                    elapsed = time.time() - rerank_start
                    self.log(f"Reranking progress: {index}/{total} questions in {elapsed:.1f}s")
        self.log("Reranking stage complete")
        self._record_gpu_memory_snapshot("after_reranking")

    def load_llm(self) -> None:
        try:
            importlib.import_module("llama_cpp")
        except ModuleNotFoundError:
            if self.paths.get("wheels_dir"):
                self._record_fallback("llama_cpp", "installed_from_local_wheel")
        llama_cpp = ensure_llama_cpp_available(self.paths.get("wheels_dir"))
        Llama = llama_cpp.Llama

        n_ctx_candidates = self.preset["llm"].get("n_ctx_candidates", [8192, 4096])
        self.llm = None
        gguf_path = self._find_gguf()
        self.component_manifest["gguf_path"] = gguf_path
        if self.preset.get("runtime", {}).get("teardown_reranker_before_llm"):
            self.teardown_reranker()
        self._record_gpu_memory_snapshot("before_llm_load")
        self.log(f"Loading MamayLM GGUF from {gguf_path}")
        for n_ctx in n_ctx_candidates:
            try:
                self.llm = Llama(model_path=gguf_path, n_gpu_layers=-1, n_ctx=n_ctx, verbose=False)
                self.llm_n_ctx = n_ctx
                if n_ctx == 4096:
                    self.preset["llm"]["max_chars_per_page"] = min(self.preset["llm"]["max_chars_per_page"], 1500)
                self.log(f"MamayLM loaded with n_ctx={n_ctx}")
                self._record_gpu_memory_snapshot("after_llm_load")
                break
            except Exception:
                self.llm = None
        if self.llm is None:
            self._record_gpu_memory_snapshot("llm_load_failed")
            raise RuntimeError("Could not load MamayLM GGUF at any configured n_ctx")
        required_n_ctx = self._contracts().get("require_n_ctx")
        if required_n_ctx is not None and int(required_n_ctx) != int(self.llm_n_ctx):
            raise RuntimeError(
                f"Preset contract requires n_ctx={required_n_ctx}, but loaded n_ctx={self.llm_n_ctx}"
            )
        self.component_manifest["llm_n_ctx"] = int(self.llm_n_ctx)
        if self.use_gpu:
            self._torch().cuda.set_device(0)

    def initial_runtime_cfg(self) -> dict:
        llm_cfg = self.preset["llm"]
        runtime_cfg = {
            "fast_mode": False,
            "passes": llm_cfg.get("base_passes", 1),
            "top_k_context": llm_cfg.get("top_k_context", 4),
            "max_chars_per_page": llm_cfg.get("max_chars_per_page", 3200),
            "enable_page_stage2": self.preset["page_selection"].get("stage2_mode", "disabled") != "disabled",
        }
        return runtime_cfg

    def build_doc_ranking_from_pages(
        self,
        ranked_pages: List[Tuple[str, int, float]],
        use_structure_doc_guard: bool = False,
    ) -> List[Tuple[str, float]]:
        retrieval_cfg = self.preset.get("retrieval", {})
        if use_structure_doc_guard and retrieval_cfg.get("structure_doc_guard_enabled"):
            by_doc = defaultdict(list)
            doc_order = []
            for doc_id, _, score in ranked_pages:
                if doc_id not in by_doc:
                    doc_order.append(doc_id)
                by_doc[doc_id].append(float(score))

            top_pages = max(1, retrieval_cfg.get("structure_doc_guard_top_pages", 2))
            secondary_weight = retrieval_cfg.get("structure_doc_guard_secondary_weight", 0.35)
            ranking = []
            for doc_id in doc_order:
                scores = by_doc[doc_id][:top_pages]
                aggregate = scores[0]
                if len(scores) > 1:
                    aggregate += secondary_weight * sum(scores[1:])
                ranking.append((doc_id, float(aggregate)))
            ranking.sort(key=lambda item: item[1], reverse=True)
            return ranking

        ranking = []
        seen = set()
        for doc_id, _, score in ranked_pages:
            if doc_id in seen:
                continue
            seen.add(doc_id)
            ranking.append((doc_id, float(score)))
        return ranking

    def build_context(
        self,
        ordered_pages: List[Tuple[str, int, float]],
        raw_candidates: List[Tuple[str, int, float]],
        doc_ranking: List[Tuple[str, float]],
    ) -> str:
        parts = []
        seen = set()
        target_pages = self.runtime_cfg.get("top_k_context", self.preset["llm"].get("top_k_context", 4))
        if target_pages <= 0:
            return ""
        if self.preset["rerank"].get("mode") == "page":
            iterator = ordered_pages
        else:
            doc_order = [doc_id for doc_id, _ in doc_ranking]
            for doc_id, _, _ in raw_candidates:
                if doc_id not in doc_order:
                    doc_order.append(doc_id)
            iterator = []
            for doc_id in doc_order:
                iterator.extend([candidate for candidate in raw_candidates if candidate[0] == doc_id])
        for doc_id, page_num, _ in iterator:
            key = (doc_id, page_num)
            if key in seen:
                continue
            text = self.page_text_index.get(key, "").strip()
            if text:
                parts.append(f"[Сторінка {page_num}]\n{text[: self.runtime_cfg['max_chars_per_page']]}")
                seen.add(key)
            if len(parts) >= target_pages:
                break
        return "\n\n".join(parts) or "[Контекст недоступний]"

    def score_mcq(self, question_row: dict, context: str, n_passes: int) -> Tuple[str, List[str]]:
        raw_prompt = build_prompt(question_row, context, self.preset["llm"].get("prompt_variant", "v4_evidence"))
        use_chat = self.preset["llm"].get("use_chat_template", False)
        votes = []
        if use_chat:
            system_prompt = self.preset["llm"].get(
                "chat_system_prompt",
                "Відповідай ЛИШЕ однією великою латинською літерою: A, B, C, D, E або F. Без пояснень.",
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": raw_prompt},
            ]
            def _call(temperature: float) -> str:
                resp = self.llm.create_chat_completion(
                    messages=messages,
                    max_tokens=4,
                    temperature=temperature,
                    stop=["\n", ".", " "],
                )
                return extract_answer(resp["choices"][0]["message"]["content"])
        else:
            def _call(temperature: float) -> str:
                resp = self.llm(raw_prompt, max_tokens=3, temperature=temperature, stop=["\n", "."])
                return extract_answer(resp["choices"][0]["text"])
        votes.append(_call(0.0))
        for _ in range(max(0, n_passes - 1)):
            votes.append(_call(self.preset["llm"].get("vote_temp", 0.5)))
        return Counter(votes).most_common(1)[0][0], votes

    def encode_query(self, text: str, max_length: int = 256) -> "np.ndarray":
        import numpy as np
        import torch.nn.functional as F

        encoded = self.bge_tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(self.device)
        with self._torch().no_grad():
            outputs = self.bge_model(**encoded)
            emb = F.normalize(outputs.last_hidden_state[:, 0], dim=-1).cpu().float().numpy()[0]
        return np.asarray(emb, dtype="float32")

    def encode_augmented_query(self, question_text: str, answer_text: str) -> "np.ndarray":
        import numpy as np
        import torch.nn.functional as F

        encoded = self.bge_tokenizer(
            [f"{question_text} {answer_text}"],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)
        with self._torch().no_grad():
            outputs = self.bge_model(**encoded)
            emb = F.normalize(outputs.last_hidden_state[:, 0], dim=-1).cpu().float().numpy()[0]
        return np.asarray(emb, dtype="float32")

    def select_page_within_doc(
        self,
        question_row: dict,
        pred_answer: str,
        ordered_pages: List[Tuple[str, int, float]],
        doc_ranking: List[Tuple[str, float]],
    ) -> Tuple[Tuple[str, int, float], dict]:
        import numpy as np

        page_cfg = self.preset["page_selection"]
        strategy = page_cfg.get("strategy", "same_doc_answer_aware")
        top_doc = doc_ranking[0][0] if doc_ranking else (ordered_pages[0][0] if ordered_pages else None)
        page_diag = {"stage2_used": False, "page_margin": None, "top_doc": top_doc, "selector_shortlist": None}
        if not ordered_pages:
            fallback_doc = next(iter(self.doc_meta.keys()))
            return (fallback_doc, 1, 0.0), page_diag
        if not top_doc:
            return ordered_pages[0], page_diag
        answer_text = question_row.get(pred_answer, "").strip()
        answer_conditioning = page_cfg.get("answer_conditioning", "predicted_answer")
        if not answer_text and answer_conditioning == "predicted_answer":
            return next((candidate for candidate in ordered_pages if candidate[0] == top_doc), ordered_pages[0]), page_diag

        question_emb = self.encode_query(question_row["Question"])
        if answer_conditioning == "disabled":
            q_emb = question_emb
        else:
            q_emb = self.encode_augmented_query(question_row["Question"], answer_text)
        page_diag["answer_conditioning"] = answer_conditioning
        if strategy == "same_doc_answer_aware":
            same_doc = [candidate for candidate in ordered_pages if candidate[0] == top_doc]
            if len(same_doc) <= 1:
                return same_doc[0] if same_doc else ordered_pages[0], page_diag
            best_score = -2.0
            best_candidate = same_doc[0]
            for candidate in same_doc:
                page_emb = self.page_embs_dict.get((candidate[0], candidate[1]))
                if page_emb is None:
                    continue
                score = float(np.dot(q_emb, page_emb))
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            return best_candidate, page_diag

        if strategy == "summary_then_answer_aware":
            all_pages = self.doc_pages_dict.get(top_doc, [])
            if len(all_pages) == 1:
                return (top_doc, all_pages[0], 0.0), page_diag
            summary_embs = self.ensure_doc_page_summary_embeddings(top_doc)
            summary_scores = []
            for page_num in all_pages:
                summary_emb = summary_embs.get((top_doc, page_num))
                if summary_emb is None:
                    continue
                summary_scores.append((top_doc, page_num, float(np.dot(question_emb, summary_emb))))
            if not summary_scores:
                return next((candidate for candidate in ordered_pages if candidate[0] == top_doc), ordered_pages[0]), page_diag
            summary_scores.sort(key=lambda row: row[2], reverse=True)
            shortlist_size = min(page_cfg.get("summary_topk", 3), len(summary_scores))
            page_diag["selector_shortlist"] = shortlist_size
            if shortlist_size > 1:
                page_diag["page_margin"] = summary_scores[0][2] - summary_scores[1][2]
            shortlist = summary_scores[:shortlist_size]
            best_candidate = shortlist[0]
            best_score = -2.0
            for candidate in shortlist:
                page_emb = self.page_embs_dict.get((candidate[0], candidate[1]))
                if page_emb is None:
                    continue
                score = float(np.dot(q_emb, page_emb))
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            return best_candidate, page_diag

        all_pages = self.doc_pages_dict.get(top_doc, [])
        if len(all_pages) == 1:
            return (top_doc, all_pages[0], 0.0), page_diag
        doc_segments = self.ensure_doc_segments(top_doc)
        page_scores = []
        for page_num in all_pages:
            key = (top_doc, page_num)
            page_segments = doc_segments.get(key)
            if page_segments:
                score = max(float(np.dot(q_emb, segment_emb)) for segment_emb in page_segments)
            elif key in self.page_embs_dict:
                score = float(np.dot(q_emb, self.page_embs_dict[key]))
            else:
                score = -2.0
            page_scores.append((top_doc, page_num, score))
        page_scores.sort(key=lambda row: row[2], reverse=True)
        if strategy == "avir_threshold":
            threshold_delta = page_cfg.get("threshold_delta", 0.02)
            shortlist = [row for row in page_scores if (page_scores[0][2] - row[2]) <= threshold_delta]
            shortlist = shortlist[: page_cfg.get("threshold_max_pages", 4)]
            page_diag["selector_shortlist"] = len(shortlist)
            ordered_page_rank = {
                candidate[1]: rank
                for rank, candidate in enumerate(candidate for candidate in ordered_pages if candidate[0] == top_doc)
            }
            shortlist.sort(key=lambda row: (ordered_page_rank.get(row[1], 10**6), -row[2], row[1]))
            if len(shortlist) > 1:
                original_scores = sorted((row[2] for row in shortlist), reverse=True)
                page_diag["page_margin"] = original_scores[0] - original_scores[1]
            return shortlist[0], page_diag

        topk = page_scores[: page_cfg.get("page_select_topk", 3)]
        if len(topk) > 1:
            page_diag["page_margin"] = topk[0][2] - topk[1][2]
        page_diag["selector_shortlist"] = len(topk)
        stage2_mode = page_cfg.get("stage2_mode", "disabled")
        if self.has_reranker and len(topk) > 1:
            if stage2_mode == "always":
                topk = self.rerank_pages(question_row["Question"], topk)
                page_diag["stage2_used"] = True
            elif (
                stage2_mode == "adaptive"
                and self.runtime_cfg.get("enable_page_stage2")
                and (topk[0][2] - topk[1][2]) < page_cfg.get("stage2_margin", 0.03)
            ):
                topk = self.rerank_pages(question_row["Question"], topk)
                page_diag["stage2_used"] = True
        if page_diag["stage2_used"]:
            self.runtime_diagnostics["page_stage2_count"] += 1
        return topk[0], page_diag

    def build_prediction_row(self, question_row: dict, pred_answer: str, pred_doc: str, pred_page: int) -> dict:
        return {
            "Question_ID": question_row["Question_ID"],
            "Correct_Answer": pred_answer,
            "Doc_ID": pred_doc,
            "Page_Num": str(pred_page),
        }

    def run(self) -> dict:
        total_start = time.time()
        self.log("Starting benchmark run")
        self.load_questions()

        stage_start = time.time()
        self.extract_all_pages()
        self.timings["stages"]["extract_seconds"] = time.time() - stage_start

        stage_start = time.time()
        self.run_retrieval()
        self.timings["stages"]["retrieval_seconds"] = time.time() - stage_start

        stage_start = time.time()
        self.run_reranking()
        self.timings["stages"]["rerank_seconds"] = time.time() - stage_start

        stage_start = time.time()
        self.load_llm()
        self.runtime_cfg = self.initial_runtime_cfg()
        predictions = []
        per_question_rows = []
        ranking_rows = []
        progress_every = max(1, int(self.preset.get("runtime", {}).get("progress_every", 25)))
        total_questions = len(self.questions)

        for index, question_row in enumerate(self.questions):
            qid = question_row["Question_ID"]
            dense_candidates = self.dense_results.get(qid, [])
            raw_candidates = self.bge_results.get(qid, [])
            ordered_pages = (
                self.reranked_results.get(qid, raw_candidates)
                if self.preset["rerank"].get("mode") == "page"
                else raw_candidates
            )
            # lock_to_dense_top_doc: re-sort ordered_pages so that dense top-1 doc's pages
            # come first. This prevents reranker-caused doc flips from degrading doc_acc
            # while still using reranker scores for within-doc page selection.
            # lock_to_dense_top_doc_min_pages: only lock when dense top-1 doc has >= N pages
            # (domain_1 proxy: all domain_1 docs have >= 25 pages; domain_2 docs < 25 pages).
            # On all splits, the reranker hurts domain_1 but helps domain_2, so the conditional
            # lock preserves domain_2 reranker benefits while fixing domain_1 doc flips.
            _page_sel_cfg = self.preset["page_selection"]
            _should_lock = should_lock_to_dense_top_doc(
                page_sel_cfg=_page_sel_cfg,
                dense_candidates=dense_candidates,
                doc_meta=self.doc_meta,
            )
            if _should_lock:
                _dense_top_doc = dense_candidates[0][0]
                _other = [(d, p, s) for d, p, s in ordered_pages if d != _dense_top_doc]
                if _page_sel_cfg.get("lock_exhaustive_dense"):
                    # Score ALL pages of the locked doc via BGE-M3 dense similarity,
                    # not just the subset returned by global top-K retrieval.
                    # Fixes cases where the answer page is missed by global retrieval
                    # (e.g., answer on page 75 of a 76-page doc not in top-20 globally).
                    import numpy as _np
                    _q_emb = self.encode_query(question_row["Question"])
                    _all_doc_pages = self.doc_pages_dict.get(_dense_top_doc, [])
                    _same = []
                    for _pg in _all_doc_pages:
                        _pg_emb = self.page_embs_dict.get((_dense_top_doc, _pg))
                        if _pg_emb is not None:
                            _same.append((_dense_top_doc, _pg, float(_np.dot(_q_emb, _pg_emb))))
                    _same.sort(key=lambda x: x[2], reverse=True)
                else:
                    _same = [(d, p, s) for d, p, s in ordered_pages if d == _dense_top_doc]
                ordered_pages = _same + _other
            doc_ranking = (
                self.doc_ranked_results.get(qid, [])
                if self.preset["rerank"].get("mode") == "doc"
                else self.build_doc_ranking_from_pages(
                    ordered_pages,
                    use_structure_doc_guard=(
                        self.preset.get("retrieval", {}).get("index_mode") == "structure_chunks"
                    ),
                )
            )
            if self.preset["runtime"].get("enable_runtime_governor"):
                changed = maybe_enable_fast_mode(
                    runtime_cfg=self.runtime_cfg,
                    runtime_limits=self.preset["runtime"],
                    done=index,
                    total=len(self.questions),
                    elapsed_total_seconds=time.time() - total_start,
                    elapsed_mcq_seconds=time.time() - stage_start,
                )
                if changed:
                    self.runtime_diagnostics["fast_mode_triggered_at"] = self.runtime_cfg.get("fast_mode_triggered_at")
                    self.runtime_diagnostics["projected_total_seconds"] = self.runtime_cfg.get("projected_total_seconds")
            context = self.build_context(ordered_pages, raw_candidates, doc_ranking)
            top_doc = doc_ranking[0][0] if doc_ranking else (ordered_pages[0][0] if ordered_pages else None)
            dense_doc_ranking = self.build_doc_ranking_from_pages(dense_candidates)
            retrieval_doc_ranking = self.build_doc_ranking_from_pages(raw_candidates)
            dense_doc_margin = compute_margin(dense_doc_ranking)
            final_doc_margin = compute_margin(doc_ranking)
            initial_page_margin = None
            if len(ordered_pages) > 1:
                initial_page_margin = ordered_pages[0][2] - ordered_pages[1][2]
            doc_guard_changed = False
            if self.preset.get("retrieval", {}).get("structure_doc_guard_enabled"):
                unguarded_doc_ranking = self.build_doc_ranking_from_pages(ordered_pages, use_structure_doc_guard=False)
                guarded_doc_ranking = self.build_doc_ranking_from_pages(ordered_pages, use_structure_doc_guard=True)
                doc_guard_changed = bool(
                    guarded_doc_ranking
                    and unguarded_doc_ranking
                    and guarded_doc_ranking[0][0] != unguarded_doc_ranking[0][0]
                )
                self.doc_guard_changed_by_qid[qid] = doc_guard_changed
                if doc_guard_changed:
                    self.runtime_diagnostics["doc_guard_changed"] += 1
            n_passes = choose_n_passes(
                doc_ranking,
                top_doc,
                self.runtime_cfg,
                self.preset["llm"],
                self.doc_pages_dict,
                page_margin=initial_page_margin,
            )
            if n_passes < self.preset["llm"].get("base_passes", 1):
                self.runtime_diagnostics["easy_pass_count"] += 1
            if n_passes > self.preset["llm"].get("base_passes", 1):
                self.runtime_diagnostics["hard_pass_count"] += 1
            pred_answer, votes = self.score_mcq(question_row, context, n_passes)
            dual_passes = self.preset["llm"].get("dual_context_passes", 0)
            if dual_passes > 0 and dense_candidates:
                dense_context = self.build_context(dense_candidates, raw_candidates, dense_doc_ranking)
                if dense_context != context:
                    _, alt_votes = self.score_mcq(question_row, dense_context, dual_passes)
                    votes = votes + alt_votes
                    pred_answer = Counter(votes).most_common(1)[0][0]
            vote_confidence = votes.count(pred_answer) / max(len(votes), 1)
            best_candidate, page_diag = self.select_page_within_doc(question_row, pred_answer, ordered_pages, doc_ranking)
            pred_doc, pred_page = best_candidate[0], best_candidate[1]
            predictions.append(self.build_prediction_row(question_row, pred_answer, pred_doc, pred_page))
            per_question_rows.append(
                {
                    "Question_ID": qid,
                    "Domain": question_row.get("Domain", "unknown"),
                    "pred_answer": pred_answer,
                    "pred_doc_id": pred_doc,
                    "pred_page_num": str(pred_page),
                    "top_doc": top_doc or "",
                    "n_passes": str(n_passes),
                    "votes": "|".join(votes),
                    "vote_confidence": f"{vote_confidence:.6f}",
                    "vote_agreement": f"{vote_confidence:.6f}",
                    "fast_mode": str(self.runtime_cfg.get("fast_mode", False)),
                    "doc_rerank_skipped": str(self.doc_rerank_skip_by_qid.get(qid, False)),
                    "doc_rerank_changed": str(self.doc_rerank_changed_by_qid.get(qid, False)),
                    "doc_guard_changed": str(doc_guard_changed),
                    "page_stage2_used": str(page_diag.get("stage2_used", False)),
                    "page_selector": self.preset["page_selection"].get("strategy", "same_doc_answer_aware"),
                    "answer_conditioning": page_diag.get(
                        "answer_conditioning",
                        self.preset["page_selection"].get("answer_conditioning", "predicted_answer"),
                    ),
                    "selector_shortlist": "" if page_diag.get("selector_shortlist") is None else str(page_diag["selector_shortlist"]),
                    "dense_doc_margin": "" if dense_doc_margin is None else f"{dense_doc_margin:.6f}",
                    "final_doc_margin": "" if final_doc_margin is None else f"{final_doc_margin:.6f}",
                    "doc_length_pages": str(len(self.doc_pages_dict.get(top_doc, []))) if top_doc else "",
                    "initial_page_margin": "" if initial_page_margin is None else f"{initial_page_margin:.6f}",
                    "top_page_margin": "" if initial_page_margin is None else f"{initial_page_margin:.6f}",
                    "page_margin": "" if page_diag.get("page_margin") is None else f"{page_diag['page_margin']:.6f}",
                }
            )
            sparse_candidates = self.sparse_results.get(qid, [])
            ranking_rows.append(
                {
                    "Question_ID": qid,
                    "Domain": question_row.get("Domain", "unknown"),
                    "true_doc_id": question_row.get("Doc_ID", ""),
                    "true_page_num": int(question_row.get("Page_Num", 1) or 1),
                    "n_pages": int(question_row.get("N_Pages", 1) or 1),
                    "pred_doc_id": pred_doc,
                    "pred_page_num": int(pred_page),
                    "dense_docs": serialize_doc_ranking(dense_doc_ranking),
                    "dense_pages": serialize_page_ranking(dense_candidates),
                    "retrieval_docs": serialize_doc_ranking(retrieval_doc_ranking),
                    "retrieval_pages": serialize_page_ranking(raw_candidates),
                    "final_docs": serialize_doc_ranking(doc_ranking),
                    "final_pages": serialize_page_ranking(ordered_pages),
                    "sparse_docs": serialize_doc_ranking(self.build_doc_ranking_from_pages(sparse_candidates)),
                    "sparse_pages": serialize_page_ranking(sparse_candidates),
                }
            )
            if (index + 1) % progress_every == 0 or (index + 1) == total_questions:
                elapsed = time.time() - stage_start
                self.log(f"MCQ progress: {index + 1}/{total_questions} questions in {elapsed:.1f}s")

        self.timings["stages"]["mcq_seconds"] = time.time() - stage_start
        self.timings["total_seconds"] = time.time() - total_start
        self.timings["stages"]["questions"] = len(self.questions)
        self.log(f"MCQ stage complete in {self.timings['stages']['mcq_seconds']:.1f}s")
        self.run_manifest = build_run_manifest(
            preset=self.preset,
            questions_path=self.questions_path,
            env=self.env,
            question_ids=[row["Question_ID"] for row in self.questions],
            run_metadata=self.run_metadata,
            component_manifest=self.component_manifest,
            extraction_manifest=self.extraction_manifest,
            fallback_events=self.fallback_events,
            use_gpu=getattr(self, "use_gpu", None),
        )

        summary = {
            "preset": self.preset["name"],
            "description": self.preset.get("description", ""),
            "reference_kaggle_score": self.preset.get("reference_kaggle_score"),
            "n_questions": len(self.questions),
            "reranker_name": self.reranker_name,
            "llm_n_ctx": self.llm_n_ctx,
            "run_info": {
                "config_hash": self.run_manifest["config_hash"],
                "git_commit": self.run_manifest["git_commit"],
                "hardware_tag": self.run_manifest["hardware_tag"],
                "fallback_count": len(self.fallback_events),
            },
            "diagnostics": {
                **self.runtime_diagnostics,
                "doc_rerank_skipped": self.runtime_diagnostics["doc_rerank_skipped"],
                "doc_rerank_changed": self.runtime_diagnostics["doc_rerank_changed"],
                "doc_guard_changed": self.runtime_diagnostics["doc_guard_changed"],
                "easy_pass_count": self.runtime_diagnostics["easy_pass_count"],
                "hard_pass_count": self.runtime_diagnostics["hard_pass_count"],
                "page_stage2_count": self.runtime_diagnostics["page_stage2_count"],
                "lazy_segment_docs": self.segment_build_stats["docs"],
                "lazy_segment_pages": self.segment_build_stats["pages"],
                "lazy_segment_segments": self.segment_build_stats["segments"],
                "structure_chunk_docs": self.structure_chunk_stats["docs"],
                "structure_chunk_pages": self.structure_chunk_stats["pages"],
                "structure_chunk_segments": self.structure_chunk_stats["chunks"],
                "fallback_count": len(self.fallback_events),
            },
            "timings": self.timings["stages"],
        }

        has_ground_truth = bool(self.questions and "Correct_Answer" in self.questions[0] and "Doc_ID" in self.questions[0])
        if has_ground_truth:
            eval_summary = evaluate_predictions(predictions, self.questions)
            summary.update(eval_summary)
            summary["ir_metrics"] = compute_ir_metrics(ranking_rows)
            score_rows = {row["Question_ID"]: row for row in build_per_question_scores(predictions, self.questions)}
            for row in per_question_rows:
                if row["Question_ID"] in score_rows:
                    row.update(score_rows[row["Question_ID"]])

        if self.output_dir:
            write_benchmark_artifacts(
                self.output_dir,
                predictions,
                per_question_rows,
                summary,
                self.timings,
                ranking_rows=ranking_rows,
                run_manifest=self.run_manifest,
            )
            write_csv(self.output_dir / "submission.csv", predictions)
            self.log(f"Artifacts written to {self.output_dir}")

        return {
            "predictions": predictions,
            "per_question_rows": per_question_rows,
            "ranking_rows": ranking_rows,
            "summary": summary,
            "timings": self.timings,
            "run_manifest": self.run_manifest,
        }


def run_pipeline_from_preset(
    preset_name: str,
    questions_path: Optional[Path | str] = None,
    output_dir: Optional[Path | str] = None,
    env: str = "local",
    n_questions: int = 0,
    runtime_budget_hours: Optional[float] = None,
    overrides: Optional[dict] = None,
    run_metadata: Optional[dict] = None,
) -> dict:
    preset = resolve_preset(preset_name, overrides=overrides)
    runner = PipelineRunner(
        preset=preset,
        questions_path=questions_path,
        output_dir=output_dir,
        env=env,
        n_questions=n_questions,
        runtime_budget_hours=runtime_budget_hours,
        run_metadata=run_metadata,
    )
    return runner.run()
