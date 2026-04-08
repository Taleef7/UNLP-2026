#!/usr/bin/env python3
"""Phase 0.5: Create doc-grouped evaluation splits and manifests.

This keeps the legacy `train.csv`/`val.csv` and leave-one-domain-out outputs,
but the primary protocol is now:
- grouped-by-Doc_ID 5-fold cross-validation
- a fixed lockbox held-out document manifest
- fold and lockbox JSON manifests for downstream benchmarking
"""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

BASE = Path("/scratch/gilbreth/tamst01/unlp2026")
DATA = BASE / "data"
SPLITS = DATA / "splits"

SEED = 42
VAL_RATIO = 0.2
N_FOLDS = 5
LOCKBOX_RATIO = 0.2
BUCKET_COUNT = 4


def load_questions(path: Path | str = DATA / "dev_questions.csv"):
    """Load questions from CSV."""
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _question_sort_key(row: dict):
    question_id = row.get("Question_ID", "")
    try:
        return (0, int(question_id))
    except (TypeError, ValueError):
        return (1, str(question_id))


def _doc_sort_key(doc: dict):
    return (-doc["question_count"], -doc["n_pages"], doc["domain"], doc["doc_id"])


def _summarize_docs(docs: Iterable[dict]) -> dict:
    domain_counts = Counter()
    bucket_counts = Counter()
    question_count = 0
    for doc in docs:
        domain_counts[doc["domain"]] += 1
        bucket_counts[doc["length_bucket"]] += 1
        question_count += doc["question_count"]
    return {
        "question_count": question_count,
        "doc_count": sum(domain_counts.values()),
        "domain_counts": dict(sorted(domain_counts.items())),
        "length_bucket_counts": dict(sorted(bucket_counts.items())),
    }


def _group_questions_by_doc(questions):
    by_doc = defaultdict(list)
    for row in questions:
        by_doc[row["Doc_ID"]].append(dict(row))
    return by_doc


def _build_doc_records(questions):
    by_doc = _group_questions_by_doc(questions)
    docs = []
    for doc_id, rows in sorted(by_doc.items()):
        rows.sort(key=_question_sort_key)
        domain = rows[0].get("Domain", "unknown")
        n_pages = max(int(row.get("N_Pages", 1) or 1) for row in rows)
        docs.append(
            {
                "doc_id": doc_id,
                "domain": domain,
                "n_pages": n_pages,
                "question_count": len(rows),
                "questions": rows,
            }
        )

    docs_by_length = sorted(docs, key=lambda doc: (doc["n_pages"], doc["doc_id"]))
    bucket_size = max(1, math.ceil(len(docs_by_length) / BUCKET_COUNT))
    for index, doc in enumerate(docs_by_length):
        bucket_index = min(BUCKET_COUNT - 1, index // bucket_size)
        doc["length_bucket"] = f"q{bucket_index + 1}"
    docs.sort(key=_doc_sort_key)
    return docs


def _partition_targets(docs: list[dict], group_count: int, target_size: float | None = None) -> dict:
    total_questions = sum(doc["question_count"] for doc in docs)
    domain_counts = Counter(doc["domain"] for doc in docs)
    bucket_counts = Counter(doc["length_bucket"] for doc in docs)
    if target_size is None:
        target_size = len(docs) / group_count if group_count else 0
    ratio = (target_size / len(docs)) if docs else 0
    return {
        "doc_count": target_size,
        "question_count": total_questions * ratio,
        "domain_counts": {domain: count * ratio for domain, count in domain_counts.items()},
        "bucket_counts": {bucket: count * ratio for bucket, count in bucket_counts.items()},
    }


def _empty_partition_state():
    return {
        "doc_count": 0,
        "question_count": 0,
        "domain_counts": Counter(),
        "bucket_counts": Counter(),
    }


def _partition_penalty(state: dict, doc: dict, targets: dict) -> float:
    projected_doc_count = state["doc_count"] + 1
    projected_question_count = state["question_count"] + doc["question_count"]
    score = 0.0
    score += ((projected_doc_count - targets["doc_count"]) / max(targets["doc_count"], 1.0)) ** 2 * 4.0
    score += (
        (projected_question_count - targets["question_count"]) / max(targets["question_count"], 1.0)
    ) ** 2 * 3.0

    for domain, target in targets["domain_counts"].items():
        projected = state["domain_counts"][domain] + (1 if doc["domain"] == domain else 0)
        score += ((projected - target) / max(target, 1.0)) ** 2 * 2.0

    for bucket, target in targets["bucket_counts"].items():
        projected = state["bucket_counts"][bucket] + (1 if doc["length_bucket"] == bucket else 0)
        score += ((projected - target) / max(target, 1.0)) ** 2

    return score


def _assign_docs_to_groups(docs: list[dict], group_count: int, seed: int):
    rng = random.Random(seed)
    groups = [[] for _ in range(group_count)]
    states = [_empty_partition_state() for _ in range(group_count)]

    strata = defaultdict(list)
    for doc in docs:
        strata[(doc["domain"], doc["length_bucket"])].append(doc)

    ordered_docs = []
    for key in sorted(strata):
        bucket_docs = list(strata[key])
        rng.shuffle(bucket_docs)
        bucket_docs.sort(key=_doc_sort_key)
        strata[key] = bucket_docs

    while True:
        advanced = False
        for key in sorted(strata):
            if strata[key]:
                ordered_docs.append(strata[key].pop(0))
                advanced = True
        if not advanced:
            break

    for doc in ordered_docs:
        best_index = 0
        best_key = None
        for index in range(group_count):
            key = (
                states[index]["domain_counts"][doc["domain"]],
                states[index]["bucket_counts"][doc["length_bucket"]],
                states[index]["question_count"],
                states[index]["doc_count"],
                index,
            )
            if best_key is None or key < best_key:
                best_key = key
                best_index = index
        groups[best_index].append(doc)
        states[best_index]["doc_count"] += 1
        states[best_index]["question_count"] += doc["question_count"]
        states[best_index]["domain_counts"][doc["domain"]] += 1
        states[best_index]["bucket_counts"][doc["length_bucket"]] += 1

    return groups, states


def _select_balanced_subset(docs: list[dict], subset_size: int, seed: int):
    if subset_size <= 0:
        return [], _empty_partition_state()
    if subset_size >= len(docs):
        state = _empty_partition_state()
        for doc in docs:
            state["doc_count"] += 1
            state["question_count"] += doc["question_count"]
            state["domain_counts"][doc["domain"]] += 1
            state["bucket_counts"][doc["length_bucket"]] += 1
        return list(docs), state

    rng = random.Random(seed)
    targets = _partition_targets(docs, 1, target_size=subset_size)
    selected = []
    selected_state = _empty_partition_state()
    remaining = list(docs)
    decorated = [(rng.random(), doc) for doc in remaining]
    decorated.sort(key=lambda item: (_doc_sort_key(item[1]), item[0]))
    remaining = [doc for _, doc in decorated]

    while len(selected) < subset_size:
        best_doc = None
        best_key = None
        for doc in remaining:
            penalty = _partition_penalty(selected_state, doc, targets)
            key = (penalty, -doc["question_count"], -doc["n_pages"], doc["doc_id"])
            if best_key is None or key < best_key:
                best_key = key
                best_doc = doc
        selected.append(best_doc)
        selected_state["doc_count"] += 1
        selected_state["question_count"] += best_doc["question_count"]
        selected_state["domain_counts"][best_doc["domain"]] += 1
        selected_state["bucket_counts"][best_doc["length_bucket"]] += 1
        remaining = [doc for doc in remaining if doc["doc_id"] != best_doc["doc_id"]]

    return selected, selected_state


def _doc_grouped_split(docs: list[dict], val_ratio: float, seed: int):
    val_size = min(max(1, round(len(docs) * val_ratio)), max(0, len(docs) - 1))
    val_docs, _ = _select_balanced_subset(docs, val_size, seed)
    val_doc_ids = {doc["doc_id"] for doc in val_docs}
    train_docs = [doc for doc in docs if doc["doc_id"] not in val_doc_ids]
    return train_docs, val_docs


def _docs_to_questions(docs: list[dict]) -> list[dict]:
    rows = []
    for doc in docs:
        rows.extend(doc["questions"])
    rows.sort(key=_question_sort_key)
    return rows


def stratified_split(questions, val_ratio=VAL_RATIO, seed=SEED):
    """Backwards-compatible train/val split built from grouped documents."""
    docs = _build_doc_records(questions)
    train_docs, val_docs = _doc_grouped_split(docs, val_ratio, seed)
    return _docs_to_questions(train_docs), _docs_to_questions(val_docs)


def leave_one_domain_out(questions):
    """Create LODO splits for each domain."""
    by_domain = defaultdict(list)
    for q in questions:
        by_domain[q["Domain"]].append(q)

    splits = {}
    domains = sorted(by_domain.keys())
    for held_out in domains:
        train = []
        for d in domains:
            if d != held_out:
                train.extend(by_domain[d])
        splits[held_out] = {
            "train": train,
            "val": by_domain[held_out],
        }
    return splits


def save_split(questions, path):
    """Save a list of question dicts to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not questions:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(questions[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(questions)


def _summarize_docs(docs: list[dict]) -> dict:
    return {
        "doc_count": len(docs),
        "question_count": sum(doc["question_count"] for doc in docs),
        "doc_ids": [doc["doc_id"] for doc in docs],
        "domain_counts": dict(Counter(doc["domain"] for doc in docs)),
        "length_bucket_counts": dict(Counter(doc["length_bucket"] for doc in docs)),
    }


def build_split_protocol(questions, seed=SEED, n_folds=N_FOLDS, lockbox_ratio=LOCKBOX_RATIO):
    """Build grouped-document CV folds and a fixed lockbox subset."""
    docs = _build_doc_records(questions)
    total_docs = len(docs)
    lockbox_count = min(max(1, round(total_docs * lockbox_ratio)), max(0, total_docs - n_folds))
    lockbox_docs, lockbox_state = _select_balanced_subset(docs, lockbox_count, seed)
    lockbox_doc_ids = {doc["doc_id"] for doc in lockbox_docs}
    remaining_docs = [doc for doc in docs if doc["doc_id"] not in lockbox_doc_ids]

    fold_groups, fold_states = _assign_docs_to_groups(remaining_docs, n_folds, seed)
    folds = []
    for fold_index, val_docs in enumerate(fold_groups):
        train_docs = [doc for doc in remaining_docs if doc["doc_id"] not in {d["doc_id"] for d in val_docs}]
        train_rows = _docs_to_questions(train_docs)
        val_rows = _docs_to_questions(val_docs)
        folds.append(
            {
                "fold_index": fold_index,
                "docs": val_docs,
                "train_docs": train_docs,
                "val_docs": val_docs,
                "train": train_rows,
                "val": val_rows,
                "train_by_doc": {doc["doc_id"]: doc["questions"] for doc in train_docs},
                "val_by_doc": {doc["doc_id"]: doc["questions"] for doc in val_docs},
                "summary": {
                    "train": _summarize_docs(train_docs),
                    "val": _summarize_docs(val_docs),
                },
            }
        )

    legacy_train_docs, legacy_val_docs = _doc_grouped_split(remaining_docs, VAL_RATIO, seed)
    legacy_train = _docs_to_questions(legacy_train_docs)
    legacy_val = _docs_to_questions(legacy_val_docs)
    lodo_splits = leave_one_domain_out(questions)

    return {
        "seed": seed,
        "n_folds": n_folds,
        "lockbox_ratio": lockbox_ratio,
        "bucket_count": BUCKET_COUNT,
        "docs": docs,
        "lockbox": {
            "docs": lockbox_docs,
            "questions": _docs_to_questions(lockbox_docs),
            "state": {
                "doc_count": lockbox_state["doc_count"],
                "question_count": lockbox_state["question_count"],
                "domain_counts": dict(sorted(lockbox_state["domain_counts"].items())),
                "bucket_counts": dict(sorted(lockbox_state["bucket_counts"].items())),
            },
        },
        "folds": folds,
        "legacy": {
            "train": legacy_train,
            "val": legacy_val,
            "train_docs": legacy_train_docs,
            "val_docs": legacy_val_docs,
            "train_summary": _summarize_docs(legacy_train_docs),
            "val_summary": _summarize_docs(legacy_val_docs),
        },
        "lodo": lodo_splits,
    }


def _protocol_manifest(protocol: dict) -> dict:
    folds = []
    for fold in protocol["folds"]:
        folds.append(
            {
                "fold_index": fold["fold_index"],
                "train_doc_ids": [doc["doc_id"] for doc in fold["train_docs"]],
                "val_doc_ids": [doc["doc_id"] for doc in fold["val_docs"]],
                "train_summary": fold["summary"]["train"],
                "val_summary": fold["summary"]["val"],
            }
        )

    lockbox_docs = protocol["lockbox"]["docs"]
    return {
        "seed": protocol["seed"],
        "n_folds": protocol["n_folds"],
        "lockbox_ratio": protocol["lockbox_ratio"],
        "bucket_count": protocol["bucket_count"],
        "doc_count": len(protocol["docs"]),
        "question_count": sum(doc["question_count"] for doc in protocol["docs"]),
        "folds": folds,
        "lockbox": {
            "doc_ids": [doc["doc_id"] for doc in lockbox_docs],
            "summary": protocol["lockbox"]["state"],
        },
        "legacy": {
            "train": protocol["legacy"]["train_summary"],
            "val": protocol["legacy"]["val_summary"],
        },
        "lodo": {
            held_out: {"train": len(split["train"]), "val": len(split["val"])}
            for held_out, split in protocol["lodo"].items()
        },
    }


def write_split_protocol(protocol: dict, output_dir: Path | str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_split(protocol["legacy"]["train"], output_dir / "train.csv")
    save_split(protocol["legacy"]["val"], output_dir / "val.csv")
    save_split(protocol["lockbox"]["questions"], output_dir / "lockbox.csv")

    for fold in protocol["folds"]:
        fold_index = fold["fold_index"]
        save_split(fold["train"], output_dir / f"fold_{fold_index}_train.csv")
        save_split(fold["val"], output_dir / f"fold_{fold_index}_val.csv")

    lodo_splits = protocol["lodo"]
    for held_out, split in lodo_splits.items():
        save_split(split["train"], output_dir / f"lodo_train_no_{held_out}.csv")
        save_split(split["val"], output_dir / f"lodo_val_{held_out}.csv")

    manifest = _protocol_manifest(protocol)
    split_info = {
        "seed": protocol["seed"],
        "val_ratio": VAL_RATIO,
        "standard": {
            "train": len(protocol["legacy"]["train"]),
            "val": len(protocol["legacy"]["val"]),
            "train_by_domain": protocol["legacy"]["train_summary"]["domain_counts"],
            "val_by_domain": protocol["legacy"]["val_summary"]["domain_counts"],
        },
        "protocol": manifest,
        "lodo": manifest["lodo"],
    }

    with open(output_dir / "split_info.json", "w", encoding="utf-8") as handle:
        json.dump(split_info, handle, indent=2, ensure_ascii=False)
    with open(output_dir / "folds.json", "w", encoding="utf-8") as handle:
        json.dump({"folds": manifest["folds"]}, handle, indent=2, ensure_ascii=False)
    with open(output_dir / "lockbox.json", "w", encoding="utf-8") as handle:
        json.dump(manifest["lockbox"], handle, indent=2, ensure_ascii=False)


def main():
    questions = load_questions()
    print(f"Total questions: {len(questions)}")

    protocol = build_split_protocol(questions, seed=SEED, n_folds=N_FOLDS, lockbox_ratio=LOCKBOX_RATIO)
    write_split_protocol(protocol, SPLITS)

    legacy_train = protocol["legacy"]["train"]
    legacy_val = protocol["legacy"]["val"]
    legacy_train_domains = Counter(row["Domain"] for row in legacy_train)
    legacy_val_domains = Counter(row["Domain"] for row in legacy_val)

    print("\nGrouped doc-fold protocol:")
    print(
        f"  Lockbox: {len(protocol['lockbox']['docs'])} docs, "
        f"{len(protocol['lockbox']['questions'])} questions"
    )
    for fold in protocol["folds"]:
        print(
            f"  Fold {fold['fold_index']}: train={len(fold['train'])}, val={len(fold['val'])}, "
            f"train_docs={len(fold['train_docs'])}, val_docs={len(fold['val_docs'])}"
        )

    print("\nLegacy train/val split from fold 0:")
    print(f"  Train: {len(legacy_train)} questions — {dict(sorted(legacy_train_domains.items()))}")
    print(f"  Val:   {len(legacy_val)} questions — {dict(sorted(legacy_val_domains.items()))}")

    lodo_splits = protocol["lodo"]
    for held_out, split in lodo_splits.items():
        print(f"  LODO {held_out}: train={len(split['train'])}, val={len(split['val'])}")

    print(f"\nSaved splits to {SPLITS}/")


if __name__ == "__main__":
    main()
