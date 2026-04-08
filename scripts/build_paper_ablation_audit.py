#!/usr/bin/env python3
"""Generate paper-facing ablation audit tables and narrative docs."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from notebooks.paper_audit import write_paper_audit_artifacts


def main() -> None:
    outputs = write_paper_audit_artifacts()
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
