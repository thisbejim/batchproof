from __future__ import annotations

import json
from pathlib import Path

from batchproof.cli import main


def test_summarize_passes_a_clean_report(tmp_path: Path, capsys: object) -> None:
    report = {
        "schema_version": 1,
        "examples": 2,
        "input_tokens": 9,
        "active_tokens": 8,
        "supervised_tokens": 6,
        "padded_tokens": 1,
        "findings": [],
    }
    path = tmp_path / "audit.jsonl"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    assert main(["summarize", str(path)]) == 0
    assert "batchproof PASS batches=1 examples=2" in capsys.readouterr().out  # type: ignore[attr-defined]


def test_summarize_fails_when_a_report_contains_errors(tmp_path: Path) -> None:
    report = {
        "schema_version": 1,
        "examples": 2,
        "input_tokens": 9,
        "active_tokens": 8,
        "supervised_tokens": 0,
        "padded_tokens": 0,
        "findings": [{"code": "supervision.empty_batch", "severity": "error", "count": 2}],
    }
    path = tmp_path / "audit.jsonl"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    assert main(["summarize", str(path)]) == 1


def test_summarize_rejects_malformed_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("{broken\n", encoding="utf-8")

    assert main(["summarize", str(path)]) == 2
