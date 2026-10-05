"""Command-line tools for batchproof reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import TextIO


def _open_report(path: str) -> tuple[TextIO, bool]:
    if path == "-":
        return sys.stdin, False
    return Path(path).open(encoding="utf-8"), True


def _summarize(paths: Iterable[str], output: TextIO) -> int:
    total_batches = 0
    total_examples = 0
    input_tokens = 0
    active_tokens = 0
    supervised_tokens = 0
    padded_tokens = 0
    errors = 0
    read_any = False

    for path in paths:
        handle, should_close = _open_report(path)
        try:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                read_any = True
                try:
                    report = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"batchproof: {path}:{line_number}: invalid JSON: {exc.msg}", file=output)
                    return 2
                if not isinstance(report, dict) or report.get("schema_version") != 1:
                    print(
                        f"batchproof: {path}:{line_number}: unsupported batch report",
                        file=output,
                    )
                    return 2
                try:
                    total_batches += 1
                    total_examples += int(report["examples"])
                    input_tokens += int(report["input_tokens"])
                    active_tokens += int(report["active_tokens"])
                    supervised_tokens += int(report["supervised_tokens"])
                    padded_tokens += int(report["padded_tokens"])
                    findings = report["findings"]
                    if not isinstance(findings, list):
                        raise ValueError("findings is not a list")
                    errors += sum(
                        int(item.get("count", 1))
                        for item in findings
                        if isinstance(item, dict) and item.get("severity") == "error"
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    print(
                        f"batchproof: {path}:{line_number}: invalid report field: {exc}",
                        file=output,
                    )
                    return 2
        finally:
            if should_close:
                handle.close()

    if not read_any:
        print("batchproof: no batch reports found", file=output)
        return 2
    status = "FAIL" if errors else "PASS"
    fraction = supervised_tokens / active_tokens if active_tokens else 0.0
    print(
        "batchproof "
        f"{status} batches={total_batches} examples={total_examples} "
        f"input_tokens={input_tokens} active_tokens={active_tokens} "
        f"supervised_tokens={supervised_tokens} "
        f"padded_tokens={padded_tokens} supervision={fraction:.1%} findings={errors}",
        file=output,
    )
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="batchproof", description="Review local training-batch audits."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    summary = commands.add_parser("summarize", help="roll up JSONL reports from training batches")
    summary.add_argument("reports", nargs="+", help="JSONL report paths (use - for stdin)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "summarize":
        return _summarize(args.reports, sys.stdout)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
