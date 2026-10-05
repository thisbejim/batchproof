"""Inspect the tensors presented to a causal language-model training step."""

from __future__ import annotations

import importlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast


def _materialize(value: object) -> object:
    """Convert common tensor/array objects without importing their libraries."""
    candidate: Any = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    tolist = getattr(candidate, "tolist", None)
    if callable(tolist):
        candidate = tolist()
    return candidate


def _matrix(value: object, field: str, *, mask: bool = False) -> list[list[int]]:
    raw = _materialize(value)
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{field} must be a one- or two-dimensional integer array")

    if not raw:
        rows: list[list[object]] = [[]]
    elif all(not isinstance(item, (list, tuple)) for item in raw):
        rows = [list(raw)]
    elif all(isinstance(item, (list, tuple)) for item in raw):
        rows = [list(item) for item in raw]
    else:
        raise ValueError(f"{field} must have a consistent one- or two-dimensional shape")

    result: list[list[int]] = []
    for row in rows:
        converted: list[int] = []
        for item in row:
            if mask and isinstance(item, bool):
                converted.append(int(item))
            elif isinstance(item, int) and not isinstance(item, bool):
                converted.append(item)
            else:
                kind = "boolean or integer 0/1" if mask else "integer"
                raise ValueError(f"{field} values must be {kind}")
        result.append(converted)
    return result


def _rank() -> int:
    value = os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))
    try:
        return int(value)
    except ValueError:
        return 0


class _AuditInputError(ValueError):
    def __init__(self, code: str, message: str, count: int = 1) -> None:
        super().__init__(message)
        self.code = code
        self.count = count


def _looks_like_tensor(value: object) -> bool:
    return (
        callable(getattr(value, "detach", None))
        and hasattr(value, "shape")
        and hasattr(value, "device")
    )


def _tensor_metrics(
    batch: Mapping[object, object],
    *,
    input_key: str,
    labels_key: str,
    attention_key: str,
    ignore_index: int,
    check_label_identity: bool,
) -> dict[str, object] | None:
    """Count tensor rows on their device and copy only the small per-row summary."""
    input_value = batch[input_key]
    labels_value = batch[labels_key]
    attention_value = batch.get(attention_key)
    if not any(
        _looks_like_tensor(value)
        for value in (input_value, labels_value, attention_value)
        if value is not None
    ):
        return None

    torch = importlib.import_module("torch")
    tensor_type = torch.Tensor
    if not isinstance(input_value, tensor_type) or not isinstance(labels_value, tensor_type):
        raise _AuditInputError(
            "tensor.mixed", "input_ids and labels must both be tensors when either is a tensor"
        )
    inputs = input_value.detach()
    labels = labels_value.detach()
    if inputs.ndim == 1:
        inputs = inputs.unsqueeze(0)
    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if inputs.ndim != 2 or labels.ndim != 2:
        raise _AuditInputError(
            "shape.dimensions", "input_ids and labels must be one- or two-dimensional"
        )
    if tuple(inputs.shape) != tuple(labels.shape):
        code = (
            "shape.batch_mismatch"
            if inputs.shape[0] != labels.shape[0]
            else "shape.sequence_mismatch"
        )
        raise _AuditInputError(code, "input and label tensor shapes differ")

    signed_types = (torch.int8, torch.int16, torch.int32, torch.int64)
    input_types = (*signed_types, torch.uint8)
    if inputs.dtype not in input_types:
        raise _AuditInputError("input.invalid", "input_ids tensor must use an integer dtype")
    if labels.dtype not in signed_types:
        raise _AuditInputError("labels.invalid", "labels tensor must use a signed integer dtype")
    if inputs.device != labels.device:
        raise _AuditInputError(
            "tensor.device_mismatch", "input_ids and labels are on different devices"
        )

    if attention_value is None:
        mask = torch.ones_like(inputs, dtype=torch.bool)
    else:
        if not isinstance(attention_value, tensor_type):
            raise _AuditInputError(
                "tensor.mixed", "attention_mask must be a tensor when input_ids are tensors"
            )
        mask = attention_value.detach()
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        mask_types = (*input_types, torch.bool)
        if mask.dtype not in mask_types:
            raise _AuditInputError(
                "attention.invalid", "attention_mask tensor must use a boolean or integer dtype"
            )
        if tuple(mask.shape) != tuple(inputs.shape):
            raise _AuditInputError(
                "shape.attention_mismatch", "attention mask shape differs from inputs"
            )
        if mask.device != inputs.device:
            raise _AuditInputError(
                "tensor.device_mismatch", "attention_mask is on a different device"
            )

    invalid_mask = (mask != 0) & (mask != 1)
    supervised_mask = labels != ignore_index
    active_by_example = mask.to(dtype=torch.int64).sum(dim=1)
    supervised_by_example = supervised_mask.sum(dim=1)
    padded_by_example = inputs.shape[1] - active_by_example
    supervised_padding_by_example = (supervised_mask & (mask == 0)).sum(dim=1)
    if check_label_identity:
        identity_by_example = (supervised_mask & (labels != inputs)).sum(dim=1)
    else:
        identity_by_example = torch.zeros_like(supervised_by_example)
    invalid_by_example = invalid_mask.sum(dim=1)
    summary = (
        torch.stack(
            (
                active_by_example,
                supervised_by_example,
                padded_by_example,
                supervised_padding_by_example,
                identity_by_example,
                invalid_by_example,
            ),
            dim=1,
        )
        .detach()
        .cpu()
        .tolist()
    )

    invalid_count = sum(row[5] for row in summary)
    if invalid_count:
        raise _AuditInputError(
            "attention.not_binary", "attention mask values must be 0 or 1", invalid_count
        )

    width = int(inputs.shape[1])
    examples = int(inputs.shape[0])
    per_example = [
        {
            "example": index,
            "input_tokens": width,
            "active_tokens": int(row[0]),
            "supervised_tokens": int(row[1]),
            "padded_tokens": int(row[2]),
        }
        for index, row in enumerate(summary)
    ]
    input_tokens = examples * width
    active_tokens = sum(int(row[0]) for row in summary)
    supervised_tokens = sum(int(row[1]) for row in summary)
    padded_tokens = sum(int(row[2]) for row in summary)
    empty_rows = sum(int(row[1]) == 0 for row in summary)
    supervised_on_padding = sum(int(row[3]) for row in summary)
    identity_mismatches = sum(int(row[4]) for row in summary)
    return {
        "examples": examples,
        "input_tokens": input_tokens,
        "active_tokens": active_tokens,
        "supervised_tokens": supervised_tokens,
        "padded_tokens": padded_tokens,
        "empty_supervision_examples": empty_rows,
        "supervised_on_padding": supervised_on_padding,
        "per_example": per_example,
        "supervision_fraction": round(supervised_tokens / active_tokens, 6)
        if active_tokens
        else 0.0,
        "identity_mismatches": identity_mismatches,
    }


class BatchAuditError(ValueError):
    """Raised after a failing batch report has been written."""

    def __init__(self, report: dict[str, object]) -> None:
        self.report = report
        findings = report.get("findings", [])
        codes = (
            [
                str(item["code"])
                for item in findings
                if isinstance(item, dict) and item.get("severity") == "error"
            ]
            if isinstance(findings, list)
            else []
        )
        batch = report.get("batch", "?")
        super().__init__(f"batchproof: batch {batch} failed: {', '.join(codes)}")


class BatchProof:
    """Audit and optionally record each post-collation training batch.

    Reports contain shape and token counts only. The input mapping is never changed.
    PyTorch is imported lazily only when a torch tensor is supplied; no other
    training framework is imported.
    """

    def __init__(
        self,
        output: str | Path | None = None,
        *,
        input_key: str = "input_ids",
        labels_key: str = "labels",
        attention_key: str = "attention_mask",
        ignore_index: int = -100,
        check_label_identity: bool = False,
        raise_on_error: bool = True,
    ) -> None:
        if not input_key or not labels_key or not attention_key:
            raise ValueError("batch field names must not be empty")
        if not isinstance(ignore_index, int) or isinstance(ignore_index, bool):
            raise ValueError("ignore_index must be an integer")
        self.input_key = input_key
        self.labels_key = labels_key
        self.attention_key = attention_key
        self.ignore_index = ignore_index
        self.check_label_identity = check_label_identity
        self.raise_on_error = raise_on_error
        self.batch_count = 0
        self.output = self._output_path(output)
        if self.output is not None:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_text("", encoding="utf-8")

    @staticmethod
    def _output_path(output: str | Path | None) -> Path | None:
        if output is None or str(output) == "-":
            return None
        path = Path(output)
        rank = os.environ.get("RANK")
        if rank is None and os.environ.get("WORLD_SIZE", "1") != "1":
            rank = os.environ.get("LOCAL_RANK", "0")
        if rank is not None:
            try:
                rank_number = int(rank)
            except ValueError:
                rank_number = 0
            path = path.with_name(f"{path.stem}.rank-{rank_number}{path.suffix}")
        return path

    def inspect(
        self,
        batch: object,
        *,
        step: int | None = None,
        split: str = "train",
    ) -> dict[str, object]:
        """Return counts for one batch and raise on a finding by default."""
        batch_number = self.batch_count
        self.batch_count += 1
        report: dict[str, object] = {
            "schema_version": 1,
            "batch": batch_number,
            "step": step,
            "rank": _rank(),
            "split": split,
            "examples": 0,
            "input_tokens": 0,
            "active_tokens": 0,
            "supervised_tokens": 0,
            "padded_tokens": 0,
            "empty_supervision_examples": 0,
            "supervised_on_padding": 0,
            "findings": [],
        }
        findings: list[dict[str, object]] = []
        report["findings"] = findings

        def finding(code: str, message: str, count: int = 1) -> None:
            findings.append({"code": code, "severity": "error", "message": message, "count": count})

        def coverage_findings(identity_mismatches: int) -> None:
            examples = cast(int, report["examples"])
            input_count = cast(int, report["input_tokens"])
            supervised_count = cast(int, report["supervised_tokens"])
            empty_rows = cast(int, report["empty_supervision_examples"])
            on_padding = cast(int, report["supervised_on_padding"])
            if not examples:
                finding("batch.empty", "batch contains no examples")
            elif input_count == 0:
                finding("input.empty", "batch contains no input tokens")
            if supervised_count == 0 and examples:
                finding("supervision.empty_batch", "batch has no supervised tokens", examples)
            if empty_rows and supervised_count > 0:
                finding(
                    "supervision.empty_examples",
                    "one or more examples have no supervised tokens",
                    empty_rows,
                )
            if on_padding:
                finding(
                    "padding.supervised",
                    "labels contribute to loss on attention-masked positions",
                    on_padding,
                )
            if identity_mismatches:
                finding(
                    "labels.not_input_identity",
                    "supervised labels differ from input tokens",
                    identity_mismatches,
                )

        if not isinstance(batch, Mapping):
            finding("batch.not_mapping", "training batch must be a mapping")
            return self._finish(report)

        if self.input_key not in batch:
            finding("input.missing", f"missing required field {self.input_key!r}")
        if self.labels_key not in batch:
            finding("labels.missing", f"missing required field {self.labels_key!r}")
        if findings:
            return self._finish(report)

        try:
            tensor_metrics = _tensor_metrics(
                batch,
                input_key=self.input_key,
                labels_key=self.labels_key,
                attention_key=self.attention_key,
                ignore_index=self.ignore_index,
                check_label_identity=self.check_label_identity,
            )
        except _AuditInputError as exc:
            finding(exc.code, str(exc), exc.count)
            return self._finish(report)
        if tensor_metrics is not None:
            identity_mismatches = cast(int, tensor_metrics.pop("identity_mismatches", 0))
            report.update(tensor_metrics)
            coverage_findings(identity_mismatches)
            return self._finish(report)

        try:
            inputs = _matrix(batch[self.input_key], self.input_key)
        except (TypeError, ValueError) as exc:
            finding("input.invalid", str(exc))
            return self._finish(report)
        try:
            labels = _matrix(batch[self.labels_key], self.labels_key)
        except (TypeError, ValueError) as exc:
            finding("labels.invalid", str(exc))
            return self._finish(report)

        if len(inputs) != len(labels):
            finding("shape.batch_mismatch", "input and label batch sizes differ")
            return self._finish(report)
        if len({len(row) for row in inputs}) > 1 or len({len(row) for row in labels}) > 1:
            finding("shape.ragged", "batch rows have unequal sequence lengths")
            return self._finish(report)
        if any(
            len(input_row) != len(label_row)
            for input_row, label_row in zip(inputs, labels, strict=True)
        ):
            finding("shape.sequence_mismatch", "input and label sequence lengths differ")
            return self._finish(report)

        masks: list[list[int]] | None = None
        if self.attention_key in batch and batch[self.attention_key] is not None:
            try:
                masks = _matrix(batch[self.attention_key], self.attention_key, mask=True)
            except (TypeError, ValueError) as exc:
                finding("attention.invalid", str(exc))
                return self._finish(report)
            if len(masks) != len(inputs) or any(
                len(mask_row) != len(input_row)
                for mask_row, input_row in zip(masks, inputs, strict=True)
            ):
                finding("shape.attention_mismatch", "attention mask shape differs from inputs")
                return self._finish(report)
            if any(item not in (0, 1) for row in masks for item in row):
                finding("attention.not_binary", "attention mask values must be 0 or 1")
                return self._finish(report)

        per_example: list[dict[str, int]] = []
        empty_rows = 0
        supervised_on_padding = 0
        padded_tokens = 0
        input_tokens = 0
        active_tokens = 0
        supervised_tokens = 0
        identity_mismatches = 0
        for index, (input_row, label_row) in enumerate(zip(inputs, labels, strict=True)):
            mask_row = masks[index] if masks is not None else [1] * len(input_row)
            active = sum(mask_row)
            supervised = sum(label != self.ignore_index for label in label_row)
            on_padding = sum(
                label != self.ignore_index and attention == 0
                for label, attention in zip(label_row, mask_row, strict=True)
            )
            if supervised == 0:
                empty_rows += 1
            if self.check_label_identity:
                identity_mismatches += sum(
                    label != self.ignore_index and label != token
                    for token, label in zip(input_row, label_row, strict=True)
                )
            input_tokens += len(input_row)
            active_tokens += active
            supervised_tokens += supervised
            padded_tokens += len(input_row) - active
            supervised_on_padding += on_padding
            per_example.append(
                {
                    "example": index,
                    "input_tokens": len(input_row),
                    "active_tokens": active,
                    "supervised_tokens": supervised,
                    "padded_tokens": len(input_row) - active,
                }
            )

        report.update(
            {
                "examples": len(inputs),
                "input_tokens": input_tokens,
                "active_tokens": active_tokens,
                "supervised_tokens": supervised_tokens,
                "padded_tokens": padded_tokens,
                "empty_supervision_examples": empty_rows,
                "supervised_on_padding": supervised_on_padding,
                "per_example": per_example,
                "supervision_fraction": round(supervised_tokens / active_tokens, 6)
                if active_tokens
                else 0.0,
            }
        )
        coverage_findings(identity_mismatches)
        return self._finish(report)

    def _finish(self, report: dict[str, object]) -> dict[str, object]:
        if self.output is not None:
            line = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        findings = report.get("findings", [])
        if self.raise_on_error and isinstance(findings, list) and findings:
            raise BatchAuditError(report)
        return report
