from __future__ import annotations

import json
from pathlib import Path

import pytest

from batchproof import BatchAuditError, BatchProof, BatchProofTrainerMixin


def test_records_valid_padded_batch_without_content(tmp_path: Path) -> None:
    output = tmp_path / "reports.jsonl"
    source = {
        "input_ids": [[10, 11, 0], [20, 21, 22]],
        "labels": [[-100, 11, -100], [20, 21, 22]],
        "attention_mask": [[1, 1, 0], [1, 1, 1]],
        "raw_text": "must not enter the report",
    }
    original = json.loads(json.dumps(source))

    report = BatchProof(output).inspect(source, step=7)

    assert report["batch"] == 0
    assert report["step"] == 7
    assert report["examples"] == 2
    assert report["input_tokens"] == 6
    assert report["active_tokens"] == 5
    assert report["supervised_tokens"] == 4
    assert report["padded_tokens"] == 1
    assert report["supervision_fraction"] == pytest.approx(4 / 5)
    assert report["findings"] == []
    assert source == original
    serialized = output.read_text(encoding="utf-8")
    assert "raw_text" not in serialized
    assert "must not enter" not in serialized
    assert "input_ids" not in serialized


def test_rejects_batch_with_no_targets_and_writes_finding(tmp_path: Path) -> None:
    output = tmp_path / "reports.jsonl"
    proof = BatchProof(output)

    with pytest.raises(BatchAuditError, match="supervision.empty_batch") as raised:
        proof.inspect({"input_ids": [[1, 2], [3, 4]], "labels": [[-100, -100], [-100, -100]]})

    assert raised.value.report["supervised_tokens"] == 0
    assert json.loads(output.read_text(encoding="utf-8"))["findings"][0]["code"] == (
        "supervision.empty_batch"
    )


def test_rejects_empty_example_inside_nonempty_batch() -> None:
    proof = BatchProof(raise_on_error=False)

    report = proof.inspect({"input_ids": [[1, 2], [3, 4]], "labels": [[1, 2], [-100, -100]]})

    assert report["supervised_tokens"] == 2
    assert report["empty_supervision_examples"] == 1
    assert report["findings"][0]["code"] == "supervision.empty_examples"  # type: ignore[index]


def test_rejects_labels_on_padding() -> None:
    proof = BatchProof(raise_on_error=False)

    report = proof.inspect(
        {
            "input_ids": [[7, 8, 0]],
            "labels": [[7, 8, 9]],
            "attention_mask": [[1, 1, 0]],
        }
    )

    assert report["supervised_on_padding"] == 1
    assert [finding["code"] for finding in report["findings"]] == [  # type: ignore[union-attr]
        "padding.supervised"
    ]


@pytest.mark.parametrize(
    ("batch", "expected_code"),
    [
        ({"input_ids": [[1, 2]], "labels": [[1]]}, "shape.sequence_mismatch"),
        (
            {"input_ids": [[1, 2]], "labels": [[1, 2]], "attention_mask": [[1]]},
            "shape.attention_mismatch",
        ),
        (
            {"input_ids": [[1]], "labels": [[1]], "attention_mask": [[2]]},
            "attention.not_binary",
        ),
        ({"input_ids": [[1.5]], "labels": [[1]]}, "input.invalid"),
        ({"labels": [[1]]}, "input.missing"),
    ],
)
def test_reports_invalid_shapes_and_values(batch: object, expected_code: str) -> None:
    report = BatchProof(raise_on_error=False).inspect(batch)
    codes = [item["code"] for item in report["findings"]]  # type: ignore[union-attr]
    assert expected_code in codes


def test_optional_causal_label_identity_check() -> None:
    proof = BatchProof(check_label_identity=True, raise_on_error=False)

    report = proof.inspect({"input_ids": [[1, 2]], "labels": [[2, -100]]})

    assert report["findings"][0]["code"] == "labels.not_input_identity"  # type: ignore[index]


def test_accepts_tensor_like_values_without_framework_imports() -> None:
    class TensorLike:
        def __init__(self, values: list[list[int]]) -> None:
            self.values = values

        def detach(self) -> TensorLike:
            return self

        def cpu(self) -> TensorLike:
            return self

        def tolist(self) -> list[list[int]]:
            return self.values

    report = BatchProof().inspect(
        {"input_ids": TensorLike([[1, 2]]), "labels": TensorLike([[1, 2]])}
    )

    assert report["input_tokens"] == 2
    assert report["findings"] == []


def test_torch_tensors_return_counts_without_token_values(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    output = tmp_path / "torch.jsonl"

    report = BatchProof(output).inspect(
        {
            "input_ids": torch.tensor([[10, 11, 0], [20, 21, 22]]),
            "labels": torch.tensor([[-100, 11, -100], [20, 21, 22]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        }
    )

    assert report["input_tokens"] == 6
    assert report["active_tokens"] == 5
    assert report["supervised_tokens"] == 4
    assert "input_ids" not in output.read_text(encoding="utf-8")


def test_distributed_ranks_get_independent_output_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    proof = BatchProof(tmp_path / "batches.jsonl")

    proof.inspect({"input_ids": [[1]], "labels": [[1]]})

    assert proof.output == tmp_path / "batches.rank-1.jsonl"
    assert json.loads(proof.output.read_text(encoding="utf-8"))["rank"] == 1


def test_trainer_mixin_observes_before_delegating() -> None:
    events: list[object] = []

    class ParentTrainer:
        def training_step(
            self, model: object, inputs: object, *args: object, **kwargs: object
        ) -> str:
            events.append(("parent", inputs))
            return "trained"

    class Trainer(BatchProofTrainerMixin, ParentTrainer):
        state = type("State", (), {"global_step": 4})()

    trainer = Trainer()
    trainer.batchproof = BatchProof()
    batch = {"input_ids": [[4, 5]], "labels": [[4, 5]]}

    result = trainer.training_step(object(), batch)

    assert result == "trained"
    assert events == [("parent", batch)]
    assert trainer.batchproof.batch_count == 1
