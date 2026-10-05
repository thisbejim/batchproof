"""Standalone example that inspects a synthetic final batch without PyTorch."""

from batchproof import BatchProof

proof = BatchProof(output="example-report.jsonl")
proof.inspect(
    {
        "input_ids": [[101, 12, 13, 0], [101, 22, 23, 24]],
        "labels": [[-100, 12, 13, -100], [-100, 23, 24, 25]],
        "attention_mask": [[1, 1, 1, 0], [1, 1, 1, 1]],
    },
    step=0,
)
print("Wrote example-report.jsonl")
