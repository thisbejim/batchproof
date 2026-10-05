# batchproof

Audit the actual batches entering causal language-model training. Catch empty supervision, labels on padding, and shape mismatches at the training step, before a run reports a misleading loss or burns accelerator time.

## Problem

Tokenized dataset rows can look correct while the live collator, truncation, padding, or packing path changes what the model receives. A batch with every label ignored can produce a `0.0` training loss and finish without an obvious warning. Debugging usually means printing tensors and counting masks by hand.

## Solution

`batchproof` observes `input_ids`, `labels`, and `attention_mask` in the actual training step, records counts and checks locally, then fails on broken supervision by default. Reports contain no token values, text, prompts, or model outputs.

```text
live training batch -> batchproof -> local JSONL audit + fail-fast checks
```

## Quick Start

Python 3.10+:

```console
python -m pip install "git+https://github.com/thisbejim/batchproof.git@v0.1.0"
```

Add the mixin before your Trainer or TRL trainer base class:

```python
from batchproof import BatchProof, BatchProofTrainerMixin
from trl import SFTTrainer


class AuditedSFTTrainer(BatchProofTrainerMixin, SFTTrainer):
    pass


trainer = AuditedSFTTrainer(model=model, args=args, train_dataset=train_dataset)
trainer.batchproof = BatchProof(output="batchproof.jsonl")
trainer.train()
```

If a batch has no trainable labels, a supervised label falls on padding, or tensor shapes do not line up, `batchproof` writes the finding and raises `BatchAuditError` before the model processes that batch. Valid batches continue unchanged.

Summarize the recorded batches:

```console
batchproof summarize batchproof.jsonl
```

## Example Output

```text
batchproof PASS batches=3 examples=6 input_tokens=15 active_tokens=12 supervised_tokens=6 padded_tokens=3 findings=0
```

For a fully masked batch, the trainer stops with a message like:

```text
batchproof: batch 4 has no supervised tokens (4 examples, 0 of 128 input tokens)
```

The JSONL report retains per-example token counts and finding codes, but no content:

```json
{"active_tokens":128,"batch":4,"examples":4,"findings":[{"code":"supervision.empty_batch","count":4,"severity":"error"}],"input_tokens":128,"padded_tokens":0,"rank":0,"schema_version":1,"split":"train","supervised_tokens":0}
```

## Checks

* `input_ids`, `labels`, and optional `attention_mask` have matching batch and sequence dimensions.
* `attention_mask` contains only boolean or integer `0`/`1` values.
* Each example and each batch has supervised labels (`labels != -100` by default).
* No supervised label falls on an attention-masked padding position.
* Optional `check_label_identity=True` verifies that supervised labels equal input tokens, the usual causal-SFT convention.
* Reports include input, supervised, and padded token counts plus per-example coverage.

Use the standalone API in another training loop:

```python
from batchproof import BatchProof

proof = BatchProof(output="batches.jsonl")
for step, batch in enumerate(train_loader):
    proof.inspect(batch, step=step)
    train_step(batch)
```

For a custom ignore index or intentionally transformed labels:

```python
proof = BatchProof(ignore_index=-1, check_label_identity=True)
```

## Supported Inputs

The audited value is a mapping with integer arrays named `input_ids` and `labels`, each shaped `[sequence]` or `[batch, sequence]`. `attention_mask` is optional. Python arrays work without any framework. When given PyTorch tensors, `batchproof` imports PyTorch lazily and counts on-device, transferring only per-example counts to the host. Reports stay useful for packed batches because they count the final tensors as delivered; they do not infer packed-sequence boundaries.

`BatchProofTrainerMixin` wraps the Trainer `training_step` method and passes the same objects to the parent method unchanged. Use it with Hugging Face Trainer or a compatible subclass. It has no Transformers or TRL dependency. For distributed runs, each global rank writes a separate `*.rank-N.jsonl` report.

Batch auditing synchronizes once per observed GPU batch to write per-example counts, which can affect throughput. Use it during a short diagnostic run, then remove the mixin before benchmarking or long training runs.

For distributed jobs, summarize rank files with:

```console
batchproof summarize batchproof.rank-*.jsonl
```

## Why

* [TRL issue #6668](https://github.com/huggingface/trl/issues/6668) reports fully masked SFT batches whose training loop displays `0.0` loss even though the raw model loss is NaN.
* [TRL issue #3927](https://github.com/huggingface/trl/issues/3927) reports assistant-only labels disappearing after truncation, including a custom dataloader path that continued with zero loss.
* [TRL issue #3705](https://github.com/huggingface/trl/issues/3705) describes sequence-length metadata getting lost in a packing path before position IDs are generated.
* Hugging Face's [training pipeline debugging guide](https://huggingface.co/docs/course/main/chapter8/4) recommends manually iterating over and inspecting batches when Trainer runs fail.

The general PyTorch preflight tool [preflight-ml](https://github.com/Rusheel86/preflight) checks data health, shape, split leakage, and model basics, but its documented checks do not count ignored causal-LM labels or audit batches inside the training loop. The static [sftmask](https://github.com/thisbejim/sftmask) tool audits exported tokenized rows before training. `batchproof` covers the live SFT target-mask contract after dynamic collation and training-step transformations have happened.

## Independence and Privacy

No signup, payment, proprietary API, model, or server is required. All checks run locally and offline. The package has no mandatory dependencies. It lazily imports PyTorch only when passed PyTorch tensors; it never imports Transformers or TRL. Reports contain dimensions, counts, rank/step metadata, and finding codes. They do not include token IDs, decoded text, prompts, labels, or model outputs. No telemetry or network requests are made.

## Development

```console
python -m pip install -e '.[dev]'
pytest
ruff check .
ruff format --check .
mypy src
python -m build
```

## License

MIT. See [LICENSE](LICENSE).
