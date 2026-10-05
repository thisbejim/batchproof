# Product specification and research

## Target developer

Research and training-infrastructure engineers who build causal language-model fine-tuning pipelines with Hugging Face Trainer/TRL, Axolotl, or a custom PyTorch loop. They change tokenization, truncation, packing, collators, or distributed input handling and need to know what the model actually receives.

## Problem and job

When a training input pipeline changes, the engineer needs to inspect the post-collation batches at the training step so they can catch missing supervision, labels on padding, and shape drift before wasting a run or trusting a misleading loss curve.

These failures recur around preprocessing and trainer changes. The cost ranges from debugging hours to wasted multi-GPU runs or silently training on no target tokens. The standard workaround is to build a temporary script, take a batch from the trainer's dataloader, print tensors, and decode token/label pairs by hand.

## Public demand evidence

* [TRL issue #6668](https://github.com/huggingface/trl/issues/6668) reports that an SFT run can finish with a displayed loss of `0.0` when truncation removes every completion target. The model's raw loss is NaN and the batch has only ignored labels. The report explicitly asks for a fail or warning.
* [TRL issue #3927](https://github.com/huggingface/trl/issues/3927) reports all-masked assistant labels after truncation, including a custom dataloader that bypassed the framework's exception and continued training with zero loss.
* [TRL issue #3705](https://github.com/huggingface/trl/issues/3705) describes a packed-training path where sequence-length metadata may be dropped before the collator builds position IDs.
* [TRL issue #3692](https://github.com/huggingface/trl/issues/3692) asks which collator is correct for padding-free packing, illustrating that the final batch behavior depends on details that are easy to miss in source rows.
* Hugging Face's [training-pipeline debugging guide](https://huggingface.co/docs/course/main/chapter8/4) recommends manually iterating through the full pipeline and inspecting a batch when debugging Trainer failures. Hugging Face users continue to ask for concrete [label-span inspection snippets](https://discuss.huggingface.co/t/automatic-100-masking-of-the-questions-in-labels/176151).

The evidence includes independent reports across truncation and packing, plus first-party debugging guidance that currently sends users to manual inspection. The incidents establish concrete failure modes, not a measured prevalence rate.

## Existing workflow and alternatives

* **Manual dataloader inspection** is the documented general approach. It can work, but each engineer writes one-off tensor printing and counting code, and it is easy to inspect a pre-collation row instead of the actual batch entering training.
* **[`preflight-ml`](https://github.com/Rusheel86/preflight)** samples PyTorch dataloaders and checks general data health, shapes, leakage, and model/gradient basics. Its published checks do not inspect ignored causal-LM labels or per-example target coverage, and its README says it does not run the full training loop. `batchproof` is narrower and observes that specific training-step contract as it runs.
* **`sftmask`** audits exported, already-tokenized causal-SFT rows before a model is loaded. It is useful earlier in the pipeline; it cannot observe runtime collation, random transforms, distributed sampling, or tensors after packing and truncation in the active training step.
* **TRL and Transformers checks** catch selected invalid configurations. The issues above show remaining paths where a batch can still carry zero supervision or incorrect packing metadata. Framework rules also cannot validate a custom collator's intended behavior for every user.

## Product thesis

For engineers debugging causal-LM training pipelines, `batchproof` audits the tensors that reach each training step and saves a compact, privacy-preserving report; unlike static dataset checks or handwritten tensor prints, it observes the post-collation batch in the live loop and can fail immediately on zero supervision or labels over padding.

## Candidates screened

* **Inference determinism/reproducibility probe:** Recent vLLM issues report temperature-zero output changes across repeated requests and cache/scheduler paths. Rejected after finding direct projects [`detllm`](https://github.com/tommasocerruti/detllm) and [`LockStep`](https://github.com/u7k4rs6/LockStep), which already generate deterministic-inference checks and reproduction packs. A new general inference determinism CLI would have a weak improvement claim.
* **GGUF tokenizer metadata parity checker:** Public llama.cpp and vLLM reports show real tokenization failures from conversion metadata. Rejected because the distinct tokenizer file/version mismatch job already has [`tokresolve`](https://github.com/thisbejim/tokresolve), and the runtime-specific portion would need to chase GGUF and engine behavior.
* **Live SFT training-batch audit:** Selected because reports and first-party debugging instructions still direct engineers to manually inspect final batches. Static exported-row checks and trainer warnings cannot observe every user-defined collator or the exact mapping passed into a training step.

## Core workflow

```text
actual training-step batch
  -> BatchProofTrainerMixin / BatchProof.inspect
  -> local JSONL counts + precise mask findings
  -> batchproof summarize run.jsonl
```

## Interface

* A small Python library inspects mappings containing `input_ids`, `labels`, and optionally `attention_mask`.
* `BatchProofTrainerMixin` adds observation to Hugging Face Trainer subclasses, including TRL trainers, without importing either framework.
* `batchproof summarize` rolls up saved JSONL batch reports for local review or CI.
* The auditor does not change, retain, or print input token values, prompts, or model outputs.

## Offline and independence

Everything works offline. The package has no mandatory dependencies. Python arrays use the standard library; PyTorch is imported lazily only for torch tensor inputs so counts can be reduced on-device. No model, GPU, API credential, account, or external service is required. There is no telemetry.

## Non-goals and limits

`batchproof` does not infer whether a prompt or assistant span should be trained, decode token IDs, validate a model's loss implementation, inspect GPU memory, inspect multimodal/image masks, or certify distributed numerical equivalence. It checks mechanical alignment and target coverage on mappings with ordinary `input_ids`, `labels`, and optional `attention_mask`. Custom targets can opt out of the usual causal label identity check. Per-batch GPU observation synchronizes once to transfer a small count summary, so the mixin is intended for short diagnostic runs rather than throughput measurement.

## Final pre-build challenge

A training engineer can add the mixin to a trainer subclass and get a report from the batches that actually reach `training_step`; a zero-target batch fails before the model computes an apparently clean zero loss. This is a narrow, deterministic hook backed by a standard-library implementation and fixture tests, not a model demo or a new training framework.

## Opportunity assessment

| Criterion | Score |
| --- | ---: |
| Problem value | 8/10 |
| Evidence of demand | 8/10 |
| Usefulness to serious AI developers | 8/10 |
| Improvement over current workflow | 8/10 |
| Standalone feasibility | 9/10 |
| Maintainability | 8/10 |

The evidence is concentrated in the Hugging Face causal-SFT ecosystem rather than broad across every training framework. A general preflight tool exists; its published checks do not cover ignored-label semantics. The improvement is specifically the live training-step view and per-example target counts, which exported-row audits and general tensor checks do not provide. Integration follows the Trainer `training_step` seam without version-specific framework imports.

## Skeptical review

This could remain a small utility used only during debugging, and many experienced teams already inspect their first batch manually. TRL may fix specific zero-loss cases upstream. But upstream checks do not cover every custom collator, packing path, or transformation; inspecting the active training step remains useful even when a particular bug is fixed. The scope is intentionally narrow, so it will not replace general experiment tracking or training validation. It passes the quality gate because the saved report and fail-fast behavior make a recurring, expensive failure mode reproducible with a few lines and no model or infrastructure dependency.
