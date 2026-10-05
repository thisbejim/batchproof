"""Framework-free mixin for Hugging Face Trainer and compatible subclasses."""

from __future__ import annotations

from typing import Any


class BatchProofTrainerMixin:
    """Audit the actual mapping passed to ``training_step``.

    Set ``trainer.batchproof`` to a :class:`batchproof.BatchProof` instance.
    Keeping the hook framework-free lets it mix into Transformers and TRL
    trainers without importing or pinning either package.
    """

    def training_step(self, model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
        proof = getattr(self, "batchproof", None)
        if proof is not None:
            state = getattr(self, "state", None)
            global_step = getattr(state, "global_step", None)
            step = global_step if isinstance(global_step, int) else None
            proof.inspect(inputs, step=step)
        return super().training_step(model, inputs, *args, **kwargs)  # type: ignore[misc]
