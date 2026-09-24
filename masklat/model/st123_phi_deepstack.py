"""Opt-in st1/st2/st3 latent additions at the inputs of Phi's first layers.

E1 is already in ``inputs_embeds``.  This module adds E2 before layer 2 and
E3 before layer 3, at the existing latent positions only.  No parameters or
state-dict keys are introduced, and unannotated forwards are unchanged.
"""

from contextlib import contextmanager
from functools import wraps
import inspect

import torch


class St123PhiDeepstackMixin:
    """Mixin for a holder whose ``llm`` is Phi3ForCausalLM or a PEFT wrapper."""

    @staticmethod
    def _validate_st123_phi_inputs(stage_latents, group_ids):
        if not isinstance(stage_latents, (tuple, list)) or len(stage_latents) != 3:
            raise ValueError("st123 Phi requires exactly (E1, E2, E3)")
        if not isinstance(group_ids, torch.Tensor) or group_ids.ndim != 2:
            raise ValueError("st123 Phi group IDs must be [B, L]")
        if min(group_ids.shape) <= 0:
            raise ValueError("st123 Phi group-ID batch and sequence must be nonempty")
        if group_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError("st123 Phi group IDs must be integer tensors")
        shape = None
        for latent in stage_latents:
            if not isinstance(latent, torch.Tensor) or latent.ndim != 3:
                raise ValueError("st123 Phi latent tables must be [B, K, D]")
            if min(latent.shape) <= 0 or (shape is not None and latent.shape != shape):
                raise ValueError("st123 Phi latent tables must have equal nonempty shapes")
            shape = latent.shape
        if shape[0] != group_ids.shape[0]:
            raise ValueError("st123 Phi latent and group-ID source batches disagree")
        if bool(group_ids.ge(shape[1]).any()):
            raise ValueError("st123 Phi group ID escapes the latent table")

    @contextmanager
    def _st123_phi_call_context(self, state):
        previous = self._active_st123_phi_call
        self._active_st123_phi_call = state
        try:
            yield
        finally:
            self._active_st123_phi_call = previous

    @contextmanager
    def _st123_phi_checkpoint_context(self, causal_model):
        """Capture call-local injection state in checkpoint recomputation.

        Transformers 4.x checkpoints in Phi3Model, whereas newer releases
        checkpoint in each decoder layer.  Adapt either layout without asking
        Phi to forward custom kwargs.  The closure outlives the outer forward,
        so loss.backward() also works after the forward's context is cleared.
        Non-reentrant checkpointing is required for the shared recurrent graph.
        """
        restored = []
        try:
            for module in causal_model.modules():
                function = getattr(module, "_gradient_checkpointing_func", None)
                if not callable(function):
                    continue
                if module.training and getattr(module, "gradient_checkpointing", False):
                    options = getattr(function, "keywords", {}) or {}
                    if options.get("use_reentrant", True):
                        raise RuntimeError(
                            "st123 Phi requires gradient_checkpointing_kwargs="
                            "{'use_reentrant': False}"
                        )

                def with_call_state(run_function, *args, _checkpoint=function, **kwargs):
                    state = self._active_st123_phi_call

                    def replay(*replay_args, **replay_kwargs):
                        with self._st123_phi_call_context(state):
                            return run_function(*replay_args, **replay_kwargs)

                    return _checkpoint(replay, *args, **kwargs)

                restored.append((module, function))
                module._gradient_checkpointing_func = with_call_state
            yield
        finally:
            for module, function in reversed(restored):
                module._gradient_checkpointing_func = function

    def _install_st123_phi_deepstack_hooks(self):
        """Install once, on this model instance only; supports later PEFT wrapping."""
        if getattr(self, "_st123_deepstack_hook_handles", None):
            return
        candidates = []
        for module in self.llm.modules():
            base = module._modules.get("model")
            layers = getattr(base, "layers", None)
            if (
                getattr(getattr(module, "config", None), "model_type", None) == "phi3"
                and layers is not None
                and "lm_head" in module._modules
            ):
                candidates.append((module, layers))
        if len(candidates) != 1 or len(candidates[0][1]) < 3:
            raise RuntimeError("st123 deepstack requires one Phi-3 causal LM with >= 3 layers")
        causal_model, layers = candidates[0]
        self._active_st123_phi_call = None
        self._active_st123_generation_latents = None
        self._active_st123_generation_group_ids = None
        self._st123_deepstack_hook_handles = []
        original_forward = causal_model.forward
        signature = inspect.signature(original_forward)

        @wraps(original_forward)
        def forward_with_st123(*args, **kwargs):
            stage_latents = kwargs.pop("masklat_st123_latents", None)
            group_ids = kwargs.pop("masklat_st123_group_ids", None)
            generation = stage_latents is None and group_ids is None
            if generation:
                stage_latents = self._active_st123_generation_latents
                group_ids = self._active_st123_generation_group_ids
            if stage_latents is None and group_ids is None:
                with self._st123_phi_call_context(None):
                    return original_forward(*args, **kwargs)
            self._validate_st123_phi_inputs(stage_latents, group_ids)
            # Binding handles both positional and keyword HF forward calls.
            bound = signature.bind_partial(*args, **kwargs).arguments
            cache = kwargs.get("past_key_values", bound.get("past_key_values"))
            if cache is None:
                past_length = 0
            elif hasattr(cache, "get_seq_length"):
                past_length = int(cache.get_seq_length())
            else:
                past_length = int(cache[0][0].shape[-2])
            state = dict(
                latents=tuple(stage_latents),
                group_ids=group_ids,
                generation=generation,
                past_length=past_length,
                cache_position=kwargs.get("cache_position", bound.get("cache_position")),
                observed=set(),
            )
            with self._st123_phi_call_context(state):
                with self._st123_phi_checkpoint_context(causal_model):
                    result = original_forward(*args, **kwargs)
                if state["observed"] != {1, 2}:
                    raise RuntimeError("Phi did not execute both st123 input hooks")
                return result

        causal_model.forward = forward_with_st123

        def input_hook(stage_index):
            def inject(module, args, kwargs):
                state = self._active_st123_phi_call
                if state is None:
                    return None
                state["observed"].add(stage_index)
                hidden = args[0] if args else kwargs.get("hidden_states")
                if hidden is None or hidden.ndim != 3:
                    raise ValueError("st123 Phi hidden states must be [B, L, D]")
                ids = state["group_ids"].to(device=hidden.device)
                table = state["latents"][stage_index]
                if hidden.shape[0] % ids.shape[0] != 0:
                    raise ValueError("st123 Phi source batch cannot expand to LLM batch")
                if hidden.shape[-1] != table.shape[-1]:
                    raise ValueError("st123 Phi latent and hidden dimensions disagree")
                if (
                    not state["generation"]
                    and state["past_length"] == 0
                    and hidden.shape[1] != ids.shape[1]
                ):
                    raise ValueError("st123 Phi hidden sequence and group IDs disagree")
                positions = state["cache_position"]
                if positions is None:
                    positions = torch.arange(
                        state["past_length"],
                        state["past_length"] + hidden.shape[1],
                        device=hidden.device,
                    )
                else:
                    positions = positions.to(device=hidden.device, dtype=torch.long)
                if positions.ndim != 1 or positions.numel() != hidden.shape[1]:
                    raise ValueError("st123 Phi cache positions must match the current sequence")
                in_prompt = positions.ge(0) & positions.lt(ids.shape[1])
                if not bool(in_prompt.any()):
                    return None  # Cached decoding: latent tokens are already in KV cache.
                selected_ids = ids[:, positions.clamp(0, ids.shape[1] - 1)]
                selected_ids = selected_ids.masked_fill(~in_prompt.unsqueeze(0), -1)
                repeat = hidden.shape[0] // ids.shape[0]
                if repeat != 1:
                    selected_ids = selected_ids.repeat_interleave(repeat, dim=0)
                    table = table.repeat_interleave(repeat, dim=0)
                valid = selected_ids.ge(0)
                if not bool(valid.any()):
                    return None
                table = table.to(device=hidden.device, dtype=hidden.dtype)
                addition = torch.gather(
                    table,
                    1,
                    selected_ids.clamp_min(0).long().unsqueeze(-1).expand(
                        -1, -1, hidden.shape[-1]
                    ),
                )
                hidden = hidden + addition.masked_fill(~valid.unsqueeze(-1), 0)
                if args:
                    return (hidden,) + args[1:], kwargs
                return args, dict(kwargs, hidden_states=hidden)

            return inject

        for stage_index in (1, 2):
            self._st123_deepstack_hook_handles.append(
                layers[stage_index].register_forward_pre_hook(
                    input_hook(stage_index), with_kwargs=True
                )
            )

    @contextmanager
    def _st123_deepstack_generation_context(self, stage_latents, group_ids):
        """Expose E1/E2/E3 to generation without extra GenerationMixin kwargs."""
        self._validate_st123_phi_inputs(stage_latents, group_ids)
        if self._active_st123_generation_latents is not None:
            raise RuntimeError("st123 generation contexts cannot be nested")
        self._active_st123_generation_latents = tuple(stage_latents)
        self._active_st123_generation_group_ids = group_ids
        try:
            yield
        finally:
            self._active_st123_generation_latents = None
            self._active_st123_generation_group_ids = None
