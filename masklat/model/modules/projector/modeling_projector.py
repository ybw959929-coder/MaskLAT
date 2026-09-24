import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from masklat.model.utils import pixel_shuffle

from .configuration_projector import DynamicProjectorConfig


class DynamicProjectorModel(PreTrainedModel):
    _auto_class = "AutoModel"
    config_class = DynamicProjectorConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["model"]

    def __init__(self, config: DynamicProjectorConfig) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        visual_hidden_size = config.visual_hidden_size * int(1 / config.downsample_ratio) ** 2
        modules = [
            nn.Linear(visual_hidden_size, config.llm_hidden_size, bias=config.bias),
        ]
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(nn.Linear(config.llm_hidden_size, config.llm_hidden_size, bias=config.bias))
        self.model = nn.Sequential(*modules)

    def enable_input_require_grads(self):
        # Match the transformers hook bookkeeping so
        # ``disable_input_require_grads`` can really detach frozen upstream
        # paths during adapter-only training.
        self.disable_input_require_grads()

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        hook = self.model.register_forward_hook(make_inputs_require_grad)
        # Transformers <=4.44 uses the singular attribute; newer releases use
        # a list.  Keep both so either implementation can inspect the module.
        self._require_grads_hook = hook
        self._require_grads_hooks = [hook]

    def disable_input_require_grads(self):
        """Remove input-gradient hooks across old and new Transformers."""

        hooks = list(getattr(self, "_require_grads_hooks", ()) or ())
        legacy_hook = getattr(self, "_require_grads_hook", None)
        if legacy_hook is not None and all(
            legacy_hook is not hook for hook in hooks
        ):
            hooks.append(legacy_hook)
        seen = set()
        for hook in hooks:
            if id(hook) not in seen:
                hook.remove()
                seen.add(id(hook))
        self._require_grads_hooks = []
        if hasattr(self, "_require_grads_hook"):
            del self._require_grads_hook

    def forward(self, x):
        if x.ndim == 4:
            if self.config.downsample_ratio != 1:
                x = pixel_shuffle(x, self.config.downsample_ratio)
            x = x.view(x.shape[0], -1, x.shape[-1])

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(self.model, x)
        else:
            layer_outputs = self.model(x)

        return layer_outputs
