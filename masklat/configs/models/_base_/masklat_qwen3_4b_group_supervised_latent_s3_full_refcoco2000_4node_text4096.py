








from mmengine.config import read_base

with read_base():
    from .masklat_qwen3_4b_group_supervised_latent_s3_full_refcoco2000_4node import *


_parent_text_max_length = max_length
_text_max_length = 4096


def _replace_text_budget(value):
    """Replace only the parent data-side truncation budget, never LLM capacity."""
    if isinstance(value, dict):
        return {
            key: (
                _text_max_length
                if key == "max_length" and child == _parent_text_max_length
                else _replace_text_budget(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_replace_text_budget(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_replace_text_budget(child) for child in value)
    return value





for _name, _value in list(globals().items()):
    if not _name.startswith("_") and isinstance(_value, (dict, list, tuple)):
        globals()[_name] = _replace_text_budget(_value)

max_length = latent_text_max_length = _text_max_length
_base_text_max_length = max_length + num_proposal_latents


assert _parent_text_max_length == 2255
assert max_length == latent_text_max_length == 4096
assert num_proposal_latents == 64
assert all(
    dataset["max_length"] == 4096
    for dataset in train_dataloader["dataset"]["datasets"]
)
assert periodic_refcoco_dataset["max_length"] == 4096
assert custom_hooks[0]["dataset"]["max_length"] == 4096
assert model["llm"].get("max_position_embeddings") is None
assert model["inject_sam_vit_tokens_to_vlm"] is False
assert batch_size == 4 and global_batch == 128
assert lr == 4e-5 and max_epochs == 2
assert model["group_loss_weight"] == 0.1
