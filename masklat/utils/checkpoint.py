import logging
import os.path as osp

from mmengine.fileio import PetrelBackend, get_file_backend
from xtuner.model.utils import guess_load_checkpoint

from masklat.utils.logging import print_log


def load_checkpoint(model, pth_model: str) -> dict:
    """Load model checkpoint."""
    if pth_model is None or not str(pth_model).strip():
        raise FileNotFoundError("Checkpoint path is required, but no path was provided")
    if not osp.exists(pth_model):
        raise FileNotFoundError(f"Checkpoint does not exist: {pth_model}")

    backend = get_file_backend(pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio

        with patch_fileio():
            state_dict = guess_load_checkpoint(pth_model)
    else:
        state_dict = guess_load_checkpoint(pth_model)

    model.load_state_dict(state_dict, strict=False)
    checkpoint_keys = tuple(state_dict.keys())
    model_keys = tuple(model.state_dict().keys())
    checkpoint_key_set = set(checkpoint_keys)
    model_key_set = set(model_keys)
    matched_keys = [key for key in checkpoint_keys if key in model_key_set]
    mismatched_keys = [
        key for key in checkpoint_keys if key not in model_key_set
    ]
    missed_keys = [
        key for key in model_keys if key not in checkpoint_key_set
    ]
    print_log(f"Load checkpoint from {pth_model}", logger="current")
    print_log(f"Matched keys: {len(matched_keys)} / {len(state_dict.keys())}", logger="current")
    if len(mismatched_keys) > 0:
        print_log(f"Mismatched keys: {mismatched_keys}", logger="current", level=logging.WARNING)
    if len(missed_keys) > 0:
        print_log(f"Missed keys: {missed_keys}", logger="current", level=logging.WARNING)
    return {
        "checkpoint_path": str(pth_model),
        "checkpoint_keys": checkpoint_keys,
        "model_keys": model_keys,
        "matched_keys": tuple(matched_keys),
        "unexpected_checkpoint_keys": tuple(mismatched_keys),
        "missing_model_keys": tuple(missed_keys),
    }
