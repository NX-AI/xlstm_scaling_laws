import json
from pathlib import Path

import torch
from safetensors.torch import load_file

# Note: These models use the same tokenizer as the xLSTM-7B,
# so we can load the tokenizer from the Hugging Face Hub without needing to save it in the checkpoint.
# e.g. tokenizer = AutoTokenizer.from_pretrained("NX-AI/xLSTM-7B")

# Note: We need to append the BOS token to the input tokens before passing them to the model, since the model was trained with a BOS token at the start of each sequence.


def load_from_pretrained(
    checkpoint_path: str | Path,
) -> dict[str, torch.Tensor]:
    """
    Load a mLSTM model from a checkpoint.

    Args:
        checkpoint_path: The path to the checkpoint.

    Returns:
        A state dict containing the model parameters.
    """

    checkpoint_path = Path(checkpoint_path)
    non_sharded_path = checkpoint_path / "model.safetensors"
    if non_sharded_path.exists():
        state_dict = load_file(non_sharded_path)
    else:
        n = 0
        sharded_path = checkpoint_path / f"model_{n}.safetensors"
        state_dict = {}
        while sharded_path.exists():
            state_dict.update(load_file(sharded_path))
            n += 1
            sharded_path = checkpoint_path / f"model_{n}.safetensors"

    return state_dict


def load_xlstm_config_from_checkpoint(ckpt_path: str | Path):
    """Load the xLSTM configuration from a checkpoint directory."""
    from transformers.models.xlstm.configuration_xlstm import xLSTMConfig

    with open(Path(ckpt_path) / "config.json") as f:
        config_dict = json.load(f)
    config_dict["hidden_size"] = config_dict.pop("embedding_dim")
    config_dict["mode"] = "inference"
    return xLSTMConfig(**config_dict)


def load_xlstm_model_from_checkpoint(ckpt_path: str | Path, load_weights: bool = True):
    from transformers.models.xlstm.modeling_xlstm import xLSTMForCausalLM

    config = load_xlstm_config_from_checkpoint(ckpt_path)
    model = xLSTMForCausalLM(config)

    if load_weights:
        state_dict = load_from_pretrained(ckpt_path)
        first_tensor = next(iter(state_dict.values()))
        print(
            f"Loading model from checkpoint with {len(state_dict)} tensors. Example tensor shape: {first_tensor.shape}, dtype: {first_tensor.dtype}"
        )
        model.load_state_dict(state_dict)

    return model
