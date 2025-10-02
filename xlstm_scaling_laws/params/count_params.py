from dataclasses import dataclass
from typing import Any

from .llama import count_llama_model_params
from .mlstm import count_mlstm_model_params


@dataclass
class ParamCountConfig:
    count_norm_layer_params: bool = True
    """Wether to count the parameters of the normalization layers."""
    count_embedding_params: bool = True
    """Wether to count the parameters of the embedding layers."""
    count_final_logits_params: bool = True
    """Wether to count the parameters of the final logits layer."""

def count_model_params(
    model_type: str,
    model_kwargs: dict[str, Any],
    config: ParamCountConfig,
) -> float:
    if model_type == "mlstm_v1":
        count_params_fn = count_mlstm_model_params
    elif model_type == "llama":
        count_params_fn = count_llama_model_params
    else:
        raise ValueError(f"model_type: {model_type} not supported!")

    return count_params_fn(
        model_kwargs=model_kwargs,
        config=config,
    )


def get_ffn_dim(model_type: str, model_kwargs: dict[str, Any]) -> int:
    if model_type == "mlstm_v1":
        from .mlstm import get_ffn_dim
    elif model_type == "llama":
        from .llama import get_ffn_dim
    else:
        raise ValueError(f"model_type: {model_type} not supported!")
    
    return get_ffn_dim(model_kwargs=model_kwargs)