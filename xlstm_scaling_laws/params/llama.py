from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .common import count_model_params, count_params_ffn_llama_layer

if TYPE_CHECKING:
    from .count_params import ParamCountConfig


def count_llama_model_params(
    model_kwargs: dict[str, Any],
    config: "ParamCountConfig",
) -> float:
    vocab_size = model_kwargs["vocab_size"]
    num_blocks = model_kwargs["num_blocks"]
    d_model = model_kwargs["embedding_dim"]
    head_dim = model_kwargs["head_dim"]

    proj_factor_ffn = model_kwargs["proj_factor_ffn"]
    ffn_multiple_of = model_kwargs["ffn_multiple_of"]

    assert d_model % head_dim == 0, "d_model must be divisible by head_dim"
    num_heads = d_model // head_dim

    d_qk = head_dim
    d_v = head_dim

    llama_block_params = count_llama_block_params(
        d_model=d_model,
        num_heads=num_heads,
        d_qk=d_qk,
        d_v=d_v,
        proj_factor_ffn=proj_factor_ffn,
        ffn_multiple_of=ffn_multiple_of,
        config=config,
    )

    model_params = count_model_params(
        total_block_params=llama_block_params,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        d_model=d_model,
        config=config,
    )

    return model_params


def count_attention_layer_params(
    d_model: int,
    num_heads: int,
    d_qk: int,
    d_v: int,
    config: "ParamCountConfig",
) -> float:
    # no biases
    qkv = d_model * num_heads * (2 * d_qk + d_v)
    outproj = d_model * d_model

    total_layer_params = qkv + outproj

    if config.count_norm_layer_params:
        # only one norm (pre-norm)
        total_layer_params += d_model  # only norm scale, no bias

    return float(total_layer_params)


def count_llama_block_params(
    d_model: int,
    num_heads: int,
    d_qk: int,
    d_v: int,
    proj_factor_ffn: float,
    ffn_multiple_of: int,
    config: "ParamCountConfig",
    count_ffn_layer_params: Callable = count_params_ffn_llama_layer,
) -> float:
    llama_layer_params = count_attention_layer_params(
        d_model=d_model,
        num_heads=num_heads,
        d_qk=d_qk,
        d_v=d_v,
        config=config,
    )

    ffn_params = count_ffn_layer_params(
        d_model=d_model,
        proj_factor=proj_factor_ffn,
        ffn_multiple_of=ffn_multiple_of,
        round_mode="floor_multiple_of",
        config=config,
    )

    return llama_layer_params + ffn_params


def get_ffn_dim(
    model_kwargs: dict[str, Any],
) -> int:
    from .common import get_ffn_dim

    return get_ffn_dim(
        d_model=model_kwargs["embedding_dim"],
        proj_factor=model_kwargs["proj_factor_ffn"],
        ffn_multiple_of=model_kwargs["ffn_multiple_of"],
        round_mode="floor_multiple_of",
    )
