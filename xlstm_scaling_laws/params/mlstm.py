from typing import TYPE_CHECKING, Any, Callable

from .common import count_model_params, count_params_ffn_llama_layer

if TYPE_CHECKING:
    from .count_params import ParamCountConfig

def count_mlstm_model_params(
    model_kwargs: dict[str, Any],
    config: "ParamCountConfig", 
) -> float:
    vocab_size = model_kwargs["vocab_size"]
    num_blocks = model_kwargs["num_blocks"]
    d_model = model_kwargs["embedding_dim"]
    num_heads = model_kwargs["num_heads"]

    proj_factor_ffn = model_kwargs["proj_factor_ffn"]
    ffn_multiple_of = model_kwargs["ffn_multiple_of"]

    chunk_size = model_kwargs["chunk_size"]
    proj_factor_qk = model_kwargs["proj_factor_qk"]

    d_qk = (d_model * proj_factor_qk) // num_heads
    d_v = d_model // num_heads
    # d_qk = d_v * proj_factor_qk

    mlstm_block_params = count_mlstm_block_params(
        d_model=d_model,
        num_heads=num_heads,
        d_qk=d_qk,
        d_v=d_v,
        proj_factor_ffn=proj_factor_ffn,
        ffn_multiple_of=ffn_multiple_of,
        config=config,
    )

    model_params = count_model_params(
        total_block_params=mlstm_block_params,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        d_model=d_model,
        config=config,
    )

    return model_params


def count_mlstm_layer_params(
    d_model: int,
    num_heads: int,
    d_qk: int,
    d_v: int,
    config: "ParamCountConfig",
) -> float:
    qkv = d_model * num_heads * (2 * d_qk + d_v)
    if_gate = 2 * d_model * num_heads + 2 * num_heads  # weight + bias
    o_gate = d_model * d_model
    outproj = d_model * d_model

    total_layer_params = qkv + if_gate + o_gate + outproj

    if config.count_norm_layer_params:
        total_layer_params += 2 * d_model
        # pre-norm + post-norm (multi-head norm)

    return float(total_layer_params)


def count_mlstm_block_params(
    d_model: int,
    num_heads: int,
    d_qk: int,
    d_v: int,
    proj_factor_ffn: float,
    ffn_multiple_of: int,
    config: "ParamCountConfig",
    count_ffn_layer_params: Callable = count_params_ffn_llama_layer,
) -> float:
    mlstm_layer_params = count_mlstm_layer_params(
        d_model=d_model,
        num_heads=num_heads,
        d_qk=d_qk,
        d_v=d_v,
        config=config,
    )

    ffn_layer_params = count_ffn_layer_params(
        d_model=d_model,
        proj_factor=proj_factor_ffn,
        ffn_multiple_of=ffn_multiple_of,
        round_mode="ceil_multiple_of",
        config=config,
    )

    return mlstm_layer_params + ffn_layer_params

def get_ffn_dim(
    model_kwargs: dict[str, Any]
) -> int:
    from .common import get_ffn_dim

    return get_ffn_dim(
        d_model=model_kwargs["embedding_dim"],
        proj_factor=model_kwargs["proj_factor_ffn"],
        ffn_multiple_of=model_kwargs["ffn_multiple_of"],
        round_mode="ceil_multiple_of",
    )
