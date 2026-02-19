import math
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .count_params import ParamCountConfig


def count_model_params(
    total_block_params: float,
    num_blocks: int,
    vocab_size: int,
    d_model: int,
    config: "ParamCountConfig",
) -> float:
    """
    Returns:
        total params
    """
    if config.count_embedding_params:
        embedding_params = vocab_size * d_model
    else:
        embedding_params = 0.0

    if config.count_final_logits_params:
        final_logits_params = vocab_size * d_model
    else:
        final_logits_params = 0.0

    total_model_params = (
        num_blocks * total_block_params + embedding_params + final_logits_params
    )

    if config.count_norm_layer_params:
        total_model_params += d_model

    return total_model_params


def get_ffn_dim(
    d_model: int,
    proj_factor: float,
    ffn_multiple_of: int,
    round_mode: Literal["ceil_multiple_of", "floor_multiple_of", "none"],
) -> float | int:
    """This function computes the feedforward dim with the possible different rounding modes.

    It actually matters wether we round up after multiplying by proj_factor or not.
    We make it configurable with round_ceil.
    The actual rounding depends on the model implementation.
    In our case for `mlstm_v1` we round up, for `llama` we round down.

    We also support not rounding at all, which could be used to estimate the flops.
    More accurate flops estimation would require to round according to the model implementation.

    Args:
        d_model: model dim
        proj_factor: factor to multiply d_model with
        ffn_multiple_of: multiple to round to
        round_ceil: round up or down

    Returns:
        The feedforward dim.
    """
    if round_mode == "ceil_multiple_of":
        round_fn = math.ceil
    elif round_mode == "floor_multiple_of":
        round_fn = math.floor
    elif round_mode == "none":
        return proj_factor * float(d_model)
    else:
        raise ValueError(f"Invalid round_mode: {round_mode}")

    d_ffn = round_fn(d_model * proj_factor)
    d_ffn = ffn_multiple_of * ((d_ffn + ffn_multiple_of - 1) // ffn_multiple_of)
    return int(d_ffn)


def count_params_ffn_llama_layer(
    d_model: int,
    proj_factor: float,
    ffn_multiple_of: int,
    round_mode: Literal["ceil_multiple_of", "floor_multiple_of", "none"],
    config: "ParamCountConfig",
) -> float:
    """
    Returns:
        total num params
    """

    d_ffn = get_ffn_dim(
        d_model=d_model,
        proj_factor=proj_factor,
        ffn_multiple_of=ffn_multiple_of,
        round_mode=round_mode,
    )
    total_params = 3 * d_model * d_ffn

    if config.count_norm_layer_params:
        total_params += d_model

    return float(total_params)
