from typing import TYPE_CHECKING, Literal

from ..params.common import get_ffn_dim

if TYPE_CHECKING:
    from .count_flops import FlopCountConfig


def count_norm_layer_flops(seq_len: int, d_model: int) -> float:
    # d for mean, d to subtract mean, d for variance, d for division
    return float(seq_len) * 4 * d_model


def count_skip_flops(seq_len: int, d_model: int) -> float:
    return float(seq_len) * d_model


def zero_flops(seq_len: int, d_model: int) -> float:
    return 0.0


# def count_flops_ffn_gpt_layer_fw(
#     seq_len: int,
#     d_model: int,
#     proj_factor: float = 4.0,
#     factor_act_fn: float = 1,
#     ffn_multiple_of: int = 64,
#     round_mode: Literal[
#         "ceil_multiple_of", "floor_multiple_of", "none"
#     ] = "ceil_multiple_of",
#     count_norm_layer_flops: Callable[[int, int], float] = zero_flops,
#     count_skip_flops: Callable[[int, int], float] = zero_flops,
# ) -> tuple[float, float, float]:
#     """
#     Returns:
#         total flops, linear layer flops, other flops
#     """

#     d_ffn = get_ffn_dim(
#         d_model=d_model,
#         proj_factor=proj_factor,
#         ffn_multiple_of=ffn_multiple_of,
#         round_mode=round_mode,
#     )

#     seq_len = float(seq_len)
#     upproj_flops = 2 * seq_len * d_model * d_ffn + seq_len * d_ffn * factor_act_fn
#     downproj_flops = 2 * seq_len * d_model * d_ffn
#     ln_flops = count_norm_layer_flops(seq_len=seq_len, d_model=d_model)
#     skip_flops = count_skip_flops(seq_len=seq_len, d_model=d_model)

#     linear_layer_flops = upproj_flops + downproj_flops
#     other_flops = ln_flops + skip_flops

#     total_flops = linear_layer_flops + other_flops

#     return total_flops, linear_layer_flops, other_flops


def count_flops_ffn_llama_layer_fw(
    seq_len: int,
    d_model: int,
    proj_factor: float,
    ffn_multiple_of: int,
    round_mode: Literal["ceil_multiple_of", "floor_multiple_of", "none"],
    config: "FlopCountConfig",
) -> tuple[float, float, float]:
    """
    Returns:
        total flops, linear layer flops, other flops
    """
    seq_len = float(seq_len)

    d_ffn = get_ffn_dim(
        d_model=d_model,
        proj_factor=proj_factor,
        ffn_multiple_of=ffn_multiple_of,
        round_mode=round_mode,
    )

    upproj_flops = 4 * seq_len * d_model * d_ffn

    gate_flops = seq_len * d_ffn + seq_len * d_ffn * config.flop_factor_ffn_act_fn

    downproj_flops = 2 * seq_len * d_model * d_ffn

    if config.include_norm_layer_flops:
        count_norm_layer_flops_fn = count_norm_layer_flops
    else:
        count_norm_layer_flops_fn = zero_flops
    ln_flops = count_norm_layer_flops_fn(seq_len=seq_len, d_model=d_model)

    if config.include_skip_flops:
        count_skip_flops_fn = count_skip_flops
    else:
        count_skip_flops_fn = zero_flops
    skip_flops = count_skip_flops_fn(seq_len=seq_len, d_model=d_model)

    linear_layer_flops = upproj_flops + downproj_flops
    other_flops = ln_flops + skip_flops + gate_flops

    total_flops = linear_layer_flops + other_flops
    return total_flops, linear_layer_flops, other_flops


def count_model_flops_fw(
    total_block_flops_fw: float,
    linear_layer_block_flops_fw: float,
    seq_mix_layer_block_flops_fw: float,
    other_block_flops: float,
    num_blocks: int,
    vocab_size: int,
    d_model: int,
    seq_len: int,
    config: "FlopCountConfig",
) -> tuple[float, float, float]:
    """
    Returns:
        total flops, linear layer flops, sequence mix flops, other flops
    """
    assert abs(
        total_block_flops_fw
        - (
            linear_layer_block_flops_fw
            + seq_mix_layer_block_flops_fw
            + other_block_flops
        )
        < 0.1
    ), (
        f"total block flops {total_block_flops_fw} does not match the sum of ",
        f"its components {linear_layer_block_flops_fw + seq_mix_layer_block_flops_fw + other_block_flops}",
    )

    seq_len = float(seq_len)
    if config.include_embedding_flops:
        embedding_flops = 2 * seq_len * vocab_size * d_model
    else:
        embedding_flops = 0.0

    if config.include_final_logit_flops:
        final_logits_flops = 2 * seq_len * vocab_size * d_model
    else:
        final_logits_flops = 0.0

    # we count embedding flops as linear flops, but could also put them in the other flops
    linear_layer_model_flops_fw = (
        num_blocks * linear_layer_block_flops_fw + embedding_flops + final_logits_flops
    )
    seq_mix_layer_model_flops_fw = num_blocks * seq_mix_layer_block_flops_fw
    other_model_flops_fw = num_blocks * other_block_flops
    if config.include_norm_layer_flops:
        other_model_flops_fw += count_norm_layer_flops(seq_len=seq_len, d_model=d_model)

    total_model_flops_fw = (
        linear_layer_model_flops_fw
        + seq_mix_layer_model_flops_fw
        + other_model_flops_fw
    )

    return (
        total_model_flops_fw,
        linear_layer_model_flops_fw,
        seq_mix_layer_model_flops_fw,
        other_model_flops_fw,
    )


def calculate_model_fwbw_flops_from_fw_flops(
    total_model_flops_fw: float,
    linear_layer_model_flops_fw: float,
    seq_mix_layer_model_flops_fw: float,
    other_model_flops_fw: float,
    config: "FlopCountConfig",
) -> tuple[float, float, float, float]:
    """
    Returns:
        total flops, linear layer flops, sequence mix flops, other flops
    """

    linear_layer_model_flops_fwbw = linear_layer_model_flops_fw * (1.0 + 2.0)
    other_model_flops_fwbw = other_model_flops_fw * (1.0 + 2.0)

    if config.bw_flop_count_mode == "total_factor_2":
        seq_mix_bw_factor = 2.0
    elif (
        config.bw_flop_count_mode
        == "factor_2_linear_custom_seqmix_factor"
    ):
        seq_mix_bw_factor = config.seq_mix_bw_flop_factor
    elif config.bw_flop_count_mode == "factor_2_linear_custom_seqmix_bw_count":
        # we count the seqmix bw flops exactly with a custom function
        seq_mix_bw_factor = -1.0
    else:
        raise ValueError(f"Invalid bw_flop_count_mode: {config.bw_flop_count_mode}")

    seq_mix_layer_model_flops_fwbw = seq_mix_layer_model_flops_fw * (
        1.0 + seq_mix_bw_factor
    )

    total_model_flops_fwbw = (
        linear_layer_model_flops_fwbw
        + seq_mix_layer_model_flops_fwbw
        + other_model_flops_fwbw
    )

    # assertion only possible for overall case, as sanity check
    if config.bw_flop_count_mode == "total_factor_2":
        assert abs(total_model_flops_fwbw - (total_model_flops_fw * 3.0)) < 0.1, (
            f"total model flops {total_model_flops_fwbw} does not match 3 times the forward pass flops {3.0 * total_model_flops_fw}",
        )

    return (
        total_model_flops_fwbw,
        linear_layer_model_flops_fwbw,
        seq_mix_layer_model_flops_fwbw,
        other_model_flops_fwbw,
    )
