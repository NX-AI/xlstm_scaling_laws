from typing import TYPE_CHECKING, Any

from .common import (
    count_flops_ffn_llama_layer_fw,
    count_model_flops_fw,
    count_norm_layer_flops,
    count_skip_flops,
    zero_flops,
)

if TYPE_CHECKING:
    from .count_flops import FlopCountConfig


def count_llama_model_flops_fw(
    model_kwargs: dict[str, Any],
    context_length: int,
    config: "FlopCountConfig",
) -> tuple[float, float, float, float]:
    """
    Count the numebr of FLOPs for the forward pass of
    a Llama style Transformer model for a single sample (i.e. global batch size = 1).

    Returns:
        total flops, linear flops, attention flops, other flops
    """
    vocab_size = model_kwargs["vocab_size"]
    num_blocks = model_kwargs["num_blocks"]
    d_model = model_kwargs["embedding_dim"]

    head_dim = model_kwargs["head_dim"]

    proj_factor_ffn = model_kwargs["proj_factor_ffn"]
    ffn_multiple_of = model_kwargs["ffn_multiple_of"]

    d_qk = head_dim
    d_v = head_dim
    num_heads = d_model // head_dim

    # count llama block flops
    (
        total_block_flops_fw,
        linear_layer_block_flops_fw,
        attn_layer_block_flops_fw,
        other_block_flops_fw,
    ) = count_flops_attn_block_fw(
        seq_len=context_length,
        d_model=d_model,
        d_qk=d_qk,
        d_v=d_v,
        num_heads=num_heads,
        proj_factor_ffn=proj_factor_ffn,
        ffn_multiple_of=ffn_multiple_of,
        config=config,
    )
    # count llama model flops
    (
        total_model_flops_fw,
        linear_layer_model_flops_fw,
        seq_mix_layer_model_flops_fw,
        other_model_flops_fw,
    ) = count_model_flops_fw(
        total_block_flops_fw=total_block_flops_fw,
        linear_layer_block_flops_fw=linear_layer_block_flops_fw,
        seq_mix_layer_block_flops_fw=attn_layer_block_flops_fw,
        other_block_flops=other_block_flops_fw,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        d_model=d_model,
        seq_len=context_length,
        config=config,
    )

    assert (
        total_model_flops_fw
        == linear_layer_model_flops_fw
        + seq_mix_layer_model_flops_fw
        + other_model_flops_fw
    ), "Total flops should be the sum of linear, seq_mix, and other flops."

    return (
        total_model_flops_fw,
        linear_layer_model_flops_fw,
        seq_mix_layer_model_flops_fw,
        other_model_flops_fw,
    )


def count_flops_attn_layer_fw(
    seq_len: int,
    d_model: int,
    num_heads: int,
    d_qk: int,
    d_v: int,
    config: "FlopCountConfig",
) -> tuple[float, float, float, float]:
    """
    Attention Layer only.

    Returns:
        total_flops, linear flops, attn flops, other flops
    """

    assert d_model // num_heads == d_v

    seq_len = float(seq_len)
    d_model = float(d_model)
    num_heads = float(num_heads)
    d_qk = float(d_qk)
    d_v = float(d_v)

    qkv_projection_flops = 2 * seq_len * d_model * num_heads * (2 * d_qk + d_v)

    if config.attention_flop_calc_mode == "chinchilla":
        # Note: here it is not accounted for causality
        qk_logit_flops = 2 * seq_len * seq_len * (d_qk * num_heads)
        softmax_flops = 3 * num_heads * seq_len * seq_len
        value_or_softmax_query_red_flops = 2 * seq_len * seq_len * (d_v * num_heads)
    elif config.attention_flop_calc_mode == "distill_scaling":
        # Note: here there is a factor of 1/2 on the qk_logit_flops and value_or_softmax_query_red_flops
        qk_logit_flops = seq_len * seq_len * (d_qk * num_heads)
        softmax_flops = 2.5 * num_heads * seq_len * seq_len
        value_or_softmax_query_red_flops = seq_len * seq_len * (d_v * num_heads)
    else:
        raise ValueError(
            f"Invalid attention_flop_calc_mode: {config.attention_flop_calc_mode}"
        )

    out_proj = 2 * seq_len * d_model * num_heads * d_v

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

    linear_layer_flops = qkv_projection_flops + out_proj
    attn_flops = qk_logit_flops + softmax_flops + value_or_softmax_query_red_flops
    other_flops = ln_flops + skip_flops

    total_attn_flops = linear_layer_flops + attn_flops + other_flops

    return total_attn_flops, linear_layer_flops, attn_flops, other_flops


def count_flops_attn_block_fw(
    seq_len: int,
    d_model: int,
    d_qk: int,
    d_v: int,
    num_heads: int,
    proj_factor_ffn: float,
    ffn_multiple_of: int,
    config: "FlopCountConfig",
) -> tuple[float, float, float, float]:
    """
    Full Attention block consisting of the attention layer and the feed forward layer.

    Returns:
        total flops, linear flops, attn flops, other flops
    """

    total_attn_flops, linear_attn_flops, attn_flops, attn_other_flops = (
        count_flops_attn_layer_fw(
            seq_len=seq_len,
            d_model=d_model,
            num_heads=num_heads,
            d_qk=d_qk,
            d_v=d_v,
            config=config,
        )
    )

    assert total_attn_flops == linear_attn_flops + attn_flops + attn_other_flops, (
        "total_attn_flops should be the sum of linear, attn, and other flops."
    )

    total_ffn_layer_flops, linear_layer_ffn_flops, other_ffn_flops = (
        count_flops_ffn_llama_layer_fw(
            seq_len=seq_len,
            d_model=d_model,
            proj_factor=proj_factor_ffn,
            ffn_multiple_of=ffn_multiple_of,
            round_mode="floor_multiple_of"
            if config.round_ffn_dim_to_multiple_of_for_flops
            else "none",
            config=config,
        )
    )

    assert total_ffn_layer_flops == linear_layer_ffn_flops + other_ffn_flops, (
        "total_ffn_layer_flops should be the sum of linear and other flops."
    )

    total_flops = total_attn_flops + total_ffn_layer_flops

    total_linear_flops = linear_attn_flops + linear_layer_ffn_flops
    total_attn_flops = attn_flops
    total_other_flops = attn_other_flops + other_ffn_flops

    assert (
        abs(total_flops - (total_linear_flops + total_attn_flops + total_other_flops))
        < 0.1
    ), (
        f"Total flops should be the sum of linear, attn, and other flops. "
        f"total_flops={total_flops}, sum of components={total_linear_flops + total_attn_flops + total_other_flops}"
    )

    return total_flops, total_linear_flops, total_attn_flops, total_other_flops
