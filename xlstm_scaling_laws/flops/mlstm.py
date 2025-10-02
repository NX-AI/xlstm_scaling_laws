"""Contains the functions to compute the FLOPs for the mlstm kernels and blocks."""

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


def count_mlstm_model_flops_fw(
    model_kwargs: dict[str, Any],
    context_length: int,
    config: "FlopCountConfig",
) -> tuple[float, float, float, float]:
    """
    Count the number of FLOPs for the forward pass of the mLSTM v1 model
    for a single sample (i.e. a single sequence) (i.e. global batch size = 1).

    Returns:
        total flops, linear flops, mlstm flops, other flops
    """
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

    # count mlstm block flops
    (
        total_block_flops_fw,
        linear_layer_block_flops_fw,
        seq_mix_layer_block_flops_fw,
        other_block_flops_fw,
    ) = count_flops_mlstm_v1_block_fw(
        seq_len=context_length,
        chunk_size=chunk_size,
        d_model=d_model,
        d_qk=d_qk,
        d_v=d_v,
        num_heads=num_heads,
        proj_factor_ffn=proj_factor_ffn,
        ffn_multiple_of=ffn_multiple_of,
        config=config,
    )
    # count model flops
    (
        total_model_flops_fw,
        linear_layer_model_flops_fw,
        seq_mix_layer_model_flops_fw,
        other_model_flops_fw,
    ) = count_model_flops_fw(
        total_block_flops_fw=total_block_flops_fw,
        linear_layer_block_flops_fw=linear_layer_block_flops_fw,
        seq_mix_layer_block_flops_fw=seq_mix_layer_block_flops_fw,
        other_block_flops=other_block_flops_fw,
        num_blocks=num_blocks,
        vocab_size=vocab_size,
        d_model=d_model,
        seq_len=context_length,
        config=config,
    )

    return (
        total_model_flops_fw,
        linear_layer_model_flops_fw,
        seq_mix_layer_model_flops_fw,
        other_model_flops_fw,
    )


def count_flops_fw_C(
    chunk_size, num_chunks, d_qk, d_v, num_heads, factor_exp, factor_max, factor_mask
) -> float:
    """Flops for a single sequence."""
    # paper formulas
    chunkwise_gates = (
        float(num_heads)
        * num_chunks
        * (0.5 * chunk_size * (chunk_size + 1) + 2 * chunk_size)
    )
    gates_and_max_state = (
        float(num_heads)
        * num_chunks
        * (3 + factor_max + factor_exp + chunk_size * (3 + 2 * factor_exp))
    )
    numerator = (
        float(num_heads)
        * num_chunks
        * (2 * d_qk * d_v + 4 * chunk_size * d_qk * d_v + 3 * chunk_size * d_qk)
    )
    denominator = num_heads * num_chunks * (2 * d_qk + 4 * chunk_size * d_qk)

    total_flops = chunkwise_gates + gates_and_max_state + numerator + denominator
    return total_flops


def count_flops_fw_H(
    chunk_size,
    num_chunks,
    d_qk,
    d_v,
    num_heads,
    factor_exp,
    factor_max,
    factor_mask,
    factor_abs,
) -> float:
    """Flops for a single sequence."""
    combined_term = (
        float(num_chunks)
        * num_heads
        * (
            (chunk_size * (chunk_size + 1)) // 2
            + chunk_size
            * chunk_size
            * (7 + factor_mask + factor_max + factor_exp + 2 * d_qk + 2 * d_v)
            + (1 + factor_max) * chunk_size
        )
    )
    # paper formulas intra part
    gate_matrix = (
        0.5 * chunk_size * (chunk_size + 1)
        + chunk_size * chunk_size * (3 + factor_mask + factor_max + factor_exp)
        + chunk_size * (1 + factor_max)
    )
    gated_attn_matrix = 2 * chunk_size * chunk_size * d_qk + 2 * chunk_size * chunk_size
    h_intra = 2 * chunk_size * chunk_size * d_v
    n_intra = 2 * chunk_size * chunk_size

    intra_flops = (
        float(num_heads)
        * num_chunks
        * (gate_matrix + gated_attn_matrix + h_intra + n_intra)
    )
    assert intra_flops == combined_term, (
        f"The two flop computations do not match. paper_formulas={intra_flops} vs. combined_term={combined_term}"
    )

    # paper formulas output combination
    output_combination = (
        float(num_heads)
        * num_chunks
        * (
            chunk_size * (1 + factor_max)
            + chunk_size * (2 + factor_abs + factor_exp + factor_max)
            + 2 * chunk_size * d_v
        )
    )

    return intra_flops + output_combination


def count_flops_mlstm_cell_chunkwise_fw__first(
    chunk_size: int,
    num_chunks: int,
    d_qk: int,
    d_v: int,
    num_heads: int,
    factor_exp: float,
    factor_max: float,
    factor_mask: float,
    factor_abs: float,
) -> tuple[float, float, float]:
    flops_fw_C = count_flops_fw_C(
        chunk_size,
        num_chunks,
        d_qk,
        d_v,
        num_heads,
        factor_exp,
        factor_max,
        factor_mask,
    )
    flops_fw_H = count_flops_fw_H(
        chunk_size,
        num_chunks,
        d_qk,
        d_v,
        num_heads,
        factor_exp,
        factor_max,
        factor_mask,
        factor_abs,
    )
    flops_fw_total = flops_fw_C + flops_fw_H
    return flops_fw_total, flops_fw_C, flops_fw_H


def count_flops_mlstm_v1_layer_fw(
    seq_len: int,
    d_model: int,
    d_qk: int,
    d_v: int,
    num_heads: int,
    chunk_size: int,
    config: "FlopCountConfig",
) -> tuple[float, float, float, float]:
    """
    mLSTM v1 layer only.

    Returns:
        total flops, linear flops, mlstm flops, other flops
    """
    num_chunks = seq_len // chunk_size
    qkvif_flops = 2 * seq_len * d_model * num_heads * (2 * d_qk + d_v + 2)
    ogate_flops = (
        2 * seq_len * d_model * num_heads * d_v
        + seq_len * num_heads * d_v * config.flop_factor_sig
    )
    out_proj = 2 * seq_len * d_model * num_heads * d_v

    if config.mlstm_fw_flop_calc_mode == "first":
        (
            mlstm_cell_flops_total,
            mlstm_cell_flops_recurrent,
            mlstm_cell_flops_parallel,
        ) = count_flops_mlstm_cell_chunkwise_fw__first(
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            d_qk=d_qk,
            d_v=d_v,
            num_heads=num_heads,
            factor_exp=config.flop_factor_exp,
            factor_max=config.flop_factor_max,
            factor_mask=config.flop_factor_mask,
            factor_abs=config.flop_factor_abs,
        )
    elif config.mlstm_fw_flop_calc_mode == "tfla":
        (
            mlstm_cell_flops_total,
            mlstm_cell_flops_recurrent,
            mlstm_cell_flops_parallel,
        ) = count_flops_mlstm_cell_chunkwise_fw__tfla(
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            d_qk=d_qk,
            d_v=d_v,
            num_heads=num_heads,
            factor_exp=config.flop_factor_exp,
            factor_log=config.flop_factor_log,
            factor_sig=config.flop_factor_sig,
            factor_max=config.flop_factor_max,
            factor_mask=config.flop_factor_mask,
            factor_abs=config.flop_factor_abs,
            causal_factor=config.mlstm_flop_causal_factor,
        )
    else:
        raise ValueError(
            f"Unknown mLSTM forward flop calculation mode: {config.mlstm_fw_flop_calc_mode}"
        )

    if config.include_norm_layer_flops:
        count_norm_layer_flops_fn = count_norm_layer_flops
    else:
        count_norm_layer_flops_fn = zero_flops

    ln_flops = count_norm_layer_flops_fn(
        seq_len=seq_len, d_model=d_model
    ) + num_heads * count_norm_layer_flops_fn(seq_len=seq_len, d_model=d_v)

    if config.include_skip_flops:
        count_skip_flops_fn = count_skip_flops
    else:
        count_skip_flops_fn = zero_flops

    skip_flops = count_skip_flops_fn(seq_len=seq_len, d_model=d_model)

    linear_layer_flops = qkvif_flops + out_proj + ogate_flops
    mlstm_flops = mlstm_cell_flops_total
    other_flops = ln_flops + skip_flops

    total_flops = linear_layer_flops + mlstm_flops + other_flops

    return total_flops, linear_layer_flops, mlstm_flops, other_flops


def count_flops_mlstm_v1_block_fw(
    seq_len: int,
    chunk_size: int,
    d_model: int,
    d_qk: int,
    d_v: int,
    num_heads: int,
    proj_factor_ffn: float,
    ffn_multiple_of: int,
    config: "FlopCountConfig",
) -> tuple[float, float, float, float]:
    """
    mLSTM v1 block.
    Consists of the mLSTM v1 layer and the feed forward layer.

    Returns:
        total flops, linear flops, mlstm flops, other flops
    """
    (
        total_mlstm_v1_layer_flops,
        mlstm_linear_layer_flops,
        mlstm_cell_flops,
        mlstm_other_flops,
    ) = count_flops_mlstm_v1_layer_fw(
        seq_len=seq_len,
        d_model=d_model,
        d_qk=d_qk,
        d_v=d_v,
        num_heads=num_heads,
        chunk_size=chunk_size,
        config=config,
    )
    assert (
        total_mlstm_v1_layer_flops
        == mlstm_linear_layer_flops + mlstm_cell_flops + mlstm_other_flops
    ), "Total flops should be the sum of linear, mlstm, and other flops. "

    total_ffn_layer_flops, linear_layer_ffn_flops, other_ffn_flops = (
        count_flops_ffn_llama_layer_fw(
            seq_len=seq_len,
            d_model=d_model,
            proj_factor=proj_factor_ffn,
            ffn_multiple_of=ffn_multiple_of,
            round_mode="ceil_multiple_of"
            if config.round_ffn_dim_to_multiple_of_for_flops
            else "none",
            config=config,
        )
    )
    assert total_ffn_layer_flops == linear_layer_ffn_flops + other_ffn_flops, (
        "Total flops should be the sum of linear and other flops. "
    )

    total_linear_layer_flops = linear_layer_ffn_flops + mlstm_linear_layer_flops
    total_mlstm_flops = mlstm_cell_flops
    total_other_flops = mlstm_other_flops + other_ffn_flops

    total_flops = total_mlstm_v1_layer_flops + total_ffn_layer_flops

    assert (
        abs(
            total_flops
            - (total_linear_layer_flops + total_mlstm_flops + total_other_flops)
        )
        < 0.1
    ), (
        f"Total flops should be the sum of linear, mlstm, and other flops. "
        f"total_flops={total_flops}, sum of components={total_linear_layer_flops + total_mlstm_flops + total_other_flops}"
    )

    return total_flops, total_linear_layer_flops, total_mlstm_flops, total_other_flops


def count_model_flops_bw_mlstm_cell_only(
    model_kwargs: dict[str, Any],
    context_length: int,
    config: "FlopCountConfig",
) -> tuple[float, float, float]:
    """Counts the mlstm_cell backward flops of a full model, i.e. the number of flops of a single mlstm cell backward pass
    are multiplied by the number of blocks in the model.
    """
    num_blocks = model_kwargs["num_blocks"]
    d_model = model_kwargs["embedding_dim"]
    num_heads = model_kwargs["num_heads"]

    chunk_size = model_kwargs["chunk_size"]
    proj_factor_qk = model_kwargs["proj_factor_qk"]

    d_qk = (d_model * proj_factor_qk) // num_heads
    d_v = d_model // num_heads

    num_chunks = context_length // chunk_size

    (
        total_mlstm_cell_bw_flops,
        recurrent_mlstm_cell_bw_flops,
        parallel_mlstm_cell_bw_flops,
    ) = count_flops_mlstm_cell_chunkwise_bw__tfla(
        chunk_size=chunk_size,
        num_chunks=num_chunks,
        d_qk=d_qk,
        d_v=d_v,
        num_heads=num_heads,
        causal_factor=config.mlstm_flop_causal_factor,
        factor_exp=config.flop_factor_exp,
        factor_max=config.flop_factor_max,
    )

    total_flops = total_mlstm_cell_bw_flops * num_blocks
    recurrent_flops = recurrent_mlstm_cell_bw_flops * num_blocks
    parallel_flops = parallel_mlstm_cell_bw_flops * num_blocks

    return total_flops, recurrent_flops, parallel_flops


def count_flops_mlstm_cell_chunkwise_bw__tfla(
    chunk_size: int,
    num_chunks: int,
    d_qk: int,
    d_v: int,
    num_heads: int,
    causal_factor: float,
    factor_exp: float,
    factor_sig: float,
    factor_log: float,
    factor_max: float,
) -> tuple[float, float, float]:
    """Compute the FLOPs for the mLSTM cell in the backward pass.

    Args:
        chunk_size:
        causal_factor: The factor for the parallel "attention" matrix in the backward pass.
                        For the Flash Linear Attention Kernels (with chunk size 64), this factor is typically 1.0 as the full attention matrix is computed.
                        For TFLA kernels with arbitrary chunk sizes this factor is betweeon 0.5 and 1.0 as only the causal part of the attention matrix is computed.

    Returns:
        total flops, recurrent flops, parallel flops
    """

    ## Recomputation of C states (recurrent forward pass)
    recomp_recurrent_gates = (
        2 * chunk_size
        + 0.5 * chunk_size * (chunk_size + 1)
        + chunk_size * (1 + factor_exp + factor_log + factor_sig)
        + 3
        + factor_max
        + factor_exp
    )

    recomp_recurrent_numerator = (
        2 * d_qk * d_v + 2 * chunk_size * d_qk * d_v + chunk_size * d_qk
    )

    recomp_recurrent_denominator = 2 * d_qk + 2 * chunk_size * d_qk

    flops_recomp_recurrent = (
        float(num_heads)
        * float(num_chunks)
        * (
            recomp_recurrent_gates
            + recomp_recurrent_numerator
            + recomp_recurrent_denominator
        )
    )

    ## Recurrent Backward Pass deltaC
    bw_recurrent_gates = (
        0.5 * chunk_size * (chunk_size + 1)
        + (chunk_size + 1) * (factor_exp + 2)
        + chunk_size * (factor_sig + factor_log)
    )
    bw_deltaC = 2 * d_qk * d_v + 2 * chunk_size * d_qk * d_v + chunk_size * d_qk

    flops_bw_recurrent = (
        float(num_heads) * float(num_chunks) * (bw_recurrent_gates + bw_deltaC)
    )

    ## Parallel Backward Pass Intra deltaQ, deltaK, deltaV
    bw_parallel_gates = chunk_size * (chunk_size + 1) + 2 * chunk_size + chunk_size * (factor_log + factor_sig)

    recomp_gate_matrix = (
        2 * chunk_size * chunk_size * d_qk + chunk_size * chunk_size * (factor_exp + 5)
    )

    bw_intra_chunk_qkv_gradients = (
        4 * chunk_size * chunk_size * (d_qk + d_v) + 3 * chunk_size * chunk_size
    )

    flops_bw_parallel_intra = (
        float(num_heads)
        * float(num_chunks)
        * (bw_parallel_gates + recomp_gate_matrix + bw_intra_chunk_qkv_gradients)
    )

    if chunk_size > 64:
        # we assume TFLA kernels for chunk sizes > 64
        flops_bw_parallel_intra *= causal_factor

    ## Parallel Backward Pass Inter deltaQ, deltaK, deltaV
    bw_inter_chunk_qkv_gradients = (
        6 * chunk_size * d_qk * d_v
        + 4 * chunk_size * d_qk
        + chunk_size * (2 * factor_exp + 3)
    )

    bw_combination_qkv_gradients = chunk_size * (2 * d_qk * d_v)

    flops_bw_parallel_inter = (
        float(num_heads)
        * float(num_chunks)
        * (bw_inter_chunk_qkv_gradients + bw_combination_qkv_gradients)
    )

    ## Computation of forget and input gate gradients
    seq_len = num_chunks * chunk_size
    d = num_heads * d_qk

    bw_forgetgate = 0.5 * seq_len * (seq_len + 1) + 5 * seq_len * d

    bw_inputgate = 3 * seq_len * d

    flops_bw_input_forget_gates = bw_forgetgate + bw_inputgate

    ### Total backward pass FLOPs
    total_flops_recurrent = flops_recomp_recurrent + flops_bw_recurrent
    total_flops_parallel = (
        flops_bw_parallel_intra + flops_bw_parallel_inter + flops_bw_input_forget_gates
    )

    total_flops = total_flops_recurrent + total_flops_parallel

    return total_flops, total_flops_recurrent, total_flops_parallel


def count_flops_mlstm_cell_chunkwise_fw__tfla(
    chunk_size: int,
    num_chunks: int,
    d_qk: int,
    d_v: int,
    num_heads: int,
    causal_factor: float,
    factor_exp: float,
    factor_log: float,
    factor_sig: float,
    factor_max: float,
    factor_mask: float,
    factor_abs: float,
) -> tuple[float, float, float]:
    ## Recurrent Computation of the C states
    recurrent_gates = (
        2 * chunk_size
        + 0.5 * chunk_size * (chunk_size + 1)
        + chunk_size * (1 + factor_exp + factor_log + factor_sig)
        + 3
        + factor_max
        + factor_exp
    )

    recurrent_C = 2 * d_qk * d_v + 2 * chunk_size * d_qk * d_v + chunk_size * d_qk
    recurrent_n = 2 * d_qk + 2 * chunk_size * d_qk

    flops_fw_recurrent = (
        float(num_heads)
        * float(num_chunks)
        * (recurrent_gates + recurrent_C + recurrent_n)
    )

    ## Parallel computation of the intra outputs
    parallel_gates = chunk_size * chunk_size * (
        3 + factor_exp + factor_max
    ) + chunk_size * (1 + factor_max)
    parallel_intra_outputs = (
        2 * chunk_size * chunk_size * (d_qk + d_v) + 3 * chunk_size * chunk_size
    )

    flops_fw_parallel_intra = (
        float(num_heads) * float(num_chunks) * (parallel_gates + parallel_intra_outputs)
    )

    if chunk_size > 64:
        # we assume TFLA kernels for chunk sizes > 64
        flops_fw_parallel_intra *= causal_factor

    # add the cumulative forgetgate flops (on these flops no causal factor is applied)
    cumulative_forget_gates = float(num_heads) * float(
        num_chunks
    ) * (0.5 * chunk_size * (chunk_size + 1) + chunk_size * (factor_log + factor_sig))
    flops_fw_parallel_intra += cumulative_forget_gates

    ## Parallel computation of the inter outputs
    parallel_inter_outputs = 2 * chunk_size * d_qk * d_v + 3 * chunk_size * d_qk

    flops_fw_parallel_inter = (
        float(num_heads) * float(num_chunks) * parallel_inter_outputs
    )
    ## Combination of inter and intra outputs
    combination_outputs = 2 * chunk_size * d_v + chunk_size * (
        1 + factor_max + factor_abs + factor_exp
    )

    flops_fw_combination = float(num_heads) * float(num_chunks) * combination_outputs

    ## Total forward pass FLOPs
    total_flops = (
        flops_fw_recurrent
        + flops_fw_parallel_intra
        + flops_fw_parallel_inter
        + flops_fw_combination
    )

    total_flops_recurrent = flops_fw_recurrent
    total_flops_parallel = (
        flops_fw_parallel_intra + flops_fw_parallel_inter + flops_fw_combination
    )

    assert total_flops == total_flops_recurrent + total_flops_parallel
    return total_flops, total_flops_recurrent, total_flops_parallel
