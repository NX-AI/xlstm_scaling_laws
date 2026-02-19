# In this file we count the combined flops for the mLSTM backbone and the mLSTM cell.
from .mlstm_backbone import count_flops_mlstm_backbone
from .mlstm_cell import (
    count_flops_mlstmexp_chunkwise_parallel,
    count_flops_mlstmexp_recurrent,
)


def count_mlstm_model_flops__chunkwise_parallel(
    batch_size,
    n_vocab,
    n_blocks,
    d_ff,
    n_headq,
    d_qk,
    d_hv,  # backbone args
    seq_len,
    factor_causal,
    chunk_size,  # cell args
    with_unembed=True,  # whether to count the final linear layer
    return_mlstm_cell_flops=False,  # whether to return the mLSTM cell flops separately
    **kwargs,  # additional arguments (not used here, but can be passed for compatibility)
):
    """Count the flops for the mLSTM model in chunkwise parallel mode
    for one sequence.

    Note: For training flops we would need to set with_unembed=True, but for inference
    we can set it to False to avoid counting the final linear layer.
    """
    # print("batch_size", batch_size, "n_vocab", n_vocab, "n_blocks", n_blocks,
    #       "d_ff", d_ff, "n_headq", n_headq, "d_qk", d_qk, "d_hv", d_hv,
    #       "seq_len", seq_len, "factor_causal", factor_causal,
    #       "chunk_size", chunk_size,
    #       "with_unembed", with_unembed,)
    flops_backbone = count_flops_mlstm_backbone(
        n_vocab=n_vocab,
        n_blocks=n_blocks,
        d_ff=d_ff,
        n_headq=n_headq,
        d_qk=d_qk,
        d_hv=d_hv,
        with_unembed=with_unembed,
    )

    flops_cell = count_flops_mlstmexp_chunkwise_parallel(
        seq_len=seq_len,
        n_headq=n_headq,
        d_qk=d_qk,
        d_hv=d_hv,
        factor_causal=factor_causal,
        chunk_size=chunk_size,
    )
    # total flops
    total_flops = seq_len * flops_backbone + n_blocks * flops_cell

    # during inference we actually compute the output logits
    # for the very last token in the sequence,
    # we add these flops in case with_unembed is False
    if not with_unembed:
        total_flops += 2 * n_headq * d_hv * n_vocab
    # print("flops", total_flops, "bs", batch_size)
    total_flops *= float(batch_size)
    if return_mlstm_cell_flops:
        return total_flops, n_blocks * flops_cell
    else:
        return total_flops


def count_mlstm_model_flops__recurrent(
    batch_size, n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv, seq_len, **kwargs
):
    flops_backbone = count_flops_mlstm_backbone(
        n_vocab=n_vocab,
        n_blocks=n_blocks,
        d_ff=d_ff,
        n_headq=n_headq,
        d_qk=d_qk,
        d_hv=d_hv,
        with_unembed=True,
    )

    flops_cell = count_flops_mlstmexp_recurrent(
        seq_len=seq_len,
        n_headq=n_headq,
        d_qk=d_qk,
        d_hv=d_hv,
    )
    # total flops
    total_flops = batch_size * (seq_len * flops_backbone + n_blocks * flops_cell)
    return total_flops
