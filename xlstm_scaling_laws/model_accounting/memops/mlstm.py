from .mlstm_backbone import count_memops_mlstm_backbone
from .mlstm_cell import (
    count_memops_mlstmexp_chunkwise_parallel,
    count_memops_mlstmexp_recurrent,
)


def count_mlstm_model_memops__chunkwise_parallel(
    batch_size,
    n_vocab,
    n_blocks,
    d_ff,
    n_headq,
    d_qk,
    d_hv,
    seq_len,
    chunk_size,
    bytes_act,
    bytes_Cmn,
    bytes_w,
    with_unembed=True,
    **kwargs,
):
    memops_backbone = count_memops_mlstm_backbone(
        batch_size=batch_size,
        seq_len=seq_len,
        n_vocab=n_vocab,
        n_blocks=n_blocks,
        d_ff=d_ff,
        n_headq=n_headq,
        d_qk=d_qk,
        d_hv=d_hv,
        bytes_act=bytes_act,
        bytes_w=bytes_w,
        with_unembed=with_unembed,
    )

    memops_cell = count_memops_mlstmexp_chunkwise_parallel(
        batch_size=batch_size,
        seq_len=seq_len,
        n_headq=n_headq,
        d_qk=d_qk,
        d_hv=d_hv,
        chunk_size=chunk_size,
        bytes_act=bytes_act,
        bytes_Cmn=bytes_Cmn,
    )

    # total memory operations
    total_memops = memops_backbone + n_blocks * memops_cell
    return total_memops


def count_mlstm_model_memops__recurrent(
    batch_size,
    n_vocab,
    n_blocks,
    d_ff,
    n_headq,
    d_qk,
    d_hv,
    seq_len,
    bytes_act,
    bytes_Cmn,
    bytes_w,
    with_unembed=True,
    **kwargs,
):
    memops_backbone = count_memops_mlstm_backbone(
        batch_size=batch_size,
        seq_len=seq_len,
        n_vocab=n_vocab,
        n_blocks=n_blocks,
        d_ff=d_ff,
        n_headq=n_headq,
        d_qk=d_qk,
        d_hv=d_hv,
        bytes_act=bytes_act,
        bytes_w=bytes_w,
        with_unembed=with_unembed,
    )

    memops_cell = count_memops_mlstmexp_recurrent(
        batch_size=batch_size,
        seq_len=seq_len,
        n_headq=n_headq,
        d_qk=d_qk,
        d_hv=d_hv,
        bytes_act=bytes_act,
        bytes_Cmn=bytes_Cmn,
    )

    # total memory operations
    total_memops = memops_backbone + n_blocks * memops_cell
    return total_memops
