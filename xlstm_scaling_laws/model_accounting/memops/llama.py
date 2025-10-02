from .attention_op import (
    count_memops_attention_generation,
    count_memops_attention_prefill,
)
from .llama_backbone import count_memops_llama_backbone


def count_llama_model_memops__prefill(
    batch_size,
    n_vocab,
    n_blocks,
    d_ff,
    n_headq,
    n_headkv,
    d_qk,
    d_hv,
    seq_len,
    bytes_act,
    bytes_w,
    with_unembed=True,
):
    memops_backbone = count_memops_llama_backbone(
        batch_size=batch_size,
        seq_len=seq_len,
        n_vocab=n_vocab,
        n_blocks=n_blocks,
        d_ff=d_ff,
        n_headq=n_headq,
        n_headkv=n_headkv,
        d_qk=d_qk,
        d_hv=d_hv,
        bytes_act=bytes_act,
        bytes_w=bytes_w,
        with_unembed=with_unembed,
    )

    memops_attn_prefill = count_memops_attention_prefill(
        batch_size=batch_size,
        seq_len=seq_len,
        d_qk=d_qk,
        d_hv=d_hv,
        n_headq=n_headq,
        n_headkv=n_headkv,
        bytes_act=bytes_act,
    )

    # total memory operations
    total_memops = memops_backbone + n_blocks * memops_attn_prefill
    return total_memops


def count_llama_model_memops__generation(
    batch_size,
    n_vocab,
    n_blocks,
    d_ff,
    n_headq,
    n_headkv,
    d_qk,
    d_hv,
    seq_len_pre,
    seq_len_gen,
    bytes_act,
    bytes_w,
    with_unembed=True,
):
    memops_backbone = count_memops_llama_backbone(
        batch_size=batch_size,
        seq_len=seq_len_gen,
        n_vocab=n_vocab,
        n_blocks=n_blocks,
        d_ff=d_ff,
        n_headq=n_headq,
        n_headkv=n_headkv,
        d_qk=d_qk,
        d_hv=d_hv,
        bytes_act=bytes_act,
        bytes_w=bytes_w,
        with_unembed=with_unembed,
    )

    memops_attn_generation = count_memops_attention_generation(
        batch_size=batch_size,
        seq_len_pre=seq_len_pre,
        seq_len_gen=seq_len_gen,
        d_qk=d_qk,
        d_hv=d_hv,
        n_headq=n_headq,
        n_headkv=n_headkv,
        bytes_act=bytes_act,
    )

    # total memory operations
    total_memops = memops_backbone + n_blocks * memops_attn_generation
    return total_memops / seq_len_gen
