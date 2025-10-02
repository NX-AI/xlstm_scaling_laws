import sympy as sp

from ..symbols import (
    T,
    T_gen,
    T_pre,
    bytes_act,
    bytes_qkv,
    d_hv,
    d_qk,
    n_headkv,
    n_headq,
)

# Note: it might be that the keys and values have to be loaded again for each query head,
# since they might not be shared across the query heads in SRAM (shared memory).
# In this case we would need to set n_headq = n_headkv.
memop_attn_prefill = T * (d_qk + d_hv) * (n_headq + n_headkv) * bytes_qkv

memop_attn_generation = (
    T_gen * (d_qk + d_hv) * n_headkv # write keys and values for T_gen tokens
    + (T_pre * T_gen + 0.5 * T_gen * (T_gen - 1)) * (d_qk + d_hv) * n_headkv # read keys and values for T_gen tokens
) * bytes_qkv

subs_dict = {bytes_qkv: bytes_act}

fn_memops_attention_prefill = sp.lambdify(
    (T, d_qk, d_hv, n_headq, n_headkv, bytes_act),
    memop_attn_prefill.subs(subs_dict),
    modules=["numpy"],
)

fn_memops_attention_generation = sp.lambdify(
    (T_pre, T_gen, d_qk, d_hv, n_headq, n_headkv, bytes_act),
    memop_attn_generation.subs(subs_dict),
    modules=["numpy"],
)


def count_memops_attention_prefill(
    batch_size,
    seq_len,
    d_qk,
    d_hv,
    n_headq,
    n_headkv,
    bytes_act,
):
    """Count the memory operations for the attention prefill."""
    return batch_size * fn_memops_attention_prefill(
        seq_len, d_qk, d_hv, n_headq, n_headkv, bytes_act
    )


def count_memops_attention_generation(
    batch_size,
    seq_len_pre,
    seq_len_gen,
    d_qk,
    d_hv,
    n_headq,
    n_headkv,
    bytes_act,
):
    """Count the memory operations for the attention generation."""
    return batch_size * fn_memops_attention_generation(
        seq_len_pre, seq_len_gen, d_qk, d_hv, n_headq, n_headkv, bytes_act
    )
