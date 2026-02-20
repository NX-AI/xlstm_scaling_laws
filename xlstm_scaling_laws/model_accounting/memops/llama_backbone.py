import sympy as sp

from ..symbols import (
    B,
    T,
    bytes_act,
    bytes_actnorm,
    bytes_qkv,
    bytes_w,
    d_ff,
    d_hv,
    d_model,
    d_qk,
    n_blocks,
    n_headkv,
    n_headq,
    n_vocab,
)

####### llama backbone activation memops ######
memops_llama_emb_act = B * T * d_model * bytes_act

## llama layer memops activations
memops_llama_prenorm_act = B * T * d_model * bytes_actnorm
# QKV projections
memops_llama_qkv_act = (
    B * T * (d_model + d_qk * n_headq + d_qk * n_headkv + d_hv * n_headkv) * bytes_qkv
)
# memops for the attention operation
memops_llama_outproj_act = B * T * (d_model + n_headq * d_hv) * bytes_act

## llama ffn memops activations
memops_llama_ffn_prenorm_act = B * T * d_model * bytes_actnorm
memops_llama_ffn_mlp_act = 3 * B * T * (d_model + d_ff) * bytes_act

## memops lm head activations
memops_llama_head_outnorm_act = B * T * d_model * bytes_actnorm
memops_llama_last_linear_act = B * T * (d_model + n_vocab) * bytes_act

###### llama backbone weight memops ######
## llama layer memops weights
memops_llama_prenorm_weights = d_model * bytes_w
memops_llama_qkv_weights = (
    d_model * (d_qk * n_headq + (d_qk + d_hv) * n_headkv) * bytes_w
)
memops_llama_outproj_weights = d_model * n_headq * d_hv * bytes_w

## llama ffn memops weights
memops_llama_ffn_prenorm_weights = d_model * bytes_w
memops_llama_ffn_mlp_weights = 3 * d_model * d_ff * bytes_w

## memops lm head weights
memops_llama_head_outnorm_weights = d_model * bytes_w
memops_llama_last_linear_weights = d_model * n_vocab * bytes_w

####### llama backbone total memops ######

memops_llama_block_act = (
    +memops_llama_prenorm_act
    + memops_llama_qkv_act
    + memops_llama_outproj_act
    + memops_llama_ffn_prenorm_act
    + memops_llama_ffn_mlp_act
)
memops_llama_block_weights = (
    memops_llama_prenorm_weights
    + memops_llama_qkv_weights
    + memops_llama_outproj_weights
    + memops_llama_ffn_prenorm_weights
    + memops_llama_ffn_mlp_weights
)

memops_llama_backbone_nounembed = memops_llama_emb_act + n_blocks * (
    memops_llama_block_act + memops_llama_block_weights
)

memops_llama_lmhead = (
    memops_llama_head_outnorm_act
    + memops_llama_last_linear_act
    + memops_llama_head_outnorm_weights
    + memops_llama_last_linear_weights
)

memops_llama_backbone_withunembed = (
    memops_llama_backbone_nounembed + memops_llama_lmhead
)


subs_dict = {bytes_actnorm: bytes_act, bytes_qkv: bytes_act}


fn_memops_llama_backbone_nounembed = sp.lambdify(
    (B, T, n_vocab, n_blocks, d_ff, n_headq, n_headkv, d_qk, d_hv, bytes_act, bytes_w),
    memops_llama_backbone_nounembed.subs(subs_dict).subs(d_model, d_hv * n_headq),
    modules=["numpy"],
)

fn_memops_llama_lmhead = sp.lambdify(
    (
        B,
        T,
        n_vocab,
        d_hv,
        n_headq,
        bytes_act,
        bytes_w,
    ),
    memops_llama_lmhead.subs(subs_dict).subs(d_model, d_hv * n_headq),
    modules=["numpy"],
)


def count_memops_llama_backbone(
    batch_size,
    seq_len,
    n_vocab,
    n_blocks,
    d_ff,
    n_headq,
    n_headkv,
    d_qk,
    d_hv,
    bytes_act,
    bytes_w,
    with_unembed=True,
):
    """Count the memory operations for the Llama backbone."""
    memops_llama_backbone_nounembed = fn_memops_llama_backbone_nounembed(
        batch_size,
        seq_len,
        n_vocab,
        n_blocks,
        d_ff,
        n_headq,
        n_headkv,
        d_qk,
        d_hv,
        bytes_act,
        bytes_w,
    )

    lm_head_seq_len = seq_len if with_unembed else 1
    memops_llama_lmhead = fn_memops_llama_lmhead(
        batch_size, lm_head_seq_len, n_vocab, d_hv, n_headq, bytes_act, bytes_w
    )
    return memops_llama_backbone_nounembed + memops_llama_lmhead
