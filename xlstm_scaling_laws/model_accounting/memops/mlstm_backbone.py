import sympy as sp

from ..symbols import (
    B,
    T,
    bytes_act,
    bytes_actnorm,
    bytes_if,
    bytes_qkv,
    bytes_w,
    d_ff,
    d_hv,
    d_model,
    d_qk,
    n_blocks,
    n_headq,
    n_vocab,
)

####### mLSTM backbone activation memops ######
memops_mlstm_emb_act = B * T * d_model * bytes_w

## mlstm layer memops activations
memops_mlstm_prenorm_act = B * T * d_model * bytes_actnorm
memops_mlstm_qkv_act = B * T * (d_model + n_headq * (2 * d_qk + d_hv)) * bytes_qkv
memops_mlstm_if_act = 2 * B * T * (d_model + n_headq) * bytes_if
# memops for the mlstm cell
memops_mlstm_ogate_act = B * T * (d_model + n_headq * d_hv) * bytes_act
memops_mlstm_outnorm_act = B * T * n_headq * d_hv * bytes_actnorm
memops_mlstm_outproj_act = B * T * (d_model + n_headq * d_hv) * bytes_act

## mlstm ffn memops activations
memops_mlstm_ffn_prenorm_act = B * T * d_model * bytes_actnorm
memops_mlstm_ffn_mlp_act = 3 * B * T * (d_model + d_ff) * bytes_act

## memops lm head activations
memops_mlstm_head_outnorm_act = B * T * d_model * bytes_actnorm
memops_mlstm_last_linear_act = B * T * (d_model + n_vocab) * bytes_act

###### mLSTM backbone weight memops ######
## mlstm layer memops weights
memops_mlstm_prenorm_weights = d_model * bytes_w
memops_mlstm_qkv_weights = d_model * n_headq * (2 * d_qk + d_hv) * bytes_w
memops_mlstm_if_weights = (2 * d_model * n_headq + 2 * n_headq) * bytes_w
memops_mlstm_ogate_weights = (d_model * n_headq * d_hv) * bytes_w
memops_mlstm_outnorm_weights = n_headq * d_hv * bytes_w
memops_mlstm_outproj_weights = d_model * n_headq * d_hv * bytes_w

## mlstm ffn memops weights
memops_mlstm_ffn_prenorm_weights = d_model * bytes_w
memops_mlstm_ffn_mlp_weights = 3 * d_model * d_ff * bytes_w

## memops lm head weights
memops_mlstm_head_outnorm_weights = d_model * bytes_w
memops_mlstm_last_linear_weights = d_model * n_vocab * bytes_w

####### mLSTM backbone total memops ######

memops_mlstm_block_act = (
    +memops_mlstm_prenorm_act
    + memops_mlstm_qkv_act
    + memops_mlstm_if_act
    + memops_mlstm_ogate_act
    + memops_mlstm_outnorm_act
    + memops_mlstm_outproj_act
    + memops_mlstm_ffn_prenorm_act
    + memops_mlstm_ffn_mlp_act
)
memops_mlstm_block_weights = (
    memops_mlstm_prenorm_weights
    + memops_mlstm_qkv_weights
    + memops_mlstm_if_weights
    + memops_mlstm_ogate_weights
    + memops_mlstm_outnorm_weights
    + memops_mlstm_outproj_weights
    + memops_mlstm_ffn_prenorm_weights
    + memops_mlstm_ffn_mlp_weights
)

memops_mlstm_backbone_nounembed = memops_mlstm_emb_act + n_blocks * (
    memops_mlstm_block_act + memops_mlstm_block_weights
)

memops_mlstm_lmhead = (
    +memops_mlstm_head_outnorm_act
    + memops_mlstm_last_linear_act
    + memops_mlstm_head_outnorm_weights
    + memops_mlstm_last_linear_weights
)

memops_mlstm_backbone_withunembed = (
    memops_mlstm_backbone_nounembed + memops_mlstm_lmhead
)

subs_dict = {bytes_actnorm: bytes_act, bytes_if: bytes_act, bytes_qkv: bytes_act}


fn_memops_mlstm_backbone_nounembed = sp.lambdify(
    (B, T, n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv, bytes_act, bytes_w),
    memops_mlstm_backbone_nounembed.subs(subs_dict).subs(d_model, d_hv * n_headq),
    modules=["numpy"],
)


fn_memops_mlstm_lmhead = sp.lambdify(
    (
        B,
        T,
        n_vocab,
        d_hv,
        n_headq,
        bytes_act,
        bytes_w,
    ),
    memops_mlstm_lmhead.subs(subs_dict).subs(d_model, d_hv * n_headq),
    modules=["numpy"],
)


def count_memops_mlstm_backbone(
    batch_size,
    seq_len,
    n_vocab,
    n_blocks,
    d_ff,
    n_headq,
    d_qk,
    d_hv,
    bytes_act,
    bytes_w,
    with_unembed=True,
):
    """Count the memory operations for the mLSTM backbone."""
    memops_mlstm_backbone_nounembed = fn_memops_mlstm_backbone_nounembed(
        batch_size,
        seq_len,
        n_vocab,
        n_blocks,
        d_ff,
        n_headq,
        d_qk,
        d_hv,
        bytes_act,
        bytes_w,
    )

    lm_head_seq_len = seq_len if with_unembed else 1
    memops_mlstm_lmhead = fn_memops_mlstm_lmhead(
        batch_size, lm_head_seq_len, n_vocab, d_hv, n_headq, bytes_act, bytes_w
    )
    return memops_mlstm_backbone_nounembed + memops_mlstm_lmhead
