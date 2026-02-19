# In this file we count all flops for the mLSTM backbone that are
# sequence length independent, i.e. these flops are computed for every token in the sequence.
import sympy as sp

from ..symbols import (
    F_norm,
    F_sig,
    F_skip,
    F_swish,
    d_ff,
    d_hv,
    d_model,
    d_qk,
    n_blocks,
    n_headq,
    n_vocab,
)

# mlstm layer flops
flop_mlstm_prenorm_skip = d_model * (F_skip + F_norm)
flop_mlstm_qkv = 2 * d_model * n_headq * (2 * d_qk + d_hv)
flop_mlstm_if_gates = 2 * d_model * n_headq + 2 * n_headq
# flops for the mLSTM cell
flop_mlstm_o_gate = 2 * d_model * n_headq * d_hv + n_headq * d_hv * F_sig
flop_mlstm_outnorm = n_headq * d_hv * F_norm
flop_mlstm_outproj = 2 * d_model * n_headq * d_hv

flop_mlstm_mlstm_layer = (
    flop_mlstm_prenorm_skip
    + flop_mlstm_qkv
    + flop_mlstm_if_gates
    + flop_mlstm_o_gate
    + flop_mlstm_outnorm
    + flop_mlstm_outproj
)

# ffn layer flops
flop_ffn_prenorm_skip = d_model * (F_skip + F_norm)
flop_ffn_mlps = 6 * d_model * d_ff
flop_ffn_act = d_ff * (1 + F_swish)

flop_ffn = flop_ffn_prenorm_skip + flop_ffn_mlps + flop_ffn_act

# final linear layer / unembedding + outnorm
flop_final_unembed = 2 * d_model * n_vocab
flop_outnorm = d_model * F_norm

flop_mlstm_backbone_blocks = n_blocks * (flop_mlstm_mlstm_layer + flop_ffn)
flop_mlstm_backbone_withfinallinear = (
    flop_mlstm_backbone_blocks + flop_outnorm + flop_final_unembed
)

# we set all the flop factors
subs_dict = {F_norm: 3, F_swish: 1, F_skip: 1, F_sig: 1}

fn_flop_mlstm_backbone_withfinallinear = sp.lambdify(
    (n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv),
    flop_mlstm_backbone_withfinallinear.subs(subs_dict).subs(d_model, d_hv * n_headq),
    modules=["numpy"],
)
fn_flop_mlstm_backbone = sp.lambdify(
    (n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv),
    flop_mlstm_backbone_blocks.subs(subs_dict).subs(d_model, d_hv * n_headq),
    modules=["numpy"],
)


def count_flops_mlstm_backbone(
    n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv, with_unembed=True
):
    if with_unembed:
        return fn_flop_mlstm_backbone_withfinallinear(
            n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv
        )
    else:
        return fn_flop_mlstm_backbone(n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv)
