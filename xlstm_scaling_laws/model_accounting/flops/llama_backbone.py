# In this file we count all flops for the Llama backbone that are 
# sequence length independent, i.e. these flops are computed for every token in the sequence.

import sympy as sp

from ..symbols import (
    F_norm,
    F_skip,
    F_swish,
    d_ff,
    d_hv,
    d_model,
    d_qk,
    n_blocks,
    n_headkv,
    n_headq,
    n_vocab,
)

# attention layer flops
flop_attn_prenorm_skip = d_model * (F_skip + F_norm)
flop_attn_qkv = 2 * d_model * (d_qk * n_headq + d_qk * n_headkv + d_hv * n_headkv)
# flops for the attention operation
flop_outproj = 2 * d_model * n_headkv * d_hv

flop_attn_layer = (
    flop_attn_prenorm_skip
    + flop_attn_qkv
    + flop_outproj
)

# ffn layer flops
flop_ffn_prenorm_skip = d_model * (F_skip + F_norm)
flop_ffn_mlps = 6 * d_model * d_ff
flop_ffn_act = d_ff * (1 + F_swish)

flop_ffn = (
    flop_ffn_prenorm_skip
    + flop_ffn_mlps
    + flop_ffn_act
)

# final linear layer / unembedding + outnorm
flop_final_unembed = 2 * d_model * n_vocab
flop_outnorm = d_model * F_norm

flop_llama_backbone_blocks = n_blocks * (
    flop_attn_layer + flop_ffn
)
flop_llama_backbone_withfinallinear = (
    flop_llama_backbone_blocks + flop_outnorm + flop_final_unembed
)

# we set all the flop factors
subs_dict = {F_norm: 3, F_swish: 1, F_skip: 1, d_model: d_qk * n_headq}

fn_flop_llama_backbone_withfinallinear = sp.lambdify(
    (n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv, n_headkv),
    flop_llama_backbone_withfinallinear.subs(subs_dict),
    modules=["numpy"],
)

fn_flop_llama_backbone = sp.lambdify(
    (n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv, n_headkv),
    flop_llama_backbone_blocks.subs(subs_dict),
    modules=["numpy"],
)

def count_flops_llama_backbone(
    n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv, n_headkv, with_unembed=True
):
    """
    Count the flops for the Llama backbone.
    
    Parameters:
        n_vocab (int): Vocabulary size.
        n_blocks (int): Number of blocks.
        d_ff (int): Feed-forward dimension.
        n_headq (int): Number of query heads.
        d_qk (int): Query key dimension.
        d_hv (int): Head value dimension.
        n_headkv (int): Number of key-value heads.
        with_unembed (bool): Whether to include the final unembedding layer flops.

    Returns:
        int: Total number of flops.
    """
    if with_unembed:
        return fn_flop_llama_backbone_withfinallinear(
            n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv, n_headkv
        )
    else:
        return fn_flop_llama_backbone(
            n_vocab, n_blocks, d_ff, n_headq, d_qk, d_hv, n_headkv
        )