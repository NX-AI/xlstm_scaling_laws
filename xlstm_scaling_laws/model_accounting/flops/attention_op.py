import sympy as sp

from ..symbols import F_causal, F_softmax, S, T, T_gen, T_pre, d_hv, d_qk, n_headq

### generic attention op flops for prefill and generation

flop_attn_logits = 2 * S * T * d_qk * F_causal
flop_attn_scores = S * T * F_softmax * F_causal
flop_attn_outputs = 2 * S * T * d_hv * F_causal

flop_attn_generic = flop_attn_logits + flop_attn_scores + flop_attn_outputs

###

# These are the flop counts for the attention operation for prefill and generation:

flop_attn_prefill = 2 * F_causal * T * T * n_headq * (d_qk + d_hv + 0.5 * F_softmax)

flop_attn_generation = (
    2
    * F_causal
    * n_headq
    * (d_qk + d_hv + 0.5 * F_softmax)
    * (T_pre * T_gen + 0.5 * T_gen * (T_gen + 1))
)

## We set all the flop factors
subs_dict = {F_causal: 0.65, F_softmax: 5.0}

fn_flop_attn_prefill = sp.lambdify(
    (T, d_qk, d_hv, n_headq),
    flop_attn_prefill.subs(subs_dict),
    modules=["numpy"],
)

fn_flop_attn_generation = sp.lambdify(
    (T_pre, T_gen, d_qk, d_hv, n_headq),
    flop_attn_generation.subs(subs_dict),
    modules=["numpy"],
)


def count_flops_attention_prefill(seq_len, d_qk, d_hv, n_headq, **kwargs):
    """
    Count the flops for the attention operation during prefill.
    """
    return fn_flop_attn_prefill(seq_len, d_qk, d_hv, n_headq)


def count_flops_attention_generation(
    seq_len_pre, seq_len_gen, d_qk, d_hv, n_headq, **kwargs
):
    """
    Count the flops for the attention operation during generation.
    """
    return fn_flop_attn_generation(seq_len_pre, seq_len_gen, d_qk, d_hv, n_headq)
