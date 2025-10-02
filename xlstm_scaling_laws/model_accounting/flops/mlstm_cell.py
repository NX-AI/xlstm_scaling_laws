import sympy as sp

from ..symbols import (
    F_abs,
    F_causal,
    F_exp,
    F_log,
    F_mask,
    F_max,
    F_sig,
    L,
    T,
    d_hv,
    d_qk,
    n_chunk,
    n_headq,
)

######################################
### Chunkwise-parallel formulation ###
######################################

# Note: These are flops per chunk, not for the full sequence.

## mLSTMexp ##

# recurrent computation of the inter chunk states
flop_cwp_exp_gates = (
    2 * L + 0.5 * L * (L + 1) + L * (1 + F_exp + F_log + F_sig) + 3 + F_max + F_exp
)
flop_cwp_exp_numerator = 2 * d_qk * d_hv + 2 * L * d_qk * d_hv + L * d_qk
flop_cwp_exp_denominator = 2 * d_qk + 2 * L * d_qk
flop_cwp_exp_reccomp_inter = (
    flop_cwp_exp_gates + flop_cwp_exp_numerator + flop_cwp_exp_denominator
)

# parallel computation of the intra chunk outputs
flop_cwp_exp_cum_fgates = 0.5 * L * (L + 1) + L * (F_log + F_sig)
flop_cwp_exp_gate_matrix = F_causal * (L**2 * (3 + F_exp + F_max) + L * (1 + F_max))
flop_cwp_exp_intra_outputs = F_causal * (2 * L**2 * (d_qk + d_hv) + 3 * L**2)
flop_cwp_exp_parcomp_intra = (
    flop_cwp_exp_cum_fgates + flop_cwp_exp_gate_matrix + flop_cwp_exp_intra_outputs
)

# parallel computation of the inter chunk outputs
flop_cwp_exp_inter_outputs = 2 * L * d_qk * d_hv + 3 * L * d_qk

# combination of inter and intra chunk outputs
flop_cwp_exp_output_comb = 2 * L * d_hv + L * (1 + F_max + F_abs + F_exp)


flop_cwp_exp_total = (
    flop_cwp_exp_reccomp_inter
    + flop_cwp_exp_parcomp_intra
    + flop_cwp_exp_inter_outputs
    + flop_cwp_exp_output_comb
)


######################################
### Recurrent formulation ###
######################################

# Note: These are flops per step, not for the full sequence.

## mLSTMexp ##

flop_rec_exp_gates = 4 + 2 * F_exp + F_log + F_sig + F_max
flop_rec_exp_cell_update = 4 * d_qk * d_hv
flop_rec_exp_denom = 6 * d_qk + d_hv + 1 + F_abs + F_max
flop_rec_exp_output = 2 * d_hv * d_qk + d_qk

flop_rec_exp_total = (
    flop_rec_exp_gates
    + flop_rec_exp_cell_update
    + flop_rec_exp_denom
    + flop_rec_exp_output
)


######################################
### Parallel formulation ###
######################################

## mLSTMexp ##
flop_par_exp_cum_fgates = 0.5 * T * (T + 1) + T * (F_log + F_sig)
flop_par_exp_gate_matrix = T**2 * (3 + F_exp + F_max + F_mask)
flop_par_exp_attn_logits = F_causal * (2 * T**2 * d_qk + 2 * T**2)
flop_par_exp_normalization = F_causal * (T**2 * (3 + F_abs) + T * (F_exp + F_max))
flop_par_exp_outputs = F_causal * 2 * T**2 * d_hv

flop_par_exp_total = (
    flop_par_exp_cum_fgates
    + flop_par_exp_gate_matrix
    + flop_par_exp_attn_logits
    + flop_par_exp_normalization
    + flop_par_exp_outputs
)


###################################
### Total flops accounting ###
###################################

# We multiply the per-step or per-chunk flops by the sequence length or number of chunks.
# We multiply the per-head flops by the number of heads.

flop_mlstm_exp_rec = n_headq * T * flop_rec_exp_total
flop_mlstm_exp_cwp = n_headq * n_chunk * flop_cwp_exp_total
flop_mlstm_exp_par = n_headq * flop_par_exp_total

# we set all the flop factors to 1
subs_dict = {F_exp: 1, F_log: 1, F_sig: 1, F_max: 1, F_mask: 1, F_abs: 1}

# simplify and substitute the flop factors
simpl_flop_rec_exp_total = sp.collect(flop_mlstm_exp_rec.subs(subs_dict), T)
simpl_flop_cwp_exp_total = sp.collect(
    sp.cancel(sp.collect(flop_mlstm_exp_cwp.subs(subs_dict), L)), L
)
simpl_flop_par_exp_total = sp.collect(flop_mlstm_exp_par.subs(subs_dict), T)

# lamdify the expressions
fn_flop_rec_exp = sp.lambdify(
    (T, n_headq, d_qk, d_hv), simpl_flop_rec_exp_total, modules=["numpy"]
)
fn_flop_cwp_exp = sp.lambdify(
    (T, n_headq, d_qk, d_hv, F_causal, L), simpl_flop_cwp_exp_total, modules=["numpy"]
)
fn_flop_par_exp = sp.lambdify(
    (T, n_headq, d_qk, d_hv, F_causal), simpl_flop_par_exp_total, modules=["numpy"]
)

def count_flops_mlstmexp_recurrent(seq_len, n_headq, d_qk, d_hv, **kwargs):
    return fn_flop_rec_exp(seq_len, n_headq, d_qk, d_hv)


def count_flops_mlstmexp_chunkwise_parallel(
    seq_len, n_headq, d_qk, d_hv, factor_causal, chunk_size, **kwargs
):
    return fn_flop_cwp_exp(seq_len, n_headq, d_qk, d_hv, factor_causal, chunk_size)


def count_flops_mlstmexp_parallel(seq_len, n_headq, d_qk, d_hv, factor_causal, **kwargs):
    return fn_flop_par_exp(seq_len, n_headq, d_qk, d_hv, factor_causal)
