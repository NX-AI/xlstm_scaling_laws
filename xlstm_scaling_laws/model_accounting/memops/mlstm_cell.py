import sympy as sp

from ..symbols import (
    L,
    T,
    bytes_act,
    bytes_Cmn,
    bytes_if,
    bytes_qkv,
    d_hv,
    d_qk,
    n_chunk,
)

######################################
### Chunkwise-parallel formulation ###
######################################

## mLSTMexp ##
memop_cwp_exp_rec_load = L * (d_qk + d_hv) * bytes_qkv + 2 * L * bytes_if
memop_cwp_exp_rec_store = (d_qk * d_hv + d_qk + 1) * bytes_Cmn

memop_cwp_exp_par_load = (
    L * (2 * d_qk + d_hv) * bytes_qkv
    + 2 * L * bytes_if
    + (d_qk * d_hv + d_qk + 1) * bytes_Cmn
)
memop_cwp_exp_par_store = L * d_hv * bytes_qkv + 2 * L * bytes_Cmn

total_memop_cwp_exp = (
    memop_cwp_exp_rec_load
    + memop_cwp_exp_rec_store
    + memop_cwp_exp_par_load
    + memop_cwp_exp_par_store
)
total_memop_cwp_exp = sp.simplify(
    sp.collect(total_memop_cwp_exp, syms=[bytes_qkv, bytes_Cmn, bytes_if])
)

#######################################
### Recurrent formulation ###
#######################################

## mLSTMexp ##
memop_rec_exp_load = (
    (2 * d_qk + d_hv) * bytes_qkv + 2 * bytes_if + (d_qk * d_hv + d_qk + 1) * bytes_Cmn
)
memop_rec_exp_store = d_hv * bytes_qkv + (d_qk * d_hv + d_qk + 1) * bytes_Cmn

total_memop_rec_exp = memop_rec_exp_load + memop_rec_exp_store
total_memop_rec_exp = sp.simplify(
    sp.collect(total_memop_rec_exp, syms=[bytes_qkv, bytes_Cmn, bytes_if])
)


######################################
### Parallel formulation ###
######################################

## mLSTMexp ##
memop_par_exp_load = T * (2 * d_qk + d_hv) * bytes_qkv + 2 * T * bytes_if
memop_par_exp_store = T * d_hv * bytes_qkv + 2 * T * bytes_Cmn

total_memop_par_exp = memop_par_exp_load + memop_par_exp_store
total_memop_par_exp = sp.simplify(
    sp.collect(total_memop_par_exp, syms=[bytes_qkv, bytes_Cmn, bytes_if])
)

###################################################
### TOTAL Memory Operations for a full sequence ###
###################################################

## mLSTMexp ##
comp_memop_cwp_exp_total = n_chunk * total_memop_cwp_exp
comp_memop_par_exp_total = total_memop_par_exp
comp_memop_rec_exp_total = T * total_memop_rec_exp

subs_dict = {
    bytes_if: bytes_act,
    bytes_qkv: bytes_act,
}

fn_memops_mlstmexp_chunkwise_parallel = sp.lambdify(
    (L, T, d_qk, d_hv, bytes_act, bytes_Cmn),
    comp_memop_cwp_exp_total.subs(subs_dict).subs(n_chunk, T / L),
    modules=["numpy"],
)

fn_memops_mlstmexp_recurrent = sp.lambdify(
    (T, d_qk, d_hv, bytes_act, bytes_Cmn),
    comp_memop_rec_exp_total.subs(subs_dict),
    modules=["numpy"],
)


def count_memops_mlstmexp_chunkwise_parallel(
    batch_size, seq_len, n_headq, d_qk, d_hv, chunk_size, bytes_act, bytes_Cmn
):
    """
    Count memory operations for mLSTMexp in chunkwise-parallel formulation for a single sequence.
    """
    return (
        batch_size
        * n_headq
        * fn_memops_mlstmexp_chunkwise_parallel(
            chunk_size,
            seq_len,
            d_qk,
            d_hv,
            bytes_act,
            bytes_Cmn,
        )
    )


def count_memops_mlstmexp_recurrent(
    batch_size, seq_len, n_headq, d_qk, d_hv, bytes_act, bytes_Cmn
):
    """
    Memory operations for mLSTMexp in recurrent formulation for a single sequence.
    """
    return (
        batch_size
        * n_headq
        * fn_memops_mlstmexp_recurrent(seq_len, d_qk, d_hv, bytes_act, bytes_Cmn)
    )
