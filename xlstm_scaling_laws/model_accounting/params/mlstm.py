from ..symbols import d_ff, d_hv, d_model, d_qk, n_blocks, n_headq, n_vocab

# embeddings + outnorm parameters
param_emb = n_vocab * d_model
param_outnorm = d_model

# mlstm parameters
param_mlstm_prenorm = d_model
param_mlstm_qkv = d_model * n_headq * (2 * d_qk + d_hv)
param_mlstm_if_gates = 2 * d_model * n_headq + 2 * n_headq
param_mlstm_o_gate = d_model * n_headq * d_hv
param_mlstm_outnorm = n_headq * d_hv
param_mlstm_outproj = d_model * n_headq * d_hv

param_mlstm_layer = (
    param_mlstm_prenorm
    + param_mlstm_qkv
    + param_mlstm_if_gates
    + param_mlstm_o_gate
    + param_mlstm_outnorm
    + param_mlstm_outproj
)

# ffn parameters
param_ffn_prenorm = d_model
param_mlps = 3 * d_model * d_ff

param_ffn = param_ffn_prenorm + param_mlps

# total parameters
param_mlstm_model_blocks = n_blocks * (param_mlstm_layer + param_ffn) + param_outnorm
param_mlstm_model_noembed = param_mlstm_model_blocks + param_emb
param_mlstm_model_withembed = param_mlstm_model_noembed + param_emb
