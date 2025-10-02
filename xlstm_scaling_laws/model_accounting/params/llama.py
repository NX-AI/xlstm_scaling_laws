from ..symbols import d_ff, d_hv, d_model, d_qk, n_blocks, n_headkv, n_headq, n_vocab

# embeddings + outnorm parameters
param_emb = n_vocab * d_model
param_outnorm = d_model

# attention parameters
param_att_prenorm = d_model
param_att_qkv = d_model * (d_qk*n_headq + (d_qk + d_hv)*n_headkv)
param_att_outproj = d_model * d_hv * n_headq

param_att_layer = (
    param_att_prenorm + param_att_qkv + param_att_outproj
)

# ffn parameters
param_ffn_prenorm = d_model
param_mlps = 3 * d_model * d_ff

param_ffn = param_ffn_prenorm + param_mlps

# total parameters
param_llama_model_blocks = (
    n_blocks * (param_att_layer + param_ffn) + param_outnorm
)
param_llama_model_noembed = (
    param_llama_model_blocks + param_emb
)
param_llama_model_withembed = (
    param_llama_model_noembed + param_emb
)