import sympy as sp

### Define the symbols ###

### DIMENSIONS ###

# batch dims
B, n_chunk = sp.symbols("B n_chunk", integer=True, positive=True)

# sequence dims
S, T, L = sp.symbols("S T L", integer=True, positive=True)
# generation and prefill sequence length
T_pre, T_gen = sp.symbols("T_pre T_gen", integer=True, positive=True)

# number of heads
(
    n_headq,
    n_headkv,
) = sp.symbols("n_headq n_headkv", integer=True, positive=True)

# head dims
d_qk = sp.symbols("d_qk", integer=True, positive=True)
d_hv = sp.symbols("d_hv", integer=True, positive=True)
# dimension factor
p_qk = sp.symbols("p_qk")

# model size
d_model = sp.symbols("d_model", integer=True, positive=True)
n_blocks = sp.symbols("n_blocks", integer=True, positive=True)
d_ff = sp.symbols("d_ffn", integer=True, positive=True)
n_vocab = sp.symbols("n_vocab", integer=True, positive=True)

### DEPENDENCIES ###

n_chunk = T / L
# d_model = n_headq * d_hv # defining this here causes issues with lambdify

### FLOPS ###

# flop factors
F_exp, F_log, F_sig, F_max, F_mask, F_abs = sp.symbols(
    "F_exp F_log F_sig F_max F_mask F_abs", real=True, positive=True
)
# causal factor
F_causal = sp.symbols("F_causal", real=True, positive=True)

# flop factors model
F_skip, F_norm = sp.symbols("F_skip F_norm", real=True, positive=True)
F_swish = sp.symbols("F_swish", real=True, positive=True)
F_softmax = sp.symbols("F_softmax", real=True, positive=True)

### MEMOPS ###

# number of bytes
bytes_qkv, bytes_Cmn, bytes_if = sp.symbols(
    "bytes_qkv bytes_Cmn bytes_if", integer=True, positive=True
)

bytes_act, bytes_w, bytes_actnorm = sp.symbols(
    "bytes_act bytes_w bytes_actnorm", integer=True, positive=True
)
