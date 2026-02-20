config_template_str_mlstm = """
# @package _global_
defaults:
  - /data@data_train.ds1: dclm_arrayrecord_train
  - /data@data_eval.ds1: dclm_arrayrecord_eval_preprocessed
  # - /data@data_eval.ds2: slimpajama_627B_arrayrecord_eval_preprocessed
  - override /parallel: mLSTMv1_7B # use fsdp #mLSTMv1_1.3B # use this for no FSDP (pure dp)
  - override /model: mLSTMv1_default
  - override /optimizer: adamw
  - override /scheduler: cosine_decay
  - override /hydra/launcher: slurm_launcher
  - _self_

# specify the deltas from the defaults:
task_name: sclaw_iso8
batch_size_per_device: 4
context_length: 8192
num_epochs: 1000
num_train_steps: 18_000
lr: LLLL

scheduler:
  warmup_steps: 750
  cooldown_steps: 1000 #2000

trainer:
  gradient_accumulate_steps: 1
  check_val_every_n_steps: VALEVERY
  log_logit_stats: true
  log_intermediates: true

checkpointing:
  monitor: dclm_perplexity

data_train:
  ds1:
    tokenizer_path: "EleutherAI/gpt-neox-20b"

data_eval:
  ds1:
    tokenizer_path: "EleutherAI/gpt-neox-20b"
  # ds2:
  #   tokenizer_path: "EleutherAI/gpt-neox-20b"

model:
  name: MMMM

  igate_bias_init_range: -10.0
  igate_preact_fixed_to: null
  cell_norm_type_v1: layernorm #layernorm #rmsnorm

  reset_at_document_boundaries: true
  vocab_size: -1
  backend: "triton_kernels"
  backend_name: "chunkwise--triton_limit_chunk"
  chunk_size: 64

  num_blocks: BBBB
  num_heads: HHHH
  embedding_dim: EEEE
  proj_factor: PPPP

profiling:
  # Not profiling to reduce time and potential failure points.
  profile_every_n_minutes: -1

base_dir: /nfs-gpu/xlstm/outputs_beck
logging_name: ${data_train.ds1.name}_${model.name}_ctx${context_length}_lr${lr}_steps${num_train_steps}_nb${model.num_blocks}_ed${model.embedding_dim}_nh${model.num_heads}_pf${model.proj_factor}

logger:
  wb_tags:
    - ${task_name}
    - ${model.name}
    - nb${model.num_blocks}_ed${model.embedding_dim}_nh${model.num_heads}_pf${model.proj_factor}
    - ${model.backend_name}
    - sclaw_iso
    - sclaw_iso_round8

# Run command:
# PYTHONPATH=. python scripts/training/train_with_hydra.py +experiment_sclaw=train_mLSTMv1_160M_dclm_cosine
hydra:
  launcher:
    nodes: 8
    additional_parameters: {
      "gpu-bind": "closest",
      "wait-all-nodes": "1",
      "time": "21-00:00:00",
      "exclusive": "",
    }
  mode: MULTIRUN
  sweeper:
    params:
      num_train_steps: SSSS
"""

config_template_str_mlstm_ctx = """
# @package _global_
defaults:
  - /data@data_train.ds1: dclm_arrayrecord_train
  - /data@data_eval.ds1: dclm_arrayrecord_eval_preprocessed
  # - /data@data_eval.ds2: slimpajama_627B_arrayrecord_eval_preprocessed
  - override /parallel: mLSTMv1_7B #mLSTMv1_7B # use fsdp #mLSTMv1_1.3B # use this for no FSDP (pure dp)
  - override /model: mLSTMv1_default
  - override /optimizer: adamw
  - override /scheduler: cosine_decay
  - override /hydra/launcher: slurm_launcher
  - _self_

# specify the deltas from the defaults:
task_name: sclaw_mlstm_ctx_iso8 #! adapt here
batch_size_per_device: BSPBSP
context_length: CCCC
num_epochs: 1000
num_train_steps: SSSS #18_000
lr: LLLL

scheduler:
  warmup_steps: 750
  cooldown_steps: 1000 #2000

trainer:
  gradient_accumulate_steps: 1
  check_val_every_n_steps: VALEVERY
  log_logit_stats: true
  log_intermediates: false

checkpointing:
  monitor: dclm_perplexity

data_train:
  ds1:
    tokenizer_path: "EleutherAI/gpt-neox-20b"

data_eval:
  ds1:
    tokenizer_path: "EleutherAI/gpt-neox-20b"
  # ds2:
  #   tokenizer_path: "EleutherAI/gpt-neox-20b"

model:
  name: MMMM

  igate_bias_init_range: -10.0
  igate_preact_fixed_to: null
  cell_norm_type_v1: layernorm #layernorm #rmsnorm

  reset_at_document_boundaries: true
  vocab_size: -1
  backend: "triton_kernels"
  backend_name: "chunkwise--triton_limit_chunk"
  chunk_size: 64

  num_blocks: BBBB
  num_heads: HHHH
  embedding_dim: EEEE
  proj_factor: PPPP

profiling:
  # Not profiling to reduce time and potential failure points.
  profile_every_n_minutes: -1

base_dir: /nfs-gpu/xlstm/outputs_beck
logging_name: ${data_train.ds1.name}_${model.name}_ctx${context_length}_lr${lr}_steps${num_train_steps}_nb${model.num_blocks}_ed${model.embedding_dim}_nh${model.num_heads}_pf${model.proj_factor}

logger:
  wb_tags:
    - ${task_name}
    - ${model.name}
    - nb${model.num_blocks}_ed${model.embedding_dim}_nh${model.num_heads}_pf${model.proj_factor}
    - ${model.backend_name}
    - sclaw_mlstm_ctx_iso_round8 #! adapt here

# Run command:
# PYTHONPATH=. python scripts/training/train_with_hydra.py +experiment_sclaw=train_mLSTMv1_160M_dclm_cosine
hydra:
  launcher:
    nodes: NNNN
    additional_parameters: {
      "gpu-bind": "closest",
      "wait-all-nodes": "1",
      "time": "21-00:00:00",
      "exclusive": "",
      "reservation": "nxai_scaling",
    }
  mode: MULTIRUN
  sweeper:
    params:
      num_train_steps: SSSS
"""

config_template_str_llama = """
# @package _global_
defaults:
  - /data@data_train.ds1: dclm_arrayrecord_train
  - /data@data_eval.ds1: dclm_arrayrecord_eval_preprocessed # for different ctx len use: dclm_arrayrecord_eval
  # - /data@data_eval.ds1: slimpajama_627B_arrayrecord_eval_preprocessed
  - override /parallel: llama1.3B # no fsdp
  - override /model: llama_default
  - override /optimizer: adamw
  - override /scheduler: cosine_decay
  - override /hydra/launcher: slurm_launcher
  - _self_

# specify the deltas from the defaults:
task_name: sclaw_llama_iso13 #! adapt here
batch_size_per_device: BSPBSP
context_length: CCCC
num_epochs: 1000
num_train_steps: SSSS #95_000
lr: LLLL

scheduler:
  warmup_steps: 750
  cooldown_steps: 1000 #2000

trainer:
  gradient_accumulate_steps: 1
  check_val_every_n_steps: VALEVERY
  log_logit_stats: false
  log_intermediates: false

checkpointing:
  monitor: dclm_perplexity

data_train:
  ds1:
    tokenizer_path: "EleutherAI/gpt-neox-20b"

data_eval:
  ds1:
    tokenizer_path: "EleutherAI/gpt-neox-20b"
  # ds2:
  #   tokenizer_path: "EleutherAI/gpt-neox-20b"

model:
  name: MMMM

  reset_at_document_boundaries: true
  vocab_size: -1

  theta: 500_000

  num_blocks: BBBB
  head_dim: HDHDHD #64 # @12 heads
  embedding_dim: EEEE

profiling:
  # Not profiling to reduce time and potential failure points.
  profile_every_n_minutes: -1

base_dir: /nfs-gpu/xlstm/outputs_beck
logging_name: ${data_train.ds1.name}_${model.name}_ctx${context_length}_lr${lr}_steps${num_train_steps}_nb${model.num_blocks}_ed${model.embedding_dim}_hd${model.head_dim}

logger:
  wb_tags:
    - ${task_name}
    - ${model.name}
    - nb${model.num_blocks}_ed${model.embedding_dim}_hd${model.head_dim}
    - sclaw_llama_iso_round13 #! adapt here


# Run command:
# PYTHONPATH=. python scripts/training/train_with_hydra.py +experiment_sclaw=train_mLSTMv1_1.4B_dclm_cosine
hydra:
  launcher:
    nodes: NNNN
    additional_parameters: {
      "gpu-bind": "closest",
      "wait-all-nodes": "1",
      "time": "21-00:00:00",
      "exclusive": "",
      "exclude": "",
      "reservation": "nxai_scaling",
    }
  mode: MULTIRUN
  sweeper:
    params:
      num_train_steps: SSSS
"""
