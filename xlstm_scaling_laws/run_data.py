import logging
import traceback
from dataclasses import dataclass
from typing import Any

from .common.base_run_data import BaseRunData, RunDataCalcConfig
from .common.wandb_run_data import WandBRunData

LOGGER = logging.getLogger(__name__)

@dataclass
class RunData(BaseRunData):

    def __repr__(self) -> str:
        return super().__repr__()

    @property
    def model_checkpoint_paths(self) -> list[str]:
        """Returns the model checkpoint paths for the run.
        It is a list, because there can be multiple checkpoints for a run (e.g., if the run was resumed from a previous checkpoint).
        
        The paths returned are the paths to the wandb root directory, which has been stored in the run directory. 
        To get the full path to the checkpoint, you need to know how the checkpoints are stored in the run directory. For example, if the checkpoints are stored in a subdirectory called "checkpoints" and the checkpoint files are named "checkpoint_{step}", then the full path to the checkpoint would be:

        Returned model checkpoint paths:
        ['/nfs-gpu/xlstm/outputs_beck/sclaw/dclm_mLSTMv1_1.4B_ctx8192_lr0.0007_steps746000_2024-12-21T11:53:25/0/wandb']

        Full path to checkpoint:
        `/nfs-gpu/users_work/beck/xlstm_sclaw_ckpt_selected/dclm_mLSTMv1_1.4B_ctx8192_lr0.0007_steps746000_2024-12-21T11:53:25/0/checkpoints/checkpoint_746000`

        Content of the checkpoint directory:
        (base) [beck@gpu-prd-controller checkpoint_746000]$ du -h
        5.0G    ./params/ocdbt.process_0/d
        5.0G    ./params/ocdbt.process_0
        8.5K    ./params/d
        5.0G    ./params
        9.6G    ./opt_state/ocdbt.process_0/d
        9.6G    ./opt_state/ocdbt.process_0
        8.5K    ./opt_state/d
        9.6G    ./opt_state
        17K     ./rng/checkpoint
        25K     ./rng
        8.5K    ./metadata
        17K     ./step/checkpoint
        25K     ./step
        8.5K    ./metrics
        15G     .
        """
        return [wbrun.metadata["root"] for wbrun in self._raw_data]

    @classmethod
    def create_run_data_from_wandb_run_data(
        cls,
        raw_run_data: WandBRunData,
        run_tag: str | None = None,
        config_calc_run_data: RunDataCalcConfig | None = None,
    ) -> "RunData":
        return create_run_data_from_wandb_run_data(
            raw_run_data=raw_run_data,
            run_tag=run_tag,
            config_calc_run_data=config_calc_run_data,
        )



# Note: Replace this function with another one, if you change the Config in the WandBRunData class.
def create_run_data_from_wandb_run_data(
    raw_run_data: WandBRunData,
    run_tag: str | None = None,
    config_calc_run_data: RunDataCalcConfig = None,
) -> RunData:
    def extract_chunk_size(backend_kwargs_str: str) -> int:
        # Example: {'backend_name': 'chunkwise--triton_xl_chunk', 'chunk_size': 64}
        try:
            chunk_size_str = (
                backend_kwargs_str.split("chunk_size")[1]
                .split(":")[1]
                .split("}")[0]
                .split(",")[0]
                .strip()
            )
        except IndexError:
            LOGGER.warning(
                (
                    f"Could not extract chunk size from backend kwargs '{backend_kwargs_str}' from run {raw_run_data}."
                    "Using default chunk size of 64."
                )
            )
            chunk_size_str = "64"
        return int(chunk_size_str)

    def get_model_type_and_kwargs(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        model_class = config["model.model_class"]
        if model_class == "xlstm_jax.models.xlstm_parallel.xlstm_lm_model.xLSTMLMModel":
            model_type = config["model.model_config.mlstm_block.mlstm.layer_type"]

            if model_type == "mlstm_v1":
                num_blocks = config["model.model_config.num_blocks"]
                embedding_dim = config["model.model_config.embedding_dim"]
                proj_factor_ffn = config[
                    "model.model_config.mlstm_block.feedforward.proj_factor"
                ]
                num_heads = config["model.model_config.mlstm_block.mlstm.num_heads"]
                proj_factor_qk = config[
                    "model.model_config.mlstm_block.mlstm.qk_dim_factor"
                ]
                chunk_size = extract_chunk_size(
                    config[
                        "model.model_config.mlstm_block.mlstm.mlstm_cell.backend.kwargs"
                    ]
                )
                vocab_size = config["model.model_config.vocab_size"]
                ffn_multiple_of = config[
                    "model.model_config.mlstm_block.mlstm.round_proj_up_to_multiple_of"
                ]

                model_kwargs = {
                    "num_blocks": num_blocks,
                    "embedding_dim": embedding_dim,
                    "proj_factor_ffn": proj_factor_ffn,
                    "num_heads": num_heads,
                    "proj_factor_qk": proj_factor_qk,
                    "chunk_size": chunk_size,
                    "vocab_size": vocab_size,
                    "ffn_multiple_of": ffn_multiple_of,
                }
            else:
                raise ValueError(f"Model type '{model_type}' not supported.")

        elif model_class == "xlstm_jax.models.llama.llama.LlamaTransformer":
            num_blocks = config["model.model_config.num_blocks"]
            embedding_dim = config["model.model_config.embedding_dim"]
            vocab_size = config["model.model_config.vocab_size"]
            head_dim = config["model.model_config.head_dim"]
            ffn_multiple_of = config["model.model_config.ffn_multiple_of"]
            proj_factor_ffn = 2.667
            model_kwargs = {
                "num_blocks": num_blocks,
                "embedding_dim": embedding_dim,
                "vocab_size": vocab_size,
                "head_dim": head_dim,
                "ffn_multiple_of": ffn_multiple_of,
                "proj_factor_ffn": proj_factor_ffn,
            }
            model_type = "llama"

        return model_type, model_kwargs

    model_type, model_kwargs = get_model_type_and_kwargs(raw_run_data.config)

    def get_context_length() -> int:
        ctx_len = raw_run_data.config.get("model.model_config.context_length", None)
        if ctx_len is None:
            ctx_len = int(raw_run_data.name.split("ctx")[1].split("_")[0])
        return ctx_len

    def get_learning_rate() -> float:
        lr_str = raw_run_data.config.get("optimizer.scheduler", None)
        if lr_str is None:
            LOGGER.warning("Could not extract learning rate, setting to NaN")
            lr = float("nan")
        else:
            lr = float(lr_str.split(",")[0].split(":")[1].strip())
        return lr
    
    def get_warmup_steps() -> int:
        wu_str = raw_run_data.config.get("optimizer.scheduler", None)
        if wu_str is None:
            LOGGER.warning("Could not extract learning rate, setting to NaN")
            wu = -1
        else:
            wu = wu_str[wu_str.find("warmup_steps"):].split(",")[0].split(":")[1].strip()
            wu = int(wu)
        return wu

    config_calc_run_data = (
        config_calc_run_data
        if config_calc_run_data is not None
        else RunDataCalcConfig()
    )

    try:
        run_data = RunData(
            name=raw_run_data.name,
            run_tag=run_tag,
            run_id=raw_run_data.id,
            model_type=model_type,
            learning_rate=get_learning_rate(),
            num_lr_warmup_steps=get_warmup_steps(),
            model_kwargs=model_kwargs,
            global_batch_size=raw_run_data.summary["dataset/"]["global_batch_size"],
            num_params=float(raw_run_data.summary["model/"]["num_params"]),
            num_train_steps=raw_run_data.summary["dataset/"]["num_train_steps"],
            context_length=get_context_length(),
            num_nodes=raw_run_data.num_nodes,
            num_gpus_per_node=raw_run_data.num_gpus_per_node,
            config_calc_run_data=config_calc_run_data,
        )
    except KeyError as e:
        # raise e
        LOGGER.warning(f"KeyError: {e}. While creating RunData from {raw_run_data}")
        run_data = RunData(
            name=raw_run_data.name,
            run_tag=run_tag,
            run_id=raw_run_data.id,
            model_type=model_type,
            learning_rate=get_learning_rate(),
            model_kwargs=model_kwargs,
            global_batch_size=float("nan"),
            num_params=float("nan"),
            num_train_steps=float("nan"),
            context_length=get_context_length(),
            num_nodes=raw_run_data.num_nodes,
            num_gpus_per_node=raw_run_data.num_gpus_per_node,
            config_calc_run_data=config_calc_run_data,
            status="failed",
        )
    return run_data