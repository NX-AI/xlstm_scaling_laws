import copy
import pathlib as Path
from collections import defaultdict
from typing import Literal

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ....flops.count_flops import FlopCountConfig, count_model_flops_fwbw
from ....params.count_params import ParamCountConfig, count_model_params


def compute_fwbw_flops_and_num_params_for_models(
    model_config_dict: dict[str, dict],
    model_type: str = "mlstm_v1",
    context_length: int = 8192,
    include_embedding_flops: bool = False,
    attention_flop_calc_mode: Literal[
        "chinchilla", "distill_scaling"
    ] = "distill_scaling",
) -> dict[str, dict]:
    """Compute the FLOPs and number of parameters for the models in the model_config_dict.

    Example:
    ```
    model_config_dict = {
        "mlstm_100M_1": {
        "num_blocks": 10,
        "embedding_dim": 640,
        "proj_factor_ffn": 2.667,
        "num_heads": 5,
        "proj_factor_qk": 0.5,
        "chunk_size": 64,
        "vocab_size": 50304,
        "ffn_multiple_of": 64,
        "global_batch_size": 128,
        "learning_rate": 0.003,
    }
    }

    ```

    Args:
        model_config_dict (dict[str, dict]): A dictionary containing the model configuration.
        model_type (str, optional): The type of model. Defaults to "mlstm_v1".
        context_length (int, optional): The context length. Defaults to 8192.
        include_embedding_flops (bool, optional): Whether to include embedding FLOPs. Defaults to False.
        attention_flop_calc_mode (str, optional): The attention FLOP calculation mode. Defaults to "distill_scaling".
        This can be either "chinchilla" or "distill_scaling".

    Returns:
        dict[str, dict]: A dictionary containing the model configuration with the FLOPs and number of parameters.

    """

    flop_count_config = FlopCountConfig(
        include_skip_flops=False,
        include_norm_layer_flops=False,
        include_embedding_flops=include_embedding_flops,
        include_final_logit_flops=True,
        attention_flop_calc_mode=attention_flop_calc_mode,  # "distill_scaling", # "chinchilla"
        mlstm_fw_flop_calc_mode="first",
        mlstm_flop_causal_factor=1.0,  # unused
        round_ffn_dim_to_multiple_of_for_flops=False,
        bw_flop_count_mode="total_factor_2",
        seq_mix_bw_flop_factor=2.5,
        flop_factor_sig=1.0,
        flop_factor_exp=1.0,
        flop_factor_ffn_act_fn=1.0,
        flop_factor_max=1.0,
        flop_factor_mask=1.0,
        flop_factor_abs=1.0,
    )

    param_count_config = ParamCountConfig(
        count_norm_layer_params=True,
        count_embedding_params=True,
        count_final_logits_params=True,
    )

    updated_model_dict = copy.deepcopy(model_config_dict)
    for model_key, model_dict in model_config_dict.items():
        updated_model_dict[model_key].update(
            {
                "num_flops_fwbw": count_model_flops_fwbw(
                    model_type=model_type,
                    context_length=context_length,
                    model_kwargs=model_dict,
                    config=flop_count_config,
                )[0],
                "num_params": count_model_params(
                    model_type=model_type,
                    model_kwargs=model_dict,
                    config=param_count_config,
                ),
            }
        )
    return updated_model_dict


def calculate_num_train_steps_per_isoflop(
    model_config_dict: dict[str, dict],
    iso_flop_counts: list[float],
    context_length: int = 8192,
    default_context_length: int = 8192,
    global_batch_size_override: int | None = None,
) -> dict[str, dict]:
    """Calculate the number of training steps per flop count for iso-flop curves.

    The model config dict must have the following keys:
    - num_flops_fwbw
    - global_batch_size
    - num_params
    - num_blocks
    - embedding_dim
    - num_heads
    - proj_factor_ffn

    Args:
        model_config_dict (dict[str, dict]): A dictionary containing the model configuration.
        iso_flop_counts (list[float]): A list of iso-flop counts.
        context_length (int, optional): The context length. Defaults to 8192.
        default_context_length (int, optional): The default context length. Defaults to 8192.
                                                This the default context length is the one used to determine the batch size.
                                                Adjust the batch size to the selected context length.
        global_batch_size_override (int | None, optional): An optional override for the global batch size. Defaults to None.

    Returns:
        dict[str, dict]: A dictionary containing the model configuration with the number of training steps per iso-flop.
    """

    train_length_per_iso_flop: dict[str, list[int]] = {}
    iso_flop_array = np.array(iso_flop_counts)

    # check that the context length and default context lengths are multiples of each other
    if context_length <= default_context_length:
        assert default_context_length % context_length == 0, (
            f"default_context_length: {default_context_length} must be a multiple of context_length: {context_length}"
        )
        ctx_len_factor = default_context_length / context_length
    else:
        assert context_length % default_context_length == 0, (
            f"context_length: {context_length} must be a multiple of default_context_length: {default_context_length}"
        )
        ctx_len_factor = default_context_length / context_length

    for model_key, model_kwargs in model_config_dict.items():
        num_flops_for_model = model_kwargs["num_flops_fwbw"]
        num_nodes = None

        # Note: we can currently change either the batch size or the context length
        if global_batch_size_override is not None:
            assert "num_nodes" in model_kwargs, (
                f"num_nodes must be in model_kwargs if global_batch_size_override is set: {global_batch_size_override}"
            )
            num_nodes = model_kwargs["num_nodes"]
            batch_size_per_device = model_kwargs["batch_size_per_device"]

            global_batch_size_for_model = global_batch_size_override
            # find the global batch size override factor
            gbs_model_kwargs = model_kwargs["global_batch_size"]
            # if the global batch size override is larger than the model kwargs, we need to adjust the number of nodes
            # if the global batch size override is smaller than the model kwargs, we need to adjust the batch size per device
            if global_batch_size_override > gbs_model_kwargs:
                assert global_batch_size_override % gbs_model_kwargs == 0, (
                    f"global_batch_size_override: {global_batch_size_override} must be a multiple of gbs_model_kwargs: {gbs_model_kwargs}"
                )
                gbs_factor = global_batch_size_override / gbs_model_kwargs
                # adjust the number of nodes
                num_nodes = int(num_nodes * gbs_factor)
            else:
                assert gbs_model_kwargs % global_batch_size_override == 0, (
                    f"gbs_model_kwargs: {gbs_model_kwargs} must be a multiple of global_batch_size_override: {global_batch_size_override}"
                )
                gbs_factor = gbs_model_kwargs / global_batch_size_override
                # adjust the batch size per device
                if num_nodes == 1:
                    batch_size_per_device = int(batch_size_per_device * gbs_factor)
                else:
                    num_nodes = int(num_nodes * gbs_factor)

        else:
            global_batch_size_for_model = model_kwargs["global_batch_size"]
            global_batch_size_for_model = int(
                global_batch_size_for_model * ctx_len_factor
            )  # adjust the batch size to the selected context length
            if "batch_size_per_device" in model_kwargs:
                # adapt the batch size per device and num nodes to the global batch size
                # so that the overall number of token per batch remains the same
                batch_size_per_device = model_kwargs["batch_size_per_device"]
                batch_size_per_device = int(batch_size_per_device * ctx_len_factor)
            else:
                batch_size_per_device = None

        training_lengths = iso_flop_array / (
            global_batch_size_for_model * num_flops_for_model
        )

        chinchilla_optimal = (model_kwargs["num_params"] * 22) / (
            global_batch_size_for_model * context_length
        )

        if "num_heads" in model_kwargs:
            model_tag = f"nb{model_kwargs['num_blocks']}_ed{model_kwargs['embedding_dim']}_nh{model_kwargs['num_heads']}_pf{model_kwargs['proj_factor_ffn']}"
        else:
            model_tag = f"nb{model_kwargs['num_blocks']}_ed{model_kwargs['embedding_dim']}_hd{model_kwargs['head_dim']}_pf{model_kwargs['proj_factor_ffn']}"

        result_dict = {
            "num_params in M": model_kwargs["num_params"] / 1e6,
            "num_flops_fwbw in 1e12": model_kwargs["num_flops_fwbw"] / 1e12,
            "global_batch_size": global_batch_size_for_model,
            "model_tag": model_tag,
        }
        train_length_per_iso_flop[model_key] = result_dict
        result_dict.update(
            {
                f"{flop:.1e}": n_train_step
                for flop, n_train_step in zip(
                    iso_flop_counts, training_lengths.tolist()
                )
            }
        )
        result_dict["chinchilla_optimal"] = chinchilla_optimal
        model_params_to_keep = [
            "num_blocks",
            "embedding_dim",
            "num_heads",
            "head_dim",
            "proj_factor_ffn",
            "learning_rate",
        ]
        for model_param in model_params_to_keep:
            if model_param in model_kwargs:
                result_dict[model_param] = model_kwargs[model_param]
        result_dict["context_length"] = context_length
        result_dict["batch_size_per_device"] = batch_size_per_device
        if "num_nodes" in model_kwargs:
            if num_nodes is not None:
                result_dict["num_nodes"] = num_nodes
            else:
                result_dict["num_nodes"] = model_kwargs["num_nodes"]

    return train_length_per_iso_flop


def create_iso_flop_train_len_df(
    model_type: str,
    model_config_dict: dict[str, dict],
    iso_flop_counts: list[float],
    context_length: int = 8192,
    include_embedding_flops: bool = False,
    attention_flop_calc_mode: Literal[
        "chinchilla", "distill_scaling"
    ] = "distill_scaling",
    global_batch_size_override: int | None = None,
) -> pd.DataFrame:
    """Create a DataFrame containing the number of training steps per iso-flop count for the models in the model_config_dict.

    Args:
        model_type: The type of model.
        model_config_dict: A dictionary containing the model configuration.
        iso_flop_counts: A list of iso-flop counts.
        context_length: The context length. Defaults to 8192.
        include_embedding_flops: Whether to include embedding FLOPs. Defaults to False.

    Returns:
        A DataFrame containing the number of training steps per iso-flop count for the models in the model_config_dict.
    """

    updated_model_dict = compute_fwbw_flops_and_num_params_for_models(
        model_config_dict=model_config_dict,
        model_type=model_type,
        context_length=context_length,
        include_embedding_flops=include_embedding_flops,
        attention_flop_calc_mode=attention_flop_calc_mode,
    )

    train_length_per_iso_flop = calculate_num_train_steps_per_isoflop(
        model_config_dict=updated_model_dict,
        iso_flop_counts=iso_flop_counts,
        context_length=context_length,
        global_batch_size_override=global_batch_size_override,
    )

    return pd.DataFrame(train_length_per_iso_flop).T.reset_index()


def generate_iso_configs_mlstm(
    run_isoflop_df: pd.DataFrame,
    config_template_str: str,
    model_name_template: str = "mLSTMv1_{sizename}",
    valevery_steps: int = 500,
    run_valevery_steps: int | None = None,
) -> dict[str, str]:
    """Generate the config files for the mlstm isoflop runs.
    Deprecated: kept for backwards compatibility. Please use `generate_iso_configs_mlstm_ctx` instead.

    Args:
        run_isoflop_df: The DataFrame containing the isoflop runs.
        config_template_str: The config template string.
        model_name_template: The model name template. Defaults to "mLSTMv1_{sizename}".
        valevery_steps: The num training step interval to which the flop budget is rounded. Defaults to 500.
        run_valevery_steps: The number of steps to validate every for the run. Defaults to None.
    """

    config_dict = defaultdict(list)
    for idx, row in run_isoflop_df.iterrows():
        config = config_template_str.replace("LLLL", f"{row['learning_rate']}")
        config = config.replace("BBBB", f"{int(row['num_blocks'])}")
        config = config.replace("HHHH", f"{int(row['num_heads'])}")
        config = config.replace("EEEE", f"{int(row['embedding_dim'])}")
        config = config.replace("PPPP", f"{row['proj_factor_ffn']}")
        if run_valevery_steps is not None:
            config = config.replace("VALEVERY", f"{run_valevery_steps}")
        else:
            config = config.replace("VALEVERY", f"{valevery_steps}")

        model_name = model_name_template.format(sizename=row["index"].split("_")[1])
        config = config.replace("MMMM", model_name)

        nsteps = row.filter(regex=r"e\+").array.astype(float)
        nsteps_rounded = (
            str(
                (np.round(nsteps / valevery_steps) * valevery_steps)
                .astype(int)
                .tolist()
            )
            .replace("[", "")
            .replace("]", "")
        )
        config = config.replace("SSSS", nsteps_rounded)
        config_dict[model_name].append(config)
    return config_dict


def generate_iso_configs_mlstm_ctx(
    run_isoflop_df: pd.DataFrame,
    config_template_str: str,
    model_name_template: str = "mLSTMv1_{sizename}",
    valevery_steps: int = 500,
    run_valevery_steps: int | None = None,
    num_gpus_per_node: int = 8,
) -> dict[str, str]:
    """Generate the config files for the mlstm isoflop runs.

    Args:
        run_isoflop_df: The DataFrame containing the isoflop runs.
        config_template_str: The config template string.
        model_name_template: The model name template. Defaults to "mLSTMv1_{sizename}".
        valevery_steps: The num training step interval to which the flop budget is rounded. Defaults to 500.
        run_valevery_steps: The number of steps to validate every for the run. Defaults to None.
    """

    config_dict = defaultdict(list)
    for idx, row in run_isoflop_df.iterrows():
        assert (
            row["global_batch_size"]
            == row["batch_size_per_device"] * row["num_nodes"] * num_gpus_per_node
        ), (
            f"batch size mismatch! global_batch_size: {row['global_batch_size']}, batch_size_per_device: {row['batch_size_per_device']}, num_nodes: {row['num_nodes']}"
        )

        config = config_template_str.replace("LLLL", f"{row['learning_rate']}")
        config = config.replace("BSPBSP", f"{int(row['batch_size_per_device'])}")
        config = config.replace("CCCC", f"{int(row['context_length'])}")
        config = config.replace("NNNN", f"{int(row['num_nodes'])}")
        config = config.replace("BBBB", f"{int(row['num_blocks'])}")
        config = config.replace("HHHH", f"{int(row['num_heads'])}")
        config = config.replace("EEEE", f"{int(row['embedding_dim'])}")
        config = config.replace("PPPP", f"{row['proj_factor_ffn']}")
        if run_valevery_steps is not None:
            config = config.replace("VALEVERY", f"{run_valevery_steps}")
        else:
            config = config.replace("VALEVERY", f"{valevery_steps}")

        model_name = model_name_template.format(sizename=row["index"].split("_")[1])
        config = config.replace("MMMM", model_name)

        nsteps = row.filter(regex=r"e\+").array.astype(float)
        nsteps_rounded = (
            str(
                (np.round(nsteps / valevery_steps) * valevery_steps)
                .astype(int)
                .tolist()
            )
            .replace("[", "")
            .replace("]", "")
        )
        config = config.replace("SSSS", nsteps_rounded)
        config_dict[model_name].append(config)
    return config_dict


def generate_iso_configs_llama(
    run_isoflop_df: pd.DataFrame,
    config_template_str: str,
    model_name_template: str = "llama_{sizename}",
    valevery_steps: int = 500,
    run_valevery_steps: int | None = None,
    num_gpus_per_node: int = 8,
) -> dict[str, str]:
    """Generate the config files for the llama isoflop runs.

    Args:
        run_isoflop_df: The DataFrame containing the isoflop runs.
        config_template_str: The config template string.
        model_name_template: The model name template. Defaults to "llama_{sizename}".
        valevery_steps: The num training step interval to which the flop budget is rounded. Defaults to 500.
        run_valevery_steps: The number of steps to validate every for the run. Defaults to None.
    """

    config_dict = defaultdict(list)
    for idx, row in run_isoflop_df.iterrows():
        assert (
            row["global_batch_size"]
            == row["batch_size_per_device"] * row["num_nodes"] * num_gpus_per_node
        ), (
            f"batch size mismatch! global_batch_size: {row['global_batch_size']}, batch_size_per_device: {row['batch_size_per_device']}, num_nodes: {row['num_nodes']}"
        )

        config = config_template_str.replace("LLLL", f"{row['learning_rate']}")
        config = config.replace("BSPBSP", f"{int(row['batch_size_per_device'])}")
        config = config.replace("CCCC", f"{int(row['context_length'])}")
        config = config.replace("NNNN", f"{int(row['num_nodes'])}")
        config = config.replace("BBBB", f"{int(row['num_blocks'])}")
        config = config.replace("HDHDHD", f"{int(row['head_dim'])}")
        config = config.replace("EEEE", f"{int(row['embedding_dim'])}")
        config = config.replace("PPPP", f"{row['proj_factor_ffn']}")

        if run_valevery_steps is not None:
            config = config.replace("VALEVERY", f"{run_valevery_steps}")
        else:
            config = config.replace("VALEVERY", f"{valevery_steps}")

        model_name = model_name_template.format(sizename=row["index"].split("_")[1])
        config = config.replace("MMMM", model_name)

        nsteps = row.filter(regex=r"e\+").array.astype(float)
        nsteps_rounded = (
            str(
                (np.round(nsteps / valevery_steps) * valevery_steps)
                .astype(int)
                .tolist()
            )
            .replace("[", "")
            .replace("]", "")
        )
        config = config.replace("SSSS", nsteps_rounded)
        config_dict[model_name].append(config)
    return config_dict


def save_iso_configs(
    config_dict: dict[str, list[str]],
    save_dir: Path,
    config_filename_template: str = "train_{model_name}_{idx}.yaml",
):
    save_dir.mkdir(exist_ok=True, parents=True)
    for model_name, configs in config_dict.items():
        for idx, config in enumerate(configs):
            og_cfg = OmegaConf.create(config)

            cfg_filename = config_filename_template.format(
                model_name=og_cfg.model.name, idx=idx
            )

            with open(save_dir / cfg_filename, "w") as f:
                f.write(config)


def filter_num_params_between(
    train_len_df: pd.DataFrame, min_num_params: float, max_num_params: float
) -> pd.DataFrame:
    return train_len_df[
        (train_len_df["num_params in M"] >= min_num_params)
        & (train_len_df["num_params in M"] <= max_num_params)
    ].sort_values(by="num_params in M", ascending=True)
