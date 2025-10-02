import pickle
from pathlib import Path
from typing import Literal, Type

import pandas as pd

from .base_run_data import (
    BaseRunData,
    FlopCountConfig,
    ParamCountConfig,
    RunDataCalcConfig,
    convert_to_run_data_dict,
    create_run_summary_table,
)


def get_default_run_data_calc_config(
    attention_flop_calc_mode: Literal["chinchilla", "distill_scaling"] = "chinchilla",
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "first",
) -> RunDataCalcConfig:
    """Get the default RunDataCalcConfig.

    Returns:
        The default RunDataCalcConfig.
    """
    return RunDataCalcConfig(
        flop_count_config=FlopCountConfig(
            include_skip_flops=False,
            include_norm_layer_flops=False,
            include_embedding_flops=False,
            include_final_logit_flops=True,
            attention_flop_calc_mode=attention_flop_calc_mode,
            mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
            mlstm_flop_causal_factor=0.75,
            round_ffn_dim_to_multiple_of_for_flops=True,  # round to multiple of 64
            bw_flop_count_mode="total_factor_2",
            seq_mix_bw_flop_factor=2.0,
        ),
        param_count_config=ParamCountConfig(
            count_norm_layer_params=True,
            count_embedding_params=True,
            count_final_logits_params=True,
        ),
    )


def load_run_data(
    wandb_run_data_dict_file: Path | dict,
    run_data_class: Type[BaseRunData],
    group_runs_by: str = "name",
    config_calc_run_data: RunDataCalcConfig = get_default_run_data_calc_config(
        attention_flop_calc_mode="chinchilla",
        mlstm_fw_flop_calc_mode="first",
    ),
) -> dict[str, list[BaseRunData]]:
    
    class ModuleRenameUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Map old module names to new module names
            module_mapping = {
                'mlstm_scaling_laws': 'xlstm_scaling_laws',
                # Add other specific mappings here
            }
            
            # Handle nested module paths
            for old_prefix, new_prefix in module_mapping.items():
                if module.startswith(old_prefix):
                    module = module.replace(old_prefix, new_prefix, 1)
                    break
                
            return super().find_class(module, name)

    if isinstance(wandb_run_data_dict_file, Path):
        with open(wandb_run_data_dict_file, "rb") as f:
            unpickler = ModuleRenameUnpickler(f)
            run_data = unpickler.load()
    elif isinstance(wandb_run_data_dict_file, dict):
        run_data = wandb_run_data_dict_file
    else:
        raise ValueError(
            f"wandb_run_data_dict_file must be a Path or dict, got {type(wandb_run_data_dict_file)}"
        )

    return convert_to_run_data_dict(
        wandb_run_data_dict=run_data,
        config_calc_run_data=config_calc_run_data,
        run_data_class=run_data_class,
        group_runs_by=group_runs_by,
    )


def load_run_summary_table(
    wandb_run_data_dict_file: Path | dict,
    run_data_class: Type[BaseRunData],
    config_calc_run_data: RunDataCalcConfig = get_default_run_data_calc_config(
        attention_flop_calc_mode="chinchilla",
        mlstm_fw_flop_calc_mode="first",
    ),
    group_runs_by: str = "name",
) -> pd.DataFrame:
    run_data_dict = load_run_data(
        wandb_run_data_dict_file=wandb_run_data_dict_file,
        config_calc_run_data=config_calc_run_data,
        run_data_class=run_data_class,
        group_runs_by=group_runs_by,
    )

    return create_run_summary_table(run_data_dict)
