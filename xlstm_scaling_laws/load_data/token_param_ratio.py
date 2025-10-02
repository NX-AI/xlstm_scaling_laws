from typing import Literal

import numpy as np
import pandas as pd

from ..common.load_data import get_default_run_data_calc_config, load_run_summary_table
from ..load_data.datafiles import RunDataSet, get_run_data_file
from ..run_data import RunData
from .postprocess import round_to_catgorical_vals


def create_token_param_ratio_data_table(
    model_data: Literal["mlstm", "llama", "combined"] = "combined",
    attention_flop_calc_mode: Literal["chinchilla", "distill_scaling"]
    | None = "distill_scaling",
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "first",
) -> pd.DataFrame:
    if attention_flop_calc_mode is None:
        attention_flop_calc_mode = "distill_scaling"

    mlstm_df = load_run_summary_table(
        wandb_run_data_dict_file=get_run_data_file(
            data_set=RunDataSet.TOKENPARAM_MLSTM
        ),
        run_data_class=RunData,
        config_calc_run_data=get_default_run_data_calc_config(
            attention_flop_calc_mode=attention_flop_calc_mode,
            mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
        ),
        group_runs_by="name",
    )
    llama_df = load_run_summary_table(
        wandb_run_data_dict_file=get_run_data_file(
            data_set=RunDataSet.TOKENPARAM_LLAMA
        ),
        run_data_class=RunData,
        config_calc_run_data=get_default_run_data_calc_config(
            attention_flop_calc_mode=attention_flop_calc_mode,
            mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
        ),
        group_runs_by="name",
    )

    ## select final data
    mlstm_scaling_law_tags = [
        "scl_mlstm_160Mv2",
        "scl_mlstm_400M",
        "scl_mlstm_830M",
        "scl_mlstm_1.4B",
        "scl_mlstm_2.7B",
        "scl_mlstm_7B",
        "dclm_mLSTMv1_7B_longrun_pretraining_final",
    ]
    llama_scaling_law_tags = [
        "scl_llama_160M",
        "scl_llama_400M",
        "scl_llama_830Mv2",
        "scl_llama_1.4Bv2",
        "scl_llama_2.7B",
        "scl_llama_7B",
    ]

    mlstm_scl_df = mlstm_df[
        mlstm_df["run_tag"].isin(mlstm_scaling_law_tags)
    ].reset_index()
    llama_scl_df = llama_df[
        llama_df["run_tag"].isin(llama_scaling_law_tags)
    ].reset_index()

    ## refine rows
    model_type_mapping = {"mlstm_v1": "mLSTM", "llama": "Llama"}
    run_tag_mapping = {
        "scl_mlstm_160Mv2": "160M",
        "scl_mlstm_400M": "400M",
        "scl_mlstm_830M": "830M",
        "scl_mlstm_1.4B": "1.4B",
        "scl_mlstm_2.7B": "2.7B",
        "scl_mlstm_7B": "7B",
        "dclm_mLSTMv1_7B_longrun_pretraining_final": "7B long",
        "scl_llama_160M": "160M",
        "scl_llama_400M": "400M",
        "scl_llama_830Mv2": "830M",
        "scl_llama_1.4Bv2": "1.4B",
        "scl_llama_2.7B": "2.7B",
        "scl_llama_7B": "7B",
    }
    # Round the token param ratios to the configured values
    token_param_ratios = [22, 44, 110, 220, 550, 1100, 2200]

    mlstm_scl_df.loc[:, "Preset Token Param Ratio"] = mlstm_scl_df[
        "token_param_ratio"
    ].apply(
        lambda x: round_to_catgorical_vals(
            x, categorical_vals=np.array(token_param_ratios), round_fraction=0.18
        )
    )
    # manually exclude one token param ratio
    mlstm_scl_df.loc[3, "Preset Token Param Ratio"] = "extra"

    mlstm_scl_df.loc[:, "Model Size"] = mlstm_scl_df["run_tag"].map(run_tag_mapping)
    mlstm_scl_df.loc[:, "Model Type"] = mlstm_scl_df["model_type"].map(
        model_type_mapping
    )

    llama_scl_df.loc[:, "Preset Token Param Ratio"] = llama_scl_df[
        "token_param_ratio"
    ].apply(
        lambda x: round_to_catgorical_vals(
            x, categorical_vals=np.array(token_param_ratios), round_fraction=0.18
        )
    )
    llama_scl_df.loc[:, "Model Size"] = llama_scl_df["run_tag"].map(run_tag_mapping)
    llama_scl_df.loc[:, "Model Type"] = llama_scl_df["model_type"].map(
        model_type_mapping
    )

    combined_scl_df = pd.concat([mlstm_scl_df, llama_scl_df]).reset_index()

    if model_data == "mlstm":
        return mlstm_scl_df
    elif model_data == "llama":
        return llama_scl_df
    elif model_data == "combined":
        return combined_scl_df
    else:
        raise ValueError(f"Model data {model_data} not supported.")
