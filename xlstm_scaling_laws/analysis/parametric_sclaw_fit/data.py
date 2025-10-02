from functools import cache
from typing import Literal

import pandas as pd

from ...load_data.token_param_ratio import create_token_param_ratio_data_table
from ...params.common import get_ffn_dim
from ..isoflop.data import create_combined_isoflop_data_table


@cache
def get_all_parametric_sclaw_fit_data_dataframe(
    attention_flop_calc_mode: Literal["chinchilla", "distill_scaling"] | None = None,
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "first",
    model_type: Literal["mlstm", "llama", "all"] = "all",
) -> pd.DataFrame:
    """Load all dataframes from the parametric scaling law fits."""
    selected_columns = [
        "experiment_set_ctx_length",
        "name",
        "run_tag",
        "model_type",
        "num_params",
        "num_tokens_training",
        "num_flops_training",
        "val/.dclm_loss",
        "token_param_ratio",
        "width_depth_ratio",
        "Preset Token Param Ratio",
        "experiment_set",
        "context_length",
        "learning_rate",
        "global_batch_size",
        "num_train_steps",
        "val/.dclm_perplexity",
        "Preset Num Params",
        "Model Size",
        "embedding_dim",
        "num_blocks",
        "num_heads",
        "proj_factor_ffn",
        "ffn_multiple_of",
        "ffn_dim",
        "head_dim_qk",
        "head_dim_v",
        "IsoFLOP",
        "train/.loss_mean",
    ]

    # load the isoflop data
    isoflop_df = create_combined_isoflop_data_table(
        attention_flop_calc_mode=attention_flop_calc_mode,
        mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
    )
    # add experiment set column
    isoflop_df["experiment_set"] = "isoflop"
    # load the token param ratio data
    token_param_ratio_df = create_token_param_ratio_data_table(
        model_data="combined",
        attention_flop_calc_mode=attention_flop_calc_mode,
        mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
    )
    # add experiment set column
    token_param_ratio_df["experiment_set"] = "tokenparam"

    # merge the dataframes
    combined_df = pd.concat(
        [isoflop_df, token_param_ratio_df], ignore_index=True, axis=0
    )
    # add width_depth_ratio column
    combined_df["width_depth_ratio"] = (
        combined_df["embedding_dim"] / (combined_df["num_blocks"])
    )

    combined_df["experiment_set_ctx_length"] = (
        combined_df["experiment_set"]
        + "_ctx"
        + combined_df["context_length"].astype(str)
    )

    # compute the ffn dim
    def _compute_ffn_dim(row):
        if row["model_type"] == "mlstm_v1":
            round_mode = "ceil_multiple_of"
        elif row["model_type"] == "llama":
            round_mode = "floor_multiple_of"
        else:
            raise ValueError(f"Unknown model type {row['model_type']}")
        
        return get_ffn_dim(
            d_model=row["embedding_dim"],
            proj_factor=row["proj_factor_ffn"],
            ffn_multiple_of=row["ffn_multiple_of"],
            round_mode=round_mode,
        )
    combined_df["ffn_dim"] = combined_df.apply(_compute_ffn_dim, axis=1)

    # compute the head dims
    def _compute_qk_head_dim(row):
        if row["model_type"] == "mlstm_v1":
            return int((row["embedding_dim"] // row["num_heads"]) * row["proj_factor_qk"])
        elif row["model_type"] == "llama":
            return float("nan")
        else:
            raise ValueError(f"Unknown model type {row['model_type']}")

    def _compute_v_head_dim(row):
        if row["model_type"] == "mlstm_v1":
            return int(row["embedding_dim"] // row["num_heads"])
        elif row["model_type"] == "llama":
            return int(row["head_dim"])
        else:
            raise ValueError(f"Unknown model type {row['model_type']}")

    combined_df["head_dim_qk"] = combined_df.apply(_compute_qk_head_dim, axis=1)
    combined_df["head_dim_v"] = combined_df.apply(_compute_v_head_dim, axis=1)
    
    # add num heads
    def _compute_num_heads(row):
        if row["model_type"] == "mlstm_v1":
            return int(row["num_heads"])
        elif row["model_type"] == "llama":
            return int(row["embedding_dim"] // row["head_dim"])
        else:
            raise ValueError(f"Unknown model type {row['model_type']}")

    combined_df["num_heads"] = combined_df.apply(_compute_num_heads, axis=1)

    # select the columns
    combined_df = combined_df[selected_columns]

    if model_type != "all":
        # filter by model type
        model_type_tag_mapping = {
            "mlstm": "mlstm_v1",
            "llama": "llama",
        }
        combined_df = combined_df[
            combined_df["model_type"] == model_type_tag_mapping[model_type]
        ]
        assert not combined_df.empty, (
            f"Run data dataframe is empty after filtering by model type {model_type}."
        )

    return combined_df


def create_param_fit_sclaw_data_table(
    model_type: Literal["mlstm", "llama"],
    attention_flop_calc_mode: Literal[
        "chinchilla", "distill_scaling"
    ] = "distill_scaling",
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "first",
    context_length: int = 8192,
    experiment_set: Literal["all", "tokenparam", "isoflop"] = "all",
    experiment_set_split: Literal["all", "all_butlong7b", "long7b"] = "all_butlong7b",
) -> pd.DataFrame:
    """Create a dataframe with the scaling law data for the given model type.

    Notes:
    - we exclude the the 7B model for the mlstm_v1 model, for later test of the fit.

    """

    run_data_df = get_all_parametric_sclaw_fit_data_dataframe(
        attention_flop_calc_mode=attention_flop_calc_mode,
        mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
    )

    # filter by model type
    model_type_tag_mapping = {
        "mlstm": "mlstm_v1",
        "llama": "llama",
    }

    run_data_df = run_data_df[
        run_data_df["model_type"] == model_type_tag_mapping[model_type]
    ]

    assert not run_data_df.empty, (
        f"Run data dataframe is empty after filtering by model type {model_type}."
    )

    # filter by context length
    run_data_df = run_data_df[run_data_df["context_length"] == context_length]

    if run_data_df.empty:
        raise ValueError(
            f"Run data dataframe is empty after filtering by context length {context_length}. ",
            "Did you set the correct context length?",
        )

    # filter by experiment set
    experiment_set_mapping = {
        "all": ["isoflop", "tokenparam"],
        "isoflop": ["isoflop"],
        "tokenparam": ["tokenparam"],
    }

    run_data_df = run_data_df[
        run_data_df["experiment_set"].isin(experiment_set_mapping[experiment_set])
    ]

    # filter by experiment set split
    if experiment_set_split == "long7b":
        run_data_df = run_data_df[
            run_data_df["run_tag"] == "dclm_mLSTMv1_7B_longrun_pretraining_final"
        ]
    elif experiment_set_split == "all_butlong7b":
        run_data_df = run_data_df[
            run_data_df["run_tag"] != "dclm_mLSTMv1_7B_longrun_pretraining_final"
        ]
    elif experiment_set_split == "all":
        # we do not filter any run
        pass
    else:
        raise ValueError(
            f"Experiment set split {experiment_set_split} is not valid. "
            "Valid values are: all_butlong7b, long7b."
        )

    return run_data_df
