import logging
from typing import Literal

import numpy as np
import pandas as pd

from ..common.load_data import get_default_run_data_calc_config, load_run_summary_table
from ..run_data import RunData
from .datafiles import RunDataSet, get_run_data_file
from .postprocess import bin_to_categorical_vals, round_to_catgorical_vals

LOGGER = logging.getLogger(__name__)


def create_isoflop_data_table(
    data_specifier: RunDataSet = "isoflop_mlstm_ctx8192",
    attention_flop_calc_mode: Literal["chinchilla", "distill_scaling"] | None = None,
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "first",
) -> pd.DataFrame:
    """Create the IsoFLOP dataframe with all isoflop runs."""

    assert "isoflop" in data_specifier, (
        f"Data specifier {data_specifier} is not an IsoFLOP data specifier. Got "
        f"{data_specifier}."
    )
    if attention_flop_calc_mode is None:
        attention_flop_calc_mode = "distill_scaling"
        if "flops_chinchilla" in data_specifier:
            attention_flop_calc_mode = "chinchilla"
    else:
        LOGGER.warning(
            f"While loading IsoFLOP Runs: Overriding attention flop calc mode to {attention_flop_calc_mode}."
        )

    isoflop_df = load_run_summary_table(
        wandb_run_data_dict_file=get_run_data_file(data_specifier),
        run_data_class=RunData,
        config_calc_run_data=get_default_run_data_calc_config(
            attention_flop_calc_mode=attention_flop_calc_mode,
            mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
        ),
        group_runs_by="none",
    )

    isoflop_df_with_isoflop_cols = add_isoflop_columns(isoflop_df)
    return isoflop_df_with_isoflop_cols


def add_isoflop_columns(run_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Add the IsoFLOP and Preset Token Param Ratio columns to the run summary table.

    Args:
        run_summary_df: The run summary table.

    Returns:
        The run summary table with the IsoFLOP and Preset Token Param Ratio columns added.
    """

    isoflop_df = run_summary_df.copy()

    # Round the training flops to the configured values
    isoflop_counts = [
        6e18,  # 160M, 400M
        1e19,
        3e19,  # 160M, 400M, 1.4B
        1e20,  # (160M), 400M, 830M, 1.4B
        6e20,  # 400M, 830M,
        3e21,
    ]

    isoflop_df.loc[:, "IsoFLOP"] = isoflop_df["num_flops_training"].apply(
        lambda x: round_to_catgorical_vals(
            x, np.array(isoflop_counts), round_fraction=0.1
        )
    )

    # Round the token param ratios to the configured values
    token_param_ratios = [22, 44, 110, 220, 550, 1100, 2200]

    isoflop_df.loc[:, "Preset Token Param Ratio"] = isoflop_df[
        "token_param_ratio"
    ].apply(
        lambda x: round_to_catgorical_vals(
            x, categorical_vals=np.array(token_param_ratios), round_fraction=0.07
        )
    )

    # Round the number of parameters to the configured values
    num_param_bins = [130e6, 180e6, 240e6, 340e6, 440e6, 700e6, 950e6, 1.2e9, 1.4e9]
    num_param_bin_names = [
        "<130M",
        "<180M",
        "<240M",
        "<340M",
        "<440M",
        "<700M",
        "<950M",
        "<1.2B",
        "<1.4B",
    ]
    # num_param_bin_names = [
    #     "120M",
    #     "160M",
    #     "200M",
    #     "300M",
    #     "400M",
    #     "600M",
    #     "800M",
    #     "1.1B",
    #     "1.4B",
    # ]

    isoflop_df.loc[:, "Preset Num Params"] = isoflop_df["num_params"].apply(
        lambda x: bin_to_categorical_vals(
            x,
            categorical_vals=np.array(num_param_bins),
            categorical_val_names=np.array(num_param_bin_names),
        )
    )
    return isoflop_df
