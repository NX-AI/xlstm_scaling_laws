from functools import cache
from typing import Literal

import pandas as pd

from ...fitting.fit_isoflop_polynomials import generate_isoflop_polynomial_fits
from ...load_data.datafiles import RunDataSet
from ...load_data.isoflop import create_isoflop_data_table


@cache
def create_combined_isoflop_data_table(
    attention_flop_calc_mode: Literal["chinchilla", "distill_scaling"] | None = None,
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "first",
) -> pd.DataFrame:
    """Create a combined IsoFLOP dataframe with all selected isoflop runs."""
    data_specifiers = [
        RunDataSet.ISOFLOP_MLSTM_CTX2048.value,
        RunDataSet.ISOFLOP_MLSTM_CTX8192.value,
        RunDataSet.ISOFLOP_MLSTM_CTX16384.value,
        RunDataSet.ISOFLOP_LLAMA_CTX2048.value,
        RunDataSet.ISOFLOP_LLAMA_CTX8192.value,
        RunDataSet.ISOFLOP_LLAMA_CTX16384.value,
    ]

    isoflop_dfs = [
        create_filtered_isoflop_data_table(
            data_specifier=data_specifier,
            return_only_selected_runs=True,
            attention_flop_calc_mode=attention_flop_calc_mode,
            mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
        )
        for data_specifier in data_specifiers
    ]
    # Concatenate the dataframes
    combined_isoflop_df = pd.concat(isoflop_dfs, ignore_index=True, axis=0)
    return combined_isoflop_df


def create_filtered_isoflop_data_table(
    data_specifier: RunDataSet = "isoflop_mlstm_ctx8192",
    return_only_selected_runs: bool = True,
    attention_flop_calc_mode: Literal["chinchilla", "distill_scaling"] | None = None,
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "first",
) -> pd.DataFrame:
    """
    Create the IsoFLOP dataframe with all selected isoflop runs.

    This method loads the dataframe for the given data specifier and filters it to
    only include the selected runs for the final plot.

    Args:
        data_specifier: The data specifier for the IsoFLOP data set.
        return_only_selected_runs: Whether to return only the selected runs for the
            final plot. If False, return all runs. Default is True.

    Returns:
        The IsoFLOP dataframe with all selected isoflop runs.
    """

    isoflop_df = create_isoflop_data_table(
        data_specifier,
        attention_flop_calc_mode=attention_flop_calc_mode,
        mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
    )

    if (
        return_only_selected_runs
        and data_specifier in data_specifier_to_filter_func_mapping
    ):
        filter_func = data_specifier_to_filter_func_mapping[data_specifier]
        isoflop_df = filter_func(isoflop_df)

    return isoflop_df


def _filter_isoflop_df_mlstm_ctx8192(raw_isoflop_df: pd.DataFrame) -> pd.DataFrame:
    isoflop_df = raw_isoflop_df.copy()

    # filter diverging runs
    isoflop_df = isoflop_df[isoflop_df["val/.dclm_loss"] < 4.0]

    # filter runs with num_flops_training < 3e20 and global_batch_size < 200
    isoflop_df = isoflop_df.loc[
        (
            (isoflop_df["num_flops_training"] < 3e20)
            & (isoflop_df["global_batch_size"] < 200.0)
        )
        | (isoflop_df["num_flops_training"] > 3e20)
    ]

    # filter runs in selected iso flop budgets
    sel_isoflop_budgets = ["6e+18", "1e+19", "3e+19", "1e+20", "6e+20"]

    isoflop_df = isoflop_df[isoflop_df["IsoFLOP"].isin(sel_isoflop_budgets)]

    # filter diverging run in 6e+20 budget
    isoflop_df = isoflop_df[
        ~((isoflop_df["IsoFLOP"] == "6e+20") & (isoflop_df["val/.dclm_loss"] > 3.0))
    ]

    return isoflop_df


def _filter_isoflop_df_llama_ctx8192(raw_isoflop_df: pd.DataFrame) -> pd.DataFrame:
    isoflop_df = raw_isoflop_df.copy()

    # filter global outliers
    isoflop_df = isoflop_df[isoflop_df["val/.dclm_loss"] < 3.5]

    # filter run with loss spike in 3e19 FLOP budget
    isoflop_df = isoflop_df[
        ~(
            # (isoflop_df["IsoFLOP"] == "3e+19")
            (
                (isoflop_df["num_flops_training"] > 2.75e19)
                & (isoflop_df["num_flops_training"] < 4.5e19)
            )
            & (
                (isoflop_df["val/.dclm_loss"] > 3.0)
                & (isoflop_df["num_params"] == 406635520.0)
            )
        )
    ]
    isoflop_df = isoflop_df[
        ~(  # (isoflop_df["IsoFLOP"] == "1e+20")
            (
                (isoflop_df["num_flops_training"] > 0.9e20)
                & (isoflop_df["num_flops_training"] < 1.4e20)
            )
            & (isoflop_df["val/.dclm_loss"] > 3.2)
        )
    ]
    return isoflop_df


def _filter_isoflop_df_llama_ctx2048(raw_isoflop_df: pd.DataFrame) -> pd.DataFrame:
    isoflop_df = raw_isoflop_df.copy()
    # filter global outliers or failed runs
    isoflop_df = isoflop_df.dropna(axis=0)

    # filter run with loss spike in 3e19 FLOP budget
    isoflop_df = isoflop_df[
        ~(
            (isoflop_df["IsoFLOP"] == "3e+19")
            & (isoflop_df["num_params"] == 1223934208.0)
            # | (isoflop_df["num_params"] == 1339894528.0)
        )
    ]
    return isoflop_df


def _filter_isoflop_df_mlstm_ctx2048(raw_isoflop_df: pd.DataFrame) -> pd.DataFrame:
    isoflop_df = raw_isoflop_df.copy()
    # filter global outliers or failed runs
    isoflop_df = isoflop_df.dropna(axis=0)

    # filter global outliers
    isoflop_df = isoflop_df[isoflop_df["val/.dclm_loss"] < 4.5]

    return isoflop_df


def _filter_isoflop_df_mlstm_ctx16384(raw_isoflop_df: pd.DataFrame) -> pd.DataFrame:
    isoflop_df = raw_isoflop_df.copy()
    # filter global outliers or failed runs
    isoflop_df = isoflop_df.dropna(axis=0)

    # filter global outliers
    isoflop_df = isoflop_df[isoflop_df["val/.dclm_loss"] < 4.5]

    return isoflop_df


def _filter_isoflop_df_llama_ctx16384(raw_isoflop_df: pd.DataFrame) -> pd.DataFrame:
    isoflop_df = raw_isoflop_df.copy()
    # filter global outliers or failed runs
    isoflop_df = isoflop_df.dropna(axis=0)

    # filter global outliers
    isoflop_df = isoflop_df[isoflop_df["val/.dclm_loss"] < 4.5]

    return isoflop_df


# Filter function mapping for different data specifiers
# to their respective filter functions.
data_specifier_to_filter_func_mapping = {
    RunDataSet.ISOFLOP_MLSTM_CTX8192.value: _filter_isoflop_df_mlstm_ctx8192,
    RunDataSet.ISOFLOP_LLAMA_CTX8192.value: _filter_isoflop_df_llama_ctx8192,
    RunDataSet.ISOFLOP_MLSTM_CTX2048.value: _filter_isoflop_df_mlstm_ctx2048,
    RunDataSet.ISOFLOP_LLAMA_CTX2048.value: _filter_isoflop_df_llama_ctx2048,
    RunDataSet.ISOFLOP_MLSTM_CTX16384.value: _filter_isoflop_df_mlstm_ctx16384,
    RunDataSet.ISOFLOP_LLAMA_CTX16384.value: _filter_isoflop_df_llama_ctx16384,
}


def get_isoflop_datapoints_for_ctx(context_length: int) -> pd.DataFrame:
    """Convenience function that returns the datapoints for plotting for a specific context length."""

    assert context_length in [2048, 8192, 16384], (
        "Context length must be either 2048, 8192 or 16384."
    )

    llama_isoflop_df = create_filtered_isoflop_data_table(
        data_specifier=f"isoflop_llama_ctx{context_length}",
        return_only_selected_runs=True,
    )
    mlstm_isoflop_df = create_filtered_isoflop_data_table(
        data_specifier=f"isoflop_mlstm_ctx{context_length}",
        return_only_selected_runs=True,
    )

    # Concatenate the two dataframes
    isoflop_df = pd.concat(
        [llama_isoflop_df, mlstm_isoflop_df], ignore_index=True, axis=0
    )
    return isoflop_df


def get_isoflop_datapoints_for_compute(compute: str, model_name=None) -> pd.DataFrame:
    """Convenience function that returns the datapoints for plotting for a specific context length."""

    assert compute in ["6e+18", "1e+19", "3e+19"], (
        "Compute must be either '6e+18', '1e+19' or '3e+19'."
    )

    def _load_llama():
        isoflop_llama_ctx2048_df = create_filtered_isoflop_data_table(
            data_specifier="isoflop_llama_ctx2048",
            return_only_selected_runs=True,
        )
        isoflop_llama_ctx8192_df = create_filtered_isoflop_data_table(
            data_specifier="isoflop_llama_ctx8192",
            return_only_selected_runs=True,
        )
        isoflop_llama_ctx16384_df = create_filtered_isoflop_data_table(
            data_specifier="isoflop_llama_ctx16384",
            return_only_selected_runs=True,
        )
        return pd.concat(
            [
                isoflop_llama_ctx2048_df,
                isoflop_llama_ctx8192_df,
                isoflop_llama_ctx16384_df,
            ],
            ignore_index=True,
            axis=0,
        )

    def _load_mlstm():
        isoflop_mlstm_ctx2048_df = create_filtered_isoflop_data_table(
            data_specifier="isoflop_mlstm_ctx2048",
            return_only_selected_runs=True,
        )
        isoflop_mlstm_ctx8192_df = create_filtered_isoflop_data_table(
            data_specifier="isoflop_mlstm_ctx8192",
            return_only_selected_runs=True,
        )
        isoflop_mlstm_ctx16384_df = create_filtered_isoflop_data_table(
            data_specifier="isoflop_mlstm_ctx16384",
            return_only_selected_runs=True,
        )
        return pd.concat(
            [
                isoflop_mlstm_ctx2048_df,
                isoflop_mlstm_ctx8192_df,
                isoflop_mlstm_ctx16384_df,
            ],
            ignore_index=True,
            axis=0,
        )

    dfs = (
        [_load_llama(), _load_mlstm()]
        if model_name is None
        else [_load_llama()]
        if model_name == "llama"
        else [_load_mlstm()]
    )

    isoflop_df = pd.concat(
        dfs,
        ignore_index=True,
        axis=0,
    )

    # convert "context_length" column to string for filtering
    isoflop_df["context_length"] = isoflop_df["context_length"].astype("string")

    # Filter the dataframe for the given compute
    isoflop_df = isoflop_df[isoflop_df["IsoFLOP"] == compute]

    return isoflop_df


def get_isoflop_polyfits_for_ctx(
    context_length: int,
    x_col: Literal["num_tokens_training", "num_params"],
    y_col: Literal["val/.dclm_loss", "train/.loss_mean"] = "val/.dclm_loss",
    model_type_col: str = "model_type",
) -> pd.DataFrame:
    """Convenience function that returns the polynomial fits for a specific context length."""

    assert context_length in [2048, 8192, 16384], (
        "Context length must be either 2048, 8192 or 16384."
    )

    assert x_col in [
        "num_tokens_training",
        "num_params",
    ], "x_col must be either 'num_tokens_training' or 'num_params'."

    model_names = ["mlstm", "llama"]
    model_types = ["mlstm_v1", "llama"]

    polyfit_dfs = []
    for i, model_name in enumerate(model_names):
        isoflop_df = create_filtered_isoflop_data_table(
            data_specifier=f"isoflop_{model_name}_ctx{context_length}",
            return_only_selected_runs=True,
        )
        isoflop_polyfit_df = generate_isoflop_polynomial_fits(
            isoflop_df=isoflop_df,
            x_col=x_col,
            y_col=y_col,
            apply_log10_to_x=True,
            return_full_output=False,
            return_dataframe=True,
        )
        isoflop_polyfit_df[model_type_col] = model_types[i]
        polyfit_dfs.append(isoflop_polyfit_df)

    combined_isoflop_polyfit_df = pd.concat(polyfit_dfs, ignore_index=True, axis=0)

    return combined_isoflop_polyfit_df


def get_isoflop_polyfits_for_compute(
    compute: str,
    x_col: Literal["num_tokens_training", "num_params"],
    y_col: Literal["val/.dclm_loss", "train/.loss_mean"] = "val/.dclm_loss",
    model_type_col: str = "model_type",
) -> pd.DataFrame:
    """Convenience function that returns the polynomial fits for a specific compute."""

    assert compute in ["6e+18", "1e+19", "3e+19"], (
        "Compute must be either '6e+18', '1e+19' or '3e+19'."
    )

    assert x_col in [
        "num_tokens_training",
        "num_params",
    ], "x_col must be either 'num_tokens_training' or 'num_params'."

    model_names = ["mlstm", "llama"]
    model_types = ["mlstm_v1", "llama"]

    polyfit_dfs = []
    for i, model_name in enumerate(model_names):
        isoflop_df = get_isoflop_datapoints_for_compute(
            compute=compute, model_name=model_name
        )
        isoflop_polyfit_df = generate_isoflop_polynomial_fits(
            isoflop_df=isoflop_df,
            x_col=x_col,
            y_col=y_col,
            isoflop_tags=["2048", "8192", "16384"],  # Use context lengths as tags
            isoflop_tag_col="context_length",
            apply_log10_to_x=True,
            return_full_output=False,
            return_dataframe=True,
        )
        isoflop_polyfit_df[model_type_col] = model_types[i]
        polyfit_dfs.append(isoflop_polyfit_df)

    combined_isoflop_polyfit_df = pd.concat(polyfit_dfs, ignore_index=True, axis=0)

    return combined_isoflop_polyfit_df


def get_isoflop_polyfits_for_all_ctx(
    x_col: Literal["num_tokens_training", "num_params"],
    y_col: Literal["val/.dclm_loss", "train/.loss_mean"] = "val/.dclm_loss",
    model_type_col: str = "model_type",
) -> pd.DataFrame:
    """Convenience function that returns the polynomial fits for all context lengths."""

    assert x_col in [
        "num_tokens_training",
        "num_params",
    ], "x_col must be either 'num_tokens_training' or 'num_params'."

    isoflop_polyfit_dfs = []
    for context_length in [2048, 8192, 16384]:
        isoflop_polyfit_df = get_isoflop_polyfits_for_ctx(
            context_length=context_length,
            x_col=x_col,
            y_col=y_col,
            model_type_col=model_type_col,
        )
        isoflop_polyfit_df["context_length"] = context_length
        isoflop_polyfit_dfs.append(isoflop_polyfit_df)

    combined_isoflop_polyfit_df = pd.concat(
        isoflop_polyfit_dfs, ignore_index=True, axis=0
    )

    return combined_isoflop_polyfit_df
