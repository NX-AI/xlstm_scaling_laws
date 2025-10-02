import logging
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.optimize import curve_fit

LOGGER = logging.getLogger(__name__)


"""In this module we fit a power law to the FLOP-Param and FLOP-Token relationship.
Similar to Approach 1&2 in the Chinchilla paper or Section 3.2.1 in the Scaling Laws paper.
"""


def power_law(x, a, alpha):
    """Power law function."""
    return a * x**alpha


def power_law_log(x, a, alpha):
    """Power law function in log space."""
    return np.log(a) + alpha * np.log(x)


def fit_power_law(
    flop_to_nparam_ntok_df: pd.DataFrame,
    x_col: str = "flop_mean",
    y_col: str = "num_tokens_training",  # "num_params"
    fit_in_log_space: bool = False,
    return_full_output: bool = False,
    curve_fit_kwargs: dict[str, Any] = {},
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    x = flop_to_nparam_ntok_df[x_col].values
    y = flop_to_nparam_ntok_df[y_col].values

    if fit_in_log_space:
        # Fit in log space
        y = np.log(y)
        power_law_func = power_law_log
    else:
        # Fit in normal space
        power_law_func = power_law

    opt_res = curve_fit(
        f=power_law_func,
        xdata=x,
        ydata=y,
        full_output=return_full_output,
        **curve_fit_kwargs,
    )

    popt, pcov = opt_res[:2]

    if return_full_output:
        remaining_output = {
            "info_dict": opt_res[2],
            "mesg": opt_res[3],
            "ier": opt_res[4],
        }
    else:
        remaining_output = {}

    return popt, pcov, remaining_output


def generate_power_law_fit(
    flop_to_nparam_ntok_df: pd.DataFrame,
    x_col: str = "flop_mean",
    y_col: str = "num_tokens_training",  # "num_params"
    model_type_col: str = "model_type",
    fit_in_log_space: bool = False,
    select_flop_range: tuple[float, float] | None = None,
    curve_fit_kwargs: dict[str, Any] = {},
) -> pd.DataFrame:
    """Fit a power law to each model type in the dataframe and return a dataframe with the fit parameters."""

    model_types = flop_to_nparam_ntok_df[model_type_col].unique()
    fit_results = []
    for model_type in model_types:
        model_type_df = flop_to_nparam_ntok_df[
            flop_to_nparam_ntok_df[model_type_col] == model_type
        ]
        if select_flop_range is not None:
            model_type_df = model_type_df[
                (model_type_df[x_col] >= select_flop_range[0])
                & (model_type_df[x_col] <= select_flop_range[1])
            ]
        if model_type_df.empty:
            LOGGER.warning(
                f"No data points for model type {model_type} in the selected FLOP range {select_flop_range}"
            )
            continue
        popt, pcov, _ = fit_power_law(
            flop_to_nparam_ntok_df=model_type_df,
            x_col=x_col,
            y_col=y_col,
            fit_in_log_space=fit_in_log_space,
            curve_fit_kwargs=curve_fit_kwargs,
        )

        a, alpha = popt
        a_std, alpha_std = np.sqrt(np.diag(pcov))
        fit_results.append(
            {
                "model_type": model_type,
                "a": a,
                "alpha": alpha,
                "a_std": a_std,
                "alpha_std": alpha_std,
            }
        )

    fit_results_df = pd.DataFrame(fit_results)
    return fit_results_df


def plot_powerlaw_fits(
    ax: Axes,
    flop_to_nparam_ntok_df: pd.DataFrame,
    powerlaw_fit_df: pd.DataFrame,
    plot_datapoints: bool = True,
    x_col: str = "flops_mean",
    y_col: str = "x_opt",  # "num_params" # "x_opt"
    model_type_optimum_style_dict: dict[str, dict[str, Any]] = {
        "llama": {"marker": "X", "color": "purple", "s": 110, "edgecolor": "black"},
        "mlstm_v1": {"marker": "o", "color": "purple", "s": 100, "edgecolor": "black"},
    },
    model_type_label_mapping: dict[str, str] = {
        "llama":    "Transformer",
        "mlstm_v1": "xLSTM         ",
    },
    model_type_fit_style_dict: dict[str, dict[str, Any]] = {
        "llama": {"color": "black", "linestyle": "--"},
        "mlstm_v1": {"color": "black", "linestyle": "-"},
    },
    isoflop_style_dicts: dict[str, dict[str, Any]]
    | None = None,  # if this is provided, its colors are used for the styledict
    xlim: tuple[float, float] = (1e17, 1e23),
    model_type_alpha_override: dict[str, float] | None = None,
    add_fit_result_to_legend_label: bool = True,
    plot_type: Literal["num_tokens_training", "num_params"] = None, # only used for legend label
    legend_kwargs: dict[str, Any] | None = {},
    num_points: int = 100,
) -> Axes:
    """Plot the power law fits for each model type in the dataframe.
    
    Args:

    
    """

    x_vals = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), num=num_points, base=10)
    for _, row in powerlaw_fit_df.iterrows():
        model_type = row["model_type"]
        a = row["a"]
        alpha = row["alpha"]

        style_dict = model_type_fit_style_dict.get(model_type, {})

        fit_res_label = ""
        if add_fit_result_to_legend_label:
            if plot_type == "num_params":
                fit_res_label = r"$a$=%.3f" % alpha + r", $A'$=%.3f" % a #r"$\alpha$=%.3f" % alpha + r", $A$=%.3f" % a #r"$\alpha = %.3f$" % alpha + r", $A = %.3f$" % a
                if a > 1e4:
                    fit_res_label = r"$a$=%.3f" % alpha + r", $A'$=%.1e" % a
                    # remove the leading 0 from the exponent
                    if a < 1e10:
                        fit_res_label = fit_res_label.split("e+")[0] + "e+" + fit_res_label.split("e+")[1][1]
            elif plot_type == "num_tokens_training":
                fit_res_label = r"$b$=%.3f" % alpha + r", $B'$=%.3f" % a #r"$\beta$=%.3f" % alpha + r", $B$=%.3f" % a #r"$\beta = %.3f$" % alpha + r", $B = %.3f$" % a
                if a > 1e4:
                    fit_res_label = r"$b$=%.3f" % alpha + r", $B'$=%.1e" % a
                    if a < 1e10:
                        # remove the leading 0 from the exponent
                        fit_res_label = fit_res_label.split("e+")[0] + "e+" + fit_res_label.split("e+")[1][1]
        
        y_vals = power_law(x_vals, a, alpha)
        ax.plot(
            x_vals,
            y_vals,
            label=f"{model_type_label_mapping.get(model_type, model_type)} "
            + fit_res_label,
            **style_dict,
            zorder=5,
        )

    if plot_datapoints:
        for model_type in flop_to_nparam_ntok_df["model_type"].unique():
            model_type_df = flop_to_nparam_ntok_df[
                flop_to_nparam_ntok_df["model_type"] == model_type
            ]
            for _, row in model_type_df.iterrows():
                style_dict = model_type_optimum_style_dict.get(model_type, {})
                if "isoflop_tag" in row and isoflop_style_dicts is not None:
                    updated_style_dict = style_dict.copy()
                    updated_style_dict["color"] = isoflop_style_dicts.get(
                        row["isoflop_tag"], style_dict["color"]
                    )["color"]
                else:
                    updated_style_dict = style_dict

                if model_type_alpha_override is not None:
                    updated_style_dict["alpha"] = model_type_alpha_override.get(
                        model_type, 1.0
                    )

                ax.scatter(
                    row[x_col],
                    row[y_col],
                    **updated_style_dict,
                    zorder=10,
                )
    if legend_kwargs is not None:
        ax.legend(**legend_kwargs)
    if x_col == "flops_mean":
        ax.set_xlabel("Compute (FLOPs)")
    elif x_col == "context_length":
        ax.set_xlabel("Context Length")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both")
    ax.grid(which="minor", linestyle="-", linewidth=0.5, color="lightgrey")
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    return ax
