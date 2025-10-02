import logging
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.optimize import curve_fit

LOGGER = logging.getLogger(__name__)


"""In this file we define the code to fit the polynomials to the isoflop data.
"""


def second_degree_polynomial(
    x: float | np.ndarray,
    a: float,
    b: float,
    c: float,
) -> float | np.ndarray:
    """Second degree polynomial function.

    Args:
        x: The input value(s).
        a: The coefficient for the quadratic term.
        b: The coefficient for the linear term.
        c: The constant term.

    Returns:
        The value of the polynomial at x.
    """
    return a * x**2 + b * x + c


def fit_isoflop_polynomial_for_isoflop_optimum(
    isoflop_df: pd.DataFrame,
    x_col: str = "num_tokens_training",
    y_col: str = "val/.dclm_loss",
    apply_log_to_x: bool = True,
    return_full_output: bool = False,
    curve_fit_kwargs: dict[str, Any] = {},
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fits a second degree polynomial to the isoflop data.

    Args:
        isoflop_df: The dataframe containing the isoflop data.
        x_col: The column containing the x data. Defaults to "num_tokens_training".
        y_col: The column containing the y data. Defaults to "val/.dclm_loss".
        apply_log_to_x: If True, applies log to x. Defaults to True.
        return_full_output: If True, returns additional output from curve_fit. Defaults to False.
        curve_fit_kwargs: Additional arguments to pass to curve_fit.
            Defaults to {}.
    Returns:
        tuple[np.ndarray, np.ndarray, dict[str, Any]]: The parameters of the fit,
            the covariance matrix, and additional output from curve_fit if requested.

    """
    x = isoflop_df[x_col].values
    if apply_log_to_x:
        x = np.log10(x)
    y = isoflop_df[y_col].values

    opt_res = curve_fit(
        f=second_degree_polynomial,
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


def generate_isoflop_polynomial_fits(
    isoflop_df: pd.DataFrame,
    x_col: str = "num_tokens_training",
    y_col: str = "val/.dclm_loss",
    isoflop_tags: list[str] | None = None,
    isoflop_tag_col: str = "IsoFLOP",
    flops_training_col: str = "num_flops_training",
    apply_log10_to_x: bool = True,
    curve_fit_kwargs: dict[str, Any] = {},
    return_full_output: bool = False,
    return_dataframe: bool = True,
) -> dict[str, dict[str, Any]] | pd.DataFrame:
    """Generates polynomial fits for the isoflop data.

    Args:
        isoflop_df: The dataframe containing the isoflop data.
        x_col: The column containing the x data. Defaults to "num_tokens_training".
        y_col: The column containing the y data. Defaults to "val/.dclm_loss".
        isoflop_tags: The isoflop tags to fit. If None, fits all isoflop tags.
            Defaults to None.
        isoflop_tag_col: The column containing the isoflop tags. Defaults to "IsoFLOP".
        apply_log_to_x: If True, applies log to x. Defaults to True.
        curve_fit_kwargs: Additional arguments to pass to curve_fit.
            Defaults to {}.
        return_full_output: If True, returns additional output from curve_fit. Defaults to False.
        return_dataframe: If True, returns a dataframe with the fits. Defaults to True.
        If False, returns a dictionary with the fits.

    Returns:
        A dictionary containing the mean isoflop budget and the fits for each isoflop tag.
        Or a dataframe with the fits for each isoflop tag.
    """
    if isoflop_tags is None:
        isoflop_tags = isoflop_df[isoflop_tag_col].unique()

    fits = {}
    for isoflop_tag in isoflop_tags:
        isoflop_tag_df = isoflop_df[isoflop_df[isoflop_tag_col] == isoflop_tag]
        isoflop_mean = isoflop_tag_df[flops_training_col].mean(axis=0)
        pop, pcov, fit_out = fit_isoflop_polynomial_for_isoflop_optimum(
            isoflop_df=isoflop_tag_df,
            x_col=x_col,
            y_col=y_col,
            apply_log_to_x=apply_log10_to_x,
            return_full_output=return_full_output,
            curve_fit_kwargs=curve_fit_kwargs,
        )
        fits[isoflop_tag] = {
            "flops_mean": isoflop_mean,
            "popt": pop,
            "pcov": pcov,
            "fit_out": fit_out,
        }

    if return_dataframe:
        isoflop_polyfit_df = create_isoflop_polyfit_summary_df(
            isoflop_polyfit_res_dict=fits,
            unapply_log10_to_x=apply_log10_to_x,
        )
        return isoflop_polyfit_df
    else:
        return fits


def create_isoflop_polyfit_summary_df(
    isoflop_polyfit_res_dict: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]],
    unapply_log10_to_x: bool = True,
) -> pd.DataFrame:
    isoflop_rows = []
    for isoflop_tag, isoflop_res_dict in isoflop_polyfit_res_dict.items():
        flop_mean = isoflop_res_dict["flops_mean"]
        popt = isoflop_res_dict["popt"]
        pcov = isoflop_res_dict["pcov"]
        a, b, c = popt
        a_std, b_std, c_std = np.sqrt(np.diag(pcov))

        # determine the minimum
        log_x_opt = -b / (2 * a)
        y_opt = a * log_x_opt**2 + b * log_x_opt + c

        if unapply_log10_to_x:
            x_opt = 10 ** (log_x_opt)
        else:
            x_opt = log_x_opt

        isoflop_row = {
            "isoflop_tag": isoflop_tag,
            "flops_mean": flop_mean,
            "x_opt": x_opt,
            "y_opt": y_opt,
            "a": a,
            "b": b,
            "c": c,
            "a_std": a_std,
            "b_std": b_std,
            "c_std": c_std,
        }
        isoflop_rows.append(isoflop_row)
    isoflop_summary_df = pd.DataFrame(isoflop_rows)
    return isoflop_summary_df


def plot_isoflop_polynomial_fits(
    ax: Axes,
    isoflop_df: pd.DataFrame,
    isoflop_polyfit_df: pd.DataFrame,
    style_dicts: dict[str, dict[str, dict]] = {},
    x_col: str = "num_tokens_training",
    num_points: int = 250,
    isoflop_tags: list[str] | None = None,
    plot_optimum: bool = True,
    use_isoflop_color_for_optimum: bool = True,
    style_dict_optimum: dict[str, Any] = {},
    legend_label_suffix: str | None = None,
) -> Axes:
    if isoflop_tags is None:
        isoflop_tags = isoflop_df["IsoFLOP"].unique()

    for isoflop_tag in isoflop_tags:
        if isoflop_tag not in isoflop_polyfit_df["isoflop_tag"].values:
            LOGGER.warning(
                f"Isoloflop tag {isoflop_tag} not found in polyfit dataframe. Skipping."
            )
            continue
        isoflop_polyfit_row = isoflop_polyfit_df[
            isoflop_polyfit_df["isoflop_tag"] == str(isoflop_tag)
        ].iloc[0]

        # get max and min x values
        if isoflop_tags[0] not in isoflop_df["IsoFLOP"].values:
            isoflop_tag_df = isoflop_df[isoflop_df["context_length"] == str(isoflop_tag)]
        else:
            isoflop_tag_df = isoflop_df[isoflop_df["IsoFLOP"] == str(isoflop_tag)]
        x_min = isoflop_tag_df[x_col].min()
        x_max = isoflop_tag_df[x_col].max()

        x_vals = np.logspace(np.log10(x_min), np.log10(x_max), num_points, base=10.0)
        log_x_vals = np.log10(x_vals)

        a = isoflop_polyfit_row["a"]
        b = isoflop_polyfit_row["b"]
        c = isoflop_polyfit_row["c"]

        y_vals_isoflop_row = second_degree_polynomial(log_x_vals, a, b, c)

        style_dict_isoflop_row = style_dicts.get(
            isoflop_tag,
            {
                "linestyle": "-",
            },
        )
        ax.plot(
            x_vals,
            y_vals_isoflop_row,
            **style_dict_isoflop_row,
            zorder=1,
            label=f"fit_{isoflop_tag}"
            if legend_label_suffix is None
            else f"fit_{isoflop_tag}_{legend_label_suffix}",
        )
        if plot_optimum:
            if use_isoflop_color_for_optimum:
                updated_style_dict_optimum = style_dict_optimum.copy()
                updated_style_dict_optimum["color"] = style_dict_isoflop_row.get(
                    "color", None
                )
            else:
                updated_style_dict_optimum = style_dict_optimum

            x_opt = isoflop_polyfit_row["x_opt"]
            y_opt = isoflop_polyfit_row["y_opt"]
            ax.scatter(
                x_opt,
                y_opt,
                **updated_style_dict_optimum,
                zorder=20,
                label=f"optimum_{isoflop_tag}_{legend_label_suffix}",
            )

    return ax
