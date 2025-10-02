from typing import Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.optimize import curve_fit

"""In this file we define the code that is used to do linear fits for the training FLOPs vs. val loss plot.
"""


def linear_fit(x, a, b):
    return a * x + b


def fit_token_param_ratio_linear_loglog(
    token_param_ratio_df: pd.DataFrame,
    x_col: str = "num_flops_training",
    y_col: str = "val/.dclm_loss",
) -> tuple[np.ndarray, np.ndarray]:
    """Fits a linear function to the log-log data.

    Args:
        token_param_ratio_df: The dataframe containing the data to fit.
        x_col: The column containing the x data. Defaults to "num_flops_training".
        y_col: The column containing the y data. Defaults to "val/.dclm_loss".

    Returns:
        tuple[np.ndarray, np.ndarray]: The parameters of the fit and the covariance matrix.
    """
    x = np.log(token_param_ratio_df[x_col].values)
    y = np.log(token_param_ratio_df[y_col].values)
    popt, pcov = curve_fit(f=linear_fit, xdata=x, ydata=y)
    return popt, pcov


def generate_linear_fits_for_token_param_ratios(
    all_token_param_ratios_df: pd.DataFrame,
    token_param_ratios: list[str],
    x_col: str = "num_flops_training",
    y_col: str = "val/.dclm_loss",
    token_param_ratio_col: str = "Preset Token Param Ratio",
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generates linear fits for the token param ratios.

    Args:
        all_token_param_ratios_df: The dataframe containing all the token param ratios.
        token_param_ratios: The token param ratios to fit.
        x_col: The column containing the x data. Defaults to "num_flops_training".
        y_col: The column containing the y data. Defaults to "val/.dclm_loss".

    Returns:
        dict[str, tuple[np.ndarray, np.ndarray]]: A dictionary containing the fits for each token param ratio.
    """
    fits = {}
    for token_param_ratio in token_param_ratios:
        token_param_ratio_df = all_token_param_ratios_df[
            all_token_param_ratios_df[token_param_ratio_col] == str(token_param_ratio)
        ]
        fits[token_param_ratio] = fit_token_param_ratio_linear_loglog(
            token_param_ratio_df, x_col, y_col
        )
    return fits


def create_slope_summary_df(
    all_token_param_ratios_df: pd.DataFrame,
    token_param_ratios: list[str],
    x_col: str = "num_flops_training",
    y_col: str = "val/.dclm_loss",
    token_param_ratio_col: str = "Preset Token Param Ratio",
) -> pd.DataFrame:
    fits = generate_linear_fits_for_token_param_ratios(
        all_token_param_ratios_df=all_token_param_ratios_df,
        token_param_ratios=token_param_ratios,
        x_col=x_col,
        y_col=y_col,
        token_param_ratio_col=token_param_ratio_col,
    )

    def _create_summary_dict_for_token_param_ratio(fit: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
        slope, intercept = fit[0]
        stderr_slope, stderr_intercept = np.diag(fit[1])
        return {"slope": slope, "intercept": intercept, "stderr_slope": stderr_slope, "stderr_intercept": stderr_intercept}
    
    summary_dict = {}
    for token_param_ratio, fit in fits.items():
        summary_dict[token_param_ratio] = _create_summary_dict_for_token_param_ratio(fit)
    return pd.DataFrame(summary_dict).T


def plot_linear_fits_for_token_param_ratios(
    ax: Axes,
    all_token_param_ratios_df: pd.DataFrame,
    token_param_ratios: list[str],
    fits: dict[str, tuple[np.ndarray, np.ndarray]],
    style_dict: dict[str, dict] = None,
    x_col: str = "num_flops_training",
    token_param_ratio_col: str = "Preset Token Param Ratio",
    xdata_non_log: np.ndarray = None,
    plot_mode: Literal[
        "interpolate", "extrapolate", "interpolate_extrapolate"
    ] = "interpolate",
) -> Axes:
    """Plots the linear fits for the token param ratios into an pre-existing axis."""

    if "extrapolate" in plot_mode:
        assert xdata_non_log is not None, (
            "xdata_non_log must be provided for extrapolation."
        )

    for token_param_ratio in token_param_ratios:
        token_param_ratio_df = all_token_param_ratios_df[
            all_token_param_ratios_df[token_param_ratio_col] == str(token_param_ratio)
        ]
        x_inter = np.log(token_param_ratio_df[x_col].values)

        if xdata_non_log is not None:
            x_extra = np.log(xdata_non_log)

        popt, _ = fits[token_param_ratio]
        if style_dict is None:
            style_dict = {}
        if "interpolate" in plot_mode:
            style_dict_interpolate = style_dict[token_param_ratio].copy()
            style_dict_interpolate["alpha"] = 1.0
            ax.plot(
                np.exp(x_inter),
                np.exp(linear_fit(x_inter, *popt)),
                **style_dict_interpolate,
            )
        if "extrapolate" in plot_mode:
            ax.plot(
                np.exp(x_extra),
                np.exp(linear_fit(x_extra, *popt)),
                linestyle="--",
                **style_dict[token_param_ratio],
            )

    return ax
