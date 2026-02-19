import copy
from typing import Any, Literal, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def create_compare_line_plot(
    summary_df: pd.DataFrame,
    x_axis: str,
    y_axis: str | list[str],
    compare_parameter: str = "",
    compare_parameter_val_selection: list[Any] = [],
    style_dict: dict[str, dict[str, Any]] = {},
    fallback_style_dict: dict[str, Any] = dict(),
    title: str = None,
    y_label: str = "",
    x_label: str = "",
    ax: Axes = None,
    grid_alpha: float = 0.3,
    xticks: list[float] = None,
    yticks: list[float] = None,
    ylim: tuple[float, float] = (),
    xlim: tuple[float, float] = (),
    legend_label_mode: Literal[
        "include_compare_param_name", "label_name_only"
    ] = "label_name_only",
    legend_args: dict[str, Any] = dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)),
    legend_order: list[str] = None,
    figsize=(2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54),
    yscale: str = None,
    xscale: str = None,
) -> Figure:
    """Function for creating a sweep summary plot.
    Allows for selecting the x- and y-axis parameters separately.
    Typical setup:
    - x-axis: A sweep parameter, e.g. `data.dataset_kwargs.rotation_angle`
    - y-axis: A metric, e.g. `Accuracy-train_step-0`
    - compare_parameter: Compare different parameter setups, e.g. `init_model_step`

    Args:
        summary_df (pd.DataFrame): The sweep summary dataframe.
        x_axis (str): Parameter (column name in summary_df) to plot on the x-axis.
        y_axis (Union[str, list[str]): Parameter (column name in summary_df) to plot on the y-axis.
                                       Can also pass multiple values.
        compare_parameter (str): The compare parameter. Plot a line for each parameter.
        compare_parameter_val_selection (list[Any], optional): If specified, plot only these values. Plots all otherwise. Defaults to [].
        title (str, optional): The title. Defaults to None.
        ax (, optional): The Axes. Defaults to None.
        ylim (tuple[float, float], optional): y-axis limist. Defaults to ().
        figsize (tuple, optional): Size of the Figure. Defaults to (2 * 12 * 1 / 2.54, 2 * 8 * 1 / 2.54).
        savefig (bool, optional): Save the figure. Defaults to False.
        yscale: scale of the y-axis.
        xscale: scale of the x-axis.

    Returns:
        Figure: The matplotlib figure.
    """
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    if not isinstance(y_axis, list):
        y_axis = [y_axis]

    # if no compare parameter given, add the column
    if compare_parameter == "":
        summary_df[""] = "data"

    if isinstance(compare_parameter, list):
        df = pd.DataFrame(summary_df)
        df["_".join(compare_parameter)] = (
            df[compare_parameter].astype(str).apply("_".join, axis=1)
        )
        df.drop(compare_parameter, axis=1, inplace=True)
        summary_df = df
        compare_parameter = "_".join(compare_parameter)

    # select rows from compare parameter
    comp_param_vals = summary_df[compare_parameter].unique()
    if compare_parameter_val_selection:
        comp_val_sel = np.array(compare_parameter_val_selection)
        comp_param_vals = np.intersect1d(comp_param_vals, comp_val_sel)
    # comp_param_vals.sort()
    # sort along x_axis
    summary_df = summary_df.sort_values(by=x_axis, axis=0)

    comp_param_str = (
        compare_parameter.split(".")[-1]
        if isinstance(compare_parameter, str)
        else "_".join(map(str, compare_parameter))
    )

    # get x and y axis
    for cpv in comp_param_vals:
        df = summary_df.loc[summary_df[compare_parameter] == cpv].drop(
            compare_parameter, axis=1
        )
        x_vals = df[x_axis].values
        for y_ax in y_axis:
            y_vals = df[y_ax].values
            if legend_label_mode == "include_compare_param_name":
                label = f"{comp_param_str}={cpv}"
            elif legend_label_mode == "label_name_only":
                label = f"{cpv}"
            else:
                raise ValueError("legend_label_mode not recognized")
            if len(y_axis) > 1:
                label += f"#{y_ax}"
            if label in style_dict:
                style = style_dict[label]
            else:
                style = copy.deepcopy(fallback_style_dict)
                style.update({"label": label})
            ax.plot(x_vals, y_vals, **style)

    if legend_args and legend_order is None:
        ax.legend(**legend_args)
    elif legend_args and legend_order is not None:
        handles, labels = ax.get_legend_handles_labels()
        label_handle_dict = dict(zip(labels, handles))
        handles = [label_handle_dict[label] for label in legend_order]
        ax.legend(handles=handles, **legend_args)

    if xscale is not None:
        ax.set_xscale(xscale)

    if yscale is not None:
        ax.set_yscale(yscale)

    if xticks is not None:
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if yticks is not None:
        ax.set_yticks(yticks)
        y_formatter = plt.ScalarFormatter()
        # y_formatter.set_scientific(False)
        # y_formatter.set_useOffset(10.0)
        ax.get_yaxis().set_major_formatter(y_formatter)

    ax.grid(alpha=grid_alpha, which="both")
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    x_axis_label = x_label if x_label else x_axis
    ax.set_xlabel(x_axis_label)
    y_axis_label = y_label if y_label else y_axis
    ax.set_ylabel(y_axis_label)
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)

    return f
