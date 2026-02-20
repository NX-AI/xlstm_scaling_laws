from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from .plot_config import FIGSIZE


def create_group_names_from_cols(
    data_df: pd.DataFrame, colnames: str, add_colname: bool = False
) -> list[str]:
    """Create group names from columns in a DataFrame."""
    group_names = []
    group_cols = data_df[colnames].astype(int)
    for i, row in group_cols.iterrows():
        group_str = ""
        for i, colname in enumerate(colnames):
            if add_colname:
                group_str += f"{colname}={row[colname]}"
            else:
                group_str += f"{row[colname]}"
            if i < len(colnames) - 1:
                group_str += "\n"
        group_names.append(group_str)
    return group_names


def create_line_plot(
    data_df: pd.DataFrame,
    group_col_names: list[str],
    title: str = None,
    plot_column_order: list[str] = None,
    style_dict: dict[str, Any] = None,
    legend_args: dict[str, Any] = dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)),
    legend_order: list[str] = None,
    figsize=FIGSIZE,
    grid_alpha: float = 0.2,
    yticks: list[float] = None,
    ylim: tuple[float, float] | None = None,
    x_label: str = "Sequence Length",
    ax: Axes = None,
    add_colname: bool = False,
):
    """Create a line plot for runtime results.
    Simliar to `create_runtime_bar_plot`, but creates a line plot instead of a bar plot.

    Args:
        data_df: DataFrame with the data to plot.
        group_col_names: List of column names to group the bars by.
                         The group names must be columns in the dataframe and are added as x-axis labels.
        title: Title of the plot. Defaults to None.
        plot_column_order: Order of the columns to plot. Defaults to None.
        style_dict: Style dictionary for the plot. Defaults to None.
        legend_args: Legend arguments. Defaults to dict(loc="lower left", bbox_to_anchor=(1.0, 0.0)).
        legend_order: Order of the legend entries. Defaults to None.
        figsize: Figure size. Defaults to FIGSIZE.
        grid_alpha: Alpha value for the grid. Defaults to 0.2.
        yticks: Y-ticks. Defaults to None.
        ylim: Y-limits. Defaults to None.
        x_label: Label for the x-axis. Defaults to "Sequence Length".
        ax: Axis for the plot. Defaults to None.
        add_colname: If True, the column name is added to the group names. Defaults to False.

    Returns:
        The figure object.

    """

    group_names = create_group_names_from_cols(
        data_df=data_df, colnames=group_col_names, add_colname=add_colname
    )
    raw_data_df = data_df.drop(columns=group_col_names)

    # x-axis locations
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=figsize)
        f.suptitle(title)
    else:
        f = ax.get_figure()
        ax.set_title(title)

    if plot_column_order is not None:
        columns = plot_column_order
    else:
        columns = raw_data_df.columns

    for col in columns:
        if style_dict is None:
            ax.plot(range(len(raw_data_df)), raw_data_df[col], label=col, marker="s")
        else:
            ax.plot(
                range(len(raw_data_df)), raw_data_df[col], marker="s", **style_dict[col]
            )

    ax.set_ylabel("Time [ms]")
    ax.set_xlabel(x_label)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xticks(range(len(raw_data_df)), group_names)
    if legend_args and legend_order is None:
        ax.legend(**legend_args)
    elif legend_args and legend_order is not None:
        handles, labels = ax.get_legend_handles_labels()
        label_handle_dict = dict(zip(labels, handles))
        handles = [label_handle_dict[label] for label in legend_order]
        ax.legend(handles=handles, **legend_args)
    ax.grid(alpha=grid_alpha, which="both")

    if yticks is not None:
        ax.set_yticks(yticks)
        # y_formatter.set_scientific(False)
        # y_formatter.set_useOffset(10.0)

    return f
