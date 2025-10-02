from typing import Any, Literal

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Colormap, LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from xlstm_scaling_laws.analysis.parametric_sclaw_fit.data import (
    get_all_parametric_sclaw_fit_data_dataframe,
)

markers = ["o", "X", "s", "+", "x", "D", "*", "h", "^", "v", "<", ">", "p"]


def create_run_data_scatter_plot(
    data_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    c_col: str,
    style_col: str,
    style_tags_legend: dict[str, str] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    clabel: str = "",
    yscale: str = "linear",
    xscale: str = "linear",
    xticks: list[float] | None = [2e9, 4e9, 1e10, 2e10, 4e10, 1e11, 4e11, 1e12, 2e12],
    xticklabels: list[str] | None = [
        "2B",
        "4B",
        "10B",
        "20B",
        "40B",
        "100B",
        "400B",
        "1T",
        "2T",
    ],
    yticks: list[float] | None = [1e8, 4e8, 1e9, 4e9, 10e9],
    yticklabels: list[str] | None = ["100M", "400M", "1B", "4B", "10B"],
    ax: Axes | None = None,
    c_norm: Normalize | None = None,
    colormap: str | Colormap = "rocket",
    legend_kwargs: dict[str, Any] = {
        "loc": "lower left",
        "ncol": 1,
        "bbox_to_anchor": (0.01, 1.01),
        "frameon": True,
        "facecolor": "white",
    },
    add_colorbar: bool = True,
    add_legend: bool = True,
    figsize: tuple[float, float] = (6, 4.5),
) -> Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    # if style_dict is None:
    #     style_dict = {
    #         tag: {"marker": marker, "label": tag, "edgecolor": "white"}
    #         for tag, marker in zip(data_df[style_col].unique(), markers)
    #     }

    # for style_tag, style_dict_for_tag in style_dict.items():
    #     # select the data for the current style tag
    #     data_df_style = data_df[data_df[style_col] == style_tag]

    #     # plot the data points
    #     ax.scatter(
    #         x=data_df_style[x_col],
    #         y=data_df_style[y_col],
    #         c=data_df_style[c_col],
    #         **style_dict_for_tag,
    #         norm=c_norm,
    #         cmap=colormap,
    #     )

    # we use seaborn instead
    ax = sns.scatterplot(
        data=data_df,
        x=x_col,
        y=y_col,
        hue=c_col,
        style=style_col,
        palette=colormap,
        ax=ax,
        hue_norm=c_norm,
        legend=True,
    )

    sns_legend_handles, sns_legend_labels = ax.get_legend_handles_labels()
    ax.legend_.remove()
    if style_tags_legend is not None:
        # create a legend for the style tags
        legend_handles = []
        legend_labels = []
        for legend_handle, legend_label_dataframe in zip(
            sns_legend_handles, sns_legend_labels
        ):
            if legend_label_dataframe in style_tags_legend:
                legend_handles.append(legend_handle)
                legend_labels.append(style_tags_legend[legend_label_dataframe])

        ax.legend(handles=legend_handles, labels=legend_labels, **legend_kwargs)
        if not add_legend:
            ax.legend_.remove()

    if add_colorbar:
        cbar = plt.colorbar(ax=ax, mappable=ScalarMappable(cmap=colormap, norm=c_norm))
        # Remove minor ticks from the colorbar
        cbar.ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        cbar.set_label(clabel)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.grid(which="both")
    ax.grid(which="minor", linestyle="-", linewidth=0.5, color="lightgrey")
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    return ax


def get_combined_run_data_scatter_plot(
    coloraxis: Literal["loss", "flops"] = "flops",
    add_axes_title: bool = True,
    colormap: str | Colormap | None = None,
    figsize: tuple[float, float] = (12, 4.5),
    legend_kwargs: dict[str, Any] = {
        "loc": "upper left",
        "bbox_to_anchor": (0.91, 0.91),
        "frameon": True,
        "facecolor": "white",
    },
) -> tuple[Figure, Axes]:
    if colormap is None:
        rocket_cmap_full = sns.color_palette(palette="rocket", as_cmap=True)
        rocket_darkside = rocket_cmap_full(
            np.linspace(0.0, 0.9, 256)
        )  # Adjust range for pink side
        rocket_dark_cmap = LinearSegmentedColormap.from_list(
            "rocket_darkside", rocket_darkside
        )
        colormap = rocket_dark_cmap
    elif isinstance(colormap, str):
        colormap = sns.color_palette(palette=colormap, as_cmap=True)
    else:
        colormap = colormap

    mlstm_df = get_all_parametric_sclaw_fit_data_dataframe(model_type="mlstm")
    llama_df = get_all_parametric_sclaw_fit_data_dataframe(model_type="llama")

    print(len(llama_df), "Llama Runs")
    print(len(mlstm_df), "xLSTM Runs")

    if coloraxis == "flops":
        c_col = "num_flops_training"
        c_label = "Training Compute (FLOPs)"
        c_norm = LogNorm(
            vmin=mlstm_df["num_flops_training"].min(),
            vmax=mlstm_df["num_flops_training"].max(),
        )
    elif coloraxis == "loss":
        c_col = "val/.dclm_loss"
        c_label = "Validation Loss"
        c_norm = Normalize(
            vmin=mlstm_df["val/.dclm_loss"].min(),
            vmax=mlstm_df["val/.dclm_loss"].max(),
        )
    else:
        raise ValueError(
            f"coloraxis must be 'loss' or 'flops', got {coloraxis} instead."
        )

    x_col = "num_tokens_training"
    xlabel = "Training Tokens"
    y_col = "num_params"
    ylabel = "Model Parameters"
    style_col = "experiment_set_ctx_length"
    style_tags_legend = {
        "isoflop_ctx2048": "IsoFLOP ctx=2048",
        "isoflop_ctx8192": "IsoFLOP ctx=8192",
        "isoflop_ctx16384": "IsoFLOP ctx=16384",
        "tokenparam_ctx8192": "TokenParam ctx=8192",
    }
    xticks = [2e9, 4e9, 1e10, 3e10, 1e11, 4e11, 1e12, 2e12]
    xticklabels = [
        "2B",
        "4B",
        "10B",
        # "20B",
        "30B",
        # "40B",
        "100B",
        "400B",
        "1T",
        "2T",
    ]
    yticks = [1e8, 4e8, 1e9, 4e9, 10e9]
    yticklabels = ["100M", "400M", "1B", "4B", "10B"]

    # fig = plt.figure(figsize=(12, 4.5))  # Adjust figure size as needed
    # gs = gridspec.GridSpec(
    #     1,
    #     3,
    #     width_ratios=[1, 1, 0.05],
    #     wspace=0.3,
    # )

    # # Create the first two scatter plots
    # ax1 = fig.add_subplot(gs[0, 0])
    # ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)

    # # Create the colorbar axis
    # cbar_ax = fig.add_subplot(gs[0, 2])

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharex=True,
        sharey=False,
        gridspec_kw={
            "wspace": 0.25,
            "hspace": 0,
            "width_ratios": [0.47, 0.58],
        },
    )
    ax1, ax2 = ax.tolist()

    ax1 = create_run_data_scatter_plot(
        data_df=llama_df,
        x_col=x_col,
        y_col=y_col,
        c_col=c_col,
        style_col=style_col,
        style_tags_legend=style_tags_legend,
        xscale="log",
        yscale="log",
        c_norm=c_norm,
        colormap=colormap,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=c_label,
        xticks=xticks,
        xticklabels=xticklabels,
        yticks=yticks,
        yticklabels=yticklabels,
        ax=ax1,
        legend_kwargs={},
        add_colorbar=False,
        add_legend=False,
    )
    ax2 = create_run_data_scatter_plot(
        data_df=mlstm_df,
        x_col=x_col,
        y_col=y_col,
        c_col=c_col,
        style_col=style_col,
        style_tags_legend=style_tags_legend,
        xscale="log",
        yscale="log",
        c_norm=c_norm,
        colormap=colormap,
        xlabel=xlabel,
        ylabel=ylabel,
        clabel=c_label,
        xticks=xticks,
        xticklabels=xticklabels,
        yticks=yticks,
        yticklabels=yticklabels,
        ax=ax2,
        legend_kwargs={},
        add_colorbar=False,
        add_legend=True,
    )

    cbar = fig.colorbar(ax=ax2, mappable=ScalarMappable(cmap=colormap, norm=c_norm))
    # Remove minor ticks from the colorbar
    cbar.ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    cbar.set_label(c_label, labelpad=2)

    if add_axes_title:
        ax1.set_title(f"Transformer ({len(llama_df)} Runs)")
        ax2.set_title(f"xLSTM ({len(mlstm_df)} Runs)")

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend_.remove()
    if style_tags_legend is not None:
        # create a legend for the style tags
        legend_handles = [Patch(color="none")]
        legend_labels = [" "]
        current_label = None
        for legend_handle, legend_label in zip(handles, labels):
            if legend_label in style_tags_legend:
                if legend_label.split("_")[0] != current_label:
                    if current_label is not None:
                        legend_handles.extend([Patch(color="none")]*2)
                        legend_labels.extend([" "]*2)
                    current_label = legend_label.split("_")[0]
                legend_handles.append(legend_handle)
                legend_labels.append(style_tags_legend[legend_label].split(" ")[1])

        fig.legend(
            handles=legend_handles, 
            labels=legend_labels,
            loc="center right",
            bbox_to_anchor=(1.08, 0.5),
            frameon=True,
            facecolor="white",
            fontsize=12,
            markerscale=1.5
        )

        # add text
        fig.text(
            1.012,
            0.735,
            r"$\mathbf{IsoFLOP}$",
            ha="center",
            va="center",
            fontsize=12,
            zorder=99,)
        fig.text(
            1.012,
            0.335,
            r"$\mathbf{Token/Param}$",
            ha="center",
            va="center",
            fontsize=12,
            zorder=99,)

    return fig
