from typing import Any, Literal

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from ..data import get_isoflop_datapoints_for_ctx, get_isoflop_polyfits_for_ctx
from .common import (
    create_isocurve_plot,
    get_context_styledict_with_colors,
    get_isoflop_styledict_with_colors,
)
from .ctx_powerlaw import create_ctx_powerlaw_plot


def get_isoflop_ctx_plot(
    x_col: Literal["num_tokens_training", "num_params"] = "num_tokens_training",
    y_col: Literal["val/.dclm_loss", "train/.loss_mean"] = "val/.dclm_loss",
    seaborn_color_palette: str = "deep",
    isoflop_style_dicts: dict[str, dict[str, dict]] | None = None,
    model_type_datapoints_style_dicts: dict[str, dict[str, dict]] = {
        "llama": {"marker": "x"},
        "mlstm_v1": {"marker": "o"},
    },
    model_type_polyfit_style_dicts: dict[str, dict[str, dict]] = {
        "llama": {"linestyle": "dashed"},
        "mlstm_v1": {"linestyle": "solid"},
    },
    model_type_optimum_style_dicts: dict[str, dict[str, Any]] = {
        "llama": {"marker": "X", "color": "purple", "s": 110, "edgecolor": "black"},
        "mlstm_v1": {"marker": "D", "color": "purple", "s": 80, "edgecolor": "black"},
    },
    axis_labels: dict[str, str] = {
        "num_tokens_training": "Training Tokens",
        "num_params": "Model Parameters",
        "val/.dclm_loss": "Validation Loss",
        "train/.loss_mean": "Training Loss",
    },
    legend_kwargs: dict[str, Any] = {
        "loc": "upper right",
        "ncol": 1,
        "bbox_to_anchor": (1.05, 0.9),
        "frameon": True,
        "facecolor": "white",
    },
    model_type_label_mapping: dict[str, str] = {
        "llama": "Transformer",
        "mlstm_v1": "xLSTM",
    },
    ylim: tuple[float, float] = (2.8, 3.5),
    figsize: tuple[float, float] = (14, 6),
) -> Figure:
    """Create the isoflop plots for the given context length."""

    def _add_isocurve_plot(ax: Axes, context_length: int) -> Axes:
        ax = create_isocurve_plot(
            ax=ax,
            isoflop_df=get_isoflop_datapoints_for_ctx(context_length=context_length),
            isoflop_polyfit_df=get_isoflop_polyfits_for_ctx(
                context_length=context_length, x_col=x_col, y_col=y_col
            ),
            x_col=x_col,
            y_col=y_col,
            isoflop_tags=isoflop_tags,
            isoflop_datapoints_style_dicts=isoflop_style_dicts,
            isoflop_polyfit_style_dicts=isoflop_style_dicts,
            model_type_datapoints_style_dicts=model_type_datapoints_style_dicts,
            model_type_polyfit_style_dicts=model_type_polyfit_style_dicts,
            model_type_optimum_style_dicts=model_type_optimum_style_dicts,
            axis_labels=axis_labels,
            use_isoflop_color_for_optimum=True,
        )
        ax.set_title(f"Context Length {context_length}")
        return ax

    isoflop_tags = ["6e+18", "1e+19", "3e+19"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figsize,
        sharex=True,
        sharey=False,
        # gridspec_kw={"wspace": 0.2, "hspace": 0.5},
    )

    if isoflop_style_dicts is None:
        isoflop_style_dicts = get_isoflop_styledict_with_colors(
            isoflop_tags=isoflop_tags, seaborn_color_palette=seaborn_color_palette
        )

    # Create the isocurve plot for context length 2048
    ax_2048 = _add_isocurve_plot(ax=axes[0], context_length=2048)
    ax_8192 = _add_isocurve_plot(ax=axes[1], context_length=8192)
    ax_16384 = _add_isocurve_plot(ax=axes[2], context_length=16384)

    # ax_2048.minorticks_on()
    # ax_8192.minorticks_on()
    # # ax_8192.grid(which="major", linestyle="-", linewidth=0.5)  # Both major and minor grids
    # ax_8192.tick_params(axis='x', which='both', direction='in', length=5)  # Major and minor ticks
    # ax_8192.tick_params(axis="x", which="minor", length=4, width=1)
    ax_2048.grid(
        which="minor", color="lightgrey", linestyle="-", linewidth=0.5
    )  # Minor grid customization
    ax_8192.grid(
        which="minor", color="lightgrey", linestyle="-", linewidth=0.5
    )  # Minor grid customization
    ax_16384.grid(
        which="minor", color="lightgrey", linestyle="-", linewidth=0.5
    )  # Minor grid customization

    ax_2048.set_ylim(ylim)
    ax_8192.set_ylim(ylim)
    ax_16384.set_ylim(ylim)
    # add a figlegend
    handles, labels = ax_8192.get_legend_handles_labels()

    legend_label_handle_map = {label: handle for handle, label in zip(handles, labels)}

    def _get_isoflop_legend_elements(legend_label_handle_map: dict) -> list:
        legend_elements = [
            Patch(color="none", label=r'$\mathbf{Compute}$'),
        ]
        # extract the unique isoflop tags from the legend labels
        isoflop_tags = set(
            float(label.split("_")[1]) for label in legend_label_handle_map.keys()
        )
        isoflop_tags = sorted(isoflop_tags)
        # extract the colors for each isoflop tag
        for isoflop in isoflop_tags:
            color = legend_label_handle_map[f"fit_{isoflop}_mlstm_v1"]._color
            legend_elements.append(
                Line2D([0], [0], linewidth=3, color=color, label=f"{isoflop}")
            )
        return legend_elements

    def _get_datapoint_legend_elements(legend_label_handle_map: dict) -> list:
        legend_elements = [
            Patch(color="none", label=r'$\mathbf{Training \ Runs}$'),
        ]
        # extract the unique model tags from the legend labels
        model_tags = set(
            "_".join(label.split("_")[2:]) for label in legend_label_handle_map.keys()
        )
        model_tags = sorted(model_tags)
        # extract the colors for each model tag
        for model_tag in model_tags:
            marker = model_type_datapoints_style_dicts.get(model_tag, {}).get(
                "marker", "o"
            )
            linestyle = model_type_polyfit_style_dicts.get(model_tag, {}).get(
                "linestyle", "solid"
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle=linestyle,
                    color="black",
                    label=model_type_label_mapping[model_tag],
                )
            )
        return legend_elements

    def _get_optima_legend_elements(legend_label_handle_map: dict) -> list:
        legend_elements = [
            Patch(color="none", label=r'$\mathbf{FLOP \ Optima}$'),
        ]
        # extract the unique model tags from the legend labels
        model_tags = set(
            "_".join(label.split("_")[2:]) for label in legend_label_handle_map.keys()
        )
        model_tags = sorted(model_tags)
        # extract the colors for each model tag
        for model_tag in model_tags:
            marker = model_type_optimum_style_dicts.get(model_tag, {}).get(
                "marker", "o"
            )
            if model_tag == "llama":
                markersize = 10
            else:
                markersize = 7
            # size = model_type_optimum_style_dict.get(model_tag, {}).get("s", 80)
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    markersize=markersize,
                    markeredgecolor="black",
                    linestyle="None",
                    color="black",
                    label=model_type_label_mapping[model_tag],
                )
            )
        return legend_elements

    legend_elements = []
    legend_elements.extend(_get_isoflop_legend_elements(legend_label_handle_map))
    legend_elements.extend(_get_datapoint_legend_elements(legend_label_handle_map))
    legend_elements.extend(_get_optima_legend_elements(legend_label_handle_map))
    # _get_isoflop_legend_elements(legend_label_handle_map)

    # _get_datapoint_legend_elements(legend_label_handle_map)

    fig.legend(handles=legend_elements, **legend_kwargs)

    return fig


def get_isoflop_powerlaw_ctx_plot(
    x_col: Literal["num_tokens_training", "num_params"] = "num_tokens_training",
    y_col: Literal["val/.dclm_loss", "train/.loss_mean"] = "val/.dclm_loss",
    seaborn_color_palette: str = "deep",
    isoflop_style_dicts: dict[str, dict[str, dict]] | None = None,
    model_type_datapoints_style_dicts: dict[str, dict[str, dict]] = {
        "llama": {"marker": "x"},
        "mlstm_v1": {"marker": "o"},
    },
    model_type_polyfit_style_dicts: dict[str, dict[str, dict]] = {
        "llama": {"linestyle": "dashed"},
        "mlstm_v1": {"linestyle": "solid"},
    },
    model_type_optimum_style_dicts: dict[str, dict[str, Any]] = {
        "llama": {"marker": "X", "color": "purple", "s": 110, "edgecolor": "black"},
        "mlstm_v1": {"marker": "o", "color": "purple", "s": 100, "edgecolor": "black"},
    },
    axis_labels: dict[str, str] = {
        "num_tokens_training": "Training Tokens",
        "num_params": "Model Parameters",
        "val/.dclm_loss": "Validation Loss",
        "train/.loss_mean": "Training Loss",
    },
    legend_kwargs: dict[str, Any] = {
        "loc": "upper right",
        "ncol": 1,
        "bbox_to_anchor": (1.05, 0.9),
        "frameon": True,
        "facecolor": "white",
    },
    model_type_label_mapping: dict[str, str] = {
        "llama": "Transformer",
        "mlstm_v1": "xLSTM",
    },
    ylim: tuple[float, float] = (2.8, 3.55),
    xlim_params: tuple[float, float] = (7e7, 2.5e9),
    xlim_tokens: tuple[float, float] = (1.5e9, 8e10),
    xticks_ctx_plots: dict[str, list[float]] = {
        "num_params": [1e8, 4e8, 1e9, 2e9],
        "num_tokens_training": [2e9, 4e9, 1e10, 2e10, 4e10],
    },
    xtick_labels_ctx_plots: dict[str, list[str]] = {
        "num_params": ["100M", "400M", "1B", "2B"],
        "num_tokens_training": ["2B", "4B", "10B", "20B", "40B"],
    },
    figsize: tuple[float, float] = (14, 6),
    legend_kwargs_powerlaw_plot: dict[str, Any] = {
        "ncols": 2,
        "columnspacing": 0.5,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.3),  # Adjust the vertical position as needed
    },
    yticks_powerlaw_plot: dict[str, list[float]] = {
        "num_params": [1e8, 2e8, 4e8, 6e8, 1e9],
        "num_tokens_training": [4e9, 6e9, 1e10, 2e10],
    },
    ytick_labels_powerlaw_plot: dict[str, list[str]] = {
        "num_params": ["100M", "200M", "400M", "600M", "1B"],
        "num_tokens_training": ["4B", "6B", "10B", "20B"],
    },
    y_axis_labelpad_powerlaw_plot: dict[str, float] = {
        "num_params": -1,
        "num_tokens_training": 0.0,
    },
    gridspec_kw={"wspace": 0.25},
) -> Figure:
    """Create the isoflop plots for the given context length."""

    fit_ctx_color_dict = get_context_styledict_with_colors([2048, 8192, 16384])

    def _add_isocurve_plot(ax: Axes, context_length: int) -> Axes:
        ax = create_isocurve_plot(
            ax=ax,
            isoflop_df=get_isoflop_datapoints_for_ctx(context_length=context_length),
            isoflop_polyfit_df=get_isoflop_polyfits_for_ctx(
                context_length=context_length, x_col=x_col, y_col=y_col
            ),
            x_col=x_col,
            y_col=y_col,
            isoflop_tags=isoflop_tags,
            isoflop_datapoints_style_dicts=isoflop_style_dicts,
            isoflop_polyfit_style_dicts=isoflop_style_dicts,
            model_type_datapoints_style_dicts=model_type_datapoints_style_dicts,
            model_type_polyfit_style_dicts=model_type_polyfit_style_dicts,
            model_type_optimum_style_dicts=model_type_optimum_style_dicts,
            axis_labels=axis_labels,
            use_isoflop_color_for_optimum=True,
            xticks=xticks_ctx_plots.get(x_col, yticks_powerlaw_plot[x_col]),
            xtick_labels=xtick_labels_ctx_plots.get(
                x_col, ytick_labels_powerlaw_plot[x_col]
            ),
        )
        ax.set_title(
            f"Context Length {context_length}", color=fit_ctx_color_dict[context_length]["color"]
        )
        return ax

    isoflop_tags = ["6e+18", "1e+19", "3e+19"]

    fig, axes = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=figsize,
        sharex=False,
        sharey=False,
        gridspec_kw=gridspec_kw,
    )

    if isoflop_style_dicts is None:
        isoflop_style_dicts = get_isoflop_styledict_with_colors(
            isoflop_tags=isoflop_tags, seaborn_color_palette=seaborn_color_palette
        )

    # Create the isocurve plot for all context lengths
    ax_2048 = _add_isocurve_plot(ax=axes[0], context_length=2048)
    ax_8192 = _add_isocurve_plot(ax=axes[1], context_length=8192)
    ax_16384 = _add_isocurve_plot(ax=axes[2], context_length=16384)

    # Create the combined powerlaw plot
    ax_powerlaw = create_ctx_powerlaw_plot(
        ax=axes[3],
        plot_type=x_col,
        y_col=y_col,
        yticks=yticks_powerlaw_plot[x_col],
        ytick_labels=ytick_labels_powerlaw_plot[x_col],
        legend_kwargs=legend_kwargs_powerlaw_plot,
        y_axis_labelpad=y_axis_labelpad_powerlaw_plot[x_col],
        flop_range_for_powerlaw_fit=(5e18, 5e19), # only use the flop budgets 6e18, 1e19, 3e19
    )
    ax_powerlaw.tick_params(axis="y", pad=-3.0)

    # ax_2048.minorticks_on()
    # ax_8192.minorticks_on()
    # # ax_8192.grid(which="major", linestyle="-", linewidth=0.5)  # Both major and minor grids
    # ax_8192.tick_params(axis='x', which='both', direction='in', length=5)  # Major and minor ticks
    # ax_8192.tick_params(axis="x", which="minor", length=4, width=1)
    ax_2048.grid(
        which="minor", color="lightgrey", linestyle="-", linewidth=0.5
    )  # Minor grid customization
    ax_8192.grid(
        which="minor", color="lightgrey", linestyle="-", linewidth=0.5
    )  # Minor grid customization
    ax_16384.grid(
        which="minor", color="lightgrey", linestyle="-", linewidth=0.5
    )  # Minor grid customization

    ax_2048.set_ylim(ylim)
    ax_8192.set_ylim(ylim)
    ax_16384.set_ylim(ylim)
    xlim = xlim_params if x_col == "num_params" else xlim_tokens
    ax_2048.set_xlim(xlim)
    ax_8192.set_xlim(xlim)
    ax_16384.set_xlim(xlim)
    # add a figlegend
    handles, labels = ax_8192.get_legend_handles_labels()

    legend_label_handle_map = {label: handle for handle, label in zip(handles, labels)}

    def _get_isoflop_legend_elements(legend_label_handle_map: dict) -> list:
        legend_elements = [
            # Patch(color="none", label="$\mathbf{Compute}$"),
            Patch(color="none", label=" "*17),
        ]
        # extract the unique isoflop tags from the legend labels
        isoflop_tags = set(
            float(label.split("_")[1]) for label in legend_label_handle_map.keys()
        )
        isoflop_tags = sorted(isoflop_tags)
        # extract the colors for each isoflop tag
        for isoflop in isoflop_tags:
            color = legend_label_handle_map[f"fit_{isoflop}_mlstm_v1"]._color
            legend_elements.append(
                Line2D([0], [0], linewidth=3, color=color, label=f"{isoflop}")
            )
        return legend_elements

    def _get_datapoint_legend_elements(legend_label_handle_map: dict) -> list:
        legend_elements = [
            # Patch(color="none", label="$\mathbf{Training\ Runs}$"),
            Patch(color="none", label=" "*17),
        ]
        # extract the unique model tags from the legend labels
        model_tags = set(
            "_".join(label.split("_")[2:]) for label in legend_label_handle_map.keys()
        )
        model_tags = sorted(model_tags)
        # extract the colors for each model tag
        for model_tag in model_tags:
            marker = model_type_datapoints_style_dicts.get(model_tag, {}).get(
                "marker", "o"
            )
            linestyle = model_type_polyfit_style_dicts.get(model_tag, {}).get(
                "linestyle", "solid"
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    linestyle=linestyle,
                    color="black",
                    label=model_type_label_mapping[model_tag],
                )
            )
        return legend_elements

    def _get_optima_legend_elements(legend_label_handle_map: dict) -> list:
        legend_elements = [
            # Patch(color="none", label=r"$\mathbf{FLOP\ Optima}$"),
            Patch(color="none", label=" "*17),
        ]
        # extract the unique model tags from the legend labels
        model_tags = set(
            "_".join(label.split("_")[2:]) for label in legend_label_handle_map.keys()
        )
        model_tags = sorted(model_tags)
        # extract the colors for each model tag
        for model_tag in model_tags:
            marker = model_type_optimum_style_dicts.get(model_tag, {}).get(
                "marker", "o"
            )
            if model_tag == "llama":
                markersize = 10
            else:
                markersize = 7
            # size = model_type_optimum_style_dict.get(model_tag, {}).get("s", 80)
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    marker=marker,
                    markeredgecolor="black",
                    linestyle="None",
                    markersize=markersize,
                    color="black",
                    label=model_type_label_mapping[model_tag],
                )
            )
        return legend_elements

    legend_elements = []
    legend_elements.extend(_get_isoflop_legend_elements(legend_label_handle_map))
    legend_elements.extend(_get_datapoint_legend_elements(legend_label_handle_map))
    legend_elements.extend(_get_optima_legend_elements(legend_label_handle_map))
    # _get_isoflop_legend_elements(legend_label_handle_map)

    # _get_datapoint_legend_elements(legend_label_handle_map)
    
    ### Look and feel of a singular legend

    rect = FancyBboxPatch((0.09, 1.01), 0.81, 0.40,
                         transform=fig.transFigure,
                         boxstyle="round,pad=0.0,rounding_size=0.004",
                         facecolor='none',
                         edgecolor='lightgrey',
                         mutation_aspect=5,
                         linewidth=0.8)
    fig.patches.append(rect)

    fig.legend(handles=legend_elements, **legend_kwargs)

    for text, offset in zip([
        r"$\mathbf{Compute}$",
        r"$\mathbf{Training\ Runs}$",
        r"$\mathbf{FLOP\ Optima}$",
    ], [0.0, 0.093, 0.208]):
        fig.text(
            0.103 + offset, 1.37,
            text,
            ha='left', va='top',
            fontsize=16
        )

    fig.lines.append(Line2D([0.41]*2, [1.045, 1.37], transform=fig.transFigure, color='lightgrey', linewidth=0.8))

    return fig
