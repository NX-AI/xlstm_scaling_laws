from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ....fitting.fit_power_law import generate_power_law_fit, plot_powerlaw_fits
from ..data import get_isoflop_datapoints_for_ctx, get_isoflop_polyfits_for_ctx
from .common import create_isocurve_plot, get_isoflop_styledict_with_colors


def get_isoflop_powerlaw_plot(
    plot_type: Literal["num_tokens_training", "num_params"],
    y_col: str = "val/.dclm_loss",
    figsize: tuple[float, float] = (12, 5),
    flop_range_for_powerlaw_fit: tuple[float, float] = (5e18, 2e20),
    flop_range_for_powerlaw_plot: tuple[float, float] = (5e18, 1e23),
    llama_alpha_isocurve: float = 0.5,
    llama_alpha_powerlaw: float = 1.0,
    axis_labels: dict[str, str] = {
        "num_tokens_training": "Training Tokens",
        "num_params": "Model Parameters",
    },
    fit_in_log_space: bool = True,
    model_type_label_mapping: dict[str, str] = {
        "llama": "Transformer",
        "mlstm_v1": "xLSTM",
    },
    model_type_datapoints_style_dicts: dict[str, dict[str, dict]] = {
        "llama": {"marker": "x"},
        "mlstm_v1": {"marker": "o"},
    },
    model_type_optimum_style_dicts: dict[str, dict[str, Any]] = {
        "llama": {"marker": "X", "color": "purple", "s": 110, "edgecolor": "black"},
        "mlstm_v1": {"marker": "o", "color": "purple", "s": 100, "edgecolor": "black"},
    },
    model_type_polyfit_style_dicts: dict[str, dict[str, dict]] = {
        "llama": {"linestyle": "dashed"},
        "mlstm_v1": {"linestyle": "solid"},
    },
    xlim_params: tuple[float, float] = (7e7, 11e9),
    xlim_tokens: tuple[float, float] = (1.5e9, 2.5e11),
    xticks_ctx_plots: dict[str, list[float]] = {
        "num_params": [1e8, 4e8, 1e9, 4e9, 10e9],
        "num_tokens_training": [2e9, 4e9, 1e10, 2e10, 4e10, 1e11, 2e11],
    },
    xtick_labels_ctx_plots: dict[str, list[str]] = {
        "num_params": ["100M", "400M", "1B", "4B", "10B"],
        "num_tokens_training": ["2B", "4B", "10B", "20B", "40B", "100B", "200B"],
    },
    yticks_powerlaw_plot: dict[str, list[float]] = {
        "num_params": [1e8, 4e8, 1e9, 4e9, 10e9, 40e9, 100e9],
        "num_tokens_training": [4e9, 1e10, 2e10, 4e10, 1e11, 2e11, 4e11],
    },
    ytick_labels_powerlaw_plot: dict[str, list[str]] = {
        "num_params": ["100M", "400M", "1B", "4B", "10B", "40B", "100B"],
        "num_tokens_training": ["4B", "10B", "20B", "40B", "100B", "200B", "400B"],
    },
    y_axis_labelpad_powerlaw_plot: dict[str, float] = {
        "num_params": 2.0,
        "num_tokens_training": 2.0,
    },
    legend_kwargs={
        "loc": "upper right",
        "ncol": 1,
        "bbox_to_anchor": (1.05, 0.9),
        "frameon": True,
        "facecolor": "white",
        # "alignment": "top",
        # "labelspacing": 1.1,
    },
    gridspec_kw: dict[str, Any] = {"wspace": 0.25},
    add_header: bool = True,
    fig: plt.Figure | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    """Create a plot for the isoflop power law."""

    isoflop_datapoints_df = get_isoflop_datapoints_for_ctx(context_length=8192)
    isoflop_polyfits_df = get_isoflop_polyfits_for_ctx(
        context_length=8192, x_col=plot_type, y_col=y_col
    )

    isoflop_powerlaw_fit_df = generate_power_law_fit(
        flop_to_nparam_ntok_df=isoflop_polyfits_df,
        x_col="flops_mean",
        y_col="x_opt",
        model_type_col="model_type",
        fit_in_log_space=fit_in_log_space,
        select_flop_range=flop_range_for_powerlaw_fit,
    )
    isoflop_tags = isoflop_polyfits_df["isoflop_tag"].unique().tolist()

    if fig is None and ax is None:
        fig, ax = plt.subplots(
            ncols=2, nrows=1, figsize=figsize, gridspec_kw=gridspec_kw
        )

    isoflop_style_dict = get_isoflop_styledict_with_colors(
        isoflop_tags=isoflop_tags, seaborn_color_palette="deep"
    )

    ax_isocurve = create_isocurve_plot(
        ax=ax[0],
        isoflop_tags=[
            "6e+18",
            "1e+19",
            "3e+19",
            "1e+20",
            "6e+20",
        ],  # plot all isoflop tags
        isoflop_df=isoflop_datapoints_df,
        isoflop_polyfit_df=isoflop_polyfits_df,
        x_col=plot_type,
        model_tags_to_plot=["llama", "mlstm_v1"],
        y_col=y_col,
        isoflop_datapoints_style_dicts=isoflop_style_dict,
        isoflop_polyfit_style_dicts=isoflop_style_dict,
        model_type_alpha_override={"llama": llama_alpha_isocurve},
        model_type_datapoints_style_dicts=model_type_datapoints_style_dicts,
        model_type_optimum_style_dicts=model_type_optimum_style_dicts,
        model_type_polyfit_style_dicts=model_type_polyfit_style_dicts,
    )

    xlim = xlim_params if plot_type == "num_params" else xlim_tokens
    xticks = xticks_ctx_plots[plot_type]
    xtick_labels = xtick_labels_ctx_plots[plot_type]
    ax_isocurve.set_xlim(xlim)
    ax_isocurve.set_xticks(xticks)
    ax_isocurve.set_xticklabels(xtick_labels)

    ax_powerlaw = plot_powerlaw_fits(
        ax=ax[1],
        flop_to_nparam_ntok_df=isoflop_polyfits_df,
        powerlaw_fit_df=isoflop_powerlaw_fit_df,
        plot_datapoints=True,
        model_type_alpha_override={"llama": llama_alpha_powerlaw},
        xlim=flop_range_for_powerlaw_plot,
        isoflop_style_dicts=isoflop_style_dict,
        add_fit_result_to_legend_label=True,
        plot_type=plot_type,
        model_type_optimum_style_dict=model_type_optimum_style_dicts,
        legend_kwargs={"fontsize": 12},
    )
    ax_powerlaw.set_ylabel(
        axis_labels[plot_type], labelpad=y_axis_labelpad_powerlaw_plot[plot_type]
    )

    ax_powerlaw.set_yticks(yticks_powerlaw_plot[plot_type])
    ax_powerlaw.set_yticklabels(ytick_labels_powerlaw_plot[plot_type])

    handles, labels = ax_isocurve.get_legend_handles_labels()
    legend_label_handle_map = {label: handle for handle, label in zip(handles, labels)}

    def _get_isoflop_legend_elements(legend_label_handle_map: dict) -> list:
        legend_elements = [
            Patch(color="none", label=" " * 16),
        ]
        # extract the unique isoflop tags from the legend labels
        isoflop_tags = {
            float(label.split("_")[1]) for label in legend_label_handle_map.keys()
        }
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
            Patch(color="none", label=" " * 16),
        ]
        # extract the unique model tags from the legend labels
        model_tags = {
            "_".join(label.split("_")[2:]) for label in legend_label_handle_map.keys()
        }
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
            Patch(color="none", label=" " * 16),
        ]
        # extract the unique model tags from the legend labels
        model_tags = {
            "_".join(label.split("_")[2:]) for label in legend_label_handle_map.keys()
        }
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
                    color="black",
                    markersize=markersize,
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

    if add_header:
        for text, offset in zip(
            [
                r"$\mathbf{Compute}$",
                r"$\mathbf{Training\ Runs}$",
                r"$\mathbf{FLOP\ Optima}$",
                # ], [0.0, 0.375, 0.57]): # for 4.6 height
            ],
            [0.01, 0.435, 0.658],
        ):  # for 4.0 height
            fig.text(
                0.932,  # 0.979, # align center
                0.86 - offset,
                text,
                ha="left",
                va="top",
                fontsize=14,
                zorder=99,
            )

    return fig
