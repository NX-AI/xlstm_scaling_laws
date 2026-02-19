from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from ....fitting.fit_power_law import generate_power_law_fit, plot_powerlaw_fits
from ..data import get_isoflop_polyfits_for_all_ctx
from .common import get_context_styledict_with_colors, get_isoflop_styledict_with_colors


def create_ctx_powerlaw_plot(
    plot_type: Literal["num_tokens_training", "num_params"],
    x_col: str = "flops_mean",
    y_col: str = "val/.dclm_loss",
    flop_range_for_powerlaw_fit: tuple[float, float] | None = (5e18, 2e20),
    flop_range_for_powerlaw_plot: tuple[float, float] = (5e18, 1e20),
    flop_range_for_powerlaw_filter: tuple[float, float] | None = (5e18, 0.5e20),
    fit_in_log_space: bool = True,
    llama_alpha_powerlaw: float = 1,
    ax: Axes | None = None,
    axis_labels: dict[str, str] = {
        "num_tokens_training": "Optimal Training Tokens",
        "num_params": "Optimal Model Parameters",
        "val/.dclm_loss": "Validation Loss",
        "train/.loss_mean": "Training Loss",
    },
    figsize: tuple[float, float] = (6.0, 4.0),
    legend_order: list[str] | None = [
        "Transformer (2048)",
        "Transformer (8192)",
        "Transformer (16384)",
        "xLSTM (2048)",
        "xLSTM (8192)",
        "xLSTM (16384)",
    ],
    legend_kwargs: dict[str, Any] | None = {
        "ncols": 2,
        "columnspacing": 0.5,
        "loc": "upper center",
        "bbox_to_anchor": (0.5, 1.3),  # Adjust the vertical position as needed
    },
    yticks: list[float] | None = None,
    ytick_labels: list[str] | None = None,
    y_axis_labelpad: float | None = None,
) -> Axes:
    """Create a power law plot for the different context length isoflop experiments."""

    all_polyfit_nparam_ntok_df = get_isoflop_polyfits_for_all_ctx(
        x_col=plot_type, y_col=y_col
    )

    isoflop_tags = all_polyfit_nparam_ntok_df["isoflop_tag"].unique().tolist()
    context_lengths = all_polyfit_nparam_ntok_df["context_length"].unique().tolist()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    if x_col == "context_length":
        elements = isoflop_tags
    elif x_col == "flops_mean":
        elements = context_lengths

    for e, elem in enumerate(elements):
        selector = "context_length" if x_col == "flops_mean" else "isoflop_tag"
        isoflop_polyfits_df = all_polyfit_nparam_ntok_df[
            all_polyfit_nparam_ntok_df[selector] == elem
        ]

        # Filter the dataframe to only include rows with FLOPs within the specified range
        # for plotting
        if flop_range_for_powerlaw_filter is not None:
            isoflop_polyfits_df = isoflop_polyfits_df[
                (isoflop_polyfits_df["flops_mean"] >= flop_range_for_powerlaw_filter[0])
                & (
                    isoflop_polyfits_df["flops_mean"]
                    <= flop_range_for_powerlaw_filter[1]
                )
            ]

        # Fit a power law to the data
        isoflop_powerlaw_fit_df = generate_power_law_fit(
            flop_to_nparam_ntok_df=isoflop_polyfits_df,
            x_col=x_col,
            y_col="x_opt",
            fit_in_log_space=fit_in_log_space,
            select_flop_range=flop_range_for_powerlaw_fit,
        )

        isoflop_style_dict = get_isoflop_styledict_with_colors(
            isoflop_tags=isoflop_tags, seaborn_color_palette="deep"
        )

        fit_ctx_color_dict = get_context_styledict_with_colors(context_lengths)

        if x_col == "context_length":
            # exchange the values of the two style dicts
            save = isoflop_style_dict.values()
            isoflop_style_dict = {
                k: v for k, v in zip(isoflop_tags, fit_ctx_color_dict.values())
            }
            fit_ctx_color_dict = {k: v for k, v in zip(context_lengths, save)}

        ax = plot_powerlaw_fits(
            ax=ax,
            flop_to_nparam_ntok_df=isoflop_polyfits_df,
            powerlaw_fit_df=isoflop_powerlaw_fit_df,
            plot_datapoints=True,
            x_col=x_col,
            model_type_alpha_override={"llama": llama_alpha_powerlaw},
            xlim=flop_range_for_powerlaw_plot,
            isoflop_style_dicts=isoflop_style_dict,
            add_fit_result_to_legend_label=True,
            plot_type=plot_type,
            model_type_label_mapping={
                "llama": f"Transformer ({elem})",
                "mlstm_v1": f"xLSTM ({elem})",
            },
            model_type_fit_style_dict={
                "llama": {
                    "color": fit_ctx_color_dict.get(
                        context_lengths[e % len(context_lengths)], {"color": "black"}
                    )["color"],
                    "linestyle": "--",
                },
                "mlstm_v1": {
                    "color": fit_ctx_color_dict.get(
                        context_lengths[e % len(context_lengths)], {"color": "black"}
                    )["color"],
                    "linestyle": "-",
                },
            },
            legend_kwargs=None,
        )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        sorted_labels = []
        for label_ord in legend_order:
            for label in by_label.keys():
                if label.startswith(label_ord):
                    sorted_labels.append(label)
                    break

        # sorted_labels = sorted(by_label.keys())
        sorted_handles = [by_label[label] for label in sorted_labels]
        ax.legend(
            sorted_handles,
            sorted_labels,
            **legend_kwargs,
        )

    ax.set_ylabel(axis_labels[plot_type], labelpad=y_axis_labelpad)
    if yticks is not None:
        ax.set_yticks(yticks)
        if ytick_labels is not None:
            ax.set_yticklabels(ytick_labels)

    # hide minor tick labels
    ax.tick_params(axis="y", which="minor", label1On=False)

    return ax
