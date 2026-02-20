from collections.abc import Callable
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from xlstm_scaling_laws.fitting.fit_parametric_loss.plot_parametric_loss_fit import (
    get_model_config_df_dict,
    get_param_fit_sclaw_data_df_dict,
)
from xlstm_scaling_laws.fitting.fit_parametric_loss.scaling_law_funcs import (
    get_first_n_fits_as_fit_fn_dict,
)


def convex_hull(points):
    pts = np.array(points)
    indices = np.arange(len(points))
    sorted_order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[sorted_order]
    indices = indices[sorted_order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, lower_idx = [], []
    for i, p in zip(indices, pts):
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
            lower_idx.pop()
        lower.append(tuple(p))
        lower_idx.append(i)
    upper, upper_idx = [], []
    for i, p in zip(indices[::-1], pts[::-1]):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
            upper_idx.pop()
        upper.append(tuple(p))
        upper_idx.append(i)
    hull = np.array(lower[:-1] + upper[:-1])
    hull_idx = np.array(lower_idx[:-1] + upper_idx[:-1])
    return hull, hull_idx


def get_pareto_frontier_single_plot(
    combined_fit_grid_df: pd.DataFrame,
    experiment_set_fit: Literal["all", "tokenparam", "isoflop"] = "tokenparam",
    experiment_set_plot_data_points: Literal[
        "all", "tokenparam", "isoflop"
    ] = "tokenparam",
    datapoints_num_param_selection: dict[str, list[float]] | None = {
        "mlstm": [
            164110224.0,
            406856896.0,
            # 841643808.0,
            841496256.0,
            # 1421232512.0,
            1420839104.0,
            2780449600.0,
            # 2781269120.0,
            6865424896.0,
        ],
        "llama": [
            162220800.0,
            406635520.0,
            834086400.0,
            1420396544.0,
            2779548160.0,
            6863196160.0,
        ],
    },
    model_configs: dict[str, pd.DataFrame] = get_model_config_df_dict(
        context_length=8192
    ),
    hull_colors: dict[str, str] = {
        "llama": "C3",
        "mlstm": "C0",
    },
    token_param_range: Literal["0-100", "0-300", "0-5000"] = "0-5000",
    model_size_colormap: Callable | None = None,
    model_size_colormap_scale: Literal[
        "linear", "log"
    ] = "log",  # scale for the color map
    x_axis_mode: Literal["num_flops", "token_param_ratio"] = "num_flops",
    x_axis_mode_to_x_col: dict[str, str] = {
        "token_param_ratio": "token_param_ratio",
        "num_tokens": "num_tokens_training",
        "num_flops": "num_flops_training",
    },
    y_col: str = "val/.dclm_loss",
    num_params_col: str = "num_params",
    figsize_single_subplot: tuple[float, float] = (5.5, 4),
    model_tags_label_map: dict[str, str] = {
        "llama": "Transformer",
        "mlstm": "xLSTM",
    },
    data_points_style_dict: dict[str, dict] = {
        "llama": {"marker": "x"},
        "mlstm": {"marker": "o"},
    },
    x_axis_labels: dict[str, str] = {
        "token_param_ratio": "Token / Param Ratio",
        "num_tokens": "Training Tokens",
        "num_flops": "Compute (FLOPs)",
    },
    x_axis_ticks: dict[str, list[float] | None] = {
        "token_param_ratio": [2, 10, 20, 50, 100, 200, 500, 1000, 2000],
        "num_tokens": None,
        "num_flops": None,
    },
    x_axis_tick_labels: dict[str, list[str] | None] = {
        "token_param_ratio": [2, 10, 20, 50, 100, 200, 500, 1000, 2000],
        "num_tokens": None,
        "num_flops": None,
    },
    xlim: dict[str, tuple[float, float]] | None = {
        "token_param_ratio": (10.0, 3000.0),
        "num_tokens": None,
        "num_flops": (1.5e18, 1.5e23),
    },
    ylim: dict[str, tuple[float, float]] | None = {
        "token_param_ratio": (2.06, 3.44),
        "num_tokens": None,
        "num_flops": (2.06, 3.44),
    },
    y_axis_label: str = "Validation Loss",
    legend_kwargs: dict = {
        "loc": "upper left",
        "bbox_to_anchor": (0.6, 0.95),
        "ncol": 1,
    },
    xscale: Literal["linear", "log"] = "log",
    yscale: Literal["linear", "log"] = "log",
    fig: Figure | None = None,
    ax: Axes | None = None,
) -> Figure:
    """Plot mLSTM and Llama scaling law fits in a single plot."""

    if fig is None or ax is None:
        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(figsize_single_subplot[0], figsize_single_subplot[1]),
            squeeze=True,
        )

    # model_tags = model_tags_label_map.keys()

    # parametric_loss_fn_dicts = {}
    # for model_tag in model_tags:
    #     model_tag_fit_df = combined_fit_grid_df.loc[
    #         model_tag, experiment_set_fit, token_param_range
    #     ]

    #     model_func_dict = get_first_n_fits_as_fit_fn_dict(
    #         result_fit_df=model_tag_fit_df,
    #         key_prefix=f"{model_tag}-{experiment_set_fit}-{token_param_range}__",
    #         n=1,
    #         return_mode="logsumexp",
    #     )
    #     parametric_loss_fn_dicts[model_tag] = model_func_dict

    param_fit_sclaw_data_df_dict = get_param_fit_sclaw_data_df_dict(
        context_length=8192, experiment_set=experiment_set_plot_data_points
    )

    # define the colormap
    if model_size_colormap is None:
        rocket_cmap_full = sns.color_palette(palette="rocket_r", as_cmap=True)
        rocket_cmap_middle = rocket_cmap_full(np.linspace(0.2, 0.8, 256))
        rocket_cmap = LinearSegmentedColormap.from_list(
            "rocket_cmap_middle", rocket_cmap_middle
        )
        model_size_colormap = rocket_cmap

    # define the maximum num params for the color scale
    max_num_params = 0.0
    min_num_params = float("inf")
    for _, model_df in model_configs.items():
        max_num_params = max(max_num_params, model_df[num_params_col].max())
        min_num_params = min(min_num_params, model_df[num_params_col].min())

    # define the color scale
    normalize_class = Normalize if model_size_colormap_scale == "linear" else LogNorm

    colormap_normalizer = normalize_class(
        vmin=min_num_params,
        vmax=max_num_params,
    )

    for model_type_key, model_df in param_fit_sclaw_data_df_dict.items():
        # if model_type_key not in parametric_loss_fn_dicts.keys():
        #     continue
        # select data points with correct num_params
        model_df = model_df[
            model_df["num_params"].isin(datapoints_num_param_selection[model_type_key])
        ]

        ax.scatter(
            x=model_df[x_axis_mode_to_x_col[x_axis_mode]],
            y=model_df[y_col],
            c=model_df[num_params_col],
            cmap=model_size_colormap,
            norm=colormap_normalizer,
            **data_points_style_dict[model_type_key],
            alpha=0.25,
        )

        points = np.array(model_df[[x_axis_mode_to_x_col[x_axis_mode], y_col]].values)

        # append minimum of both axes to close the polygon and remove at the end
        points = np.vstack(
            [
                points,
                np.array([points[:, 0].max(), points[:, 1].max()]),
            ]
        )
        pareto_points = convex_hull(points)[0][:-1]
        ax.plot(
            pareto_points[:, 0],
            pareto_points[:, 1],
            color=hull_colors[model_type_key],
            linewidth=2.0,
            label=model_tags_label_map[model_type_key],
            **data_points_style_dict[model_type_key],
        )

        fill_points = np.vstack(
            [
                np.array([pareto_points[0, 0], ylim[x_axis_mode][1]]),
                pareto_points,
                np.array([xlim[x_axis_mode][1], pareto_points[-1, 1]]),
            ]
        )

        ax.fill_between(
            fill_points[:, 0],
            fill_points[:, 1],
            [max(fill_points[:, 1])] * len(fill_points[:, 1]),
            color=hull_colors[model_type_key],
            alpha=0.2,
        )

    ax.arrow(
        0.10,
        0.43,
        0.00,
        -0.20,
        transform=ax.transAxes,
        length_includes_head=True,
        head_width=0.015,
        head_length=0.03,
        fc="black",
        ec="black",
        lw=2,
    )
    ax.text(
        0.10,
        0.45,
        "Better",
        transform=ax.transAxes,
        va="bottom",
        ha="center",
        fontsize=12,
        color="black",
    )

    ax.arrow(
        0.40,
        0.13,
        -0.20,
        0.00,
        transform=ax.transAxes,
        length_includes_head=True,
        head_width=0.015,
        head_length=0.03,
        fc="black",
        ec="black",
        lw=2,
    )
    ax.text(
        0.42,
        0.13,
        "Cheaper",
        transform=ax.transAxes,
        va="center",
        ha="left",
        fontsize=12,
        color="black",
    )

    # set plot look and feel
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    xlabel = x_axis_labels[x_axis_mode]
    if xscale == "log":
        xlabel += " (log scale)"
    if yscale == "log":
        y_axis_label += " (log scale)"
        ax.yaxis.grid(which="minor", visible=True)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))

    if x_axis_ticks[x_axis_mode] is not None:
        ax.set_xticks(x_axis_ticks[x_axis_mode])
    if x_axis_tick_labels[x_axis_mode] is not None:
        ax.set_xticklabels(x_axis_tick_labels[x_axis_mode])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(y_axis_label)

    if xlim is not None:
        ax.set_xlim(xlim[x_axis_mode])
    if ylim is not None:
        ax.set_ylim(ylim[x_axis_mode])

    ax.grid(which="minor", linestyle="-", linewidth=0.5, color="lightgrey")
    if legend_kwargs is not None:
        ax.legend(
            **legend_kwargs,
        )

    return fig
