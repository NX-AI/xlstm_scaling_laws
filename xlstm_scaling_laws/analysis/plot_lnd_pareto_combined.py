from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from xlstm_scaling_laws.analysis.parametric_sclaw_fit.plot.plot_scaling_law_fit_single import (
    get_scaling_law_lnd_fit_single_plot,
)
from xlstm_scaling_laws.analysis.tokenparam.plot_pareto_frontier import (
    get_pareto_frontier_single_plot,
)


def plot_lnd_pareto_combined(
    combined_fit_grid_df: pd.DataFrame,
    experiment_set_plot_data_points: Literal[
        "all", "tokenparam", "isoflop"
    ] = "tokenparam",
    experiment_set_fit="tokenparam",
    x_axis_mode="num_flops",
    figsize: tuple[float, float] = (10, 4),
    model_tags_label_map={
        "llama": "Transformer",
        "mlstm": "xLSTM",
    },
    data_points_style_dict={
        "llama": {"marker": "x"},
        "mlstm": {"marker": "o"},
    },
    fit_linestyles=["dashed", "solid"],
):

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    # Plot Pareto Frontier
    fig = get_pareto_frontier_single_plot(
        combined_fit_grid_df=combined_fit_grid_df,
        experiment_set_fit=experiment_set_fit,
        experiment_set_plot_data_points=experiment_set_plot_data_points,
        x_axis_mode=x_axis_mode,
        model_tags_label_map=model_tags_label_map,
        data_points_style_dict=data_points_style_dict,
        fig=fig,
        ax=axs[0],
    )

    fig = get_scaling_law_lnd_fit_single_plot(
        combined_fit_grid_df=combined_fit_grid_df,
        linestyles=fit_linestyles,
        experiment_set_fit=experiment_set_fit,
        experiment_set_plot_data_points=experiment_set_plot_data_points,
        datapoints_num_param_selection=None,
        x_axis_mode=x_axis_mode,
        model_tags_label_map=model_tags_label_map,
        data_points_style_dict=data_points_style_dict,
        fig=fig,
        ax=axs[1],
        add_header=False,
    )

    for text, offset in zip(
        [
            r"$\mathbf{Model \ Sizes}$",
            r"$\mathbf{Empirical \ Data}$",
            r"$\mathbf{L(N,D) \ Fits}$",
        ],
        [0.0, 0.395, 0.57],
    ):
        fig.text(
            0.920,  # 0.979, #
            0.87 - offset,
            text,
            ha="left",
            va="top",
            fontsize=11,
            zorder=99,
        )

    return fig
