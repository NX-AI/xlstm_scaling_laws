from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ....fitting.fit_parametric_loss.plot_parametric_loss_fit import (
    plot_parametric_loss_fit,
)
from ....fitting.fit_parametric_loss.scaling_law_funcs import (
    get_first_n_fits_as_fit_fn_dict,
)


def get_scaling_law_fit_grid_plot(
    combined_fit_grid_df: pd.DataFrame,
    experiment_sets: list[Literal["all", "tokenparam", "isoflop"]],
    linestyles: list[str] = ["dotted", "dashdot", "solid"],
    figsize_single_subplot: tuple[float, float] = (6, 4.5),
    model_tags_title_map: dict[str, str] = {
        "llama": "Llama",
        "mlstm": "xLSTM",
    },
    title_fontsize: int = None,
    token_param_ranges_label_map: list[str] = {
        "0-100": r"$\leq 100$ Token/Param",
        "0-300": r"$\leq 300$ Token/Param",
        "0-5000": r"All Data",
    },
    data_points_style_dict: dict[str, dict] = {
        "llama": {"marker": "x"},
        "mlstm": {"marker": "o"},
    },
    legend_kwargs: dict = {
        "loc": "upper left",
        "bbox_to_anchor": (0.91, 0.9),
        "ncol": 1,
    },
) -> Figure:
    """Compare xLSTM and Llama scaling law fits (columns)
    on different experiement sets (all, tokenparam, isoflop) in different rows.
    
    Vary the token param range for each experiment set and model.    
    """
    nrows = len(experiment_sets)
    ncols = 2

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize_single_subplot[0], nrows * figsize_single_subplot[1]),
        squeeze=False,
    )

    model_tags = model_tags_title_map.keys()

    for i, model_tag in enumerate(model_tags):
        for j, experiment_set in enumerate(experiment_sets):
            model_expset_df = combined_fit_grid_df.loc[model_tag, experiment_set]

            tok_param_range_dfs = {
                tokparam_range: model_expset_df.loc[tokparam_range]
                for tokparam_range in token_param_ranges_label_map.keys()
            }

            model_tag_fit_funcs = {}
            for token_param_range in token_param_ranges_label_map.keys():
                tokparam_func_dict = get_first_n_fits_as_fit_fn_dict(
                    tok_param_range_dfs[token_param_range],
                    key_prefix=f"{model_tag}-{experiment_set}-{token_param_range}__",
                    n=1,
                    return_mode="logsumexp",
                )
                model_tag_fit_funcs.update(tokparam_func_dict)

            parametric_loss_fn_dicts = {
                model_tag: model_tag_fit_funcs,
            }
            plot_parametric_loss_fit(
                parametric_sclaw_funcs=parametric_loss_fn_dicts,
                model_size_colormap_scale="log",
                sclaw_func_linestyles=linestyles,
                x_axis_mode="token_param_ratio",
                plot_mode="compare_sclaw",
                yscale="log",
                xscale="log",
                ax=axs[j, i],
                legend_kwargs=None,  # no legend
                data_points_style_dict=data_points_style_dict,
            )
            axs[j, i].set_title(
                f"{model_tags_title_map[model_tag]}", fontsize=title_fontsize
            )

    def _get_linestyle_legend_elements():
        legend_elements = []
        legend_elements.append(
            Patch(color="none", label=r"$\mathbf{L(N,D) \ Token \ Param \ Range}$")
        )
        for (token_param_range, token_param_range_label), linestyle in zip(
            token_param_ranges_label_map.items(), linestyles
        ):
            legend_elements.append(
                plt.Line2D(
                    [],
                    [],
                    label=token_param_range_label,
                    linestyle=linestyle,
                    color="black",
                )
            )
        return legend_elements

    def _extract_color_legend_elements(ax: Axes):
        def _extract_model_size_from_label(label: str):
            try:
                return label.split("__")[0].split("_")[1]
            except IndexError:
                return None

        model_size_color_map = {}

        handles, labels = ax.get_legend_handles_labels()

        for ax_handle, ax_label in zip(handles, labels):
            model_size = _extract_model_size_from_label(ax_label)
            if model_size is not None:
                color = ax_handle.get_color()
                model_size_color_map[model_size] = color

        legend_elements = []
        legend_elements.append(Patch(color="none", label=r"$\mathbf{Model \ Sizes}$"))
        for model_size, color in model_size_color_map.items():
            legend_elements.append(
                plt.Line2D(
                    [],
                    [],
                    label=model_size,
                    linestyle="-",
                    linewidth=2,
                    color=color,
                )
            )
        return legend_elements

    def _get_marker_legend_elements():
        legend_elements = []
        legend_elements.append(
            Patch(color="none", label=r"$\mathbf{Empirical \ Data \ Points}$")
        )
        for model_tag, style in data_points_style_dict.items():
            legend_elements.append(
                plt.Line2D(
                    [],
                    [],
                    label=f"{model_tags_title_map[model_tag]} Data Points",
                    linestyle="None",
                    marker=style["marker"],
                    color="black",
                )
            )
        return legend_elements

    legend_elements = []
    legend_elements.extend(_extract_color_legend_elements(axs[0, 0]))
    legend_elements.extend(_get_marker_legend_elements())
    legend_elements.extend(_get_linestyle_legend_elements())
    fig.legend(
        handles=legend_elements,
        **legend_kwargs,
    )

    return fig


def get_scaling_law_fit_func_comparison_plot(
    combined_fit_grid_gamma_dict_df: dict[Literal["gamma", "nogamma"], pd.DataFrame],
    experiment_sets: list[Literal["all", "tokenparam", "isoflop"]],
    token_param_range: Literal["0-100", "0-300", "0-5000"] = "0-5000",
    linestyles: list[str] = ["solid", "dotted", "dashdot", "dotted"],
    x_axis_mode: Literal["num_flops", "token_param_ratio"] = "num_flops",
    figsize_single_subplot: tuple[float, float] = (6, 4.5),
    llama_alpha: float = 1.0,
    model_tags_title_map: dict[str, str] = {
        "llama": "Transformer",
        "mlstm": "xLSTM",
    },
    title_fontsize: int = None,
    gamma_label_map: list[str] = {
        "nogamma": r"Chinchilla ($\gamma=1.0$)",
        "gamma": r"Distillation Scaling Law ($\gamma$ fitted)",
    },
    data_points_style_dict: dict[str, dict] = {
        "llama": {"marker": "x"},
        "mlstm": {"marker": "o"},
    },
    legend_kwargs: dict = {
        "loc": "upper left",
        "bbox_to_anchor": (0.91, 0.9),
        "ncol": 1,
    },
) -> Figure:
    """Compare xLSTM and Llama scaling law fits (columns)
    on different experiement sets (token param range, isoflop) in different rows.
    
    Vary the scaling law fit function for each experiment set and model.
    Compare chinchilla fit with and without the gamma parameter.    
    """
    nrows = len(experiment_sets)
    ncols = 2

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(ncols * figsize_single_subplot[0], nrows * figsize_single_subplot[1]),
        squeeze=False,
    )

    model_tags = model_tags_title_map.keys()

    for i, model_tag in enumerate(model_tags):
        for j, experiment_set in enumerate(experiment_sets):

            gamma_fit_dfs = {
                gamma_key: combined_fit_grid_gamma_dict_df[gamma_key].loc[model_tag, experiment_set, token_param_range]
                for gamma_key in gamma_label_map.keys()
            }

            gamma_fit_funcs = {}
            for gamma_key in gamma_label_map.keys():
                gamma_fit_func_dict = get_first_n_fits_as_fit_fn_dict(
                    gamma_fit_dfs[gamma_key],
                    key_prefix=f"{gamma_key}-{model_tag}-{experiment_set}-{token_param_range}__",
                    n=1,
                    return_mode="logsumexp",
                )
                gamma_fit_funcs.update(gamma_fit_func_dict)

            parametric_loss_fn_dicts = {
                model_tag: gamma_fit_funcs,
            }
            plot_parametric_loss_fit(
                parametric_sclaw_funcs=parametric_loss_fn_dicts,
                model_size_colormap_scale="log",
                sclaw_func_linestyles=linestyles,
                x_axis_mode=x_axis_mode,
                plot_mode="compare_sclaw",
                yscale="log",
                xscale="log",
                ax=axs[j, i],
                legend_kwargs=None,  # no legend
                data_points_style_dict=data_points_style_dict,
                llama_alpha=llama_alpha,
            )
            axs[j, i].set_title(
                f"{model_tags_title_map[model_tag]}", fontsize=title_fontsize
            )

    def _get_linestyle_legend_elements():
        legend_elements = []
        legend_elements.append(
            Patch(color="none", label=r"$\mathbf{L(N,D) \ Fit \ Function}$")
        )
        for (gamma_key, gamma_label), linestyle in zip(
            gamma_label_map.items(), linestyles
        ):
            legend_elements.append(
                plt.Line2D(
                    [],
                    [],
                    label=gamma_label,
                    linestyle=linestyle,
                    color="black",
                )
            )
        return legend_elements

    def _extract_color_legend_elements(ax: Axes):
        def _extract_model_size_from_label(label: str):
            try:
                return label.split("__")[0].split("_")[1]
            except IndexError:
                return None

        model_size_color_map = {}

        handles, labels = ax.get_legend_handles_labels()

        for ax_handle, ax_label in zip(handles, labels):
            model_size = _extract_model_size_from_label(ax_label)
            if model_size is not None:
                color = ax_handle.get_color()
                model_size_color_map[model_size] = color

        legend_elements = []
        legend_elements.append(Patch(color="none", label=r"$\mathbf{Model \ Sizes}$"))
        for model_size, color in model_size_color_map.items():
            legend_elements.append(
                plt.Line2D(
                    [],
                    [],
                    label=model_size,
                    linestyle="-",
                    linewidth=2,
                    color=color,
                )
            )
        return legend_elements

    def _get_marker_legend_elements():
        legend_elements = []
        legend_elements.append(
            Patch(color="none", label=r"$\mathbf{Empirical \ Data \ Points}$")
        )
        for model_tag, style in data_points_style_dict.items():
            legend_elements.append(
                plt.Line2D(
                    [],
                    [],
                    label=f"{model_tags_title_map[model_tag]} Data Points",
                    linestyle="None",
                    marker=style["marker"],
                    color="black",
                )
            )
        return legend_elements

    legend_elements = []
    legend_elements.extend(_extract_color_legend_elements(axs[0, 0]))
    legend_elements.extend(_get_marker_legend_elements())
    legend_elements.extend(_get_linestyle_legend_elements())
    fig.legend(
        handles=legend_elements,
        **legend_kwargs,
    )

    return fig


# TODO add huber delta ablation plot