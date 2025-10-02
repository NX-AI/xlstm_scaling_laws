"""We plot xLSTM and Llama scaling law fits in a single plot."""

from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ....fitting.fit_parametric_loss.plot_parametric_loss_fit import (
    get_param_fit_sclaw_data_df_dict,
    plot_parametric_loss_fit,
)
from ....fitting.fit_parametric_loss.scaling_law_funcs import (
    ScalingLawReturnMode,
    get_first_n_fits_as_fit_fn_dict,
)


def get_scaling_law_lnd_fit_single_plot(
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
    token_param_range: Literal["0-100", "0-300", "0-5000"] = "0-5000",
    linestyles: list[str] = ["dashed", "solid", "dashdot", "dotted"],
    x_axis_mode: Literal["num_flops", "token_param_ratio", "num_tokens"] = "num_flops",
    llama_alpha: float = 0.5,
    figsize_single_subplot: tuple[float, float] = (5.5, 4),
    model_tags_label_map: dict[str, str] = {
        "llama": "Llama",
        "mlstm": "xLSTM",
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
    plot_fit_token_param_ratio_range: tuple[float, float] = (11.0, 2500.0),
    lnd_fit_func_mode: ScalingLawReturnMode = "logsumexp",
    add_header: bool = True,
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

    model_tags = model_tags_label_map.keys()

    parametric_loss_fn_dicts = {}
    for model_tag in model_tags:
        model_tag_fit_df = combined_fit_grid_df.loc[
            model_tag, experiment_set_fit, token_param_range
        ]

        model_func_dict = get_first_n_fits_as_fit_fn_dict(
            result_fit_df=model_tag_fit_df,
            key_prefix=f"{model_tag}-{experiment_set_fit}-{token_param_range}__",
            n=1,
            return_mode=lnd_fit_func_mode,
        )
        parametric_loss_fn_dicts[model_tag] = model_func_dict

    ax = plot_parametric_loss_fit(
        parametric_sclaw_funcs=parametric_loss_fn_dicts,
        param_fit_sclaw_data_df_dict=get_param_fit_sclaw_data_df_dict(
            context_length=8192, experiment_set=experiment_set_plot_data_points
        ),
        model_size_colormap_scale="log",
        sclaw_func_linestyles=linestyles,
        x_axis_mode=x_axis_mode,
        plot_mode="compare_models",
        yscale="log",
        xscale="log",
        ax=ax,
        legend_kwargs=None,  # no legend
        data_points_style_dict=data_points_style_dict,
        param_fit_sclaw_data_num_param_selection=datapoints_num_param_selection,
        token_param_ratio_range=plot_fit_token_param_ratio_range,
        llama_alpha=llama_alpha,
    )

    def _get_linestyle_legend_elements():
        legend_elements = []
        legend_elements.append(Patch(color="none", label=" "*16))
        for (model_key, model_label), linestyle in zip(
            model_tags_label_map.items(), linestyles
        ):
            legend_elements.append(
                plt.Line2D(
                    [],
                    [],
                    label=model_label,
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
        legend_elements.append(Patch(color="none", label=" "*16))
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
            Patch(color="none", label=" "*16)
        )
        for model_tag, style in data_points_style_dict.items():
            legend_elements.append(
                plt.Line2D(
                    [],
                    [],
                    label=f"{model_tags_label_map[model_tag]}",
                    linestyle="None",
                    marker=style["marker"],
                    color="black",
                )
            )
        return legend_elements

    legend_elements = []
    legend_elements.extend(_extract_color_legend_elements(ax))
    legend_elements.extend(_get_marker_legend_elements())
    legend_elements.extend(_get_linestyle_legend_elements())
    fig.legend(
        handles=legend_elements,
        **legend_kwargs,
    )

    if add_header:
        for text, offset in zip([
            r"$\mathbf{Model \ Sizes}$",
            r"$\mathbf{Empirical \ Data}$",
            r"$\mathbf{L(N,D) \ Fits}$",
        ], [0.0, 0.395, 0.57]):
            fig.text(
                0.932, #0.979, #
                0.87 - offset,
                text,
                ha='left', va='top',
                fontsize=11,
                zorder=99,
            )

    return fig
