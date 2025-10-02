from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from xlstm_scaling_laws.fitting.fit_tokenparam_slopes import (
    generate_linear_fits_for_token_param_ratios,
    plot_linear_fits_for_token_param_ratios,
)
from xlstm_scaling_laws.load_data import create_token_param_ratio_data_table

# markers = ["o", "X", "s", "+", "x", "D"] #, "*", "h", "^", "v", "<", ">", "p"]


def create_training_flop_token_multiplier_loglog_plot_with_fits(
    data_df: pd.DataFrame,
    figsize: tuple[float, float] = (8, 6),
    y_axis_log: bool = True,
    fit_mode: Literal[
        "none", "interpolate", "extrapolate", "interpolate_extrapolate"
    ] = "interpolate",
    marker_size: int = 80,
    last_token_param_ratio_is_extra: bool = True,
    ax: plt.Axes | None = None,
    legend_kwargs: dict[str, Any] | None = dict(
        loc="center left", bbox_to_anchor=(1, 0.5)
    ),
) -> Figure:
    from matplotlib.ticker import FuncFormatter

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
    else:
        fig = ax.get_figure()

    if last_token_param_ratio_is_extra:
        color_palette = sns.color_palette("rocket", n_colors=7)[
            : (len(data_df["Preset Token Param Ratio"].unique()) - 1)
        ]
        color_palette = color_palette + [(0.8, 0.8, 0.8)]  # add grey for extra
        # color_palette = sns.color_palette("deep")
    else:
        color_palette = sns.color_palette("rocket", n_colors=7)[
            : len(data_df["Preset Token Param Ratio"].unique())
        ]

    def generate_key(x: pd.Series):
        return x.replace("extra", "inf").astype(float)

    ax = sns.scatterplot(
        data=data_df.sort_values(by="Preset Token Param Ratio", key=generate_key),
        x="num_flops_training",
        y="val/.dclm_loss",
        hue="Preset Token Param Ratio",
        style="Model Size",
        markers=True,
        palette=color_palette,
        s=marker_size,
        ax=ax,
        zorder=10,
        legend=True,
    )

    # plot fits
    # find possible token_param_ratios
    # token_param_ratios = [22, 44, 110, 220, 550, 1100]
    if fit_mode in ["interpolate", "interpolate_extrapolate", "extrapolate"]:
        token_param_ratios = []
        for tok_param_ratio in data_df["Preset Token Param Ratio"].unique():
            if tok_param_ratio == "extra":
                continue
            if len(data_df[data_df["Preset Token Param Ratio"] == tok_param_ratio]) > 1:
                token_param_ratios.append(tok_param_ratio)

        fits = generate_linear_fits_for_token_param_ratios(
            all_token_param_ratios_df=data_df, token_param_ratios=token_param_ratios
        )
        color_dict = {
            token_param_ratio: {"color": color, "alpha": 0.5}
            for token_param_ratio, color in zip(
                token_param_ratios, color_palette[: len(token_param_ratios)]
            )
        }

        xdata_non_log = np.linspace(
            data_df["num_flops_training"].min(),
            data_df["num_flops_training"].max(),
            100,
        )

        ax = plot_linear_fits_for_token_param_ratios(
            all_token_param_ratios_df=data_df,
            token_param_ratios=token_param_ratios,
            fits=fits,
            style_dict=color_dict,
            ax=ax,
            xdata_non_log=xdata_non_log,
            plot_mode=fit_mode,
        )

    # sns.despine()
    if legend_kwargs is not None:
        ax.legend(**legend_kwargs)
    ax.set_xscale("log")
    ax.grid(which="both")
    ax.grid(which="minor", linestyle="-", linewidth=0.5, color="lightgrey")
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    if y_axis_log:
        from matplotlib.ticker import FuncFormatter

        ax.set_yscale("log")
        ax.set_ylim(2.05, 3.55)
        ax.set_yticks(np.arange(2.2, 3.4, 0.2))
        ax.yaxis.grid(which="minor", visible=True)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1f}"))
        ax.set_ylabel("Validation Loss (log scale)")
    else:
        ax.set_ylabel("Validation Loss")
    ax.set_xlabel("Compute (FLOPs) (log scale)")
    return fig


def get_combined_flop_loss_loglog_slope_plot(
    legend_kwargs: dict[str, Any] = {
        "loc": "upper left",
        "bbox_to_anchor": (0.91, 0.91),
        "frameon": True,
        "facecolor": "white",
        "ncol": 2,
        "columnspacing": 0.1,
        "markerscale": 1.2,
    },
    figsize: tuple[float, float] = (12, 4.5),
    ylim: tuple[float, float] = (2.1, 3.4),
    xlim: tuple[float, float] = (3.5e18, 4e22),
) -> Figure:
    mlstm_df_final = create_token_param_ratio_data_table(
        model_data="mlstm", mlstm_fw_flop_calc_mode="tfla"
    )
    llama_df_final = create_token_param_ratio_data_table(
        model_data="llama", attention_flop_calc_mode="distill_scaling"
    )
    # remove extra token param rows and 7B long run from data
    mlstm_df_final = mlstm_df_final[~(mlstm_df_final["Model Size"] == "7B long")]
    mlstm_df_final = mlstm_df_final[
        ~(mlstm_df_final["Preset Token Param Ratio"] == "extra")
    ]
    llama_df_final = llama_df_final[
        ~(llama_df_final["Preset Token Param Ratio"] == "extra")
    ]

    fig, ax = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharex=True,
        sharey=False,
        # gridspec_kw={
        #     "wspace": 0.2,
        #     "hspace": 0,
        #     "width_ratios": [0.47, 0.58],
        # },
    )
    ax1, ax2 = ax.tolist()

    _ = create_training_flop_token_multiplier_loglog_plot_with_fits(
        ax=ax1,
        data_df=llama_df_final,
        fit_mode="interpolate_extrapolate",
        last_token_param_ratio_is_extra=False,
        legend_kwargs=None,
    )

    _ = create_training_flop_token_multiplier_loglog_plot_with_fits(
        ax=ax2,
        data_df=mlstm_df_final,
        fit_mode="interpolate_extrapolate",
        last_token_param_ratio_is_extra=False,
        legend_kwargs=None,
    )

    ax1.set_title("Transformer")
    ax2.set_title("xLSTM")

    ax1.legend_.remove()
    ax2.legend_.remove()
    ax1.set_ylim(ylim)
    ax2.set_ylim(ylim)
    ax1.set_xlim(xlim)
    ax2.set_xlim(xlim)
    handles, labels = ax2.get_legend_handles_labels()
    style_tags_legend = {
        "Preset Token Param Ratio": "Token/Param\nRatio",
        "Model Size": "Model Size",
    }
    # create a legend for the style tags
    legend_handles = []
    legend_labels = []
    for legend_handle, legend_label_dataframe in zip(handles, labels):
        legend_handles.append(legend_handle)
        legend_labels.append(style_tags_legend.get(legend_label_dataframe, legend_label_dataframe))
    
    legend = fig.legend(handles=legend_handles, labels=legend_labels, **legend_kwargs)
    for text, label in zip(legend.get_texts(), labels):
        if label in style_tags_legend:
            text.set_fontweight('bold')


    return fig