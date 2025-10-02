from typing import Any, Literal

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes

from ....fitting.fit_isoflop_polynomials import plot_isoflop_polynomial_fits


def get_isoflop_styledict_with_colors(
    isoflop_tags: list[str], seaborn_color_palette: str = "rocket_r"
) -> dict[str, dict[str, Any]]:
    """Creates a dictionary of style dictionaries for the isoflop tags."""

    style_dict = {}

    color_palette = sns.color_palette(seaborn_color_palette, n_colors=len(isoflop_tags))

    for i, isoflop_tag in enumerate(isoflop_tags):
        style_dict[isoflop_tag] = {
            "color": color_palette[i],
        }
    return style_dict

def get_context_styledict_with_colors(
    context_lengths: list[int], seaborn_color_palette: str = "Purples"
) -> dict[int, dict[str, Any]]:
    """Creates a dictionary of style dictionaries for the context lengths."""

    style_dict = {}
    n_ctxs = len(context_lengths)

    color_palette = sns.color_palette(sns.color_palette("Purples", n_ctxs + 1)[1:], n_colors=n_ctxs)

    for i, context_length in enumerate(context_lengths):
        style_dict[context_length] = {
            "color": color_palette[i],
        }
    return style_dict


def create_isocurve_plot(
    ax: Axes | None = None,
    isoflop_tags: list[str] = None,
    context_lengths: list[int] | None = None,
    isoflop_df: pd.DataFrame = None,
    isoflop_polyfit_df: pd.DataFrame = None,
    x_col: Literal["num_tokens_training", "num_params"] = "num_tokens_training",
    y_col: Literal["val/.dclm_loss", "train/.loss_mean"] = "val/.dclm_loss",
    model_type_col: str = "model_type",
    model_tags_to_plot: list[str] | None = None,
    isoflop_datapoints_style_dicts: dict[str, dict[str, dict]] | None = None,
    model_type_datapoints_style_dicts: dict[str, dict[str, dict]] = {
        "llama": {"marker": "x"},
        "mlstm_v1": {"marker": "o"},
    },
    isoflop_polyfit_style_dicts: dict[str, dict[str, dict]] | None = None,
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
    model_type_alpha_override: dict[str, float] | None = None,
    use_isoflop_color_for_optimum: bool = True,
    xticks: list[float] | None = None,
    xtick_labels: list[str] | None = None,
    figsize: tuple[float, float] = (8., 6.),
) -> Axes:
    """Creates the isocurve plot for the given isoflop dataframe."""

    if model_tags_to_plot is None:
        model_tags_to_plot = isoflop_polyfit_df[model_type_col].unique()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None

    assert isoflop_tags is not None or context_lengths is not None, \
          "Either isoflop_tags or context_lengths must be provided."
    elements = isoflop_tags if isoflop_tags is not None else context_lengths

    for model_tag in model_tags_to_plot:
        # select the isoflop data for the current model tag
        isoflop_model_df = isoflop_df[isoflop_df[model_type_col] == model_tag]

        # plot the isoflop data data points and fits
        for elem in elements:
            # plot the isoflop data points
            # select the isoflop data for the current model tag
            if isoflop_tags is not None:
                isoflop_tag_df = isoflop_model_df[
                    isoflop_model_df["IsoFLOP"] == elem
                ]
            else:
                isoflop_tag_df = isoflop_model_df[
                    isoflop_model_df["context_length"] == elem
                ]

            x_data = isoflop_tag_df[x_col]
            y_data = isoflop_tag_df[y_col]

            isoflop_datapoint_style_dict = isoflop_datapoints_style_dicts.get(
                elem, {}
            )

            if model_type_alpha_override is not None:
                # override the alpha value for the model type
                isoflop_datapoint_style_dict["alpha"] = model_type_alpha_override.get(
                    model_tag, 1.0
                )

            ax.scatter(
                x=x_data,
                y=y_data,
                label=f"datapoints_{elem}_{model_tag}",
                **isoflop_datapoint_style_dict,
                **model_type_datapoints_style_dicts.get(model_tag, {}),
            )

        # update the style dicts for the model type
        updated_isoflop_polyfit_style_dicts = {}
        for elem in elements:
            updated_isoflop_polyfit_style_dicts[elem] = {
                **isoflop_polyfit_style_dicts.get(elem, {}),
                **model_type_polyfit_style_dicts.get(model_tag, {}),
            }
            if model_type_alpha_override is not None:
                # override the alpha value for the model type
                updated_isoflop_polyfit_style_dicts[elem]["alpha"] = (
                    model_type_alpha_override.get(model_tag, 1.0)
                )

        updated_model_type_optimum_style_dict = model_type_optimum_style_dicts.get(
            model_tag, {}
        )
        if model_type_alpha_override is not None:
            # override the alpha value for the model type
            updated_model_type_optimum_style_dict["alpha"] = (
                model_type_alpha_override.get(model_tag, 1.0)
            )

        # plot the isoflop fits
        ax = plot_isoflop_polynomial_fits(
            ax=ax,
            isoflop_df=isoflop_model_df,
            isoflop_polyfit_df=isoflop_polyfit_df[
                isoflop_polyfit_df[model_type_col] == model_tag
            ],
            style_dicts=updated_isoflop_polyfit_style_dicts,
            x_col=x_col,
            num_points=250,
            plot_optimum=True,
            isoflop_tags=[f"{e}" for e in elements],
            use_isoflop_color_for_optimum=use_isoflop_color_for_optimum,
            style_dict_optimum=updated_model_type_optimum_style_dict,
            legend_label_suffix=model_tag,
        )

    ax.set_xscale("log")
    ax.set_xlabel(axis_labels[x_col])
    ax.set_ylabel(axis_labels[y_col])
    ax.grid(which="both")
    ax.grid(which="minor", linestyle="-", linewidth=0.5, color="lightgrey")
    ax.xaxis.grid(True)
    ax.yaxis.grid(True)

    if xticks is not None:
        ax.set_xticks(xticks)
    if xtick_labels is not None:
        ax.set_xticklabels(xtick_labels)

    if fig is not None:
        # we are creating a new figure
        # therefore add a legend
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    return ax