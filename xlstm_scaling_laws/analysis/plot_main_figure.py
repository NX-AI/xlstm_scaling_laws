from typing import Any, Literal
import copy

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.collections import PathCollection

from .isoflop.plot.isocurve_powerlaw import get_isoflop_powerlaw_plot
from .parametric_sclaw_fit.plot.plot_scaling_law_fit_single import get_scaling_law_lnd_fit_single_plot

def get_main_figure_plot(
    figsize: tuple[float, float] = (10, 4),
    combined_fit_grid_df: pd.DataFrame | None = None,
    linestyles: list[str]=["dashed", "solid"],
    experiment_set_fit: Literal["all", "tokenparam", "isoflop"] = "tokenparam",
    datapoints_num_param_selection: dict[str, list[float]] | None = None,
    x_axis_mode: Literal["num_flops", "token_param_ratio"] = "num_flops",
    model_tags_label_map: dict[str, str] = {
        "llama": "Llama",
        "mlstm": "xLSTM",
    },
    data_points_style_dict: dict[str, dict] = {
        "llama": {"marker": "x"},
        "mlstm": {"marker": "o"},
    },
    ) -> Figure:

    fig = plt.figure(figsize=figsize)

    # Manually set positions (left, bottom, width, height)
    ax1 = fig.add_axes([0.05, 0.1, 0.37, 0.7])
    ax2 = fig.add_axes([0.505, 0.1, 0.37, 0.7])
    ax3 = fig.add_axes([0.975, 0.1, 0.37, 0.7])

    ax = [ax1, ax2, ax3]

    fig = get_scaling_law_lnd_fit_single_plot(
        combined_fit_grid_df=combined_fit_grid_df,
        experiment_set_fit=experiment_set_fit,
        experiment_set_plot_data_points="tokenparam",
        datapoints_num_param_selection=datapoints_num_param_selection,
        x_axis_mode=x_axis_mode,
        linestyles=linestyles,
        figsize_single_subplot=(6, 4.5),
        model_tags_label_map=model_tags_label_map,
        data_points_style_dict=data_points_style_dict,
        legend_kwargs={
            "loc": "upper left",
            "bbox_to_anchor": (1.05, 1),
            "borderaxespad": 0.0,
            "fontsize": 12,
            "frameon": False,
            "handlelength": 2.5,
            "handleheight": 0.5,
            "labelspacing": 0.5,
            "borderpad": 0.5,
        },
        fig=fig,
        ax=ax[0],
        add_header=False,
    )

    handles_lnd, labels_lnd = ax[0].get_legend_handles_labels()
    fig.legends.clear()

    fig = get_isoflop_powerlaw_plot(
        plot_type="num_params",
        y_col = "val/.dclm_loss",
        fig=fig,
        ax=ax[1:],
        add_header=False,
    )

    handles_isoflop, labels_isoflop = ax[1].get_legend_handles_labels()
    fig.legends.clear()

    all_handles = handles_lnd + handles_isoflop
    all_labels = labels_lnd + labels_isoflop

    e_hand, e_lab = Patch(color="none"), " "*12

    handles, labels = [], []

    def _add_empty_row(handles, labels, override_e_lab=None):
        handles.append(e_hand)
        if override_e_lab is not None:
            labels.append(override_e_lab)
        else:
            labels.append(e_lab)
        return handles, labels
    
    handles, labels = _add_empty_row(handles, labels)

    # first two rows
    handles += [h for h, l in zip(all_handles, all_labels) if l.startswith("mlstm_")]
    labels += [l.split("_")[1] for l in all_labels if l.startswith("mlstm_")]
    
    handles.insert(4, e_hand)
    labels.insert(4, e_lab)

    # empty row
    for i in range(4):
        handles, labels = _add_empty_row(handles, labels, override_e_lab=" "*6)

    # third row - extract llama and mlstm handle and label
    handles, labels = _add_empty_row(handles, labels)
    handles += [h for h, l in zip(all_handles, all_labels) if l == "llama"]
    labels += ["Llama"]
    handles += [h for h, l in zip(all_handles, all_labels) if l == "mlstm"]
    labels += ["xLSTM"]
    handles, labels = _add_empty_row(handles, labels)

    # fourth row
    handles, labels = _add_empty_row(handles, labels)
    handles += [[h for h, l in zip(all_handles, all_labels) if l.startswith("llama_")][0]]
    labels += ["Llama"]
    handles += [[h for h, l in zip(all_handles, all_labels) if l.startswith("mlstm_")][0]]
    labels += ["xLSTM"]
    handles, labels = _add_empty_row(handles, labels)

    # fifth (empty) row
    for i in range(4):
        handles, labels = _add_empty_row(handles, labels, override_e_lab=" "*6)

    # sixth and seventh row
    handles, labels = _add_empty_row(handles, labels)
    handles += [h for h, l in zip(all_handles, all_labels) if "fit_" in l and "mlstm" in l]
    labels += [l.split("_")[1] for l in all_labels if "fit_" in l and "mlstm" in l]
    handles.insert(-2, e_hand)
    labels.insert(-2, e_lab)
    handles, labels = _add_empty_row(handles, labels)

    # eighth row
    handles, labels = _add_empty_row(handles, labels)
    handles += [[h for h, l in zip(all_handles, all_labels) if "optimum_" in l and "llama" in l][0]]
    labels += ["Llama"]
    handles += [[h for h, l in zip(all_handles, all_labels) if "optimum_" in l and "mlstm" in l][0]]
    labels += ["xLSTM"]
    handles, labels = _add_empty_row(handles, labels)

    legend_handles = []
    black_handles = [12, 13, 14, 15, 16, 17, 18, 19, 36, 37, 38, 39]
    sizes = [32 ** 0.5, 110 ** 0.5]
    for i, h in enumerate(handles):

        try:
            color = h.get_color() if i not in black_handles else "black"
        except AttributeError:
            color = "black"

        if isinstance(h, Line2D):
            if color == "black":
                lh = Line2D(
                    [0], [0], color=color, linestyle=h.get_linestyle(), 
                    linewidth=2
                )
            else:
                lh = Patch(
                    color=color, label=labels[i], 
                    edgecolor=color, linewidth=1
                )
        elif isinstance(h, PathCollection):
            size = h.get_sizes()[0] ** 0.5
            # set to closest in sizes
            size = min(sizes, key=lambda x: abs(x - size))


            lh = Line2D(
                [0], [0],
                marker=h.get_paths()[0],                                    
                linestyle='none',                              
                markerfacecolor=color,                         
                markeredgecolor=color,                         
                markersize=size,                                
                alpha=1.0                                      
            )
        else:
            lh = h
        legend_handles.append(lh)

    fig.legend(
        handles=legend_handles,
        labels=labels,
        loc="upper center",
        bbox_to_anchor=(0.698, 1.22),
        ncol=9,
        fontsize=12,
        frameon=True,
        handlelength=2.5,
        handleheight=0.5,
        labelspacing=0.5,
        borderpad=0.9,
    )

    fig.lines.append(Line2D([0.41]*2, [0.92, 1.17], transform=fig.transFigure, color='lightgrey', linewidth=1, zorder=99))
    fig.lines.append(Line2D([0.83]*2, [0.92, 1.17], transform=fig.transFigure, color='lightgrey', linewidth=1, zorder=99))

    for text, offset in zip([
        r"$\mathbf{Model\ Size\ (Params)}$",
        r"$\mathbf{Datapoints}$",
        r"$\mathbf{Fits}$",
        r"$\mathbf{Compute\ (FLOPs)}$",
        r"$\mathbf{FLOP\ Optima}$",
    ], [-0.050, 0.365, 0.545, 0.82, 1.07]):
        fig.text(
            0.125 + offset, 1.17,
            text,
            ha='left', va='top',
            fontsize=14,
            zorder=99,
        )

    # text that is 90° rotated
    for text, offset in zip([
        "IsoParam",
        "Common",
        "Common",
        "IsoFLOP",
    ], [0.005, 0.035, 0.425, 0.455]):
        fig.text(
            0.39 + offset, 1.05,
            text,
            ha='center', va='center',
            fontsize=12,
            rotation=90,
            color="grey",
            zorder=99,
        )

    ax[0].set_title("IsoParam", color="grey", fontsize=14)
    ax[1].set_title("IsoFLOP", color="grey", fontsize=14)
    ax[2].set_title("IsoFLOP", color="grey", fontsize=14)

    return fig

