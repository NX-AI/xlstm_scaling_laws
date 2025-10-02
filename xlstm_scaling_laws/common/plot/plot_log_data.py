import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .plot_config import get_plot_mpl_context


def plot_log_keys_dfs(
    log_key_data_df: pd.DataFrame,
    style_dict: dict[str, dict],
    x_axis_col: str,
    ax: Axes = None,
    ylim: tuple[float, float] = None,
    xlim: tuple[float, float] = None,
    grid_alpha: float = 0.5,
    x_label: str = "Step",
    y_label: str = "Loss",
    yscale: str = "linear",
    scientific_xticks: bool = True,
    legend_kwargs: dict = None,
    y_labelpad: float | None = None,
) -> Figure:
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    for key, style in style_dict.items():
        ax.plot(log_key_data_df[x_axis_col], log_key_data_df[key], **style)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label, labelpad=y_labelpad)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.grid(alpha=grid_alpha)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_yscale(yscale)

    if scientific_xticks:
        import matplotlib.ticker as ticker

        ax.xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style="sci", axis="x", scilimits=(3, 3))
    if legend_kwargs is not None:
        ax.legend(**legend_kwargs)

    fig = ax.get_figure()
    return fig


def create_learning_curve_gradnorm_plot(
    loss_df: pd.DataFrame, gradnorm_df: pd.DataFrame
) -> Figure:
    blue = "#165b89ff"
    light_blue = "#80a8b3ff"

    style_dict = {
        "train/.loss_max": {
            "color": blue,
            "alpha": 0.5,
            "label": "Maximum (over 50 steps)",
        },
        "train/.loss_mean": {"color": blue, "label": "Mean (over 50 steps)"},
    }

    style_dict_gradnorm = {
        "train/.grad_norm_max": {"color": blue, "alpha": 0.5},
        "train/.grad_norm_mean": {"color": blue},
    }

    n_col = 2
    fig_height = 5
    with get_plot_mpl_context():
        fig, ax = plt.subplots(
            1,
            n_col,
            figsize=(1.5 * fig_height * n_col, fig_height),
            sharex=True,
            gridspec_kw={"wspace": 0.24, "hspace": 0.5},
        )
        fig = fig = plot_log_keys_dfs(
            loss_df,
            style_dict,
            x_axis_col="_step",
            ylim=(2.07, 3.05),
            x_label="Step",
            y_label="Training Loss",
            legend_kwargs=None,
            ax=ax[0],
        )
        fig = plot_log_keys_dfs(
            gradnorm_df,
            style_dict_gradnorm,
            x_axis_col="_step",
            # ylim=(0, 4.0),
            x_label="Step",
            y_label="Gradient Norm\n(logscale)",
            y_labelpad=-2,
            yscale="log",
            ax=ax[1],
        )

        legend_kwargs = {
            "loc": "outside upper center",
            "ncol": 2,
            # "bbox_to_anchor": (0.0, 1.0, 0.0, 1.0),
            "frameon": False,
            "facecolor": "white",
            # "alignment": "top",
            "labelspacing": 1.1,
        }
        handles, labels = ax[0].get_legend_handles_labels()
        fig.legend(handles, labels, **legend_kwargs)
    return fig


def create_xlstm7b_loss_gradnorm_plot() -> Figure:
    from ...load_data.datafiles import RunDataSet, get_run_data_dict

    raw_mlstm_data = get_run_data_dict(RunDataSet.TOKENPARAM_MLSTM)
    mlstm_7b_run_data = raw_mlstm_data["dclm_mLSTMv1_7B_longrun_pretraining_final"][0]

    loss_max_df = mlstm_7b_run_data.logs["train/.loss_max"]
    loss_mean_df = mlstm_7b_run_data.logs["train/.loss_mean"]

    gradnorm_max_df = mlstm_7b_run_data.logs["train/.grad_norm_max"]
    gradnorm_mean_df = mlstm_7b_run_data.logs["train/.grad_norm_mean"]

    loss_df = pd.merge(loss_max_df, loss_mean_df, on="_step")
    gradnorm_df = pd.merge(gradnorm_max_df, gradnorm_mean_df, on="_step")
    fig = create_learning_curve_gradnorm_plot(loss_df, gradnorm_df)
    return fig
