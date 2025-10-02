import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


def create_token_param_ratio_plot(
    data_df: pd.DataFrame,
    figsize: tuple[float, float] = (8, 6),
    context: str = "paper",
    fontscale: float = 1.0,
    y_axis_log: bool = True,
) -> Figure:
    sns.set_style("whitegrid")
    sns.set_context(context=context, font_scale=fontscale)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax = sns.lineplot(
        data=data_df,
        x="token_param_ratio",
        y="val/.dclm_loss",
        hue="Model Size",
        style="Model Type",
        markers=True,
        palette=sns.color_palette(
            "rocket_r", n_colors=len(data_df["Model Size"].unique())
        ), #sns.color_palette("deep"),
        ax=ax,
    )
    sns.despine()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xscale("log")
    ticks = [25, 50, 100, 200, 300, 500, 1000, 2000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)

    if y_axis_log:
        from matplotlib.ticker import FuncFormatter
        ax.set_yscale("log")
        ax.yaxis.grid(which="minor", visible=True)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.set_ylabel("Validation Loss (logscale)")
    else:
        ax.set_ylabel("Validation Loss")
    ax.set_xlabel("Token / Param Ratio (logscale)")
    return fig


def create_num_token_training_plot(
    data_df: pd.DataFrame,
    figsize: tuple[float, float] = (8, 6),
    context: str = "paper",
    fontscale: float = 1.0,
    y_axis_log: bool = True,
) -> Figure:
    from matplotlib.ticker import LogLocator, ScalarFormatter
    sns.set_style("whitegrid")
    sns.set_context(context=context, font_scale=fontscale)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()
    ax = sns.lineplot(
        data=data_df,
        x="num_tokens_training",
        y="val/.dclm_loss",
        hue="Model Size",
        style="Model Type",
        markers=True,
        palette=sns.color_palette(
            "rocket_r", n_colors=len(data_df["Model Size"].unique())
        ),
        ax=ax,
    )
    sns.despine()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xscale("log")

    # Set the formatter for the y-axis to use scientific notation with a fixed exponent of 10^9
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((9, 9))  # Fix the exponent to 10^9
    ax.xaxis.set_major_formatter(formatter)

    # Optionally, you can also set the label to show the fixed exponent
    ax.xaxis.get_offset_text().set_visible(False)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))

    if y_axis_log:
        from matplotlib.ticker import FuncFormatter
        ax.set_yscale("log")
        ax.yaxis.grid(which="minor", visible=True)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.set_ylabel("Validation Loss (logscale)")
    else:
        ax.set_ylabel("Validation Loss")
    ax.set_xlabel("Billion ($10^9$) Tokens  (logscale)")

    return fig


def create_training_flop_plot(
    data_df: pd.DataFrame,
    figsize: tuple[float, float] = (8, 6),
    context: str = "paper",
    fontscale: float = 1.0,
    y_axis_log: bool = True,
) -> Figure:
    from matplotlib.ticker import FuncFormatter
    sns.set_style("whitegrid")
    sns.set_context(context=context, font_scale=fontscale)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    color_palette = sns.color_palette(
        "rocket_r", n_colors=len(data_df["Model Size"].unique())
    )
    # color_palette = sns.color_palette("deep")

    ax = sns.lineplot(
        data=data_df,
        x="num_flops_training",
        y="val/.dclm_loss",
        hue="Model Size",
        style="Model Type",
        markers=True,
        palette=color_palette,
        ax=ax,
    )

    sns.despine()
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.set_xscale("log")
    ax.xaxis.grid(True)
    if y_axis_log:
        from matplotlib.ticker import FuncFormatter
        ax.set_yscale("log")
        ax.yaxis.grid(which="minor", visible=True)
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda x, _: f'{x:.1f}'))
        ax.set_ylabel("Validation Loss (logscale)")
    else:
        ax.set_ylabel("Validation Loss")
    ax.set_xlabel("Training FLOPs (logscale)")
    return fig