"""In this module we write a function to plot the results of the parametric loss fit.


We plot the validation loss over the token parameter ratio for preselected model sizes
for the mlstm and llama models.

We use Figure 5 from Beyond Chinchilla-Optimal: Accounting for Inference in Language Modeling Scaling Laws paper
as reference for the plot (https://arxiv.org/abs/2401.00448).
"""

import copy
from itertools import cycle
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
from matplotlib.ticker import FuncFormatter

from ...analysis.parametric_sclaw_fit.data import create_param_fit_sclaw_data_table
from ...flops.count_flops import FlopCountConfig, count_model_flops_fwbw
from ...params.count_params import ParamCountConfig, count_model_params


def compute_fwbw_flops_and_num_params_for_models(
    model_config_dict: dict[str, dict],
    model_type: str = "mlstm_v1",
    context_length: int = 8192,
    include_embedding_flops: bool = False,
    attention_flop_calc_mode: Literal[
        "chinchilla", "distill_scaling"
    ] = "distill_scaling",
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "tfla",
    round_ffn_dim_to_multiple_of_for_flops: bool = True,
) -> dict[str, dict]:
    """Compute the FLOPs and number of parameters for the models in the model_config_dict.

    Example:
    ```
    model_config_dict = {
        "mlstm_100M_1": {
        "num_blocks": 10,
        "embedding_dim": 640,
        "proj_factor_ffn": 2.667,
        "num_heads": 5,
        "proj_factor_qk": 0.5,
        "chunk_size": 64,
        "vocab_size": 50304,
        "ffn_multiple_of": 64,
        "global_batch_size": 128,
        "learning_rate": 0.003,
    }
    }

    ```

    Args:
        model_config_dict (dict[str, dict]): A dictionary containing the model configuration.
        model_type (str, optional): The type of model. Defaults to "mlstm_v1".
        context_length (int, optional): The context length. Defaults to 8192.
        include_embedding_flops (bool, optional): Whether to include embedding FLOPs. Defaults to False.
        attention_flop_calc_mode (str, optional): The attention FLOP calculation mode. Defaults to "distill_scaling".
        This can be either "chinchilla" or "distill_scaling".
        mlstm_fw_flop_calc_mode (str, optional): The MLSTM forward FLOP calculation mode. Defaults to "tfla".
        round_ffn_dim_to_multiple_of_for_flops (bool, optional): Whether to round the FFN dimension to a multiple of the FFN multiple. Defaults to True.

    Returns:
        dict[str, dict]: A dictionary containing the model configuration with the FLOPs and number of parameters.
    """

    flop_count_config = FlopCountConfig(
        include_skip_flops=False,
        include_norm_layer_flops=False,
        include_embedding_flops=include_embedding_flops,
        include_final_logit_flops=True,
        attention_flop_calc_mode=attention_flop_calc_mode,
        mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
        mlstm_flop_causal_factor=0.75,
        round_ffn_dim_to_multiple_of_for_flops=round_ffn_dim_to_multiple_of_for_flops,
        bw_flop_count_mode="total_factor_2",
        seq_mix_bw_flop_factor=2.5,
        flop_factor_sig=1.0,
        flop_factor_exp=1.0,
        flop_factor_ffn_act_fn=1.0,
        flop_factor_max=1.0,
        flop_factor_mask=1.0,
        flop_factor_abs=1.0,
    )

    param_count_config = ParamCountConfig(
        count_norm_layer_params=True,
        count_embedding_params=True,
        count_final_logits_params=True,
    )

    updated_model_dict = copy.deepcopy(model_config_dict)
    for model_key, model_dict in model_config_dict.items():
        updated_model_dict[model_key].update(
            {
                "num_flops_fwbw": count_model_flops_fwbw(
                    model_type=model_type,
                    context_length=context_length,
                    model_kwargs=model_dict,
                    config=flop_count_config,
                )[0],
                "num_params": count_model_params(
                    model_type=model_type,
                    model_kwargs=model_dict,
                    config=param_count_config,
                ),
            }
        )
    return updated_model_dict


def get_model_config_df_dict(
    context_length: int = 8192,
    include_embedding_flops: bool = False,
    attention_flop_calc_mode: Literal[
        "chinchilla", "distill_scaling"
    ] = "distill_scaling",
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "tfla",
    round_ffn_dim_to_multiple_of_for_flops: bool = True,
) -> dict[str, pd.DataFrame]:
    mlstm_model_configs = {
        "mlstm_160M": {
            "num_blocks": 12,
            "embedding_dim": 768,
            "proj_factor_ffn": 2.667,
            "num_heads": 6,
            "proj_factor_qk": 0.5,
            "chunk_size": 64,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 128,
            "learning_rate": 0.003,
        },
        "mlstm_400M": {
            "num_blocks": 24,
            "embedding_dim": 1024,
            "proj_factor_ffn": 2.667,
            "num_heads": 4,
            "proj_factor_qk": 0.5,
            "chunk_size": 64,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 128,
            "learning_rate": 0.003,
        },
        "mlstm_830M": {
            "num_blocks": 24,
            "embedding_dim": 1536,
            "proj_factor_ffn": 2.667,
            "num_heads": 6,  # 4,
            "proj_factor_qk": 0.5,
            "chunk_size": 64,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 256,
            "learning_rate": 0.001,
        },
        "mlstm_1.4B": {
            "num_blocks": 24,
            "embedding_dim": 2048,
            "proj_factor_ffn": 2.667,
            "num_heads": 8,
            "proj_factor_qk": 0.5,
            "chunk_size": 64,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 256,
            "learning_rate": 0.0008,
        },
        "mlstm_2.7B": {
            "num_blocks": 32,
            "embedding_dim": 2560,
            "proj_factor_ffn": 2.667,
            "num_heads": 5,
            "proj_factor_qk": 0.5,
            "chunk_size": 64,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 512,
            "learning_rate": 0.0007,
        },
        "mlstm_7B": {
            "num_blocks": 32,
            "embedding_dim": 4096,
            "proj_factor_ffn": 2.667,
            "num_heads": 8,
            "proj_factor_qk": 0.5,
            "chunk_size": 64,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 512,
            "learning_rate": 0.0005,
        },
    }
    llama_model_configs = {
        "llama_160M": {
            "num_blocks": 12,
            "embedding_dim": 768,
            "proj_factor_ffn": 2.667,
            "head_dim": 64,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 128,
            "batch_size_per_device": 16,
            "learning_rate": 0.003,
        },
        "llama_400M": {
            "num_blocks": 24,
            "embedding_dim": 1024,
            "proj_factor_ffn": 2.667,
            "head_dim": 64,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 128,
            "batch_size_per_device": 16,
            "learning_rate": 0.003,
        },
        "llama_830M": {
            "num_blocks": 24,
            "embedding_dim": 1536,
            "proj_factor_ffn": 2.667,
            "head_dim": 128,  # 4,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 256,
            "batch_size_per_device": 16,
            "learning_rate": 0.001,
        },
        "llama_1.4B": {
            "num_blocks": 24,
            "embedding_dim": 2048,
            "proj_factor_ffn": 2.667,
            "head_dim": 128,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 256,
            "batch_size_per_device": 16,
            "learning_rate": 0.0008,
        },
        "llama_2.7B": {
            "num_blocks": 32,
            "embedding_dim": 2560,
            "proj_factor_ffn": 2.667,
            "head_dim": 128,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 512,
            "batch_size_per_device": 8,
            "learning_rate": 0.0007,
        },
        "llama_7B": {
            "num_blocks": 32,
            "embedding_dim": 4096,
            "proj_factor_ffn": 2.667,
            "head_dim": 128,
            "vocab_size": 50304,
            "ffn_multiple_of": 64,
            "global_batch_size": 512,
            "batch_size_per_device": 4,
            "learning_rate": 0.0007,
        },
    }

    mlstm_model_configs = compute_fwbw_flops_and_num_params_for_models(
        mlstm_model_configs,
        model_type="mlstm_v1",
        context_length=context_length,
        include_embedding_flops=include_embedding_flops,
        attention_flop_calc_mode=attention_flop_calc_mode,
        mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
        round_ffn_dim_to_multiple_of_for_flops=round_ffn_dim_to_multiple_of_for_flops,
    )
    llama_model_configs = compute_fwbw_flops_and_num_params_for_models(
        llama_model_configs,
        model_type="llama",
        context_length=context_length,
        include_embedding_flops=include_embedding_flops,
        attention_flop_calc_mode=attention_flop_calc_mode,
        mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
        round_ffn_dim_to_multiple_of_for_flops=round_ffn_dim_to_multiple_of_for_flops,
    )
    return {
        "mlstm": pd.DataFrame.from_dict(mlstm_model_configs, orient="index"),
        "llama": pd.DataFrame.from_dict(llama_model_configs, orient="index"),
    }


def get_param_fit_sclaw_data_df_dict(
    context_length: int = 8192,
    attention_flop_calc_mode: Literal[
        "chinchilla", "distill_scaling"
    ] = "distill_scaling",
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "tfla",
    experiment_set: Literal["all", "tokenparam", "isoflop"] = "tokenparam",
):
    return {
        "mlstm": create_param_fit_sclaw_data_table(
            model_type="mlstm",
            attention_flop_calc_mode=attention_flop_calc_mode,
            mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
            context_length=context_length,
            experiment_set=experiment_set,
            experiment_set_split="all",
        ),
        "llama": create_param_fit_sclaw_data_table(
            model_type="llama",
            attention_flop_calc_mode=attention_flop_calc_mode,
            mlstm_fw_flop_calc_mode=mlstm_fw_flop_calc_mode,
            context_length=context_length,
            experiment_set=experiment_set,
            experiment_set_split="all",
        ),
    }


def plot_parametric_loss_fit(
    parametric_sclaw_funcs: dict[
        str, dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]
    ],  # level 1: mlstm / llama, level 2: different fit functions
    model_configs: dict[str, pd.DataFrame] = get_model_config_df_dict(
        context_length=8192
    ),  # level 1: mlstm / llama, level 2: model size config
    param_fit_sclaw_data_df_dict: dict[
        str, pd.DataFrame
    ] = get_param_fit_sclaw_data_df_dict(context_length=8192, experiment_set="tokenparam"),
    param_fit_sclaw_data_num_param_selection: dict[str, list[float]] | None = {
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
    num_points: int = 250,
    token_param_ratio_range: tuple[float, float] = (11.0, 2500.0),
    sclaw_func_linestyles: list[str] = ["solid", "dashed", "dotted", "dashdot", (0, (1, 10)), (0, (3, 5, 1, 5, 1, 5))],
    model_size_colormap: Callable | None = None,
    model_size_colormap_scale: Literal[
        "linear", "log"
    ] = "log",  # scale for the color map
    num_flops_fwbw_col: str = "num_flops_fwbw",
    context_length: int = 8192,
    num_params_col: str = "num_params",
    llama_alpha: float = 0.5,
    x_axis_mode: Literal[
        "token_param_ratio", "num_tokens", "num_flops"
    ] = "token_param_ratio",
    x_axis_mode_to_x_col: dict[str, str] = {
        "token_param_ratio": "token_param_ratio",
        "num_tokens": "num_tokens_training",
        "num_flops": "num_flops_training",
    },
    y_col: str = "val/.dclm_loss",
    plot_mode: Literal["compare_models", "compare_sclaw"] = "compare_models",
    plot_data_points: bool = True,
    data_points_style_dict: dict[str, dict] = {
        "mlstm": {"marker": "o"},
        "llama": {"marker": "x"},
    },
    xscale: Literal["linear", "log"] = "log",
    yscale: Literal["linear", "log"] = "log",
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
    legend_kwargs: dict[str, str] | None = {
        "loc": "upper left",
        "bbox_to_anchor": (1.1, 1),
        "fontsize": 8,
    },
    figsize: tuple[float, float] = (5, 4),
    ax: Axes = None,
) -> Axes:
    """

    Args:
        plot_mode: compare_models assumes only one scaling law function per model type (label is the first key in the dict)
        compare_sclaw assumes multiple scaling law functions per model type (label is the second key in the dict)

    """

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # define the maximum num params for the color scale
    max_num_params = 0.0
    min_num_params = float("inf")
    for model_type, model_df in model_configs.items():
        max_num_params = max(max_num_params, model_df[num_params_col].max())
        min_num_params = min(min_num_params, model_df[num_params_col].min())

    # define the colormap
    if model_size_colormap is None:
        rocket_cmap_full = sns.color_palette(palette="rocket", as_cmap=True)
        rocket_cmap_middle = rocket_cmap_full(np.linspace(0.2, 0.8, 256))
        rocket_cmap = LinearSegmentedColormap.from_list(
            "rocket_cmap_middle", rocket_cmap_middle
        )
        model_size_colormap = rocket_cmap

    # define the color scale
    normalize_class = Normalize if model_size_colormap_scale == "linear" else LogNorm

    colormap_normalizer = normalize_class(
        vmin=min_num_params,
        vmax=max_num_params,
    )

    token_param_ratios = np.logspace(
        np.log10(token_param_ratio_range[0]),
        np.log10(token_param_ratio_range[1]),
        num_points,
        base=10.0,
    )

    sclaw_func_linestyle_iterator = cycle(sclaw_func_linestyles)

    # combine the parametric_sclaw_funcs dict into a single dict with keys as tuples
    # we iterate over the dict for plotting
    parametric_sclaw_funcs_single_key = {
        (key1, key2): sclaw_func
        for key1, dicts1 in parametric_sclaw_funcs.items()
        for key2, sclaw_func in dicts1.items()
    }

    for (
        model_type_key,
        sclaw_func_key,
    ), sclaw_func in parametric_sclaw_funcs_single_key.items():
        model_type_config_df = model_configs[model_type_key]
        linestyle = next(sclaw_func_linestyle_iterator)

        for model_key, model_config in model_type_config_df.iterrows():
            nparams = model_config[num_params_col]
            num_flops_fwbw = model_config[num_flops_fwbw_col]

            ntoks = nparams * token_param_ratios
            num_flops_total = num_flops_fwbw * (ntoks / context_length)

            # get the color for the model size
            color = model_size_colormap(colormap_normalizer(nparams))

            # compute the y-values for the scaling law function
            y_vals = sclaw_func(np.broadcast_to(nparams, ntoks.shape), ntoks)

            if x_axis_mode == "token_param_ratio":
                x_vals = token_param_ratios
            elif x_axis_mode == "num_tokens":
                x_vals = ntoks
            elif x_axis_mode == "num_flops":
                x_vals = num_flops_total
            else:
                raise ValueError(
                    f"x_axis_mode must be 'token_param_ratio', 'num_tokens' or 'num_flops', got {x_axis_mode} instead."
                )

            if plot_mode == "compare_models":
                label = model_key
            elif plot_mode == "compare_sclaw":
                label = model_key + "__" + sclaw_func_key
            else:
                raise ValueError(
                    f"plot_mode must be 'compare_models' or 'compare_sclaw', got {plot_mode} instead."
                )
            ax.plot(
                x_vals,
                y_vals,
                label=label,
                color=color,
                linestyle=linestyle,
                alpha=llama_alpha if model_type_key == "llama" else 1.0,
            )

    if plot_data_points:
        for model_type_key, model_df in param_fit_sclaw_data_df_dict.items():
            if model_type_key not in parametric_sclaw_funcs.keys():
                continue
            # select data points with correct num_params
            if param_fit_sclaw_data_num_param_selection is not None:
                # select data points with correct num_params
                model_df = model_df[
                    model_df["num_params"].isin(
                        param_fit_sclaw_data_num_param_selection[model_type_key]
                    )
                ]

            ax.scatter(
                x=model_df[x_axis_mode_to_x_col[x_axis_mode]],
                y=model_df[y_col],
                c=model_df[num_params_col],
                cmap=model_size_colormap,
                norm=colormap_normalizer,
                label=model_type_key,
                **data_points_style_dict[model_type_key],
                alpha=llama_alpha if model_type_key == "llama" else 1.0,
            )

    # set the x and y axis scales
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
    return ax
