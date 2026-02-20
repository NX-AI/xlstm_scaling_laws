from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Literal

import numpy as np
import pandas as pd

from ...analysis.parametric_sclaw_fit.data import create_param_fit_sclaw_data_table
from ..common.metrics import calculate_metrics
from .scaling_law_funcs import log_scaling_law_hoffmann


## Loss functions
def quadratic_loss(x: np.ndarray) -> np.ndarray:
    """Quadratic loss function."""
    return 0.5 * x**2


def huber_loss(x: np.ndarray, delta: float) -> np.ndarray:
    """Huber-Loss function.
    This function is quadratic for small values of x and linear for large values of x.
    """
    return np.where(np.abs(x) <= delta, 0.5 * x**2, delta * (np.abs(x) - 0.5 * delta))


## Objective function
def objective_huber_log_scl_hoffmann(
    a: float | np.ndarray,
    b: float | np.ndarray,
    e: float | np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
    gamma: float | np.ndarray,
    nparams: np.ndarray,
    ntoks: np.ndarray,
    y_logloss: np.ndarray,
    huber_delta: float,
    add_logsumexp: bool = True,
    reduce_loss: Literal["sum", "mean", "none"] = "sum",
) -> np.ndarray:
    loss = log_scaling_law_hoffmann(
        nparams,
        ntoks,
        a,
        b,
        e,
        alpha,
        beta,
        gamma=gamma,
        return_mode="logsumexp" if add_logsumexp else "sum",
    )

    if reduce_loss == "none":

        def reduce_fn(x, *args, **kwargs):
            return x
    elif reduce_loss == "sum":
        reduce_fn = np.sum
    else:
        reduce_fn = np.mean

    while y_logloss.ndim < 2:
        y_logloss = y_logloss[:, np.newaxis]

    total_loss = reduce_fn(huber_loss(y_logloss - loss, delta=huber_delta), axis=0)
    return total_loss


## SciPy Objective functions
def scipy_objective_huber_log_scl_hoffmann(
    x: np.ndarray,
    nparams: np.ndarray,
    ntoks: np.ndarray,
    y_logloss: np.ndarray,
    huber_delta: float,
    add_logsumexp: bool = True,
    fit_gamma: bool = False,
    reduce_loss: Literal["sum", "mean", "none"] = "sum",
) -> np.ndarray:
    if fit_gamma:
        a, b, e, alpha, beta, gamma = x
    else:
        a, b, e, alpha, beta = x
        gamma = 1.0

    total_loss = objective_huber_log_scl_hoffmann(
        a,
        b,
        e,
        alpha,
        beta,
        gamma,
        nparams,
        ntoks,
        y_logloss,
        huber_delta,
        add_logsumexp,
        reduce_loss,
    )
    return total_loss


@dataclass
class HoffmannScalingLawObjectiveConfig:
    huber_delta: float = 1e-3
    use_logsumexp: bool = True
    fit_gamma: bool = False
    reduce_loss: Literal["sum", "mean", "none"] = "sum"

    target_loss: Literal["train", "val"] = "val"
    """The target loss to use for the scaling law. Can be "train" or "val"."""

    model_type: Literal["mlstm", "llama"] = "mlstm"
    """The model type to use for the scaling law. Can be "mlstm" or "llama"."""

    attention_flop_calc_mode: Literal["chinchilla", "distill_scaling"] = (
        "distill_scaling"
    )
    """The attention flop calculation mode to use for the scaling law. Can be "chinchilla" or "distill_scaling"."""
    context_length: int = 8192
    """The context length to use for the scaling law. Can be 2048, 8192 or 16384."""

    experiment_set: Literal["all", "tokenparam", "isoflop"] = "all"
    """The experiment set to use for the scaling law. Can be "all", "tokenparam" or "isoflop"."""
    experiment_set_split: Literal["all", "all_butlong7b", "long7b"] = "all_butlong7b"
    """The experiment set split to use for the scaling law. Can be "all", "all_butlong7b" or "long7b"."""

    token_param_range: list[float, float] | None = None
    """The range of token param ratios to use for the scaling law. Can be None or a list of two floats.
    If None, the full range is used.
    If a list of two floats, the range is used to filter the data.
    """


def get_scaling_law_objective_func(
    config: HoffmannScalingLawObjectiveConfig,
    bootstrap_seed: int = -1,
) -> tuple[Callable, pd.DataFrame]:
    """Get the scaling law objective function for the given configuration.
    Args:
        config: The configuration for the scaling law.
        bootstrap_seed: The seed to use for bootstrapping. If -1, no bootstrapping is used.
            We always sample with replacement, the same number of samples as the original data.
    Returns:
        tuple[Callable, pd.DataFrame]: The scaling law objective function and the data used for the scaling law.
    """
    df = create_param_fit_sclaw_data_table(
        model_type=config.model_type,
        attention_flop_calc_mode=config.attention_flop_calc_mode,
        context_length=config.context_length,
        experiment_set=config.experiment_set,
        experiment_set_split=config.experiment_set_split,
    )

    # Filter the data
    if config.token_param_range is not None:
        df = df[
            (df["token_param_ratio"] >= config.token_param_range[0])
            & (df["token_param_ratio"] <= config.token_param_range[1])
        ]

    if bootstrap_seed > -1:
        rng = np.random.default_rng(bootstrap_seed)
        df = df.sample(
            n=df.shape[0],
            replace=True,
            random_state=rng,
        )

    # create the objective function
    if config.target_loss == "train":
        y_logloss = np.log(df["train/.loss_mean"].to_numpy())
    elif config.target_loss == "val":
        y_logloss = np.log(df["val/.dclm_loss"].to_numpy())
    else:
        raise ValueError(
            f"target_loss must be 'train' or 'val', got {config.target_loss} instead."
        )

    objective_func = partial(
        scipy_objective_huber_log_scl_hoffmann,
        nparams=df["num_params"].to_numpy(),
        ntoks=df["num_tokens_training"].to_numpy(),
        y_logloss=y_logloss,
        huber_delta=config.huber_delta,
        add_logsumexp=config.use_logsumexp,
        fit_gamma=config.fit_gamma,
        reduce_loss=config.reduce_loss,
    )

    return objective_func, df


@dataclass
class ScalingLawValidationConfig:
    metrics: list[str] = field(
        default_factory=lambda: ["r_squared", "mse", "rmse", "mae"]
    )
    experiment_sets: list[Literal["all", "tokenparam", "isoflop"]] = field(
        default_factory=lambda: ["all", "tokenparam", "isoflop"]
    )


def get_scaling_law_validation_func(
    config_obj_func: HoffmannScalingLawObjectiveConfig,
    config_val_func: ScalingLawValidationConfig,
) -> Callable:
    """Get the scaling law validation function for the given configuration.
    Validation is NOT in log space.

    Args:
        config: The configuration for the scaling law.
    Returns:
        Callable: The scaling law validation function.
    """
    # create teh dataframe dict

    df_dict = {
        exp_set: create_param_fit_sclaw_data_table(
            model_type=config_obj_func.model_type,
            attention_flop_calc_mode=config_obj_func.attention_flop_calc_mode,
            context_length=config_obj_func.context_length,
            experiment_set=exp_set,
            experiment_set_split="all",
        )
        for exp_set in config_val_func.experiment_sets
    }

    # Filter the data
    if config_obj_func.token_param_range is not None:
        df_dict = {
            exp_set: df[
                (df["token_param_ratio"] >= config_obj_func.token_param_range[0])
                & (df["token_param_ratio"] <= config_obj_func.token_param_range[1])
            ]
            for exp_set, df in df_dict.items()
        }

    # create the validation function
    if config_obj_func.target_loss == "train":
        y_true_col = "train/.loss_mean"
    elif config_obj_func.target_loss == "val":
        y_true_col = "val/.dclm_loss"
    else:
        raise ValueError(
            f"target_loss must be 'train' or 'val', got {config_obj_func.target_loss} instead."
        )

    def validation_func(
        a: float | np.ndarray,
        b: float | np.ndarray,
        e: float | np.ndarray,
        alpha: float | np.ndarray,
        beta: float | np.ndarray,
        gamma: float | np.ndarray = 1.0,
    ) -> dict[str, float]:
        """Validation function for the scaling law."""
        all_exp_set_metrics = {}
        for exp_set, df in df_dict.items():
            y_pred = np.exp(
                log_scaling_law_hoffmann(
                    df["num_params"].to_numpy(),
                    df["num_tokens_training"].to_numpy(),
                    a,
                    b,
                    e,
                    alpha,
                    beta,
                    gamma=gamma,
                    return_mode="logsumexp",
                )
            )
            y_true = df[y_true_col].to_numpy()

            metrics_dict = calculate_metrics(
                y_true=y_true, y_pred=y_pred, metrics=config_val_func.metrics
            )
            for key, value in metrics_dict.items():
                all_exp_set_metrics[f"{exp_set}#{key}"] = value
        return all_exp_set_metrics

    return validation_func
