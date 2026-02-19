from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from xlstm_scaling_laws.load_data.inference_time import load_inference_time_data
from xlstm_scaling_laws.model_accounting.inference_time_model.llama_runtime_model import (
    predict_runtime_llama_step_time,
    predict_runtime_llama_ttft,
)
from xlstm_scaling_laws.model_accounting.inference_time_model.mlstm_runtime_model import (
    predict_runtime_mlstm_step_time,
    predict_runtime_mlstm_ttft,
)


def huber_loss(x: np.ndarray, delta: float) -> np.ndarray:
    """Huber-Loss function.
    This function is quadratic for small values of x and linear for large values of x.
    """
    if delta == 0:
        return np.abs(x)
    return np.where(np.abs(x) <= delta, 0.5 * x**2, delta * (np.abs(x) - 0.5 * delta))
    # return 0.5 * x**2


def objective_huber_inference_time_model(
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
    rho: float | np.ndarray,
    eps: float | np.ndarray,
    eps_bp: float | np.ndarray,
    data_df: pd.DataFrame,
    huber_delta: float,
    runtime_model_mode: str = "attainable_flops_logsumexp",
    alpha_0: float | None = None,
    beta_0: float | None = None,
    rho_0: float | None = None,
    eps_0: float | None = None,
    eps_bp_0: float | None = None,
    factor_causal: float = 0.75,
    bytes_act: int = 2,  # bfloat16
    bytes_Cmn: int = 4,  # float32
    bytes_w: int = 2,  # bfloat16
    reduce_loss: Literal["sum", "mean", "none"] = "sum",
    squeeze_loos: bool = False,
    fit_model: Literal["xlstm", "llama2"] = "xlstm",
    fit_data: Literal["ttft", "step_time"] = "ttft",
):
    """Objective function for the inference time model."""

    if fit_model == "xlstm" and fit_data == "ttft":
        pred_df = predict_runtime_mlstm_ttft(
            ttft_df=data_df,
            rho=rho,
            alpha=alpha,
            beta=beta,
            eps=eps,
            eps_bp=eps_bp,
            rho_0=rho_0,
            alpha_0=alpha_0,
            beta_0=beta_0,
            eps_0=eps_0,
            eps_bp_0=eps_bp_0,
            mode=runtime_model_mode,
            factor_causal=factor_causal,
            bytes_act=bytes_act,
            bytes_Cmn=bytes_Cmn,
            bytes_w=bytes_w,
        )
    elif fit_model == "xlstm" and fit_data == "step_time":
        pred_df = predict_runtime_mlstm_step_time(
            step_time_df=data_df,
            rho=rho,
            alpha=alpha,
            beta=beta,
            eps=eps,
            eps_bp=eps_bp,
            rho_0=rho_0,
            alpha_0=alpha_0,
            beta_0=beta_0,
            eps_0=eps_0,
            eps_bp_0=eps_bp_0,
            mode=runtime_model_mode,
            factor_causal=factor_causal,
            bytes_act=bytes_act,
            bytes_Cmn=bytes_Cmn,
            bytes_w=bytes_w,
        )
    elif fit_model == "llama2" and fit_data == "ttft":
        pred_df = predict_runtime_llama_ttft(
            ttft_df=data_df,
            rho=rho,
            alpha=alpha,
            beta=beta,
            eps=eps,
            eps_bp=eps_bp,
            rho_0=rho_0,
            alpha_0=alpha_0,
            beta_0=beta_0,
            eps_0=eps_0,
            eps_bp_0=eps_bp_0,
            mode=runtime_model_mode,
            bytes_act=bytes_act,
            bytes_w=bytes_w,
        )
    elif fit_model == "llama2" and fit_data == "step_time":
        pred_df = predict_runtime_llama_step_time(
            step_time_df=data_df,
            rho=rho,
            alpha=alpha,
            beta=beta,
            eps=eps,
            eps_bp=eps_bp,
            rho_0=rho_0,
            alpha_0=alpha_0,
            beta_0=beta_0,
            eps_0=eps_0,
            eps_bp_0=eps_bp_0,
            mode=runtime_model_mode,
            bytes_act=bytes_act,
            bytes_w=bytes_w,
        )
    else:
        raise NotImplementedError(
            f"Objective function for fit_model={fit_model} and fit_data={fit_data} not implemented."
        )

    y_pred = pred_df[("pred", "runtime")].to_numpy()
    y_target_runtime_measured = pred_df[("measured_data", "runtime")].to_numpy()

    if runtime_model_mode.startswith("log_"):
        y_target_runtime_measured = np.log(y_target_runtime_measured)

    loss = y_pred - y_target_runtime_measured

    if reduce_loss == "none":

        def reduce_fn(x, *args, **kwargs):
            return x
    elif reduce_loss == "sum":
        reduce_fn = np.sum
    else:
        reduce_fn = np.mean

    while loss.ndim < 2:
        loss = loss[:, np.newaxis]

    total_loss = reduce_fn(huber_loss(loss, delta=huber_delta), axis=0)
    if squeeze_loos:
        total_loss = total_loss.squeeze()
    return total_loss


@dataclass
class InferenceModelObjectiveConfig:
    huber_delta: float = 1e-3
    filter_zero_prefill: bool = True
    fit_data: Literal["ttft", "step_time"] = "ttft"
    fit_model: Literal["xlstm", "llama2"] = "xlstm"
    runtime_model_mode: str = "attainable_flops_logsumexp"
    alpha_0: float | None = None
    beta_0: float | None = None
    rho_0: float | None = None
    eps_0: float | None = None
    eps_bp_0: float | None = None
    factor_causal: float = 0.75
    bytes_act: int = 2  # bfloat16
    bytes_Cmn: int = 4  # float32
    bytes_w: int = 2  # bfloat16
    reduce_loss: Literal["sum", "mean", "none"] = "sum"
    squeeze_loos: bool = False
    override_df: pd.DataFrame | None = None


def get_inference_time_model_objective_func(
    config: InferenceModelObjectiveConfig, bootstrap_seed: int = 0
):
    """Returns the objective function for the inference time model."""

    if config.override_df is not None:
        # If an override DataFrame is provided, use it directly
        df = config.override_df
    else:
        # load the data
        ttft_df, step_df = load_inference_time_data(config.fit_model)

        if config.fit_data == "ttft":
            df = ttft_df
        elif config.fit_data == "step_time":
            df = step_df

        df = df.dropna()
        if config.filter_zero_prefill:
            df = df[df[("input_params", "prefill")] > 0]

    if bootstrap_seed > -1:
        rng = np.random.default_rng(bootstrap_seed)
        df = df.sample(
            n=df.shape[0],
            replace=True,
            random_state=rng,
        )

    def objective_func(x: np.ndarray) -> np.ndarray:
        """Objective function for the inference time model."""

        kwargs = dict(
            data_df=df,
            huber_delta=config.huber_delta,
            runtime_model_mode=config.runtime_model_mode,
            alpha_0=config.alpha_0,
            beta_0=config.beta_0,
            rho_0=config.rho_0,
            eps_0=config.eps_0,
            eps_bp_0=config.eps_bp_0,
            factor_causal=config.factor_causal,
            bytes_act=config.bytes_act,
            bytes_Cmn=config.bytes_Cmn,
            bytes_w=config.bytes_w,
            reduce_loss=config.reduce_loss,
            squeeze_loos=config.squeeze_loos,
        )
        if "linear_flops" in config.runtime_model_mode:
            alpha = x[0]
            beta = None
            rho = None
            eps = x[1]
            eps_bp = x[2]
        elif "linear_memops" in config.runtime_model_mode:
            alpha = None
            beta = x[0]
            rho = None
            eps = x[1]
            eps_bp = x[2]
        else:
            if len(x) == 4:
                alpha = x[0]
                beta = x[1]
                rho = None
                eps = x[2]
                eps_bp = x[3]
            elif len(x) == 5:
                alpha = x[0]
                beta = x[1]
                rho = x[2]
                eps = x[3]
                eps_bp = x[4]
            else:
                raise ValueError(
                    f"Expected 4 or 5 parameters, got {len(x)} instead. x: {x}"
                )
        kwargs.update(
            alpha=alpha,
            beta=beta,
            rho=rho,
            eps=eps,
            eps_bp=eps_bp,
        )

        return objective_huber_inference_time_model(
            **kwargs, fit_model=config.fit_model, fit_data=config.fit_data
        )

    return objective_func, df
