from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
import scipy.special as sp

ScalingLawReturnMode = Literal["logsumexp", "sum", "individual"]


## Parametric scaling law function
def log_scaling_law_hoffmann(
    nparams: np.ndarray,
    ntoks: np.ndarray,
    a: float | np.ndarray,
    b: float | np.ndarray,
    e: float | np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
    gamma: float | np.ndarray = 1.0,
    return_mode: Literal[ScalingLawReturnMode] = "logsumexp",
) -> np.ndarray:
    """Log Loss calculated by the parametric form used for Chinchilla scaling laws.

    Args:
        N: Number of parameters (n, 1).
        D: Number of tokens (n, 1).
        a: Scaling parameter.
        b: Scaling parameter.
        e: offset parameter.
        alpha: Param exponent parameter.
        beta: Token exponent parameter.
        gamma: Exponent for the scaling law D, N dependent part. Defaults to 1.0.
               Taken from Distillation Scaling Laws paper: https://arxiv.org/abs/2502.08606
        return_mode: Return mode for the scaling law. Can be "logsumexp", "sum" or "individual".
                     Defaults to "logsumexp".
    """
    while nparams.ndim < 2:
        nparams = nparams[:, np.newaxis]

    while ntoks.ndim < 2:
        ntoks = ntoks[:, np.newaxis]

    a_term = a - alpha * np.log(nparams)
    b_term = b - beta * np.log(ntoks)
    e_term = np.tile(e, nparams.shape)

    if return_mode == "sum":
        return a_term + gamma * (b_term + e_term)
    elif return_mode == "logsumexp":
        ab_lse = gamma * sp.logsumexp(
            np.concatenate([a_term, b_term], axis=1), axis=1, keepdims=True
        )
        lse = sp.logsumexp(
            np.concatenate([ab_lse, e_term], axis=1), axis=1, keepdims=True
        )
        return lse
    elif return_mode == "individual":
        return np.concatenate([gamma * a_term, gamma * b_term, e_term], axis=1)


def get_lnd_scaling_law_hoffmann_fn(
    a: float | np.ndarray,
    b: float | np.ndarray,
    e: float | np.ndarray,
    alpha: float | np.ndarray,
    beta: float | np.ndarray,
    gamma: float | np.ndarray = 1.0,
    return_mode: ScalingLawReturnMode = "logsumexp",
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Get the L(N, D) scaling law function for Hoffmann et al. Chinchilla.
    The returned function takes two arguments: nparams and ntoks, and returns the (non-log) loss.
    """
    return lambda nparams, ntoks: np.exp(
        log_scaling_law_hoffmann(
            nparams,
            ntoks,
            a=a,
            b=b,
            e=e,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            return_mode=return_mode,
        )
    )


def get_first_n_fits_as_fit_fn_dict(
    result_fit_df: pd.DataFrame,
    key_prefix: str,
    n: int = 3,
    return_mode: ScalingLawReturnMode = "logsumexp",
) -> dict:
    """
    Get the first n fits as a dictionary.
    """
    sel_df = result_fit_df[:n]
    fitted_params_list = sel_df["optim_params"].to_dict(orient="records")
    optim_results_list = sel_df["optim_results"].to_dict(orient="records")
    val_results_list = sel_df["val_results"].to_dict(orient="records")

    # Convert the list of dictionaries to a single dictionary
    fitted_params_dict = {}
    for fit_params, optim_results, val_results in zip(
        fitted_params_list, optim_results_list, val_results_list
    ):
        fitted_params_dict[
            f"{key_prefix}"
            + "_".join([f"{key}{value:.3f}" for key, value in fit_params.items()])
            + f"__loss{optim_results['loss']:.3e}"
            + "_".join(
                [
                    f"{key}{value:.2f}"
                    for key, value in val_results.items()
                    if "tokenparam" in key
                ]
            )
        ] = get_lnd_scaling_law_hoffmann_fn(**fit_params, return_mode=return_mode)
    return fitted_params_dict
