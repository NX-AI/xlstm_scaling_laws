"""This module contains generic optimization functionality.
It uses scipy.optimize to perform optimization of a given objective function for a
given set of parameters over a grid of initialization values.

It enables bootstrapping, i.e. repeatedly sampling the data with replacement
and fitting the model to each sample.

Collects the results of the optimization and returns them in a pandas DataFrame.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional

import numpy as np
import pandas as pd
import scipy.optimize as spopt
from tqdm import tqdm

from .initialization import generate_initialization_sweep, x_to_param_dict

LOGGER = logging.getLogger(__name__)

ObjFuncConfig = Any
ValFuncConfig = Any


@dataclass
class OptimizationConfig:
    objective_func_config: ObjFuncConfig = None
    """The configuration for the objective function to be optimized."""

    validation_func_config: ValFuncConfig = None
    """The configuration for the validation function to be used for the optimization."""

    initialization_grid: dict[str, list[float]] = None
    """The grid of initialization values to be used for the optimization."""

    num_bootstrap_samples: int = -1
    """The number of bootstrap samples to be used for the optimization. 
    Default is -1, which means no bootstrapping."""
    bootstrap_seed: int = 1
    """The seed to be used to generate the seeds each for bootstrapping sample. Default is 1."""

    scipy_optim_module: Literal["least_squares", "minimize"] = "minimize"
    """The scipy optimization module to be used. Default is 'minimize'.
    Can be 'least_squares' for least squares optimization."""

    method: str = "L-BFGS-B"
    """The optimization method to be used. Default is L-BFGS-B."""
    tol: float | None = None
    """The tolerance for the optimization. Default is None, which means the solver specific default tolerance is used."""

    other_optimization_kwargs: dict[str, float] = field(default_factory=dict)
    """Other optimization kwargs to be passed to the optimization function."""


def run_optimization(
    config: OptimizationConfig,
    objective_func_generator: Callable[
        [ObjFuncConfig, Optional[int]], tuple[Callable, pd.DataFrame]
    ],
    validation_func_generator: Callable[
        [ObjFuncConfig, ValFuncConfig], Callable
    ] = None,
) -> pd.DataFrame:
    """Run the optimization for the given configuration.

    Args:
        config: The configuration for the optimization.
        objective_func_generator: The function to generate the objective function.
        validation_funcs_generator: The function to generate the validation functions.
            Default is None, which means no validation functions are used.
            If provided, the validation functions will be used to validate the optimization
            results.
            The generator function returns validation function that takes the optim params dict as input
            and returns a dict of metric values.

    Returns:
       The results of the optimization.
    """
    LOGGER.info("Running optimization with config: %s", config)

    # Create the objective function
    objective_func, df = objective_func_generator(config.objective_func_config)
    LOGGER.info(f"Created objective function with {len(df)} samples.")

    # Create the validation functions
    if validation_func_generator is not None:
        validation_func = validation_func_generator(
            config.objective_func_config, config.validation_func_config
        )
        LOGGER.info("Created validation function")
    else:
        validation_func = None
        LOGGER.info("No validation functions will be used.")

    # Generate the initialization grid
    initializations = generate_initialization_sweep(config.initialization_grid)
    LOGGER.info(f"Generated {len(initializations)} initializations.")

    # Generate the bootstrap seeds for the samples
    if config.num_bootstrap_samples > 0:
        rng = np.random.default_rng(config.bootstrap_seed)
        bootstrap_seeds = rng.integers(
            low=0,
            high=2**32 - 1,
            size=config.num_bootstrap_samples,
        )
        LOGGER.info(f"Generated {len(bootstrap_seeds)} bootstrap seeds.")
    else:
        bootstrap_seeds = None
        LOGGER.info("No bootstrapping will be used.")

    optim_results: list[tuple[int, int, dict[str, float], spopt.OptimizeResult]] = []

    def _minimize(func) -> spopt.OptimizeResult:
        """Wrapper for the scipy.optimize.minimize or least squares function."""
        if config.scipy_optim_module == "minimize":
            return spopt.minimize(
                fun=func,
                x0=init.x0,
                method=config.method,
                tol=config.tol,
                **config.other_optimization_kwargs,
            )
        elif config.scipy_optim_module == "least_squares":
            return spopt.least_squares(
                fun=func,
                x0=init.x0,
                method=config.method,
                ftol=config.tol,
                **config.other_optimization_kwargs,
            )
        else:
            raise ValueError(
                f"Unknown scipy optimization module: {config.scipy_optim_module}. "
                "Must be 'minimize' or 'least_squares'."
            )

    # Run the optimization
    i = 0
    for init in tqdm(initializations, desc="Running optimization"):
        result = _minimize(objective_func)
        # If validation function is provided, validate the result
        if validation_func is not None and result.success:
            validation_result = validation_func(
                **x_to_param_dict(x=result.x, order=init.order)
            )
        else:
            validation_result = {}

        # bootstrap_idx = 0 is the original data
        optim_results.append((i, 0, init.params, result, validation_result))

        if bootstrap_seeds is not None:
            # Sample the data with replacement
            LOGGER.info(
                f"Running bootstrapping for init {i + 1}/{len(initializations)}: {init.params}"
            )
            j = 0
            for bs_seed in tqdm(bootstrap_seeds, desc="Bootstrapping samples"):
                # Create the objective function with the bootstrap seed
                bs_objective_func, _ = objective_func_generator(
                    config.objective_func_config,
                    bootstrap_seed=bs_seed,
                )

                result = _minimize(bs_objective_func)
                # If validation function is provided, validate the result
                if validation_func is not None:
                    validation_result = validation_func(
                        **x_to_param_dict(x=result.x, order=init.order)
                    )
                else:
                    validation_result = {}

                optim_results.append((i, j + 1, init.params, result, validation_result))
                j += 1
        i += 1
    LOGGER.info(f"Optimization finished with {len(optim_results)} results.")

    ret_df = _create_optim_result_summary_df(optim_results)

    return ret_df


def _create_optim_result_summary_df(
    optim_results: list[tuple[int, int, dict[str, float], spopt.OptimizeResult]],
) -> pd.DataFrame:
    def _optim_res_to_dict(optim_res: spopt.OptimizeResult) -> dict[str, float]:
        return {
            "loss": optim_res.fun,
            "message": optim_res.message,
            "success": optim_res.success,
            "jac": optim_res.jac,
            "nfev": optim_res.nfev,
            "njev": optim_res.njev,
            "status": optim_res.status,
        }

    idxes_list = []
    optim_param_results_list = []
    # optim_param_results_power_ab_list = []
    init_list = []
    optim_results_list = []
    val_results_list = []

    for optim_result in optim_results:
        idxes_list.append(
            {"init_idx": optim_result[0], "bootstrap_idx": optim_result[1]}
        )
        optim_param_result_dict = x_to_param_dict(
            x=optim_result[3].x, order=optim_result[2].keys()
        )
        optim_param_results_list.append(optim_param_result_dict)

        # alpha = optim_param_result_dict["alpha"]
        # beta = optim_param_result_dict["beta"]
        # optim_param_results_power_ab_list.append(
        #     {
        #         "power_a": beta / (alpha + beta),
        #         "power_b": alpha / (alpha + beta),
        #     }
        # )

        init_list.append(optim_result[2])
        optim_results_list.append(_optim_res_to_dict(optim_result[3]))
        val_results_list.append(optim_result[4])

    idxes_df = pd.DataFrame(idxes_list)
    optim_param_results_df = pd.DataFrame(optim_param_results_list)
    # optim_param_results_power_ab_df = pd.DataFrame(optim_param_results_power_ab_list)
    init_df = pd.DataFrame(init_list)
    optim_results_df = pd.DataFrame(optim_results_list)
    val_results_df = pd.DataFrame(val_results_list)

    optim_results_df = pd.concat(
        [
            idxes_df,
            val_results_df,
            optim_param_results_df,
            optim_results_df,
            init_df,
        ],
        axis=1,
        keys=["idxes", "val_results", "optim_params", "optim_results", "init_params"],
    )
    return optim_results_df
