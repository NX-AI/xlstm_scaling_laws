"""In this module we define the run the scaling law fit grids of the scaling laws over all our data.

A single scaling law fit grid runs the following grid:
- fit for mlstm and llama
- fit for different experiment sets: "all", "isoflop", "tokenparam"
- fit different token param ratio ranges
-> within each fit in the grid we sweep over a grid of initializations
-> we could also enable bootstrapping, but we don't do that here

We run each fit grid for different configurations:
- use logsumexp for the parametric loss: True/False
- add a gamma parameter to the parametric loss: True/False
- huber deltas: 0.0, 1e-2, 1e-3, 1e-4, 1e-5
"""

import logging
import pickle
from itertools import product
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ...fitting.fit_parametric_loss.run_fit_parametric_loss import (
    FitParametricLossConfig,
    HoffmannScalingLawObjectiveConfig,
    ScalingLawValidationConfig,
    fit_parametric_loss,
)

LOGGER = logging.getLogger(__name__)

# Notes on optimal initialization grid:
# We do an initial fit with a small initialization grid and report the results:
# mLSTM -- tokenparam:
# - with gamma: mlstm-a16.22_b17.31_e0.11_alpha0.73_beta0.67_gamma0.24__loss8.557e-05r_squared-69.21_mse0.20_rmse0.44_mae0.36
# - without gamma: mlstm-a3.65_b4.54_e0.23_alpha0.17_beta0.23__loss1.983e-04r_squared-68.47_mse0.19_rmse0.44_mae0.36
# Llama -- tokenparam:
# - with gamma: llama-a11.98_b13.35_e0.01_alpha0.53_beta0.51_gamma0.29__loss7.148e-05r_squared-60.57_mse0.18_rmse0.43_mae0.35
# - without gamma: llama-a2.88_b3.60_e-0.15_alpha0.13_beta0.17__loss1.385e-04r_squared-60.83_mse0.18_rmse0.43_mae0.35

# optimization parameters
init_grid_without_gamma = {
    "a": [0.0, 5.0, 10.0, 15.0, 20.0],
    "b": [0.0, 5.0, 10.0, 15.0, 20.0],
    "e": [-1.0, -0.5, 0.0, 0.5, 1.0],  # 1.5
    "alpha": [0.0, 0.2, 0.5, 1.0],  # 1.5
    "beta": [0.0, 0.2, 0.5, 1.0],  # 1.5
}
# init_grid_without_gamma = {
#     "a": [0.0, 5.0],# 10.0, 15.0, 20.0],
#     "b": [0.0, 5.0],# 10.0, 15.0, 20.0],
#     "e": [0.0],#[-1.0, -0.5, 0.0, 0.5, 1.0], # 1.5
#     "alpha": [0.0],#[0.0, 0.2, 0.5, 1.0], # 1.5
#     "beta": [0.0],#[0.0, 0.2, 0.5, 1.0], # 1.5
# }
init_grid_with_gamma = {
    **init_grid_without_gamma,
    "gamma": [0.0, 0.5, 1.0, 1.5],
}
optimization_method = "L-BFGS-B"
# relation to default params:
# smaller tolerance, more line search iterations
optimization_kwargs = {
    "options": {
        "ftol": 1e-32,
        "gtol": 1e-32,
        "maxls": 200,
    },
}


def run_fit_grid(
    # vary params
    use_logsumexp: bool = True,
    huber_delta: float = 1e-3,
    fit_gamma: bool = True,
    context_length: int = 8192,
    # fixed (default) params
    target_loss: str = "val",
    optimization_method: str = optimization_method,
    optimization_kwargs: dict = optimization_kwargs,
    init_grid_with_gamma: dict = init_grid_with_gamma,
    init_grid_without_gamma: dict = init_grid_without_gamma,
    num_bootstrap_samples: int = -1,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    model_types = ["mlstm", "llama"]
    experiment_sets = ["all", "tokenparam", "isoflop"]
    token_param_ranges = [
        [0, 100],
        [0, 300],
        [0, 5000],
    ]  # the last one is the full range

    def _tok_param_range_to_str(token_param_range: list[float, float]) -> str:
        return f"{token_param_range[0]}-{token_param_range[1]}"

    init_grid = init_grid_with_gamma if fit_gamma else init_grid_without_gamma

    LOGGER.info(
        f"Running fit grid with the following parameters: use_logsumexp={use_logsumexp}, huber_delta={huber_delta}, fit_gamma={fit_gamma}, context_length={context_length}, target_loss={target_loss}, optimization_method={optimization_method}, optimization_kwargs={optimization_kwargs}, init_grid_with_gamma={init_grid_with_gamma}, init_grid_without_gamma={init_grid_without_gamma}, num_bootstrap_samples={num_bootstrap_samples}"
    )
    LOGGER.info(f"Model types: {model_types}")
    LOGGER.info(f"Experiment sets: {experiment_sets}")
    LOGGER.info(f"Token param ranges: {token_param_ranges}")

    total_fit_grid_dict = {}
    for i, model_type in enumerate(model_types):
        LOGGER.info(f"Running model_type ({i + 1}/{len(model_types)}): {model_type}")
        model_fit_grid_dict = {}
        for j, experiment_set in enumerate(experiment_sets):
            LOGGER.info(
                f"Running experiment_set ({j + 1}/{len(experiment_sets)}): {experiment_set}"
            )
            experiment_set_fit_grid_dict = {}
            for k, token_param_range in enumerate(token_param_ranges):
                LOGGER.info(
                    f"Running token_param_range ({k + 1}/{len(token_param_ranges)}): {token_param_range}"
                )
                token_param_range_str = _tok_param_range_to_str(token_param_range)

                fit_cfg = FitParametricLossConfig(
                    objective_func_config=HoffmannScalingLawObjectiveConfig(
                        model_type=model_type,
                        attention_flop_calc_mode="distill_scaling",
                        context_length=context_length,
                        experiment_set=experiment_set,
                        experiment_set_split="all",
                        token_param_range=token_param_range,
                        target_loss=target_loss,
                        huber_delta=huber_delta,
                        fit_gamma=fit_gamma,
                        use_logsumexp=use_logsumexp,
                    ),
                    validation_func_config=ScalingLawValidationConfig(),
                    initialization_grid=init_grid,
                    num_bootstrap_samples=num_bootstrap_samples,
                    bootstrap_seed=1,
                    method=optimization_method,
                    tol=None,
                    other_optimization_kwargs=optimization_kwargs,
                )

                ret_df = fit_parametric_loss(config=fit_cfg)
                # add the fit result to the dict
                experiment_set_fit_grid_dict[token_param_range_str] = ret_df
                LOGGER.info(
                    f"Finished fit for token_param_range {token_param_range_str} with {len(ret_df)} results."
                )

            model_fit_grid_dict[experiment_set] = experiment_set_fit_grid_dict
            LOGGER.info(
                f"Finished fits for experiment_set {experiment_set} with {len(experiment_set_fit_grid_dict)} experiment sets."
            )
        total_fit_grid_dict[model_type] = model_fit_grid_dict
        LOGGER.info(
            f"Finished fits for model_type {model_type} with {len(model_fit_grid_dict)} models."
        )

    return total_fit_grid_dict


def filename_from_params(
    use_logsumexp: bool,
    huber_delta: float,
    fit_gamma: bool,
    context_length: int,
    prefix: str = "",
) -> str:
    return f"{prefix}fit_grid__uselogsumexp-{use_logsumexp}_huberdelta-{huber_delta}_fitgamma-{fit_gamma}_ctx-{context_length}.pkl"


def params_from_filename(filename: str) -> dict[str, Any]:
    # filename = "fit_grid__uselogsumexp-{use_logsumexp}_huberdelta-{huber_delta}_fitgamma-{fit_gamma}_ctx-{context_length}.pkl"
    params = {}
    params["use_logsumexp"] = filename.split("uselogsumexp-")[1].split("_")[0] == "True"
    params["huber_delta"] = float(filename.split("huberdelta-")[1].split("_")[0])
    params["fit_gamma"] = filename.split("fitgamma-")[1].split("_")[0] == "True"
    params["context_length"] = int(filename.split("ctx-")[1].split(".pkl")[0])
    return params


def run_fit_grids(
    save_dir: Path,
    param_combination_mode: Literal["all", "single"] = "all",
    use_logsumexp: bool = True,
    huber_delta: float = 1e-3,
    fit_gamma: bool = True,
    context_length: int = 8192,
) -> None:
    """Run the fit grid and save the results to the given directory.
    Args:
        save_dir: The directory to save the results to.
        param_combination_mode: The parameter combination mode to use. Can be "all" or "single".
            If "all", all combinations of parameters are used.
            If "single", only a single combination of parameters is used.
        use_logsumexp: Whether to use logsumexp for the parametric loss.
        huber_delta: The huber delta to use for the parametric loss.
        fit_gamma: Whether to fit gamma for the parametric loss.
        context_length: The context length to use for the scaling law. Can be 2048, 8192 or 16384.
    """
    # create the save dir if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    if param_combination_mode == "single":
        LOGGER.info(
            f"Running fit grid with the following parameters: use_logsumexp={use_logsumexp}, huber_delta={huber_delta}, fit_gamma={fit_gamma}, context_length={context_length}"
        )
        # run the fit grid
        fit_grid_results = run_fit_grid(
            use_logsumexp=use_logsumexp,
            huber_delta=huber_delta,
            fit_gamma=fit_gamma,
            context_length=context_length,
        )
        # save the results
        filename = filename_from_params(
            use_logsumexp=use_logsumexp,
            huber_delta=huber_delta,
            fit_gamma=fit_gamma,
            context_length=context_length,
        )
        with open(save_dir / filename, "wb") as f:
            pickle.dump(fit_grid_results, f)

    elif param_combination_mode == "all":
        use_logsumexp_list = [True]  # [True, False]
        fit_gamma_list = [True, False]
        huber_delta_list = [1e-3, 1e-4, 5.0, 1.0, 1e-1, 1e-2, 1e-5]

        param_combinations = list(
            product(use_logsumexp_list, fit_gamma_list, huber_delta_list)
        )

        for i, (use_logsumexp, fit_gamma, huber_delta) in enumerate(param_combinations):
            LOGGER.info(
                f"Running fit grid {i + 1}/{len(param_combinations)} with the following parameters: use_logsumexp={use_logsumexp}, huber_delta={huber_delta}, fit_gamma={fit_gamma}, context_length={context_length}"
            )
            # run the fit grid
            fit_grid_results = run_fit_grid(
                use_logsumexp=use_logsumexp,
                huber_delta=huber_delta,
                fit_gamma=fit_gamma,
                context_length=context_length,
            )
            # save the results
            filename = filename_from_params(
                use_logsumexp=use_logsumexp,
                huber_delta=huber_delta,
                fit_gamma=fit_gamma,
                context_length=context_length,
            )
            with open(save_dir / filename, "wb") as f:
                pickle.dump(fit_grid_results, f)
            LOGGER.info(
                f"Finished fit grid {i + 1}/{len(param_combinations)} with the following parameters: use_logsumexp={use_logsumexp}, huber_delta={huber_delta}, fit_gamma={fit_gamma}, context_length={context_length}"
            )
    else:
        raise ValueError(
            f"Unknown param_combination_mode {param_combination_mode}. Must be 'all' or 'single'."
        )
    LOGGER.info(f"Finished running fit grid. Results saved to {save_dir}.")


def _find_file(
    save_dir: Path,
    use_logsumexp: bool = True,
    huber_delta: float = 1e-3,
    fit_gamma: bool = True,
    context_length: int = 8192,
    prefix: str = "",
) -> Path:
    huber_delta_strs = [
        f"{huber_delta:.0e}",
        f"{huber_delta:.1f}",
        f"{huber_delta:.2f}",
        f"{huber_delta:.3f}",
        f"{huber_delta:.4f}",
    ]

    searched_filenames = []

    file_found = False
    for huber_delta_str in huber_delta_strs:
        search_filename = filename_from_params(
            use_logsumexp=use_logsumexp,
            huber_delta=huber_delta_str,
            fit_gamma=fit_gamma,
            context_length=context_length,
            prefix=prefix,
        )
        file = Path(save_dir) / search_filename
        searched_filenames.append(search_filename)
        if file.exists():
            file_found = True
            break

    if not file_found:
        raise FileNotFoundError(
            f"File not found with names: {str(searched_filenames)}. Please check the save_dir and parameters."
        )

    return file


def load_scaling_law_fit_grid_result(
    save_dir: Path,
    use_logsumexp: bool = True,
    huber_delta: float = 1e-3,
    fit_gamma: bool = True,
    context_length: int = 8192,
) -> dict[str, dict[str, dict[str, pd.DataFrame]]]:
    # filename = "fit_grid__uselogsumexp-{use_logsumexp}_huberdelta-{huber_delta}_fitgamma-{fit_gamma}_ctx-{ctx}.pkl"
    file = _find_file(
        save_dir=save_dir,
        use_logsumexp=use_logsumexp,
        huber_delta=huber_delta,
        fit_gamma=fit_gamma,
        context_length=context_length,
        prefix="",
    )
    with open(file, "rb") as f:
        grid_result = pickle.load(f)

    return grid_result


def combine_fit_grid_results_into_df(
    grid_result: dict[str, dict[str, dict[str, pd.DataFrame]]],
    topk: int = -1,
    sort_by_col: tuple[str] = ("optim_results", "loss"),
    ascending: bool = True,
) -> pd.DataFrame:
    """
    Combines all results into a single dataframe with a multi-index, for easy access to all results.

    If topk > 0, the topk rows are selected for each model_type, experiment_set, and token_param_range.

    grid_result structure: {model_type: {experiment_set: {token_param_range: df}}}
    """

    tuples_for_index = []
    dfs = []

    for model_type, model_type_dict in grid_result.items():
        for experiment_set, experiment_set_dict in model_type_dict.items():
            for token_param_range, df in experiment_set_dict.items():
                # add columns to df
                df["model_type"] = model_type
                df["experiment_set"] = experiment_set
                df["token_param_range"] = token_param_range

                df = df.sort_values(by=sort_by_col, ascending=ascending)
                if topk > 0:
                    df = df.head(topk)

                tuples_for_index.extend(
                    [(model_type, experiment_set, token_param_range)] * len(df)
                )
                dfs.append(df)

    # concatenate all dfs
    combined_df = pd.concat(dfs, axis=0)
    combined_df = combined_df.reset_index(drop=True)
    combined_df = combined_df.set_index(
        pd.MultiIndex.from_tuples(
            tuples_for_index,
            names=["model_type", "experiment_set", "token_param_range"],
        )
    )

    return combined_df


def load_combined_fit_grid_df(
    save_dir: Path,
    use_logsumexp: bool = True,
    huber_delta: float = 1e-3,
    fit_gamma: bool = True,
    context_length: int = 8192,
):
    """Load the combined fit grid dataframe from the given directory.
    Args:
        save_dir: The directory to load the results from.
        use_logsumexp: Whether to use logsumexp for the parametric loss.
        huber_delta: The huber delta to use for the parametric loss.
        fit_gamma: Whether to fit gamma for the parametric loss.
        context_length: The context length to use for the scaling law. Can be 2048, 8192 or 16384.
    """
    file = _find_file(
        save_dir=save_dir,
        use_logsumexp=use_logsumexp,
        huber_delta=huber_delta,
        fit_gamma=fit_gamma,
        context_length=context_length,
        prefix="combined_df__",
    )
    combined_df = pd.read_pickle(file)
    return combined_df
