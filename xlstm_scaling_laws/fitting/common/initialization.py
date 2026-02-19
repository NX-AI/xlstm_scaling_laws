from dataclasses import dataclass

import numpy as np


@dataclass
class ParametricFitInitialization:
    """
    Class to hold the initialisation parameters for the parametric fit.
    """

    # Initialisation parameters
    params: dict[str, float] = None

    order: list[str] = None

    @property
    def x0(self) -> np.ndarray:
        """
        Returns the initialisation parameters as a numpy array.
        """
        if self.params is None:
            raise ValueError("Initialisation parameters not set.")
        return np.array([self.params[key] for key in self.order])


def generate_initialization_sweep(
    init_grid: dict[str, list[float]],
) -> list[ParametricFitInitialization]:
    """
    Generates a list of initialisation parameters for the parametric fit.
    """
    initialisations = []
    keys = list(init_grid.keys())
    values = [init_grid[key] for key in keys]

    # Generate all combinations of initialisation parameters
    for combination in np.array(np.meshgrid(*values)).T.reshape(-1, len(keys)):
        params = {key: value for key, value in zip(keys, combination)}
        initialisations.append(ParametricFitInitialization(params=params, order=keys))

    return initialisations


def x_to_param_dict(x: np.ndarray, order: list[str]) -> dict[str, float]:
    """
    Converts the (initialization) parameters from a numpy array to a dictionary.
    """
    return {key: value for key, value in zip(order, x)}


def param_dict_to_str(param_dict: dict[str, float]) -> str:
    """
    Converts the (initialization) parameters from a dictionary to a string.
    """
    return "_".join([f"{key}{value:.2f}" for key, value in param_dict.items()])
