"""This module contains metrics for evaluating the fit of a model to data."""

import numpy as np


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared value for a given set of true and predicted values.

    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The R-squared value.
    """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Squared Error (MSE) for a given set of true and predicted values.

    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The Mean Squared Error (MSE).
    """
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Root Mean Squared Error (RMSE) for a given set of true and predicted values.

    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The Root Mean Squared Error (RMSE).
    """
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Absolute Error (MAE) for a given set of true and predicted values.

    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.

    Returns:
        float: The Mean Absolute Error (MAE).
    """
    return np.mean(np.abs(y_true - y_pred))


metric_funcs = {
    "r_squared": r_squared,
    "mse": mse,
    "rmse": rmse,
    "mae": mae,
}


def calculate_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, metrics: list[str] | None = None
) -> dict[str, float]:
    """Calculate the specified metrics for a given set of true and predicted values.

    Args:
        y_true (np.ndarray): The true values.
        y_pred (np.ndarray): The predicted values.
        metrics (list[str]): A list of metric names to calculate.
            If None, all available metrics will be calculated.

    Returns:
        dict[str, float]: A dictionary containing the calculated metrics.
    """
    if metrics is None:
        metrics = list(metric_funcs.keys())

    results = {}
    for metric in metrics:
        if metric in metric_funcs:
            results[metric] = metric_funcs[metric](y_true, y_pred)
        else:
            raise ValueError(f"Metric '{metric}' is not supported.")
    return results
