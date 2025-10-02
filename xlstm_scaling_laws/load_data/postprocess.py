import numpy as np


def round_to_catgorical_vals(
    numerical_val: float,
    categorical_vals: np.ndarray,
    round_fraction: float = 0.1,
) -> str:
    """Thus function rounds a numerical value to the nearest categorical value.

    Typically applied to a column of numerical values in a DataFrame.

    Example:
    ```python

    isoflop_counts = [
        6e18, # 160M, 400M
        1e19,
        3e19, # 160M, 400M, 1.4B
        1e20, # (160M), 400M, 830M, 1.4B
        6e20, # 400M, 830M,
        3e21,
    ]
    isoflop_df.loc[:, "isoflop"] = isoflop_df["num_flops_training"].apply(
        lambda x: round_to_catgorical_vals(x, np.array(isoflop_counts), round_fraction=0.1)
    )
    ```

    Args:
        numerical_val: The numerical value to round to the nearest categorical value.
        categorical_vals: The categorical values to round to.
        round_fraction: The fraction of the categorical value to round to.

    Returns:
        The nearest categorical value to the numerical value.
    """
    diff = np.abs(categorical_vals - numerical_val)
    boundary_vals = categorical_vals * round_fraction

    within_boundary = diff < boundary_vals
    if np.all(~within_boundary):
        return "extra"

    selected_categorical_val = categorical_vals[np.argmin(np.abs(boundary_vals - diff))]
    return str(selected_categorical_val.item())


def bin_to_categorical_vals(
    numerical_val: float,
    categorical_vals: np.ndarray,
    categorical_val_names: np.ndarray = None,
) -> str:
    bin_idx = np.digitize(numerical_val, categorical_vals)

    # bin_idx = np.clip(bin_idx, 0, len(categorical_vals))
    if bin_idx >= len(categorical_vals):
        bin_idx = len(categorical_vals) - 1

    if categorical_val_names is None:
        return str(categorical_vals[bin_idx].item())
    else:
        return str(categorical_val_names[bin_idx].item())
