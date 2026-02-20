import numpy as np
import pandas as pd

from xlstm_scaling_laws.analysis.isoflop.data import (
    get_isoflop_polyfits_for_all_ctx,
    get_isoflop_polyfits_for_ctx,
)
from xlstm_scaling_laws.fitting.fit_power_law import generate_power_law_fit


def _fit_powerlaw_params_for_plot(
    polyfit_df: pd.DataFrame,
    plot_type: str,
    source_plot: str,
    context_length: int | None,
    flop_range_for_powerlaw_fit: tuple[float, float],
    fit_in_log_space: bool,
) -> pd.DataFrame:
    fit_df = generate_power_law_fit(
        flop_to_nparam_ntok_df=polyfit_df,
        x_col="flops_mean",
        y_col="x_opt",
        model_type_col="model_type",
        fit_in_log_space=fit_in_log_space,
        select_flop_range=flop_range_for_powerlaw_fit,
    )

    fit_df = fit_df.copy()
    fit_df["source_plot"] = source_plot
    fit_df["plot_type"] = plot_type
    fit_df["context_length"] = context_length
    fit_df["fit_in_log_space"] = fit_in_log_space
    fit_df["flop_fit_min"] = flop_range_for_powerlaw_fit[0]
    fit_df["flop_fit_max"] = flop_range_for_powerlaw_fit[1]
    fit_df["equation"] = "x_opt = A * FLOPs^alpha"

    fit_mask = (polyfit_df["flops_mean"] >= flop_range_for_powerlaw_fit[0]) & (
        polyfit_df["flops_mean"] <= flop_range_for_powerlaw_fit[1]
    )
    fit_counts = (
        polyfit_df.loc[fit_mask]
        .groupby("model_type")
        .size()
        .rename("num_points_fit")
        .reset_index()
    )
    fit_df = fit_df.merge(fit_counts, on="model_type", how="left")

    return fit_df


def create_scaling_law_parameters_dataframe(
    y_col: str = "val/.dclm_loss",
    fit_in_log_space: bool = True,
) -> pd.DataFrame:
    """
    Collect all power-law fit parameters shown in this notebook's plots.

    Includes:
    - The first two plots (`get_isoflop_powerlaw_plot`) for context length 8192
      with fit range [5e18, 2e20]
    - The combined per-context power-law plot in `get_isoflop_powerlaw_ctx_plot`
      for contexts 2048/8192/16384 with fit/filter range [5e18, 5e19]
    """

    results: list[pd.DataFrame] = []

    first_two_fit_range = (5e18, 2e20)
    for plot_type in ["num_params", "num_tokens_training"]:
        polyfit_8192_df = get_isoflop_polyfits_for_ctx(
            context_length=8192,
            x_col=plot_type,
            y_col=y_col,
        )
        results.append(
            _fit_powerlaw_params_for_plot(
                polyfit_df=polyfit_8192_df,
                plot_type=plot_type,
                source_plot="get_isoflop_powerlaw_plot",
                context_length=8192,
                flop_range_for_powerlaw_fit=first_two_fit_range,
                fit_in_log_space=fit_in_log_space,
            )
        )

    ctx_fit_range = (5e18, 5e19)
    for plot_type in ["num_params", "num_tokens_training"]:
        all_ctx_polyfits_df = get_isoflop_polyfits_for_all_ctx(
            x_col=plot_type,
            y_col=y_col,
        )
        for context_length in sorted(all_ctx_polyfits_df["context_length"].unique()):
            per_ctx_df = all_ctx_polyfits_df[
                all_ctx_polyfits_df["context_length"] == context_length
            ]
            per_ctx_df = per_ctx_df[
                (per_ctx_df["flops_mean"] >= ctx_fit_range[0])
                & (per_ctx_df["flops_mean"] <= ctx_fit_range[1])
            ]

            results.append(
                _fit_powerlaw_params_for_plot(
                    polyfit_df=per_ctx_df,
                    plot_type=plot_type,
                    source_plot="get_isoflop_powerlaw_ctx_plot",
                    context_length=int(context_length),
                    flop_range_for_powerlaw_fit=ctx_fit_range,
                    fit_in_log_space=fit_in_log_space,
                )
            )

    scaling_law_params_df = pd.concat(results, ignore_index=True)
    scaling_law_params_df = scaling_law_params_df[
        [
            "source_plot",
            "plot_type",
            "context_length",
            "model_type",
            "a",
            "alpha",
            "a_std",
            "alpha_std",
            "num_points_fit",
            "fit_in_log_space",
            "flop_fit_min",
            "flop_fit_max",
            "equation",
        ]
    ].sort_values(by=["source_plot", "plot_type", "context_length", "model_type"])

    return scaling_law_params_df.reset_index(drop=True)


def estimate_optimal_flops_tokens_table(
    model_params: list[float] | np.ndarray,
    model_type: str,
    scaling_params_df: pd.DataFrame | None = None,
    validation_rtol: float = 1e-10,
) -> pd.DataFrame:
    """
    Estimate compute-optimal training FLOPs and tokens for a set of model sizes.

    The returned table has:
    - rows: `num_params`
    - columns: MultiIndex with levels (outer -> inner)
        1) `source_plot`
        2) `context_length`
        3) `plot_type`

    Value semantics by inner-most `plot_type`:
    - `num_flops_training`: estimated optimal training FLOPs
    - `num_tokens_training`: estimated optimal training tokens
    """

    if scaling_params_df is None:
        scaling_params_df = create_scaling_law_parameters_dataframe()

    required_cols = {
        "source_plot",
        "plot_type",
        "context_length",
        "model_type",
        "a",
        "alpha",
    }
    missing_cols = required_cols - set(scaling_params_df.columns)
    if missing_cols:
        raise ValueError(
            f"scaling_params_df is missing required columns: {sorted(missing_cols)}"
        )

    model_type_map = {
        "mlstm": "mlstm_v1",
        "mlstm_v1": "mlstm_v1",
        "llama": "llama",
    }
    if model_type not in model_type_map:
        raise ValueError("model_type must be one of: 'llama', 'mlstm', 'mlstm_v1'.")
    resolved_model_type = model_type_map[model_type]

    model_df = scaling_params_df[
        scaling_params_df["model_type"] == resolved_model_type
    ].copy()
    if model_df.empty:
        raise ValueError(
            f"No scaling-law parameters found for model_type={model_type}."
        )

    params_law_df = model_df[model_df["plot_type"] == "num_params"][
        ["source_plot", "context_length", "a", "alpha"]
    ].rename(columns={"a": "a_params", "alpha": "alpha_params"})

    tokens_law_df = model_df[model_df["plot_type"] == "num_tokens_training"][
        ["source_plot", "context_length", "a", "alpha"]
    ].rename(columns={"a": "a_tokens", "alpha": "alpha_tokens"})

    if params_law_df.empty or tokens_law_df.empty:
        raise ValueError(
            "Expected both num_params and num_tokens_training scaling laws for the selected model type."
        )

    # Ensure a unique law per (source_plot, context_length)
    for law_name, law_df in [
        ("num_params", params_law_df),
        ("num_tokens_training", tokens_law_df),
    ]:
        duplicates = law_df.duplicated(
            subset=["source_plot", "context_length"], keep=False
        )
        if duplicates.any():
            dup_rows = law_df.loc[duplicates, ["source_plot", "context_length"]]
            raise ValueError(
                f"Found duplicate {law_name} laws for some (source_plot, context_length): "
                f"{dup_rows.drop_duplicates().to_dict(orient='records')}"
            )

    laws_df = params_law_df.merge(
        tokens_law_df,
        on=["source_plot", "context_length"],
        how="inner",
    )
    if laws_df.empty:
        raise ValueError(
            "Could not match num_params and num_tokens_training laws for the selected model type."
        )

    numeric_cols = ["a_params", "alpha_params", "a_tokens", "alpha_tokens"]
    if not np.isfinite(laws_df[numeric_cols].to_numpy()).all():
        raise ValueError("Found non-finite scaling-law coefficients.")
    if (laws_df[["a_params", "a_tokens"]] <= 0).any().any():
        raise ValueError("Scaling-law prefactors must be > 0.")
    if (laws_df[["alpha_params", "alpha_tokens"]] == 0).any().any():
        raise ValueError("Scaling-law exponents must be non-zero.")

    rows = []
    for num_params in model_params:
        num_params = float(num_params)
        if not np.isfinite(num_params) or num_params <= 0:
            raise ValueError("All model_params values must be finite and > 0.")

        for _, law in laws_df.iterrows():
            est_flops = (num_params / law["a_params"]) ** (1.0 / law["alpha_params"])
            est_tokens = law["a_tokens"] * (est_flops ** law["alpha_tokens"])

            if (not np.isfinite(est_flops)) or est_flops <= 0:
                raise ValueError("Estimated FLOPs is non-finite or non-positive.")
            if (not np.isfinite(est_tokens)) or est_tokens <= 0:
                raise ValueError("Estimated tokens is non-finite or non-positive.")

            # Self-consistency check: plugging inferred FLOPs back into params law
            # must reconstruct the original num_params (up to numerical tolerance).
            reconstructed_params = law["a_params"] * (est_flops ** law["alpha_params"])
            rel_err = abs(reconstructed_params - num_params) / num_params
            if rel_err > validation_rtol:
                raise ValueError(
                    "Scaling-law inversion failed consistency check with "
                    f"relative error {rel_err:.3e} (> {validation_rtol:.1e})."
                )

            rows.append(
                {
                    "num_params": num_params,
                    "source_plot": law["source_plot"],
                    "context_length": int(law["context_length"]),
                    "plot_type": "num_flops_training",
                    "estimate": est_flops,
                }
            )
            rows.append(
                {
                    "num_params": num_params,
                    "source_plot": law["source_plot"],
                    "context_length": int(law["context_length"]),
                    "plot_type": "num_tokens_training",
                    "estimate": est_tokens,
                }
            )

    long_df = pd.DataFrame(rows)

    wide_df = long_df.pivot_table(
        index="num_params",
        columns=["source_plot", "context_length", "plot_type"],
        values="estimate",
        aggfunc="first",
    ).sort_index(axis=1)

    wide_df.columns = wide_df.columns.set_names(
        ["source_plot", "context_length", "plot_type"]
    )
    wide_df.index.name = "num_params"

    return wide_df


def _format_short_scale(
    value: float,
    suffixes: tuple[str, ...],
    decimals: int = 2,
) -> str:
    if pd.isna(value):
        return ""

    sign = "-" if value < 0 else ""
    abs_value = float(abs(value))

    if abs_value == 0:
        return "0"

    idx = 0
    while abs_value >= 1000 and idx < len(suffixes) - 1:
        abs_value /= 1000.0
        idx += 1

    if abs_value >= 100:
        value_str = f"{abs_value:.0f}"
    elif abs_value >= 10:
        value_str = f"{abs_value:.1f}"
    else:
        value_str = f"{abs_value:.{decimals}f}"

    if "." in value_str:
        value_str = value_str.rstrip("0").rstrip(".")
    return f"{sign}{value_str}{suffixes[idx]}"


def _format_scientific_lower(value: float, decimals: int = 2) -> str:
    if pd.isna(value):
        return ""
    if value == 0:
        return "0"

    sci = f"{float(value):.{decimals}e}"
    mantissa, exponent = sci.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exponent_int = int(exponent)
    return f"{mantissa}e{exponent_int}"


def format_estimated_table_with_fit_params(
    estimates_df: pd.DataFrame,
    scaling_params_df: pd.DataFrame,
    model_type: str,
    alpha_decimals: int = 4,
    coeff_decimals: int = 4,
) -> pd.DataFrame:
    """
    enrich each plot_type header with the corresponding power-law parameters (A, alpha).

    No extra columns are added and std columns are not used.
    """

    if (
        not isinstance(estimates_df.columns, pd.MultiIndex)
        or estimates_df.columns.nlevels != 3
    ):
        raise ValueError(
            "estimates_df must have a 3-level MultiIndex columns: "
            "(source_plot, context_length, plot_type)."
        )

    model_type_map = {
        "mlstm": "mlstm_v1",
        "mlstm_v1": "mlstm_v1",
        "llama": "llama",
    }
    if model_type not in model_type_map:
        raise ValueError("model_type must be one of: 'llama', 'mlstm', 'mlstm_v1'.")
    resolved_model_type = model_type_map[model_type]

    laws = scaling_params_df[
        scaling_params_df["model_type"] == resolved_model_type
    ].copy()
    if laws.empty:
        raise ValueError(f"No scaling law rows found for model_type={model_type}.")

    required_cols = ["source_plot", "context_length", "plot_type", "a", "alpha"]
    missing = set(required_cols) - set(laws.columns)
    if missing:
        raise ValueError(
            f"scaling_params_df is missing required columns: {sorted(missing)}"
        )

    formatted_df = estimates_df.copy()

    formatted_df.index = [
        _format_short_scale(v, suffixes=("", "K", "M", "B", "T"), decimals=2)
        for v in formatted_df.index
    ]
    formatted_df.index.name = estimates_df.index.name

    new_columns = []
    for source_plot, context_length, estimate_plot_type in formatted_df.columns:
        if estimate_plot_type == "num_flops_training":
            law_plot_type = "num_params"
        elif estimate_plot_type == "num_tokens_training":
            law_plot_type = "num_tokens_training"
        else:
            raise ValueError(f"Unknown estimate plot_type: {estimate_plot_type}")

        law_match = laws[
            (laws["source_plot"] == source_plot)
            & (laws["context_length"].astype(int) == int(context_length))
            & (laws["plot_type"] == law_plot_type)
        ]
        if len(law_match) != 1:
            raise ValueError(
                "Expected exactly one scaling-law row for "
                f"(source_plot={source_plot}, context_length={context_length}, plot_type={law_plot_type}), "
                f"found {len(law_match)}."
            )

        law_row = law_match.iloc[0]
        a_str = f"{float(law_row['a']):.{coeff_decimals}f}"
        alpha_str = f"{float(law_row['alpha']):.{alpha_decimals}f}"

        plot_type_with_params = f"{estimate_plot_type} [A={a_str}, alpha={alpha_str}]"
        new_columns.append((source_plot, int(context_length), plot_type_with_params))

    formatted_df.columns = pd.MultiIndex.from_tuples(
        new_columns,
        names=estimates_df.columns.names,
    )

    for col in formatted_df.columns:
        if col[2].startswith("num_flops_training"):
            formatted_df[col] = formatted_df[col].map(
                lambda x: _format_scientific_lower(x, decimals=2)
            )
        elif col[2].startswith("num_tokens_training"):
            formatted_df[col] = formatted_df[col].map(
                lambda x: _format_short_scale(
                    x, suffixes=("", "K", "M", "B", "T"), decimals=2
                )
            )

    return formatted_df


def _parse_param_index_to_float(index: pd.Index) -> pd.Series | None:
    suffix_multipliers = {
        "K": 1e3,
        "M": 1e6,
        "B": 1e9,
        "G": 1e9,
        "T": 1e12,
    }

    parsed_values = []
    for value in index:
        if isinstance(value, (int, float, np.integer, np.floating)):
            parsed_values.append(float(value))
            continue

        value_str = str(value).strip().upper().replace(",", "")
        if not value_str:
            return None

        suffix = value_str[-1]
        if suffix in suffix_multipliers:
            number_part = value_str[:-1]
            try:
                parsed_values.append(float(number_part) * suffix_multipliers[suffix])
            except ValueError:
                return None
        else:
            try:
                parsed_values.append(float(value_str))
            except ValueError:
                return None

    return pd.Series(parsed_values, index=index, dtype=float)


def _to_float_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    suffix_multipliers = {
        "K": 1e3,
        "M": 1e6,
        "B": 1e9,
        "G": 1e9,
        "T": 1e12,
    }

    def parse_value(value: object) -> float:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)

        value_str = str(value).strip().upper().replace(",", "")
        if not value_str:
            raise ValueError("Empty value encountered while parsing numeric series.")

        suffix = value_str[-1]
        if suffix in suffix_multipliers:
            return float(value_str[:-1]) * suffix_multipliers[suffix]
        return float(value_str)

    return series.map(parse_value).astype(float)


def _is_num_tokens_plot_type(plot_type: object) -> bool:
    return str(plot_type).startswith("num_tokens_training")


def add_token_param_ratio_plot_type(estimates_df: pd.DataFrame) -> pd.DataFrame:
    """Add a `token_param_ratio` plot_type computed as num_tokens_training / num_params."""
    ratio_columns = {}
    index_num_params = _parse_param_index_to_float(estimates_df.index)

    for source_plot, context_length, plot_type in estimates_df.columns:
        if not _is_num_tokens_plot_type(plot_type):
            continue

        tokens_series = _to_float_series(
            estimates_df[(source_plot, context_length, plot_type)]
        )
        num_params_key = (source_plot, context_length, "num_params")
        if num_params_key in estimates_df.columns:
            denom_num_params = _to_float_series(estimates_df[num_params_key])
        elif index_num_params is not None:
            denom_num_params = index_num_params
        else:
            raise ValueError(
                "Cannot infer num_params denominator: no 'num_params' column and non-numeric index."
            )

        ratio_col = (source_plot, context_length, "token_param_ratio")
        ratio_columns[ratio_col] = tokens_series / denom_num_params

    if not ratio_columns:
        raise ValueError(
            "No 'num_tokens_training' columns found; cannot compute token_param_ratio."
        )

    ratio_df = pd.DataFrame(ratio_columns, index=estimates_df.index)
    ratio_df.columns = pd.MultiIndex.from_tuples(ratio_df.columns)

    return pd.concat([estimates_df, ratio_df], axis=1).sort_index(axis=1)


def format_token_param_ratio_for_display(
    estimates_df: pd.DataFrame, decimals: int = 1
) -> pd.DataFrame:
    """Round only token_param_ratio columns for cleaner notebook display."""
    formatted_df = estimates_df.copy()
    ratio_columns = [
        col
        for col in formatted_df.columns
        if len(col) == 3 and col[2] == "token_param_ratio"
    ]
    if ratio_columns:
        formatted_df.loc[:, ratio_columns] = formatted_df.loc[:, ratio_columns].round(
            decimals
        )
    return formatted_df
