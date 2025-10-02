from functools import partial
from typing import Callable, Literal, TypeAlias, Union

import numpy as np

CountType: TypeAlias = Union[float, int, np.ndarray]
CountArgs: TypeAlias = dict[str, CountType]
CountFn: TypeAlias = Callable[[CountArgs], CountType]


RuntimeModel: TypeAlias = Callable[[CountFn, CountFn, CountArgs, CountArgs], Callable]


def roofline_model(arithmetic_intensity, alpha, beta):
    """
    Calculate the roofline model for a given arithmetic intensity.

    Parameters:
    - alpha: FLOP/s of the accelerator
    - beta: Memory bandwidth in bytes/s of the accelerator
    - arithmetic_intensity: FLOPs/byte of the algorithm

    Returns:
    - Attainable FLOPs based on the roofline model.
    """
    return np.minimum(alpha, beta * arithmetic_intensity)


def roofline_model_logsumexp(
    arithmetic_intensity: float, alpha: float, beta: float, rho: float
) -> float:
    """
    Calculate the roofline model using the log-sum-exp trick.

    Parameters:
    - alpha: FLOP/s of the accelerator
    - beta: Memory bandwidth in bytes/s of the accelerator
    - arithmetic_intensity: FLOPs/byte of the algorithm
    - rho: A scaling factor for the model

    Returns:
    - Attainable FLOPs based on the roofline model using log-sum-exp.
    """
    max_c = -rho * np.where(
        alpha < beta * arithmetic_intensity,
        alpha,
        beta * arithmetic_intensity,
    )

    attainable_flops = -(1 / rho) * (
        max_c
        + np.log(
            0.5
            * (
                np.exp(-rho * alpha - max_c)
                + np.exp(-rho * beta * arithmetic_intensity - max_c)
            )
        )
    )

    return attainable_flops


def _softmin_stable(x1: float, x2: float, rho: float) -> float:
    max_c = -rho * np.minimum(x1, x2)
    softmin = -(1 / rho) * (
        max_c
        + np.log(
            0.5 * (np.exp(-rho * x1 - max_c) + np.exp(-rho * x2 - max_c))
        )
    )
    return softmin

def _softmax_stable(x1: float, x2: float, rho: float) -> float:
    max_c = rho * np.maximum(x1, x2)
    softmax = (1 / rho) * (
        max_c
        + np.log(
            0.5 * (np.exp(rho * x1 - max_c) + np.exp(rho * x2 - max_c))
        )
    )
    return softmax

def _get_runtime_model_arg_or_offset(count_args: CountArgs, key: str) -> CountType | None:
    """Get a count argument from the count_args dictionary."""
    if key not in count_args:
        return None
    
    offset = count_args.get(f"{key}_0", 0.0)
    if offset is None:
        return count_args[key]

    return count_args[key] + offset

def runtime_model_attainable_flops_logsumexp(
    fn_flops_algo: CountFn,
    fn_memops_algo: CountFn,
    count_args: CountArgs,
    runtime_model_args: CountArgs,
) -> CountType:
    assert "rho" in runtime_model_args, "Runtime model args must include 'rho'"
    assert "alpha" in runtime_model_args, "Runtime model args must include 'alpha'"
    assert "beta" in runtime_model_args, "Runtime model args must include 'beta'"

    rho = _get_runtime_model_arg_or_offset(runtime_model_args, "rho")
    # Accelerator FLOP speed: maximum FLOPs / second of the GPU
    alpha = _get_runtime_model_arg_or_offset(runtime_model_args, "alpha")
    # Accelerator memory bandwidth: maximum bytes / second of the GPU
    beta = _get_runtime_model_arg_or_offset(runtime_model_args, "beta")
    eps = _get_runtime_model_arg_or_offset(runtime_model_args, "eps")

    # Calculate the arithmetic intensity of the algorithm
    flops_algo = fn_flops_algo(**count_args)
    memops_algo = fn_memops_algo(**count_args)
    arithmetic_intensity = flops_algo / memops_algo

    # attainable_flops is the maximum FLOPs that can be achieved given the runtime model parameters
    attainable_flops = roofline_model_logsumexp(
        arithmetic_intensity=arithmetic_intensity,
        alpha=alpha,
        beta=beta,
        rho=rho,
    )

    runtime = flops_algo / attainable_flops + eps

    return {
        "runtime": runtime,
        "arithmetic_intensity": arithmetic_intensity,
        "attainable_flops": attainable_flops,
        "flops_algo": flops_algo,
        "memops_algo": memops_algo,
    }

def runtime_model_attainable_flops_min(
    fn_flops_algo: CountFn,
    fn_memops_algo: CountFn,
    count_args: CountArgs,
    runtime_model_args: CountArgs,
) -> CountType:
    assert "alpha" in runtime_model_args, "Runtime model args must include 'alpha'"
    assert "beta" in runtime_model_args, "Runtime model args must include 'beta'"

    # Accelerator FLOP speed: maximum FLOPs / second of the GPU
    alpha = _get_runtime_model_arg_or_offset(runtime_model_args, "alpha")
    # Accelerator memory bandwidth: maximum bytes / second of the GPU
    beta = _get_runtime_model_arg_or_offset(runtime_model_args, "beta")
    eps = _get_runtime_model_arg_or_offset(runtime_model_args, "eps")

    # Calculate the arithmetic intensity of the algorithm
    flops_algo = fn_flops_algo(**count_args)
    memops_algo = fn_memops_algo(**count_args)
    arithmetic_intensity = flops_algo / memops_algo

    # attainable_flops is the maximum FLOPs that can be achieved given the runtime model parameters
    attainable_flops = roofline_model(
        arithmetic_intensity=arithmetic_intensity,
        alpha=alpha,
        beta=beta,
    )

    runtime = flops_algo / attainable_flops + eps

    return {
        "runtime": runtime,
        "arithmetic_intensity": arithmetic_intensity,
        "attainable_flops": attainable_flops,
        "flops_algo": flops_algo,
        "memops_algo": memops_algo,
    }

def _runtime_model_linear_flops_memops_generic(
    fn_flops_algo: CountFn,
    fn_memops_algo: CountFn,
    count_args: CountArgs,
    runtime_model_args: CountArgs,
    mode: Literal["flops", "memops", "max_flops_memops", "sum_flops_memops"] = "max_flops_memops",
) -> CountType:
    # Accelerator FLOP speed: maximum FLOPs / second of the GPU
    alpha = _get_runtime_model_arg_or_offset(runtime_model_args, "alpha")
    # Accelerator memory bandwidth: maximum bytes / second of the GPU
    beta = _get_runtime_model_arg_or_offset(runtime_model_args, "beta")
    eps = _get_runtime_model_arg_or_offset(runtime_model_args, "eps")
    eps_bp = _get_runtime_model_arg_or_offset(runtime_model_args, "eps_bp")
    # Calculate the arithmetic intensity of the algorithm
    flops_algo = fn_flops_algo(**count_args)
    memops_algo = fn_memops_algo(**count_args)
    arithmetic_intensity = flops_algo / memops_algo

    # constant and batch size / prefill dependent overhead
    # overhead = eps_bp * count_args["batch_size"] * count_args["seq_len"] + eps
    overhead = eps 

    if mode == "flops":
        runtime = flops_algo / alpha + overhead
    elif mode == "memops":
        runtime = memops_algo / beta + overhead + count_args["batch_size"] * eps_bp
    elif mode == "max_flops_memops":
        runtime = np.maximum(flops_algo / alpha, memops_algo / beta) + overhead
    elif mode == "sum_flops_memops":
        runtime = flops_algo / alpha + memops_algo / beta + overhead
    elif mode == "smooth_max_flops_memops":
        rho = _get_runtime_model_arg_or_offset(runtime_model_args, "rho")
        runtime = _softmax_stable(
            x1=flops_algo / alpha,
            x2=memops_algo / beta,
            rho=rho,
        ) + overhead
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "runtime": runtime,
        "arithmetic_intensity": arithmetic_intensity,
        "flops_algo": flops_algo,
        "memops_algo": memops_algo,
    }


# for log space models alpha and beta are in log space

def _runtime_model_log_linear_flops_memops_generic(
    fn_flops_algo: CountFn,
    fn_memops_algo: CountFn,
    count_args: CountArgs,
    runtime_model_args: CountArgs,
    mode: Literal["flops", "memops", "max_flops_memops", "sum_flops_memops"] = "max_flops_memops",
) -> CountType:
    # Accelerator FLOP speed: maximum FLOPs / second of the GPU
    alpha = _get_runtime_model_arg_or_offset(runtime_model_args, "alpha")
    # Accelerator memory bandwidth: maximum bytes / second of the GPU
    beta = _get_runtime_model_arg_or_offset(runtime_model_args, "beta")
    eps = _get_runtime_model_arg_or_offset(runtime_model_args, "eps")
    eps_bp = _get_runtime_model_arg_or_offset(runtime_model_args, "eps_bp")

    # Calculate the arithmetic intensity of the algorithm
    flops_algo = fn_flops_algo(**count_args)
    memops_algo = fn_memops_algo(**count_args)
    arithmetic_intensity = flops_algo / memops_algo

    # constant and batch size / prefill dependent overhead
    # overhead = eps_bp * count_args["batch_size"] * count_args["seq_len"] + eps
    overhead = eps
    
    if mode == "flops":
        runtime_log = np.log(flops_algo / alpha + overhead)
    elif mode == "memops":
        runtime_log = np.log(memops_algo / beta + overhead + count_args["batch_size"] * eps_bp)
    elif mode == "max_flops_memops":
        runtime_log = np.log(np.maximum(flops_algo / alpha, memops_algo / beta) + overhead)
    elif mode == "sum_flops_memops":
        runtime_log = np.log(flops_algo / alpha + memops_algo / beta + overhead)
    elif mode == "smooth_max_flops_memops":
        rho = _get_runtime_model_arg_or_offset(runtime_model_args, "rho")
        runtime_log = np.log(
            _softmax_stable(
                x1=flops_algo / alpha,
                x2=memops_algo / beta,
                rho=rho,
            ) + overhead
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "runtime": runtime_log,
        "arithmetic_intensity": arithmetic_intensity,
        "flops_algo": flops_algo,
        "memops_algo": memops_algo,
    }

def runtime_model_log_attainable_flops_min(
    fn_flops_algo: CountFn,
    fn_memops_algo: CountFn,
    count_args: CountArgs,
    runtime_model_args: CountArgs,
) -> CountType:
    assert "alpha" in runtime_model_args, "Runtime model args must include 'alpha'"
    assert "beta" in runtime_model_args, "Runtime model args must include 'beta'"
    assert "eps" in runtime_model_args, "Runtime model args must include 'eps'"

    # Accelerator FLOP speed: maximum FLOPs / second of the GPU
    alpha = _get_runtime_model_arg_or_offset(runtime_model_args, "alpha")
    # Accelerator memory bandwidth: maximum bytes / second of the GPU
    beta = _get_runtime_model_arg_or_offset(runtime_model_args, "beta")
    eps = _get_runtime_model_arg_or_offset(runtime_model_args, "eps")

    # Calculate the arithmetic intensity of the algorithm
    flops_algo = fn_flops_algo(**count_args)
    memops_algo = fn_memops_algo(**count_args)
    arithmetic_intensity = flops_algo / memops_algo

    # attainable_flops is the maximum FLOPs that can be achieved given the runtime model parameters
    attainable_flops_log = np.minimum(alpha, beta + np.log(flops_algo) - np.log(memops_algo))

    runtime_log = np.log(flops_algo) - attainable_flops_log

    return {
        "runtime": runtime_log,
        "arithmetic_intensity": arithmetic_intensity,
        "attainable_flops": attainable_flops_log,
        "flops_algo": flops_algo,
    }

def runtime_model_log_attainable_flops_logsumexp(
    fn_flops_algo: CountFn,
    fn_memops_algo: CountFn,
    count_args: CountArgs,
    runtime_model_args: CountArgs,
) -> CountType:
    assert "rho" in runtime_model_args, "Runtime model args must include 'rho'"
    assert "alpha" in runtime_model_args, "Runtime model args must include 'alpha'"
    assert "beta" in runtime_model_args, "Runtime model args must include 'beta'"

    rho = _get_runtime_model_arg_or_offset(runtime_model_args, "rho")
    # Accelerator FLOP speed: maximum FLOPs / second of the GPU
    alpha = _get_runtime_model_arg_or_offset(runtime_model_args, "alpha")
    # Accelerator memory bandwidth: maximum bytes / second of the GPU
    beta = _get_runtime_model_arg_or_offset(runtime_model_args, "beta")

    # Calculate the arithmetic intensity of the algorithm
    flops_algo = fn_flops_algo(**count_args)
    memops_algo = fn_memops_algo(**count_args)
    arithmetic_intensity = flops_algo / memops_algo

    # attainable_flops is the maximum FLOPs that can be achieved given the runtime model parameters
    attainable_flops_log = _softmin_stable(
        x1=alpha,
        x2=beta + np.log(flops_algo) - np.log(memops_algo),
        rho=rho,
    )

    runtime_log = np.log(flops_algo) - attainable_flops_log

    return {
        "runtime": runtime_log,
        "arithmetic_intensity": arithmetic_intensity,
        "attainable_flops": attainable_flops_log,
        "flops_algo": flops_algo,
        "memops_algo": memops_algo,
    }

_runtime_model_registry: dict[str, Callable] = {
    "attainable_flops_logsumexp": runtime_model_attainable_flops_logsumexp,
    "attainable_flops_min": runtime_model_attainable_flops_min,
    "linear_max_flops_memops": partial(
        _runtime_model_linear_flops_memops_generic,
        mode="max_flops_memops",),
    "linear_flops": partial(
        _runtime_model_linear_flops_memops_generic,
        mode="flops",),
    "linear_memops": partial(
        _runtime_model_linear_flops_memops_generic,
        mode="memops",),
    "linear_sum_flops_memops": partial(
        _runtime_model_linear_flops_memops_generic,
        mode="sum_flops_memops",),
    "smooth_max_flops_memops": partial(
        _runtime_model_linear_flops_memops_generic,
        mode="smooth_max_flops_memops",),
    "log_attainable_flops_min": runtime_model_log_attainable_flops_min,
    "log_attainable_flops_logsumexp": runtime_model_log_attainable_flops_logsumexp,
    "log_linear_max_flops_memops": partial(
        _runtime_model_log_linear_flops_memops_generic,
        mode="max_flops_memops",),
    "log_linear_flops": partial(
        _runtime_model_log_linear_flops_memops_generic,
        mode="flops",),
    "log_linear_memops": partial(
        _runtime_model_log_linear_flops_memops_generic,
        mode="memops",),
    "log_linear_sum_flops_memops": partial(
        _runtime_model_log_linear_flops_memops_generic,
        mode="sum_flops_memops",),
    "log_smooth_max_flops_memops": partial(
        _runtime_model_log_linear_flops_memops_generic,
        mode="smooth_max_flops_memops",),
}


def get_runtime_model(name: str) -> RuntimeModel:
    """
    Get a runtime model by name.
    """
    if name not in _runtime_model_registry:
        raise ValueError(f"Runtime model '{name}' is not registered.")
    return _runtime_model_registry[name]
