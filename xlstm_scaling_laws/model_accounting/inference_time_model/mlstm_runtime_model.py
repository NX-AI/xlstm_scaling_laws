import numpy as np
import pandas as pd

from ..flops.mlstm import (
    count_mlstm_model_flops__chunkwise_parallel,
    count_mlstm_model_flops__recurrent,
)
from ..memops.mlstm import (
    count_mlstm_model_memops__chunkwise_parallel,
    count_mlstm_model_memops__recurrent,
)
from .generic import CountType, get_runtime_model


def runtime_model_mlstm_ttft(
    # input args
    batchsize: CountType,
    prefill: CountType,
    # model args
    embedding_dim: int,
    num_blocks: int,
    num_heads: int,
    qk_dim_factor: float,
    vocab_size: int,
    chunk_size: int,
    ffn_dim: int,
    # algorithm args
    factor_causal: float,
    bytes_act: int,
    bytes_Cmn: int,
    bytes_w: int,
    # runtime model args
    rho: float,
    alpha: float,
    beta: float,
    eps: float,
    eps_bp: float,
    rho_0: float | None = None,
    alpha_0: float | None = None,
    beta_0: float | None = None,
    eps_0: float | None = None,
    eps_bp_0: float | None = None,
    # runtime model mode
    mode: str = "attainable_flops_logsumexp",
) -> dict:
    """Predict the time to first token (TTFT) for a given mLSTM model configuration in seconds."""
    runtime_model = get_runtime_model(mode)

    count_args = {}
    input_args = {
        "batch_size": batchsize,
        "seq_len": prefill,
    }

    d_hv = embedding_dim // num_heads
    d_qk = int(qk_dim_factor * d_hv)
    model_args = {
        "n_vocab": vocab_size,
        "n_blocks": num_blocks,
        "d_ff": ffn_dim,
        "n_headq": num_heads,
        "d_qk": d_qk,
        "d_hv": d_hv,
        "chunk_size": chunk_size,
        "with_unembed": False,  # we do not compute the unembedding in the TTFT model (only for the last token)
    }

    algo_args = {
        "factor_causal": factor_causal,
        "bytes_act": bytes_act,
        "bytes_Cmn": bytes_Cmn,
        "bytes_w": bytes_w,
    }

    count_args.update(input_args)
    count_args.update(model_args)
    count_args.update(algo_args)

    runtime_model_args = {
        "rho": rho,
        "alpha": alpha,
        "beta": beta,
        "eps": eps,
        "eps_bp": eps_bp,
        "rho_0": rho_0,
        "alpha_0": alpha_0,
        "beta_0": beta_0,
        "eps_0": eps_0,
        "eps_bp_0": eps_bp_0,
    }
    ret = runtime_model(
        fn_flops_algo=count_mlstm_model_flops__chunkwise_parallel,
        fn_memops_algo=count_mlstm_model_memops__chunkwise_parallel,
        count_args=count_args,
        runtime_model_args=runtime_model_args,
    )
    return ret


def predict_runtime_mlstm_ttft(
    ttft_df: pd.DataFrame,
    # runtime model args
    rho: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    eps: float | None = None,
    eps_bp: float | None = None,
    rho_0: float | None = None,
    alpha_0: float | None = None,
    beta_0: float | None = None,
    eps_0: float | None = None,
    eps_bp_0: float | None = None,
    # runtime model mode
    mode: str = "attainable_flops_logsumexp",
    add_suffix_to_col: bool = False,
    # algorithm args
    factor_causal: float = 0.75,
    bytes_act: int = 2,  # bfloat16
    bytes_Cmn: int = 4,  # float32
    bytes_w: int = 2,  # bfloat16
) -> pd.DataFrame:
    """Predict the time to first token (TTFT) for mLSTM model configurations in milliseconds."""

    if add_suffix_to_col:
        smode = "".join([w[0] for w in mode.split("_")])
        suffix = f"_r{rho}-a{alpha:.3e}-b{beta:.3e}-{smode}"
    else:
        suffix = ""

    def _apply_runtime_model(row: pd.Series) -> float:
        """Apply the runtime model to a single row of the DataFrame."""

        ret = runtime_model_mlstm_ttft(
            batchsize=row[("input_params", "batchsize")],
            prefill=row[("input_params", "prefill")],
            embedding_dim=row[("model_params", "embedding_dim")],
            num_blocks=row[("model_params", "num_blocks")],
            num_heads=row[("model_params", "num_heads")],
            qk_dim_factor=row[("model_params", "qk_dim_factor")],
            vocab_size=row[("model_params", "vocab_size")],
            chunk_size=row[("model_params", "chunk_size")],
            ffn_dim=row[("model_params", "ffn_dim")],
            factor_causal=factor_causal,
            bytes_act=bytes_act,
            bytes_Cmn=bytes_Cmn,
            bytes_w=bytes_w,
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
            # runtime model mode
            mode=mode,
        )
        # convert to milliseconds
        if "log_" in mode:
            predicted_runtime_msecs = ret.pop("runtime") + np.log(np.array(1000.0))
        else:
            predicted_runtime_msecs = ret.pop("runtime") * np.array(1000.0)
        row[(f"pred{suffix}", "runtime")] = predicted_runtime_msecs
        for r in ret:
            row[(f"pred{suffix}", r)] = ret[r]
        return row

    pred_ttft_df = ttft_df.apply(_apply_runtime_model, axis=1)

    return pred_ttft_df


def runtime_model_mlstm_step_time(
    # input args
    batchsize: CountType,
    # model args
    embedding_dim: int,
    num_blocks: int,
    num_heads: int,
    qk_dim_factor: float,
    vocab_size: int,
    chunk_size: int,
    ffn_dim: int,
    # algorithm args
    factor_causal: float,
    bytes_act: int,
    bytes_Cmn: int,
    bytes_w: int,
    # runtime model args
    rho: float,
    alpha: float,
    beta: float,
    eps: float,
    eps_bp: float,
    rho_0: float | None = None,
    alpha_0: float | None = None,
    beta_0: float | None = None,
    eps_0: float | None = None,
    eps_bp_0: float | None = None,
    # runtime model mode
    mode: str = "attainable_flops_logsumexp",
) -> dict:
    """Predict the average step time for a given mLSTM model configuration in seconds.
    We use the assumption that the step time is independent of the prefill length.
    """
    runtime_model = get_runtime_model(mode)

    count_args = {}
    input_args = {
        "batch_size": batchsize,
        "seq_len": 1,  # we predict the step time for a single token (prefill independent)
    }

    d_hv = embedding_dim // num_heads
    d_qk = int(qk_dim_factor * d_hv)
    model_args = {
        "n_vocab": vocab_size,
        "n_blocks": num_blocks,
        "d_ff": ffn_dim,
        "n_headq": num_heads,
        "d_qk": d_qk,
        "d_hv": d_hv,
        "chunk_size": chunk_size,
        # "with_unembed": True,  # we compute the unembedding in the step time model
    }

    algo_args = {
        "factor_causal": factor_causal,
        "bytes_act": bytes_act,
        "bytes_Cmn": bytes_Cmn,
        "bytes_w": bytes_w,
    }

    count_args.update(input_args)
    count_args.update(model_args)
    count_args.update(algo_args)

    runtime_model_args = {
        "rho": rho,
        "alpha": alpha,
        "beta": beta,
        "eps": eps,
        "eps_bp": eps_bp,
        "rho_0": rho_0,
        "alpha_0": alpha_0,
        "beta_0": beta_0,
        "eps_0": eps_0,
        "eps_bp_0": eps_bp_0,
    }
    ret = runtime_model(
        fn_flops_algo=count_mlstm_model_flops__recurrent,
        fn_memops_algo=count_mlstm_model_memops__recurrent,
        count_args=count_args,
        runtime_model_args=runtime_model_args,
    )
    return ret


def predict_runtime_mlstm_step_time(
    step_time_df: pd.DataFrame,
    # runtime model args
    rho: float | None = None,
    alpha: float | None = None,
    beta: float | None = None,
    eps: float | None = None,
    eps_bp: float | None = None,
    rho_0: float | None = None,
    alpha_0: float | None = None,
    beta_0: float | None = None,
    eps_0: float | None = None,
    eps_bp_0: float | None = None,
    # runtime model mode
    mode: str = "attainable_flops_logsumexp",
    add_suffix_to_col: bool = False,
    # algorithm args
    factor_causal: float = 0.75,
    bytes_act: int = 2,  # bfloat16
    bytes_Cmn: int = 4,  # float32
    bytes_w: int = 2,  # bfloat16
) -> pd.DataFrame:
    """Predict the step time for mLSTM model configurations in milliseconds."""

    if add_suffix_to_col:
        smode = "".join([w[0] for w in mode.split("_")])
        suffix = f"_r{rho}-a{alpha:.3e}-b{beta:.3e}-{smode}"
    else:
        suffix = ""

    def _apply_runtime_model(row: pd.Series) -> float:
        """Apply the runtime model to a single row of the DataFrame."""

        ret = runtime_model_mlstm_step_time(
            batchsize=row[("input_params", "batchsize")],
            embedding_dim=row[("model_params", "embedding_dim")],
            num_blocks=row[("model_params", "num_blocks")],
            num_heads=row[("model_params", "num_heads")],
            qk_dim_factor=row[("model_params", "qk_dim_factor")],
            vocab_size=row[("model_params", "vocab_size")],
            chunk_size=row[("model_params", "chunk_size")],
            ffn_dim=row[("model_params", "ffn_dim")],
            factor_causal=factor_causal,
            bytes_act=bytes_act,
            bytes_Cmn=bytes_Cmn,
            bytes_w=bytes_w,
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
            # runtime model mode
            mode=mode,
        )
        # convert to milliseconds
        if "log_" in mode:
            predicted_runtime_msecs = ret.pop("runtime") + np.log(np.array(1000.0))
        else:
            predicted_runtime_msecs = ret.pop("runtime") * np.array(1000.0)
        row[(f"pred{suffix}", "runtime")] = predicted_runtime_msecs
        for r in ret:
            row[(f"pred{suffix}", r)] = ret[r]
        return row

    pred_step_time_df = step_time_df.apply(_apply_runtime_model, axis=1)

    return pred_step_time_df
