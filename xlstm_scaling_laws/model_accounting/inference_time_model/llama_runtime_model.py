import numpy as np
import pandas as pd

from ..flops.llama import (
    count_llama_model_flops__generation,
    count_llama_model_flops__prefill,
)
from ..memops.llama import (
    count_llama_model_memops__generation,
    count_llama_model_memops__prefill,
)
from .generic import CountType, get_runtime_model


def runtime_model_llama_ttft(
    # input args
    batchsize: CountType,
    prefill: CountType,
    # model args
    embedding_dim: int,
    num_blocks: int,
    vocab_size: int,
    ffn_dim: int,
    n_headq: int,
    n_headkv: int,
    d_hv: int,
    d_qk: int,
    # algorithm args
    bytes_act: int,
    bytes_w: int,
    # runtime model args
    rho: float,
    alpha: float,
    beta: float,
    eps: float,
    eps_bp: float | None = None,
    rho_0: float | None = None,
    alpha_0: float | None = None,
    beta_0: float | None = None,
    eps_0: float | None = None,
    eps_bp_0: float | None = None,
    # runtime model mode
    mode: str = "attainable_flops_logsumexp",
) -> dict:
    """Predict the time to first token (TTFT) for a given Llama model configuration in seconds."""
    
    runtime_model = get_runtime_model(mode)
    count_args = {}
    input_args = {
        "batch_size": batchsize,
        "seq_len": prefill,
    }

    assert embedding_dim / n_headq == d_hv, "d_hv must be embedding_dim / n_headq"

    model_args = {
        "n_vocab": vocab_size,
        "n_blocks": num_blocks,
        "d_ff": ffn_dim,
        "n_headq": n_headq,
        "n_headkv": n_headkv,
        "d_qk": d_qk,
        "d_hv": d_hv,
        "with_unembed": False,  # we do not compute the unembedding in the TTFT model (only for the last token)
    }

    algo_args = {
        "bytes_act": bytes_act,
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
        fn_flops_algo=count_llama_model_flops__prefill,
        fn_memops_algo=count_llama_model_memops__prefill,
        count_args=count_args,
        runtime_model_args=runtime_model_args,
    )
    return ret


def runtime_model_llama_step_time(
    # input args
    batchsize: CountType,
    seq_len_pre: CountType,
    seq_len_gen: CountType,
    # model args
    embedding_dim: int,
    num_blocks: int,
    n_headq: int,
    n_headkv: int,
    d_qk: int,
    d_hv: int,
    vocab_size: int,
    ffn_dim: int,
    # algorithm args
    bytes_act: int,
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
    """Predict the average step time for a given Llama configuration in seconds.
    """
    runtime_model = get_runtime_model(mode)

    assert embedding_dim / n_headq == d_hv, "d_hv must be embedding_dim / n_headq"

    count_args = {}
    input_args = {
        "batch_size": batchsize,
        "seq_len_pre": seq_len_pre,
        "seq_len_gen": seq_len_gen,
    }

    model_args = {
        "n_vocab": vocab_size,
        "n_blocks": num_blocks,
        "d_ff": ffn_dim,
        "n_headq": n_headq,
        "n_headkv": n_headkv,
        "d_qk": d_qk,
        "d_hv": d_hv,
        "with_unembed": True,  # we compute the unembedding in the step time model
    }

    algo_args = {
        "bytes_act": bytes_act,
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
        fn_flops_algo=count_llama_model_flops__generation,
        fn_memops_algo=count_llama_model_memops__generation,
        count_args=count_args,
        runtime_model_args=runtime_model_args,
    )
    return ret


def predict_runtime_llama_ttft(
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
    bytes_act: int = 2, # bfloat16
    bytes_w: int = 2, # bfloat16
) -> pd.DataFrame:
    """Predict the time to first token (TTFT) for Llama model configuration in milliseconds."""
    
    if add_suffix_to_col:
        smode = "".join([w[0] for w in mode.split("_")])
        suffix = f"_r{rho}-a{alpha:.3e}-b{beta:.3e}-{smode}"
    else:
        suffix = ""


    def _apply_runtime_model(row: pd.Series) -> float:
        """Apply the runtime model to a single row of the DataFrame."""

        # We already hardcode MHA (e.g. n_headq = n_headkv) in the model params
        head_dim = row[("model_params", "embedding_dim")] / row[("model_params", "num_heads")]

        ret = runtime_model_llama_ttft(
            batchsize=row[("input_params", "batchsize")],
            prefill=row[("input_params", "prefill")],
            embedding_dim=row[("model_params", "embedding_dim")],
            num_blocks=row[("model_params", "num_blocks")],
            n_headq=row[("model_params", "num_heads")],
            n_headkv=row[("model_params", "num_heads")],
            d_qk=head_dim,
            d_hv=head_dim,
            vocab_size=row[("model_params", "vocab_size")],
            ffn_dim=row[("model_params", "ffn_dim")],
            bytes_act=bytes_act,
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

def predict_runtime_llama_step_time(
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
    bytes_act: int = 2, # bfloat16
    bytes_w: int = 2, # bfloat16
) -> pd.DataFrame:
    """Predict the step time for Llama model configurations in milliseconds."""

    if add_suffix_to_col:
        smode = "".join([w[0] for w in mode.split("_")])
        suffix = f"_r{rho}-a{alpha:.3e}-b{beta:.3e}-{smode}"
    else:
        suffix = ""

    

    def _apply_runtime_model(row: pd.Series) -> float:
        """Apply the runtime model to a single row of the DataFrame."""

        head_dim = row[("model_params", "embedding_dim")] / row[("model_params", "num_heads")]

        ret = runtime_model_llama_step_time(
            batchsize=row[("input_params", "batchsize")],
            seq_len_pre=row[("input_params", "prefill")],
            seq_len_gen=100, # assume that prefill dominates
            embedding_dim=row[("model_params", "embedding_dim")],
            num_blocks=row[("model_params", "num_blocks")],
            n_headq=row[("model_params", "num_heads")],
            n_headkv=row[("model_params", "num_heads")],
            d_qk=head_dim,
            d_hv=head_dim,
            vocab_size=row[("model_params", "vocab_size")],
            ffn_dim=row[("model_params", "ffn_dim")],
            bytes_act=bytes_act,
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