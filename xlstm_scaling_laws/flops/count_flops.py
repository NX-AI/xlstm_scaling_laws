import logging
from dataclasses import dataclass
from typing import Any, Literal

from .common import calculate_model_fwbw_flops_from_fw_flops
from .llama import count_llama_model_flops_fw
from .mlstm import count_mlstm_model_flops_fw, count_model_flops_bw_mlstm_cell_only

LOGGER = logging.getLogger(__name__)


@dataclass
class FlopCountConfig:
    include_skip_flops: bool = False
    """Include the FLOPs for the skip connections."""
    include_norm_layer_flops: bool = False
    """Include the FLOPs for the normalization layers."""
    include_embedding_flops: bool = False
    """Include the FLOPs for the embedding layers.
    If True, FLOPs are computed like a linear layer, i.e. 2 * embedding_dim * vocab_size (* seq_len).
    If False, FLOPs are set to 0.

    Note: In the chinchilla paper the embedding FLOPs are included, but more realistic would be to not include them. 
    See here: https://jax-ml.github.io/scaling-book/transformers/
    """
    include_final_logit_flops: bool = True
    """Include the FLOPs for the final logits layer.
    If True, FLOPs are computed like a linear layer, i.e. 2 * embedding_dim * vocab_size (* seq_len).
    If False, FLOPs are set to 0.
    """
    attention_flop_calc_mode: Literal["chinchilla", "distill_scaling"] = "chinchilla"
    """The mode for calculating the FLOPs for the attention operation.
    - 'chinchilla': Use the chinchilla paper calculation. (does not account for causality)
    - 'distill_scaling': Use the calculation from the Apple paper Distillation Scaling Laws. (accounts for causality)
    """
    mlstm_fw_flop_calc_mode: Literal["first", "tfla"] = "first"
    """The mode for calculating the FLOPs for the mlstm operation.
    - 'first': Use the first calculation.
    - 'tfla': Use the second calculation, which is a cleaner version that also accounts for TFLA causality.
    """
    mlstm_flop_causal_factor: float = 0.75
    """The factor to account for the causality in the mlstm flops.
    It is used if `mlstm_fw_flop_calc_mode` is set to 'tfla'
    and the chunk_size is greater than 64.

    Should vary between 0.5 and 1.0.
    """
    round_ffn_dim_to_multiple_of_for_flops: bool = False
    """Round the FFN dimension according to the model implementation (typically to a multiple of 64).
    If False, the FFN dimension is not rounded, i.e. the value d_model * proj_factor_ffn is used.
    If True, the FFN dimension is rounded according to configuration and model implementation.
    """
    bw_flop_count_mode: Literal[
        "total_factor_2",
        "factor_2_linear_custom_seqmix_factor",
        "factor_2_linear_custom_seqmix_bw_count",
    ] = "total_factor_2"
    """Use the backward flop factor of 2 approximation for the overall backward pass FLOPs.

    - `factor_2_linear_custom_seqmix_factor`: the backward flop factor for the sequence mix flops (i.e. attention or mlstm flops)
        is set to `seq_mix_bw_flop_factor`. For the linear flops the backward flop factor is set to 2.
    - `total_factor_2`: the total backward flop factor is set to 2.
    - `factor_2_linear_custom_seqmix_bw_count`: the linear and other flops are counted with a backward flop factor of 2, but the sequence mix flops
        are counted with a custom backward count method. This is useful for models where the sequence mix backward flops are not 2x the forward flops.
    """
    seq_mix_bw_flop_factor: float = 2.5
    """The backward flop factor for the sequence mix flops (i.e. attention or mlstm flops).
    It is used if `total_bw_flop_factor_2_approx` is set to False.
    """

    # factors for flop calculation
    flop_factor_sig: float = 1.0
    flop_factor_exp: float = 1.0
    flop_factor_log: float = 1.0
    flop_factor_ffn_act_fn: float = 1.0
    flop_factor_max: float = 1.0
    flop_factor_mask: float = 1.0
    flop_factor_abs: float = 1.0

    def to_config_name(self, model_type: str, fw_only: bool = False) -> str:
        """Create a string representation for the config name, depending on the model_type."""
        string_repr = ""
        if model_type == "mlstm_v1":
            string_repr += f"mlstm_fw-{self.mlstm_fw_flop_calc_mode}"
            string_repr += f"--causal_factor-{self.mlstm_flop_causal_factor}"
        elif model_type == "llama":
            string_repr += f"llama_fw-{self.attention_flop_calc_mode}"
        else:
            raise ValueError
        
        string_repr += f"--round_ffn-{self.round_ffn_dim_to_multiple_of_for_flops}"

        if not fw_only:
            string_repr += f"--bw-{self.bw_flop_count_mode}"
            if self.bw_flop_count_mode == "factor_2_linear_custom_seqmix_factor":
                string_repr += f"-seq_mix_bw_factor-{self.seq_mix_bw_flop_factor}"
            
            string_repr += f"--incl_emb-{self.include_embedding_flops}--incl_logit-{self.include_final_logit_flops}"
        return string_repr


def count_model_flops_fw(
    model_type: str,
    model_kwargs: dict[str, Any],
    context_length: int,
    config: FlopCountConfig,
    num_params: int = 0,
) -> tuple[float, float, float, float]:
    """Count the number of FLOPs for the forward pass for a single sample (i.e. global batch size = 1).

    Returns:
        total flops, linear flops, seq_mix flops, other flops
    """
    if model_type == "mlstm_v1":
        count_flops_fn = count_mlstm_model_flops_fw
    elif model_type == "llama":
        count_flops_fn = count_llama_model_flops_fw
    else:
        LOGGER.warning(
            f"Model type '{model_type}' not supported. Using the FLOP count heuristic."
        )
        return fw_flop_count_heuristic(
            context_length=context_length, num_params=num_params
        )

    return count_flops_fn(
        model_kwargs=model_kwargs,
        context_length=context_length,
        config=config,
    )


def count_model_flops_fwbw(
    model_type: str,
    model_kwargs: dict[str, Any],
    context_length: int,
    config: FlopCountConfig,
    num_params: int = 0,
):
    """Count the number of FLOPs for the forward and backward pass for a single sample (i.e. global batch size = 1).

    Uses the bw computation mode specified in `bw_flop_count_mode`.

    Returns:
        total flops, linear flops, seq_mix flops, other flops
    """

    total_flops_fw, linear_flops_fw, seq_mix_flops_fw, other_flops_fw = (
        count_model_flops_fw(
            model_type=model_type,
            model_kwargs=model_kwargs,
            context_length=context_length,
            num_params=num_params,
            config=config,
        )
    )

    # estimate the backward flops from the forward flops
    total_flops_fwbw, linear_flops_fwbw, seq_mix_flops_fwbw, other_flops_fwbw = (
        calculate_model_fwbw_flops_from_fw_flops(
            total_model_flops_fw=total_flops_fw,
            linear_layer_model_flops_fw=linear_flops_fw,
            seq_mix_layer_model_flops_fw=seq_mix_flops_fw,
            other_model_flops_fw=other_flops_fw,
            config=config,
        )
    )

    if config.bw_flop_count_mode == "factor_2_linear_custom_seqmix_bw_count":
        assert seq_mix_flops_fwbw == 0.0, "The sequence mix flops should be 0.0."

        if model_type == "mlstm_v1":
            seq_mix_flops_bw_total, seq_mix_flops_bw_recurrent, seq_mix_flops_bw_parallel = count_model_flops_bw_mlstm_cell_only(
                model_kwargs=model_kwargs,
                context_length=context_length,
                config=config,
            )

            seq_mix_flops_fwbw = seq_mix_flops_bw_total
        else:
            raise NotImplementedError(
                f"The custom sequence mix backward flop count is not implemented for model: {model_type}"
            )
        total_flops_fwbw = linear_flops_fwbw + seq_mix_flops_fwbw + other_flops_fwbw

    return total_flops_fwbw, linear_flops_fwbw, seq_mix_flops_fwbw, other_flops_fwbw




def fw_flop_count_heuristic(
    context_length: int,
    num_params: int,
    **kwargs,
) -> tuple[float, float, float, float]:
    """Heuristic from the 'Scaling Laws for Neural Language Models' paper (https://arxiv.org/abs/2001.08361).
    C = 2 * D * N, where C is the number of FLOPs, D is the number of parameters,
    and N is the number of training tokens, i.e.

    The fwbw flops are estimated to be 6 * D * N, with the assumption that the backward is 2x the forward pass.
    So the forward pass is estimated to be 2 * D * N.

    FLOPs = 2 * num_params
    """
    return (2 * num_params * context_length, 0.0, 0.0, 0.0)


def fwbw_flop_count_heuristic(
    context_length: int,
    num_params: int,
    **kwargs,
) -> tuple[float, float, float]:
    """Heuristic from the 'Scaling Laws for Neural Language Models' paper (https://arxiv.org/abs/2001.08361).
    C = 6 * D * N, where C is the number of FLOPs, D is the number of parameters,
    and N is the number of training tokens, i.e.

    FLOPs = 6 * num_params
    """
    return (6 * num_params * context_length, 0.0, 0.0, 0.0)
