import copy
import pathlib as Path
from collections import defaultdict

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from ...flops.count_flops import (
    FlopCountConfig,
    count_model_flops_fw,
    count_model_flops_fwbw,
)
from ...params.count_params import ParamCountConfig, count_model_params


def make_mlstm_fw_flop_comparison(
    model_config_dict: dict[str, dict],
    context_length: int = 8192,
    include_embedding_flops: bool = False,
    include_final_logit_flops: bool = True,
) -> pd.DataFrame:
    """Compare the two mLSTM FLOP counting methods for the forward pass."""
    model_type = "mlstm_v1"

    flop_count_cfgs_fw = [
        FlopCountConfig(
            include_skip_flops=False,
            include_norm_layer_flops=False,
            include_embedding_flops=include_embedding_flops,
            include_final_logit_flops=include_final_logit_flops,
            attention_flop_calc_mode="chinchilla",
            mlstm_fw_flop_calc_mode="first",
            mlstm_flop_causal_factor=1.0,  # unused
            round_ffn_dim_to_multiple_of_for_flops=False,
            bw_flop_count_mode="total_factor_2",
            seq_mix_bw_flop_factor=2.5,
            flop_factor_sig=1.0,
            flop_factor_exp=1.0,
            flop_factor_ffn_act_fn=1.0,
            flop_factor_max=1.0,
            flop_factor_mask=1.0,
            flop_factor_abs=1.0,
        ),
        FlopCountConfig(
            include_skip_flops=False,
            include_norm_layer_flops=False,
            include_embedding_flops=include_embedding_flops,
            include_final_logit_flops=include_final_logit_flops,
            attention_flop_calc_mode="chinchilla",
            mlstm_fw_flop_calc_mode="tfla",
            mlstm_flop_causal_factor=0.75,  # unused for chunk size <= 64
            round_ffn_dim_to_multiple_of_for_flops=False,
            bw_flop_count_mode="total_factor_2",
            seq_mix_bw_flop_factor=2.5,
            flop_factor_sig=1.0,
            flop_factor_exp=1.0,
            flop_factor_ffn_act_fn=1.0,
            flop_factor_max=1.0,
            flop_factor_mask=1.0,
            flop_factor_abs=1.0,
        ),
        FlopCountConfig(
            include_skip_flops=False,
            include_norm_layer_flops=False,
            include_embedding_flops=include_embedding_flops,
            include_final_logit_flops=include_final_logit_flops,
            attention_flop_calc_mode="chinchilla",
            mlstm_fw_flop_calc_mode="first",
            mlstm_flop_causal_factor=1.0,  # unused
            round_ffn_dim_to_multiple_of_for_flops=True,
            bw_flop_count_mode="total_factor_2",
            seq_mix_bw_flop_factor=2.5,
            flop_factor_sig=1.0,
            flop_factor_exp=1.0,
            flop_factor_ffn_act_fn=1.0,
            flop_factor_max=1.0,
            flop_factor_mask=1.0,
            flop_factor_abs=1.0,
        ),
        FlopCountConfig(
            include_skip_flops=False,
            include_norm_layer_flops=False,
            include_embedding_flops=include_embedding_flops,
            include_final_logit_flops=include_final_logit_flops,
            attention_flop_calc_mode="chinchilla",
            mlstm_fw_flop_calc_mode="tfla",
            mlstm_flop_causal_factor=1.0,  # unused
            round_ffn_dim_to_multiple_of_for_flops=True,
            bw_flop_count_mode="total_factor_2",
            seq_mix_bw_flop_factor=2.5,
            flop_factor_sig=1.0,
            flop_factor_exp=1.0,
            flop_factor_ffn_act_fn=1.0,
            flop_factor_max=1.0,
            flop_factor_mask=1.0,
            flop_factor_abs=1.0,
        ),
    ]

    flop_param_dict = {}
    for model_key, model_dict in model_config_dict.items():
        flop_param_dict[model_key] = {}
        # add model flop counts fw
        for i, flop_cfg in enumerate(flop_count_cfgs_fw):
            flop_param_dict[model_key][
                f"#FLOPs_{i}_fw-{flop_cfg.to_config_name(model_type=model_type)}"
            ] = count_model_flops_fw(
                model_type=model_type,
                context_length=context_length,
                model_kwargs=model_dict,
                config=flop_cfg,
            )[0]

    df = (
        pd.DataFrame(flop_param_dict)
        .T.reset_index()
        .rename(columns={"index": "model_name"})
    )

    return df

def make_mlstm_fwbw_flop_bw_apporox_comparison(
    model_config_dict: dict[str, dict],
    context_length: int = 8192,
    include_embedding_flops: bool = False,
    include_final_logit_flops: bool = True,
) -> pd.DataFrame:
    """Compare the two mLSTM FLOP counting methods for the forward pass."""
    model_type = "mlstm_v1"

    flop_count_cfgs_fw = [
        FlopCountConfig(
            include_skip_flops=False,
            include_norm_layer_flops=False,
            include_embedding_flops=include_embedding_flops,
            include_final_logit_flops=include_final_logit_flops,
            attention_flop_calc_mode="chinchilla",
            mlstm_fw_flop_calc_mode="tfla",
            mlstm_flop_causal_factor=0.75,  # unused
            round_ffn_dim_to_multiple_of_for_flops=True,
            bw_flop_count_mode="factor_2_linear_custom_seqmix_bw_count",
            seq_mix_bw_flop_factor=2.5,
            flop_factor_sig=1.0,
            flop_factor_exp=1.0,
            flop_factor_ffn_act_fn=1.0,
            flop_factor_max=1.0,
            flop_factor_mask=1.0,
            flop_factor_abs=1.0,
        ),
        FlopCountConfig(
            include_skip_flops=False,
            include_norm_layer_flops=False,
            include_embedding_flops=include_embedding_flops,
            include_final_logit_flops=include_final_logit_flops,
            attention_flop_calc_mode="chinchilla",
            mlstm_fw_flop_calc_mode="tfla",
            mlstm_flop_causal_factor=0.75,  # unused for chunk size <= 64
            round_ffn_dim_to_multiple_of_for_flops=True,
            bw_flop_count_mode="total_factor_2",
            seq_mix_bw_flop_factor=2.5,
            flop_factor_sig=1.0,
            flop_factor_exp=1.0,
            flop_factor_ffn_act_fn=1.0,
            flop_factor_max=1.0,
            flop_factor_mask=1.0,
            flop_factor_abs=1.0,
        ),
        FlopCountConfig(
            include_skip_flops=False,
            include_norm_layer_flops=False,
            include_embedding_flops=include_embedding_flops,
            include_final_logit_flops=include_final_logit_flops,
            attention_flop_calc_mode="chinchilla",
            mlstm_fw_flop_calc_mode="first",
            mlstm_flop_causal_factor=1.0,  # unused
            round_ffn_dim_to_multiple_of_for_flops=True,
            bw_flop_count_mode="factor_2_linear_custom_seqmix_factor",
            seq_mix_bw_flop_factor=2.5,
            flop_factor_sig=1.0,
            flop_factor_exp=1.0,
            flop_factor_ffn_act_fn=1.0,
            flop_factor_max=1.0,
            flop_factor_mask=1.0,
            flop_factor_abs=1.0,
        ),
        FlopCountConfig(
            include_skip_flops=False,
            include_norm_layer_flops=False,
            include_embedding_flops=include_embedding_flops,
            include_final_logit_flops=include_final_logit_flops,
            attention_flop_calc_mode="chinchilla",
            mlstm_fw_flop_calc_mode="tfla",
            mlstm_flop_causal_factor=1.0,  # unused
            round_ffn_dim_to_multiple_of_for_flops=True,
            bw_flop_count_mode="factor_2_linear_custom_seqmix_factor",
            seq_mix_bw_flop_factor=3.0,
            flop_factor_sig=1.0,
            flop_factor_exp=1.0,
            flop_factor_ffn_act_fn=1.0,
            flop_factor_max=1.0,
            flop_factor_mask=1.0,
            flop_factor_abs=1.0,
        ),
    ]

    flop_param_dict = {}
    for model_key, model_dict in model_config_dict.items():
        flop_param_dict[model_key] = {}
        # add model flop counts fw
        for i, flop_cfg in enumerate(flop_count_cfgs_fw):
            flop_param_dict[model_key][
                f"#FLOPs_{i}_fwbw-{flop_cfg.to_config_name(model_type=model_type)}"
            ] = count_model_flops_fwbw(
                model_type=model_type,
                context_length=context_length,
                model_kwargs=model_dict,
                config=flop_cfg,
            )[0]

    df = (
        pd.DataFrame(flop_param_dict)
        .T.reset_index()
        .rename(columns={"index": "model_name"})
    )

    return df


def make_mlstm_fw_flop_comparison_with_approximations(
    model_config_dict: dict[str, dict],
    context_length: int = 8192,
    include_embedding_flops: bool = False,
    include_final_logit_flops: bool = True,
) -> pd.DataFrame:
    """Compare the two mLSTM FLOP counting methods for the forward pass."""
    model_type = "mlstm_v1"

    flop_count_cfgs_fw = [
        FlopCountConfig(
            include_skip_flops=False,
            include_norm_layer_flops=False,
            include_embedding_flops=include_embedding_flops,
            include_final_logit_flops=include_final_logit_flops,
            attention_flop_calc_mode="chinchilla",
            mlstm_fw_flop_calc_mode="tfla",
            mlstm_flop_causal_factor=1.0,  # unused
            round_ffn_dim_to_multiple_of_for_flops=True,
            bw_flop_count_mode="total_factor_2",
            seq_mix_bw_flop_factor=2.5,
            flop_factor_sig=1.0,
            flop_factor_exp=1.0,
            flop_factor_ffn_act_fn=1.0,
            flop_factor_max=1.0,
            flop_factor_mask=1.0,
            flop_factor_abs=1.0,
        ),
    ]

    flop_param_dict = {}
    for model_key, model_dict in model_config_dict.items():
        flop_param_dict[model_key] = {}
        # add model flop counts fw
        for i, flop_cfg in enumerate(flop_count_cfgs_fw):
            flop_param_dict[model_key][
                f"#FLOPs_{i}_fw-{flop_cfg.to_config_name(model_type=model_type)}"
            ] = count_model_flops_fwbw(
                model_type=model_type,
                context_length=context_length,
                model_kwargs=model_dict,
                config=flop_cfg,
            )[0]
        # add param counts
        flop_param_dict[model_key].update(
            {
                "#params-all": count_model_params(
                    model_type=model_type,
                    model_kwargs=model_dict,
                    config=ParamCountConfig(
                        count_norm_layer_params=True,
                        count_embedding_params=True,
                        count_final_logits_params=True,
                    ),
                ),
                "#params-noembed": count_model_params(
                    model_type=model_type,
                    model_kwargs=model_dict,
                    config=ParamCountConfig(
                        count_norm_layer_params=True,
                        count_embedding_params=False,
                        count_final_logits_params=True,
                    ),
                ),
                "#params-noembed-nologit": count_model_params(
                    model_type=model_type,
                    model_kwargs=model_dict,
                    config=ParamCountConfig(
                        count_norm_layer_params=True,
                        count_embedding_params=False,
                        count_final_logits_params=False,
                    ),
                ),
            }
        )

    df = (
        pd.DataFrame(flop_param_dict)
        .T.reset_index()
        .rename(columns={"index": "model_name"})
    )

    return df