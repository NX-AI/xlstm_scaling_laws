import argparse
import logging
import pickle
from pathlib import Path

from xlstm_scaling_laws.common.wandb_run_data import download_wandb_run_data_per_tag

LOGGER = logging.getLogger(__name__)

mlstm_tokenparam_scaling_law_tags = [
    "scl_mlstm_160M",
    "scl_mlstm_160Mv2",
    "scl_mlstm_400M",
    "scl_mlstm_830M",
    "scl_mlstm_1.4B",
    "scl_mlstm_2.7B",
    "scl_mlstm_7B",
    "scl_mlstm_rerun",  # collection of all reruns (as backup)
    "dclm_mLSTMv1_7B_longrun_pretraining_final",
]
llama_tokenparam_scaling_law_tags = [
    "scl_llama_160M",
    "scl_llama_400M",
    "scl_llama_830M",
    "scl_llama_830Mv2",
    "scl_llama_1.4B",
    "scl_llama_1.4Bv2",
    "scl_llama_2.1B",
    "scl_llama_2.7B",
    "scl_llama_7B",
]
## Round 1 tags:
mlstm_isoflop_scaling_law_tags = [
    ["nb10_ed640_nh5_pf2.667", "sclaw_iso"],
    ["nb13_ed640_nh5_pf2.667", "sclaw_iso"],
    ["nb16_ed640_nh5_pf2.667", "sclaw_iso"],
    ["nb12_ed768_nh6_pf2.667", "sclaw_iso"],
    ["nb15_ed768_nh6_pf2.667", "sclaw_iso"],
    ["nb18_ed768_nh6_pf2.667", "sclaw_iso"],
    ["nb24_ed1024_nh4_pf2.667", "sclaw_iso"],
    ["nb27_ed1024_nh4_pf2.667", "sclaw_iso"],
    ["nb30_ed1024_nh4_pf2.667", "sclaw_iso"],
    ["nb24_ed1280_nh5_pf2.667", "sclaw_iso"],
    ["nb27_ed1280_nh5_pf2.667", "sclaw_iso"],
    ["nb30_ed1280_nh5_pf2.667", "sclaw_iso"],
    ["nb24_ed1536_nh6_pf2.667", "sclaw_iso"],
    ["nb27_ed1536_nh6_pf2.667", "sclaw_iso"],
    ["nb30_ed1536_nh6_pf2.667", "sclaw_iso"],
    ["nb24_ed1792_nh7_pf2.667", "sclaw_iso"],
    ["nb27_ed1792_nh7_pf2.667", "sclaw_iso"],
    ["nb30_ed1792_nh7_pf2.667", "sclaw_iso"],
    ["nb24_ed2048_nh8_pf2.667", "sclaw_iso"],
    ["nb27_ed2048_nh8_pf2.667", "sclaw_iso"],
    ["nb30_ed2048_nh8_pf2.667", "sclaw_iso"],
    ["nb32_ed2560_nh10_pf2.667", "sclaw_iso"],
    ["nb35_ed2560_nh10_pf2.667", "sclaw_iso"],
    ["nb38_ed2560_nh10_pf2.667", "sclaw_iso"],
]
## Round 2 tags:
mlstm_isoflop_scaling_law_tags2 = [
    ["nb12_ed896_nh7_pf2.667", "sclaw_iso"],
    ["nb15_ed896_nh7_pf2.667", "sclaw_iso"],
    ["nb18_ed896_nh7_pf2.667", "sclaw_iso"],
    ["nb21_ed896_nh7_pf2.667", "sclaw_iso"],
    ["nb24_ed896_nh7_pf2.667", "sclaw_iso"],
    ["nb24_ed1152_nh9_pf2.667", "sclaw_iso"],
    ["nb27_ed1152_nh9_pf2.667", "sclaw_iso"],
    ["nb30_ed1152_nh9_pf2.667", "sclaw_iso"],
    ["nb24_ed1408_nh11_pf2.667", "sclaw_iso"],
    ["nb27_ed1408_nh11_pf2.667", "sclaw_iso"],
    ["nb30_ed1408_nh11_pf2.667", "sclaw_iso"],
]
## Round 3 tags:
mlstm_isoflop_scaling_law_tags3 = [
    ["nb27_ed896_nh7_pf2.667", "sclaw_iso"],
    ["nb21_ed1024_nh4_pf2.667", "sclaw_iso"],
    ["nb18_ed1024_nh4_pf2.667", "sclaw_iso"],
    ["nb33_ed2048_nh8_pf2.667", "sclaw_iso"],
    ["nb36_ed2048_nh8_pf2.667", "sclaw_iso"],
    ["nb24_ed2304_nh9_pf2.667", "sclaw_iso"],
    ["nb27_ed2304_nh9_pf2.667", "sclaw_iso"],
    ["nb30_ed2304_nh9_pf2.667", "sclaw_iso"],
    ["nb33_ed2304_nh9_pf2.667", "sclaw_iso"],
]
## Round 5 Round 6 tags:
mlstm_isoflop_scaling_law_tags5_6 = ["sclaw_iso_round5", "sclaw_iso_round6"]
## Round 7 Round 8 tags:
mlstm_isoflop_scaling_law_tags7_8 = ["sclaw_iso_round7", "sclaw_iso_round8"]

mlstm_isoflop_scaling_law_tags_all = [
    *mlstm_isoflop_scaling_law_tags,
    *mlstm_isoflop_scaling_law_tags2,
    *mlstm_isoflop_scaling_law_tags3,
    *mlstm_isoflop_scaling_law_tags5_6,
    *mlstm_isoflop_scaling_law_tags7_8,
]

mlstm_isoflop_ctx_2048_tags = [
    "sclaw_mlstm_ctx_iso1",
    "sclaw_mlstm_ctx_iso2",
    "sclaw_mlstm_ctx_iso3",
]
mlstm_isoflop_ctx_16384_tags = [
    "sclaw_mlstm_ctx_iso4",
    "sclaw_mlstm_ctx_iso5",
    "sclaw_mlstm_ctx_iso6",
]
mlstm_isoflop_ctx_8192_large_gbs256_tags = [
    "sclaw_mlstm_ctx_iso7",
    "sclaw_mlstm_ctx_iso8",
]

llama_isoflop_ctx8192_chinchilla_tags = [
    "sclaw_llama_iso1",
    "sclaw_llama_iso2",
    "sclaw_llama_iso3",
]
llama_isoflop_ctx8192_tags = [
    "sclaw_llama_iso4",
    "sclaw_llama_iso5",
    "sclaw_llama_iso6",
    "sclaw_llama_iso13",
]
llama_isoflop_ctx2048_tags = [
    "sclaw_llama_iso7",
    "sclaw_llama_iso8",
    "sclaw_llama_iso9",
]
llama_isoflop_ctx16384_tags = [
    "sclaw_llama_iso10",
    "sclaw_llama_iso11",
    "sclaw_llama_iso12",
]


run_data_set_to_run_tag_mapping = {
    "isoflop_mlstm_ctx8192": mlstm_isoflop_scaling_law_tags_all,
    "isoflop_mlstm_ctx8192_large_gbs256": mlstm_isoflop_ctx_8192_large_gbs256_tags,
    "isoflop_mlstm_ctx16384": mlstm_isoflop_ctx_16384_tags,
    "isoflop_mlstm_ctx2048": mlstm_isoflop_ctx_2048_tags,
    "isoflop_llama_ctx8192": llama_isoflop_ctx8192_tags,
    "isoflop_llama_ctx8192_flops_chinchilla": llama_isoflop_ctx8192_chinchilla_tags,
    "isoflop_llama_ctx16384": llama_isoflop_ctx16384_tags,
    "isoflop_llama_ctx2048": llama_isoflop_ctx2048_tags,
    "tokenparam_mlstm": mlstm_tokenparam_scaling_law_tags,
    "tokenparam_llama": llama_tokenparam_scaling_law_tags,
}

log_keys = [
    "model/.num_params",
    "dataset/.global_batch_size",
    "train/.optimizer/lr",
    "train/.loss_mean",
    "train/.loss_max",
    "val/.dclm_loss",
    "val/.dclm_perplexity",
    "val/.spaj627B_AR_loss",
    "val/.spaj627B_AR_perplexity",
    "train/.step_time",
    "train/.param_norm",
    "train/.grad_norm_mean",
    "train/.grad_norm_max",
]


def download_and_save_data(wandb_tags: list[str], data_file: Path):
    LOGGER.info(f"Downloading wandb runs for tags: {wandb_tags}")

    run_data = download_wandb_run_data_per_tag(
        path="xlstm/xlstm_jax",
        tags=wandb_tags,
        order="+summary_metrics._step",
        log_keys=log_keys,
    )

    LOGGER.info(f"Saving wandb run data to {data_file}")
    with open(data_file, "wb") as f:
        pickle.dump(run_data, f)

    LOGGER.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s][%(name)s:%(lineno)d][%(levelname)s] - %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--runset", help="The runset to download data for.", type=str, default="all"
    )

    args = parser.parse_args()

    runset = args.runset

    if "," in runset:
        runsetlist = runset.split(",")
    else:
        runsetlist = [runset]

    from xlstm_scaling_laws.load_data.datafiles import get_run_data_file, run_data_sets

    possible_runset_args = ["all", *run_data_sets]
    for rs in runsetlist:
        assert rs in possible_runset_args, (
            f"Invalid runset. Must be one of: {possible_runset_args}. Got: {rs}"
        )
        LOGGER.info(f"Downloading data for runset: {runset}")

        if rs == "all":
            for rds in run_data_sets:
                LOGGER.info(f"Downloading data for runset: {rds}")
                run_data_file = get_run_data_file(rds)
                download_and_save_data(
                    wandb_tags=run_data_set_to_run_tag_mapping[rds],
                    data_file=run_data_file,
                )
        else:
            run_data_file = get_run_data_file(rs)
            download_and_save_data(
                wandb_tags=run_data_set_to_run_tag_mapping[rs], data_file=run_data_file
            )
