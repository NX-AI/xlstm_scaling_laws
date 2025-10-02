from enum import StrEnum
from pathlib import Path
from typing import Literal

from ..common.load_data import get_default_run_data_calc_config, load_run_data
from ..run_data import RunData

repo_dir = Path(__file__).resolve().parents[2]

data_dir = repo_dir / "data"


class RunDataSet(StrEnum):
    TOKENPARAM_MLSTM = "tokenparam_mlstm"
    TOKENPARAM_LLAMA = "tokenparam_llama"
    ISOFLOP_MLSTM_CTX8192 = "isoflop_mlstm_ctx8192"
    ISOFLOP_MLSTM_CTX16384 = "isoflop_mlstm_ctx16384"
    ISOFLOP_MLSTM_CTX2048 = "isoflop_mlstm_ctx2048"
    ISOFLOP_LLAMA_CTX8192 = "isoflop_llama_ctx8192"
    ISOFLOP_LLAMA_CTX16384 = "isoflop_llama_ctx16384"
    ISOFLOP_LLAMA_CTX2048 = "isoflop_llama_ctx2048"

    ISOFLOP_LLAMA_CTX8192_FLOPCOUNT_CHINCHILLA = (
        "isoflop_llama_ctx8192_flops_chinchilla"
    )

    ISOFLOP_MLSTM_CTX8192_LARE_GBS256 = "isoflop_mlstm_ctx8192_large_gbs256"

run_data_sets = [member.value for member in RunDataSet]


def get_run_data_file(data_set: RunDataSet) -> Path:
    """Get the run data file for the given data set."""
    return data_dir / f"{data_set}_wandb_data.pkl"


def get_run_data_dict(
    data_set: RunDataSet,
    attention_flop_calc_mode: Literal["chinchilla", "distill_scaling"] = "distill_scaling",
) -> dict[str, list[RunData]]:
    """Get the run data dictionary for the given run data set."""

    if "isoflop" in data_set:
        group_by = "none"
    else:
        group_by = "name"   

    run_data_dict = load_run_data(
        wandb_run_data_dict_file=get_run_data_file(data_set),
        run_data_class=RunData,
        config_calc_run_data=get_default_run_data_calc_config(
            attention_flop_calc_mode=attention_flop_calc_mode
        ),
        group_runs_by=group_by,
    )
    return run_data_dict
