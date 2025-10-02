import logging
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal, Type

import pandas as pd

from ..flops.count_flops import (
    FlopCountConfig,
    count_model_flops_fw,
    count_model_flops_fwbw,
)
from ..params.count_params import ParamCountConfig, count_model_params
from .wandb_run_data import WandBRunData

LOGGER = logging.getLogger(__name__)


@dataclass
class RunDataCalcConfig:
    """Config class that determines how specific metrics are computed or extracted from the raw run data."""

    ## flops calculation
    flop_count_config: FlopCountConfig = field(default_factory=FlopCountConfig)

    ## param calculation
    param_count_config: ParamCountConfig = field(default_factory=ParamCountConfig)


@dataclass  # TODO try frozen = true (we actually don't want to change after creation, if something changes we create a new)
class BaseRunData:
    """Data class to store the data of a single run or multiple runs with the same name.
    It can be created from a raw WandBRunData object or a list of WandBRunData objects.
    It holds a full copy of the data an can then serve as a data container for further analysis.
    """

    name: str
    run_tag: str | None

    model_type: str
    model_kwargs: dict[str, Any]

    global_batch_size: int
    num_params: float
    """Number of model parameters, logged during training."""
    num_train_steps: int
    context_length: int
    """Number of tokens in the context."""
    learning_rate: float

    num_nodes: int
    num_gpus_per_node: int

    num_lr_warmup_steps: int | None = None
    """Number of learning rate warmup steps. If -1 or None then no warmup is used."""

    num_params_calculated: float | None = None
    """Number of model parameters, calculated from the model_kwargs."""

    num_total_flops_fwbw: float | None = None
    """Number of FLOPs for forward and backward pass for a single sample (i.e. global batch size = 1)
    with the specified context length.
    """
    num_linear_flops_fwbw: float | None = None
    num_seq_mix_flops_fwbw: float | None = None
    num_other_flops_fwbw: float | None = None

    num_total_flops_fw: float | None = None
    """Number of FLOPs for the forward pass for a single sample (i.e. global batch size = 1)"""
    num_linear_flops_fw: float | None = None
    num_seq_mix_flops_fw: float | None = None
    num_other_flops_fw: float | None = None

    logs: dict[str, pd.DataFrame] | None = None
    """Run logs as a dictionary of pandas DataFrames."""

    _raw_data: list[WandBRunData] | None = None
    """The raw data as a list of WandBRunData objects.
    There can be several runs with the same name, which are restarted runs belonging to the same experiment.
    """

    status: Literal["finished", "failed"] = "finished"

    config_calc_run_data: RunDataCalcConfig = field(default_factory=RunDataCalcConfig)

    def __post_init__(self):
        (
            self.num_total_flops_fwbw,
            self.num_linear_flops_fwbw,
            self.num_seq_mix_flops_fwbw,
            self.num_other_flops_fwbw,
        ) = self.count_flops_fwbw(config_calc_run_data=self.config_calc_run_data)

        (
            self.num_total_flops_fw,
            self.num_linear_flops_fw,
            self.num_seq_mix_flops_fw,
            self.num_other_flops_fw,
        ) = self.count_flops_fw(config_calc_run_data=self.config_calc_run_data)
        self.num_params_calculated = self.count_num_params(
            config_calc_run_data=self.config_calc_run_data
        )

    def count_flops_fwbw(
        self, config_calc_run_data: RunDataCalcConfig
    ) -> tuple[float, float, float, float]:
        return count_model_flops_fwbw(
            model_type=self.model_type,
            model_kwargs=self.model_kwargs,
            context_length=self.context_length,
            num_params=self.num_params,
            config=config_calc_run_data.flop_count_config,
        )

    def count_flops_fw(
        self, config_calc_run_data: RunDataCalcConfig
    ) -> tuple[float, float, float, float]:
        return count_model_flops_fw(
            model_type=self.model_type,
            model_kwargs=self.model_kwargs,
            context_length=self.context_length,
            num_params=self.num_params,
            config=config_calc_run_data.flop_count_config,
        )

    def count_num_params(self, config_calc_run_data: RunDataCalcConfig) -> float:
        try:
            num_model_params = count_model_params(
                model_type=self.model_type,
                model_kwargs=self.model_kwargs,
                config=config_calc_run_data.param_count_config,
            )
        except ValueError:
            LOGGER.warning(
                f"Could not calculate the number of model parameters for run '{self.name}'."
            )
            num_model_params = float("nan")
        return num_model_params

    def final_value_of_log(self, key: str) -> float:
        """Get the final value of a log key."""
        if key not in self.logs.keys():
            LOGGER.warning(
                f"Log key '{key}' not found in run '{self.name}'. Returning NaN."
            )
            return float("nan")

        return float(self.logs[key][key].iloc[-1])

    def plot_log(self, key: str, ax=None):
        """Plot a log key."""
        ax = self.logs[key].plot(x="_step", y=key, ax=ax)
        ax.set_title(key)
        return ax

    @property
    def log_keys(self) -> list[str]:
        return list(self.logs.keys())

    @property
    def config(self) -> dict[str, Any]:
        return self._raw_data[0].config

    @property
    def final_train_metrics(self) -> dict[str, float]:
        """Final training metrics."""
        train_metric_keys = ["train/.loss_mean"]
        return {key: self.final_value_of_log(key) for key in train_metric_keys}

    @property
    def final_val_metrics(self) -> dict[str, float]:
        """Final validation metrics."""
        val_metric_keys = [
            "val/.dclm_loss",
            "val/.dclm_perplexity",
            "val/.spaj627B_AR_loss",
            "val/.spaj627B_AR_perplexity",
        ]
        return {
            key: self.final_value_of_log(key)
            for key in val_metric_keys
            if key in self.log_keys
        }

    @property
    def step_time(self) -> float:
        """Mean step time in seconds. The first 10 entries are ignored."""
        if "train/.step_time" not in self.logs.keys():
            LOGGER.warning(
                f"Log key 'train/.step_time' not found in run '{self.name}'. Returning NaN."
            )
            return float("nan")
        return float(
            self.logs["train/.step_time"]["train/.step_time"].iloc[-100:].mean()
        )

    @property
    def num_runs(self) -> int:
        return len(self._raw_data)

    @property
    def runtime(self) -> float:
        """Runtime in seconds."""
        return sum(run.runtime for run in self._raw_data)

    @classmethod
    def create_run_data_from_wandb_run_data(
        cls,
        raw_run_data: WandBRunData,
        run_tag: str | None = None,
        config_calc_run_data: RunDataCalcConfig | None = None,
    ) -> "BaseRunData":
        raise NotImplementedError

    @classmethod
    def from_wandb_runs(
        cls,
        wandb_runs: WandBRunData | list[WandBRunData],
        run_tag: str | None = None,
        config_calc_run_data: RunDataCalcConfig | None = None,
    ) -> "BaseRunData":
        """If a list is passed, they must have the same name.
        The remaining data is taken from the first run in the list.
        """
        if not isinstance(wandb_runs, list):
            wandb_runs = [wandb_runs]

        run_data = cls.create_run_data_from_wandb_run_data(
            raw_run_data=wandb_runs[0],
            run_tag=run_tag,
            config_calc_run_data=config_calc_run_data,
        )

        wandb_runs = sorted(wandb_runs, key=lambda x: x.created_at)

        # combine all logs from multiple runs
        logs = {}
        log_keys = wandb_runs[0].logs.keys()
        for log_key in log_keys:
            key_logs = []
            for run in wandb_runs:
                assert log_key in run.logs, (
                    f"Log key '{log_key}' not found in run '{run.name}'"
                )
                key_logs.append(run.logs[log_key])

            log_df = pd.concat(key_logs).reset_index(level=0, drop=True)
            logs[log_key] = log_df

        run_data.logs = logs
        run_data._raw_data = wandb_runs

        return run_data

    @property
    def num_tokens_training(self) -> float:
        return (
            float(self.global_batch_size) * self.context_length * self.num_train_steps
        )

    @property
    def num_flops_training(self) -> float:
        """Number of FLOPs for the entire training run."""
        return self.num_total_flops_fwbw * self.global_batch_size * self.num_train_steps

    @property
    def token_param_ratio(self) -> float:
        return self.num_tokens_training / self.num_params

    @property
    def tokens_per_sec(self) -> float:
        return (self.global_batch_size * self.context_length) / self.step_time

    @property
    def tokens_per_sec_per_device(self) -> float:
        return self.tokens_per_sec / (self.num_nodes * self.num_gpus_per_node)

    @property
    def flops_per_sec(self) -> float:
        return self.num_total_flops_fwbw / self.step_time

    @property
    def flops_per_sec_per_device(self) -> float:
        return self.flops_per_sec / (self.num_nodes * self.num_gpus_per_node)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(name={self.name}, run_tag={self.run_tag}, "
            f"model_type={self.model_type}, num_params={self.num_params}, "
            f"num_train_steps={self.num_train_steps}, context_length={self.context_length}, "
            f"runtime={self.runtime}, num_nodes={self.num_nodes}, num_gpus_per_node={self.num_gpus_per_node})"
        )

    def __str__(self):
        return self.__repr__()

    def get_data_summary(self) -> dict[str, Any]:
        try:
            return {
                "name": self.name,
                "run_tag": self.run_tag,
                "model_type": self.model_type,
                "global_batch_size": self.global_batch_size,
                "learning_rate": self.learning_rate,
                "context_length": self.context_length,
                "num_params": self.num_params,
                "num_params_calculated": self.num_params_calculated,
                "num_train_steps": self.num_train_steps,
                "num_lr_warmup_steps": self.num_lr_warmup_steps,
                "runtime": self.runtime,
                "num_nodes": self.num_nodes,
                "num_gpus_per_node": self.num_gpus_per_node,
                "num_flops_fwbw": self.num_total_flops_fwbw,
                "num_linear_flops_fwbw": self.num_linear_flops_fwbw,
                "num_seq_mix_flops_fwbw": self.num_seq_mix_flops_fwbw,
                "seq_mix_flop_ratio": self.num_seq_mix_flops_fwbw
                / self.num_total_flops_fwbw,
                "linear_flop_ratio": self.num_linear_flops_fwbw
                / self.num_total_flops_fwbw,
                "num_flops_fw": self.num_total_flops_fw,
                "num_linear_flops_fw": self.num_linear_flops_fw,
                "num_seq_mix_flops_fw": self.num_seq_mix_flops_fw,
                "num_tokens_training": self.num_tokens_training,
                "num_flops_training": self.num_flops_training,
                "step_time": self.step_time,
                "token_param_ratio": self.token_param_ratio,
                "tokens_per_sec": self.tokens_per_sec,
                "tokens_per_sec_per_device": self.tokens_per_sec_per_device,
                "flops_per_sec": self.flops_per_sec,
                "flops_per_sec_per_device": self.flops_per_sec_per_device,
                **self.final_train_metrics,
                **self.final_val_metrics,
                **self.model_kwargs,
                "status": self.status,
            }
        except Exception as e:
            LOGGER.warning(
                f"Could not create data summary for run {self.name}, first id={self._raw_data[0].id}. Error: {e}"
            )
            LOGGER.debug(traceback.format_exc())
            return {
                "name": self.name,
                "run_tag": self.run_tag,
                "model_type": self.model_type,
            }


def convert_to_run_data(
    wandb_run_data_runs: list[WandBRunData],
    run_data_class: Type[BaseRunData],
    run_tag: str | None = None,
    config_calc_run_data: RunDataCalcConfig = None,
    group_runs_by: Literal["none", "name"] = True,
) -> list[BaseRunData]:
    """
    Converts a list of WandBRunData objects to a list of RunData objects.
    Grouping is done by the run name.

    Args:
        wandb_run_data_runs: List of WandBRunData objects.
        run_tag: Optional tag for the run.
        config_calc_run_data: Config for the RunData object.
        group_by_name: If True, group the runs by their name.
            If False, create a RunData object for each run.
    Returns:
        List of RunData objects.
    """
    run_data_list = []
    if group_runs_by == "name":
        wb_run_groups = defaultdict(list)
        for run in wandb_run_data_runs:
            wb_run_groups[run.name].append(run)

        for name, runs in wb_run_groups.items():
            run_data = run_data_class.from_wandb_runs(
                wandb_runs=runs,
                run_tag=run_tag,
                config_calc_run_data=config_calc_run_data,
            )
            run_data_list.append(run_data)
    elif group_runs_by == "none":
        # If no grouping is done, create a RunData object for each run.
        for wb_run in wandb_run_data_runs:
            run_data_list.append(
                run_data_class.from_wandb_runs(
                    wandb_runs=wb_run,
                    run_tag=run_tag,
                    config_calc_run_data=config_calc_run_data,
                )
            )
    else:
        raise ValueError(
            f"Invalid value for group_runs_by: {group_runs_by}. Must be 'name' or 'none'."
        )

    return run_data_list


def convert_to_run_data_dict(
    wandb_run_data_dict: dict[str, list[WandBRunData]],
    run_data_class: Type[BaseRunData],
    config_calc_run_data: RunDataCalcConfig = None,
    group_runs_by: Literal["none", "name"] = True,
) -> dict[str, list[BaseRunData]]:
    run_data_dict = {}
    for run_tag, wandb_runs in wandb_run_data_dict.items():
        run_data_list = convert_to_run_data(
            wandb_run_data_runs=wandb_runs,
            run_data_class=run_data_class,
            run_tag=run_tag,
            config_calc_run_data=config_calc_run_data,
            group_runs_by=group_runs_by,
        )
        run_data_dict[run_tag] = run_data_list

    return run_data_dict


def create_run_summary_table(
    run_data_dict: dict[str, list[BaseRunData]],
) -> pd.DataFrame:
    run_summary_dicts = []

    for tag, runs in run_data_dict.items():
        for run in runs:
            run_summary_dicts.append(run.get_data_summary())

    return pd.DataFrame(run_summary_dicts)
