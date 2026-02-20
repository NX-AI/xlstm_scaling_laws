import copy
import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import wandb
from tqdm import tqdm

LOGGER = logging.getLogger(__file__)


@dataclass
class WandBRunData:
    name: str | None
    id: str | None
    path: list[str] | None = None
    created_at: str | None = None
    config: dict | None = None
    metadata: dict | None = None
    summary: dict | None = None
    logs: dict[str, pd.DataFrame] | None = None
    step_key: str = "_step"

    @property
    def runtime(self) -> float:
        """Runtime in seconds.
        If the runtime is not available, return NaN.
        """
        try:
            return self.summary["_runtime"]
        except KeyError:
            return float("nan")

    @property
    def num_nodes(self) -> int:
        """Number of nodes."""
        return int(self.metadata["slurm"]["job_num_nodes"])

    @property
    def num_gpus_per_node(self) -> int:
        """Number of GPUs per node."""
        return len(self.metadata["slurm"]["job_gpus"].split(","))

    def __repr__(self):
        return f"WandBRunData(name={self.name}, id={self.id}, path={self.path}, created_at={self.created_at})"

    @staticmethod
    def from_api(
        wandb_run: Any,
        log_keys: list[str] = None,
        samples: int = int(1e6),
        x_axis: str = "_step",
    ) -> "WandBRunData":
        # Note: we try to create new objects or deepcopy of them here to avoid any side effects
        # if there are references within the wandb_run object.
        name = str(wandb_run.name)
        run_id = str(wandb_run.id)
        path = list(wandb_run.path)
        created_at = str(wandb_run.created_at)
        config = copy.deepcopy(wandb_run.config)
        metadata = copy.deepcopy(wandb_run.metadata)
        summary = copy.deepcopy(wandb_run.summary._json_dict)
        if log_keys is not None:
            logs = {}
            for key in log_keys:
                key_data: pd.DataFrame = wandb_run.history(
                    samples=samples, keys=[key], x_axis=x_axis, pandas=True
                )
                if key_data.empty:
                    LOGGER.info(f"No data for log_key {key} logged. Skipping.")
                else:
                    logs[key] = key_data

        else:
            logs = {}

        return WandBRunData(
            name=name,
            id=run_id,
            path=path,
            created_at=created_at,
            config=config,
            summary=summary,
            metadata=metadata,
            logs=logs,
        )


def download_wandb_run_data(
    path: str,
    filters: dict = None,
    order: str = None,
    log_keys: list[str] = None,
    samples: int = int(1e6),
    x_axis: str = "_step",
) -> list[WandBRunData]:
    api = wandb.Api()

    runs = api.runs(path, filters=filters, order=order)

    wandb_runs = []
    for run in runs:
        wandb_runs.append(
            WandBRunData.from_api(
                wandb_run=run, log_keys=log_keys, samples=samples, x_axis=x_axis
            )
        )

    return wandb_runs


def download_wandb_run_data_per_tag(
    path: str,
    tags: list[str | list[str]],  # enable filtering by one or more tags
    order: str = None,
    log_keys: list[str] = None,
    samples: int = int(1e6),
    x_axis: str = "_step",
) -> dict[str, list[WandBRunData]]:
    wandb_runs = {}
    for tag in tqdm(tags):
        if isinstance(tag, list):
            filter_list = [{"tags": {"$in": [t]}} for t in tag]
            filters = {"$and": filter_list}
            tag_key = ",".join(tag)
        else:
            filters = {"tags": {"$eq": tag}}
            tag_key = tag

        tag_runs = download_wandb_run_data(
            path=path,
            filters=filters,
            order=order,
            log_keys=log_keys,
            samples=samples,
            x_axis=x_axis,
        )
        wandb_runs[tag_key] = tag_runs

    return wandb_runs
