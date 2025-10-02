from dataclasses import dataclass, field
from typing import Any, Callable

from ..common.run_fit import OptimizationConfig, run_optimization
from .objective_funcs import (
    HoffmannScalingLawObjectiveConfig,
    ScalingLawValidationConfig,
    get_scaling_law_objective_func,
    get_scaling_law_validation_func,
)


@dataclass
class FitParametricLossConfig(OptimizationConfig):
    objective_func_config: HoffmannScalingLawObjectiveConfig = field(
        default_factory=HoffmannScalingLawObjectiveConfig
    )
    validation_func_config: ScalingLawValidationConfig = field(
        default_factory=ScalingLawValidationConfig
    )


def fit_parametric_loss(config: FitParametricLossConfig) -> Any:
    return run_optimization(
        config=config,
        objective_func_generator=get_scaling_law_objective_func,
        validation_func_generator=get_scaling_law_validation_func,
    )
