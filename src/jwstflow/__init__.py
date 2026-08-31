"""jwstflow: a lightweight, YAML-driven orchestrator for the JWST calibration pipeline."""

from . import qafig
from .config import Config, ConfigError, config_from_dict, load_config
from .engine.runner import Runner, run_config
from .steps.base import RunContext, Step, StepParams, register_step

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ConfigError",
    "RunContext",
    "Runner",
    "Step",
    "StepParams",
    "__version__",
    "config_from_dict",
    "load_config",
    "qafig",
    "register_step",
    "run_config",
]
