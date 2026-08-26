"""jwstflow: a lightweight, YAML-driven orchestrator for the JWST calibration pipeline."""

from .config import Config, ConfigError, config_from_dict, load_config
from .engine.runner import Runner, run_config
from .steps.base import RunContext, Step, register_step

__version__ = "0.1.0"

__all__ = [
    "Config",
    "ConfigError",
    "RunContext",
    "Runner",
    "Step",
    "__version__",
    "config_from_dict",
    "load_config",
    "register_step",
    "run_config",
]
