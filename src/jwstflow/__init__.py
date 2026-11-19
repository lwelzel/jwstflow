"""jwstflow: a lightweight, YAML-driven orchestrator for the JWST calibration pipeline."""

from importlib.metadata import PackageNotFoundError, version

from . import qa
from .config import Config, ConfigError, config_from_dict, load_config
from .engine.runner import Runner, run_config
from .steps import RunContext, Step, StepParams, register_step

try:
    __version__ = version("jwstflow")
except PackageNotFoundError:  # a bare source checkout on PYTHONPATH, not an install
    __version__ = "0+unknown"

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
    "qa",
    "register_step",
    "run_config",
]
