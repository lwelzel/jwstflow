"""Driving jwstflow from Python (scripts, notebooks, Hydra apps).

jwstflow's YAML format is a plain dictionary validated by pydantic, so any
composition tool can produce it. Three examples:

    python python_api.py native      # load_config + overrides + run subset
    python python_api.py omegaconf   # compose with OmegaConf, hand the dict over
    python python_api.py hydra       # a @hydra.main app (pip install hydra-core)
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).parent


def native() -> None:
    from jwstflow import Runner, load_config

    cfg = load_config(
        HERE / "custom_steps.yaml",
        overrides=["parallel.backend=serial", "stages.calwebb_spec3.parameters.steps.cube_build.coord_system=ifualign"],
    )
    print(cfg.stage("calwebb_spec3").parameters)

    runner = Runner(cfg, only=["calwebb_detector1", "calwebb_spec2"], dry_run=True)
    for stage, tasks in runner.plan().items():  # {stage name: [Task, ...]}
        print(stage, [t.label for t in tasks])
    # runner = Runner(cfg); summary = runner.run(); print(summary.ok)


def omegaconf() -> None:
    from omegaconf import OmegaConf  # pip install omegaconf

    from jwstflow import config_from_dict
    from jwstflow.config.loader import load_raw

    base = OmegaConf.create(load_raw(HERE / "custom_steps.yaml"))  # `extends:` already resolved
    site = OmegaConf.create({"parallel": {"backend": "dask", "scheduler": "tcp://10.0.0.1:8786"}})
    cli = OmegaConf.from_dotlist(["parallel.workers=16"])
    merged = OmegaConf.to_container(OmegaConf.merge(base, site, cli), resolve=True)
    cfg = config_from_dict(merged, base_dir=HERE)
    print(cfg.parallel)


def hydra_app() -> None:
    import hydra  # pip install hydra-core
    from omegaconf import DictConfig, OmegaConf

    from jwstflow import Runner, config_from_dict

    @hydra.main(version_base=None, config_path=str(HERE), config_name="custom_steps")
    def main(hcfg: DictConfig) -> None:
        # Hydra resolves its own ${...} interpolations; jwstflow validates the result.
        data = OmegaConf.to_container(hcfg, resolve=True)
        data.pop("extends", None)  # Hydra does not know jwstflow's `extends`; use its defaults list instead
        cfg = config_from_dict(data, base_dir=HERE)
        Runner(cfg).run()

    main()


if __name__ == "__main__":
    {"native": native, "omegaconf": omegaconf, "hydra": hydra_app}[sys.argv[1] if len(sys.argv) > 1 else "native"]()
