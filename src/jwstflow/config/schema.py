"""Configuration schema for jwstflow.

Every key a user can put in a YAML file is declared here as a pydantic model.
This gives us, for free:

* validation with readable error messages (``jwstflow validate``),
* defaults and documentation in one place,
* a JSON Schema (``jwstflow schema``) that editors use for autocompletion.

Field docstrings are written for the *user* because they end up in the JSON
Schema and in ``jwstflow schema --markdown``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

# Name of the pseudo-stage that refers to downloaded/raw input data.
RAW_STAGE = "raw"


class _Base(BaseModel):
    """Common settings: forbid unknown keys so typos are caught early."""

    model_config = ConfigDict(extra="forbid", validate_default=True)


# ----------------------------------------------------------------------------
# Run-level settings
# ----------------------------------------------------------------------------


class CRDSConfig(_Base):
    """Calibration Reference Data System settings (exported as environment variables)."""

    path: Path = Field(
        default=Path("~/crds_cache"),
        description="Local CRDS cache directory (CRDS_PATH). '~' is expanded.",
    )
    server_url: str = Field(
        default="https://jwst-crds.stsci.edu",
        description="CRDS server (CRDS_SERVER_URL).",
    )
    context: str | None = Field(
        default=None,
        description=(
            "Pipeline mapping to use, e.g. 'jwst_1364.pmap'. "
            "None: let CRDS pick the server default at run time. "
            "'latest': resolve the current operational context once and pin it "
            "in the run manifest so the run is reproducible."
        ),
    )
    prefetch: bool = Field(
        default=False,
        description="Pre-download all reference files for the inputs before running "
        "(runs `crds bestrefs --sync-references`). Useful before going offline / to a cluster node.",
    )
    disable_steppars: bool = Field(
        default=False,
        description="Set STPIPE_DISABLE_CRDS_STEPPARS=true to ignore CRDS parameter-reference files "
        "(pars-*). Normally you want them ON, they are the instrument team's recommended defaults.",
    )
    readonly_cache: bool = Field(
        default=False,
        description="Set CRDS_READONLY_CACHE=1 (e.g. shared read-only cache on a cluster).",
    )


class ParallelConfig(_Base):
    """How tasks (files / associations) are distributed over CPU cores."""

    backend: Literal["serial", "process", "dask"] = Field(
        default="process",
        description="'serial' runs everything in this process (best for debugging / notebooks), "
        "'process' uses a spawn-based process pool, 'dask' uses dask.distributed "
        "(local cluster, or an existing scheduler via `scheduler`).",
    )
    workers: int = Field(default=4, ge=1, description="Number of concurrent tasks.")
    scheduler: str | None = Field(
        default=None,
        description="dask scheduler address (e.g. 'tcp://10.0.0.1:8786'). None -> LocalCluster.",
    )
    threads_per_worker: int = Field(
        default=1,
        ge=1,
        description="OMP/MKL/OPENBLAS thread limit exported to each worker, to avoid "
        "oversubscription when running many tasks at once.",
    )
    allow_nested_multiprocessing: bool = Field(
        default=False,
        description="The jwst docs say step-level multiprocessing (maximum_cores) must NOT be "
        "combined with running several exposures in parallel. jwstflow refuses that "
        "combination unless this is True.",
    )


class LoggingConfig(_Base):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    console: bool = Field(default=True, description="Also log to the terminal.")
    per_task_logs: bool = Field(
        default=True,
        description="Write one log file per task under <root>/logs/<stage>/. "
        "The full stpipe/jwst/CRDS output of each task lands there.",
    )


class CheckpointConfig(_Base):
    """Checkpointing / resume behaviour."""

    fingerprint: Literal["fast", "content"] = Field(
        default="fast",
        description="'fast' fingerprints inputs by path+size+mtime, 'content' hashes file contents "
        "(slow for GB-sized ramps, but immune to `touch`).",
    )
    require_outputs: bool = Field(
        default=True,
        description="A task only counts as complete if all outputs it recorded still exist.",
    )


# ----------------------------------------------------------------------------
# Data acquisition
# ----------------------------------------------------------------------------

Instrument = Literal["NIRSPEC", "MIRI"]


class DownloadConfig(_Base):
    """Fetch data from MAST. Files land in <root>/raw (or `dest`)."""

    enabled: bool = True
    backend: Literal["astroquery", "jwst_mast_query"] = Field(
        default="astroquery",
        description="'astroquery' uses astroquery.mast.Observations (default, tested). "
        "'jwst_mast_query' shells out to STScI's jwst_download.py.",
    )
    program: int = Field(description="Proposal / program ID, e.g. 1234.")
    observations: list[int] | None = Field(
        default=None, description="Observation numbers to fetch (None = all)."
    )
    instrument: Instrument
    modes: list[str] | None = Field(
        default=None,
        description="MAST instrument sub-modes, e.g. ['IFU'], ['MSA'], ['SLIT'], ['IMAGE'], "
        "['SLITLESS']. They are combined as '<instrument>/<mode>'. None = all modes.",
    )
    products: list[str] = Field(
        default=["UNCAL"],
        description="MAST productSubGroupDescription values, e.g. ['UNCAL'] or ['RATE', 'CAL'].",
    )
    calib_level: list[int] | None = Field(
        default=None, description="Restrict to these MAST calibration levels (1, 2, 3)."
    )
    filename_patterns: list[str] | None = Field(
        default=None,
        description="Only download products whose filename matches one of these globs, "
        "e.g. ['*_nrs1_*'].",
    )
    dest: Path | None = Field(default=None, description="Download directory (default <root>/raw).")
    token_env: str = Field(
        default="MAST_API_TOKEN",
        description="Environment variable holding a MAST token (needed for proprietary data).",
    )
    exclusive_only: bool = Field(
        default=False, description="Only fetch proprietary (exclusive access) products."
    )
    query_extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments passed straight to Observations.query_criteria.",
    )
    product_filters_extra: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra keyword arguments passed straight to Observations.filter_products.",
    )
    extra_args: list[str] = Field(
        default_factory=list,
        description="Extra command line arguments for the jwst_mast_query backend.",
    )


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------


class InputSpec(_Base):
    """Where a stage gets its input files and how to select them."""

    stage: str | None = Field(
        default=None,
        description=f"Name of the upstream stage whose output directory is scanned. "
        f"Use '{RAW_STAGE}' for downloaded/raw data. Mutually exclusive with `path`.",
    )
    path: Path | None = Field(
        default=None, description="Explicit directory to scan instead of an upstream stage."
    )
    pattern: str = Field(default="*.fits", description="Glob pattern applied inside the directory.")
    recursive: bool = False
    exclude: list[str] = Field(
        default_factory=list, description="Glob patterns to drop (matched against file names)."
    )
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="FITS primary-header filters, e.g. {EXP_TYPE: [NRS_IFU], BKGDTARG: false}. "
        "A scalar means equality, a list means membership, a string starting with "
        "'regex:' is matched as a regular expression.",
    )

    @model_validator(mode="after")
    def _one_source(self) -> InputSpec:
        if (self.stage is None) == (self.path is None):
            raise ValueError("InputSpec needs exactly one of `stage` or `path`")
        return self


class MemberRule(_Base):
    """Selects non-science association members (background, imprint, ...)."""

    exptype: Literal["background", "imprint", "psf", "target_acquisition"] = "background"
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Header filters identifying candidate members, e.g. {BKGDTARG: true}.",
    )
    match_on: list[str] = Field(
        default_factory=lambda: ["DETECTOR", "GRATING", "FILTER"],
        description="Header keywords that must be equal between science exposure and member.",
    )
    differ_on: list[str] = Field(
        default_factory=list,
        description="Header keywords that must DIFFER between science exposure and member, "
        "e.g. [PATT_NUM] to use the other nod positions as background.",
    )


class AssociationConfig(_Base):
    """Turn the input files into JSON association files (one task per association)."""

    mode: Literal["per_exposure", "group", "official"] = Field(
        description="'per_exposure': one Level-2 association per science exposure "
        "(+ matched background/imprint members). "
        "'group': one Level-3 association per group of exposures sharing `group_by` keys. "
        "'official': run the jwst association generator on a pool built from the headers "
        "(reproduces DMS behaviour, needs the jwst package).",
    )
    level: Literal[2, 3] = Field(description="Association level (2 for *2 pipelines, 3 for *3).")
    asn_type: str | None = Field(
        default=None,
        description="asn_type written into the file (spec2, image2, spec3, image3, ...). "
        "Defaults from the stage's step alias when possible.",
    )
    rule: str | None = Field(
        default=None,
        description="asn_rule written into the file. Default: DMSLevel2bBase / DMS_Level3_Base.",
    )
    science_filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Header filters selecting science members (applied on top of the stage inputs). "
        "Default excludes BKGDTARG=True and IS_IMPRT=True.",
    )
    members: list[MemberRule] = Field(
        default_factory=list,
        description="Rules adding background/imprint members to each product.",
    )
    group_by: list[str] = Field(
        default_factory=lambda: ["PROGRAM", "OBSERVTN", "INSTRUME", "GRATING", "FILTER"],
        description="(group mode) header keywords defining one Level-3 product.",
    )
    product_name: str = Field(
        default="jw{PROGRAM}-o{OBSERVTN}_{TARGPROP}_{INSTRUME}_{OPTELEM}",
        description="(group mode) product name template. Any header keyword is available; "
        "{OPTELEM} expands to grating-filter (NIRSpec) or channel-band / filter (MIRI). "
        "Lower-cased and sanitised automatically.",
    )
    relative_paths: bool = Field(
        default=True, description="Write member paths relative to the association file."
    )
    official_rules: Literal["level2", "level3"] | None = Field(
        default=None,
        description="(official mode) which rule set to use. Defaults from `level`.",
    )


class StageConfig(_Base):
    """One unit of the workflow: a jwst pipeline/step or a user-defined step."""

    name: str = Field(description="Unique stage name; also the output sub-directory.")
    step: str = Field(
        description="What to run. Either a built-in alias (detector1, image2, spec2, image3, "
        "spec3, tso3, assign_wcs, ...), a name registered through the 'jwstflow.steps' entry "
        "point, or a dotted path 'my_pkg.module:MyStep' / 'my_pkg.module.my_function'. "
        "Any stpipe Step/Pipeline subclass works, so do subclasses of the official pipelines."
    )
    enabled: bool = True
    inputs: list[InputSpec] = Field(
        default_factory=list,
        description="Input selectors. Several can be combined (e.g. science + backgrounds).",
    )
    association: AssociationConfig | None = Field(
        default=None,
        description="If set, inputs are grouped into association files and each association "
        "becomes one task. Required for the *3 pipelines and for spec2 with backgrounds/imprints.",
    )
    batch: Literal["per_file", "all"] | None = Field(
        default=None,
        description="'per_file': one task per input (parallel). 'all': a single task receives "
        "every input (for user steps that combine files, e.g. a summary plot). Default: the "
        "step's own `batch` attribute (jwstflow.Step subclasses / functions can declare it), "
        "which is 'per_file' for jwst steps and pipelines.",
    )
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="For stpipe steps: keyword arguments passed to Step.call(); the nested "
        "`steps: {jump: {rejection_threshold: 5}}` form works for pipelines. "
        "For user steps: keyword arguments of run().",
    )
    save_results: bool = Field(
        default=True, description="(stpipe steps) set save_results=True on the call."
    )
    output_dir: Path | None = Field(
        default=None, description="Override the output directory (default <root>/<name>)."
    )
    parallel: bool = Field(default=True, description="Allow this stage's tasks to run concurrently.")
    workers: int | None = Field(
        default=None, ge=1, description="Override the global worker count for this stage."
    )
    force: bool = Field(default=False, description="Ignore checkpoints and always rerun.")
    on_error: Literal["fail", "continue"] = Field(
        default="continue",
        description="'continue': a failing task is recorded and the rest keep going; "
        "downstream stages see only successful outputs. 'fail': abort the run.",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Extra ordering constraints besides those implied by `inputs`.",
    )
    tags: list[str] = Field(default_factory=list, description="Free-form labels for --tag selection.")

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not v or any(c in v for c in " /\\:") or v == RAW_STAGE:
            raise ValueError(
                f"stage name {v!r} must be non-empty, without spaces or slashes, and not {RAW_STAGE!r}"
            )
        return v

    @model_validator(mode="after")
    def _association_needs_inputs(self) -> StageConfig:
        if self.association is not None and not self.inputs:
            raise ValueError(f"stage {self.name!r}: association requires at least one input")
        return self


# ----------------------------------------------------------------------------
# Top level
# ----------------------------------------------------------------------------


class Config(_Base):
    """Top-level jwstflow configuration (one YAML file = one run)."""

    version: Literal[1] = Field(default=1, description="Config schema version.")
    name: str = Field(description="Run name, used in logs and manifests.")
    root: Path = Field(description="Run directory; every stage writes to <root>/<stage>.")
    description: str | None = None
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables exported to all workers.",
    )
    env_file: list[Path] = Field(
        default_factory=list,
        description=(
            "dotenv file(s) loaded before the configuration is interpolated and validated, "
            "relative to this YAML file (e.g. one holding CRDS_PATH/CRDS_SERVER_URL). "
            "A `.env` next to the YAML, in any parent directory, or in the current directory "
            "is picked up automatically; so is $JWSTFLOW_ENV_FILE. Variables already set in "
            "the shell win over the files."
        ),
    )
    env_file_override: bool = Field(
        default=False,
        description=(
            "If true, values from the explicit `env_file` entries replace variables that are "
            "already set in the shell (default: the shell wins, as with python-dotenv)."
        ),
    )
    plugins: list[Path] = Field(
        default_factory=list,
        description=(
            "Python files or directories with custom steps, relative to this YAML file. "
            "Files become importable by their stem (`step: my_steps:MyStep`), directories are "
            "added to sys.path (also in worker processes). Alternatively write the file path "
            "straight into `step:` as `./my_steps.py:MyStep`."
        ),
    )
    crds: CRDSConfig = Field(default_factory=CRDSConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig)
    download: DownloadConfig | None = None
    stages: list[StageConfig] = Field(default_factory=list)

    @field_validator("root", mode="before")
    @classmethod
    def _expand_root(cls, v: Any) -> Any:
        return Path(str(v)).expanduser() if v is not None else v

    @field_validator("env_file", "plugins", mode="before")
    @classmethod
    def _one_or_many_paths(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, (str, Path)):
            v = [v]
        return [Path(str(x)).expanduser() for x in v]

    @model_validator(mode="after")
    def _check_stages(self, info: ValidationInfo) -> Config:
        names: list[str] = []
        for st in self.stages:
            if st.name in names:
                raise ValueError(f"duplicate stage name {st.name!r}")
            names.append(st.name)
        known = set(names) | {RAW_STAGE}
        for st in self.stages:
            for inp in st.inputs:
                if inp.stage is not None and inp.stage not in known:
                    raise ValueError(
                        f"stage {st.name!r} reads from unknown stage {inp.stage!r} "
                        f"(known: {sorted(known)})"
                    )
            for dep in st.depends_on:
                if dep not in known:
                    raise ValueError(f"stage {st.name!r} depends_on unknown stage {dep!r}")
            if st.name in st.depends_on or any(i.stage == st.name for i in st.inputs):
                raise ValueError(f"stage {st.name!r} depends on itself")
        if self.download is None and any(
            i.stage == RAW_STAGE for st in self.stages for i in st.inputs
        ):
            # Not an error: users may drop files in <root>/raw by hand.
            pass
        return self

    @model_validator(mode="after")
    def _check_nested_multiprocessing(self) -> Config:
        """Refuse `maximum_cores` + multi-worker stages (jwst docs: mutually exclusive)."""
        if self.parallel.allow_nested_multiprocessing or self.parallel.backend == "serial":
            return self
        for st in self.stages:
            workers = st.workers or self.parallel.workers
            if not st.parallel or workers <= 1:
                continue
            for where, value in _walk(st.parameters):
                if where[-1] == "maximum_cores" and str(value).lower() not in ("none", "1"):
                    raise ValueError(
                        f"stage {st.name!r}: parameters.{'.'.join(where)}={value!r} enables "
                        "step-level multiprocessing while the stage also runs "
                        f"{workers} tasks in parallel. The jwst pipeline forbids nesting these. "
                        "Either set `parallel: false` / `workers: 1` on the stage, use "
                        "maximum_cores: none, or set parallel.allow_nested_multiprocessing: true."
                    )
        return self

    # convenience -----------------------------------------------------------------
    def stage(self, name: str) -> StageConfig:
        for st in self.stages:
            if st.name == name:
                return st
        raise KeyError(name)

    def stage_dir(self, name: str) -> Path:
        if name == RAW_STAGE:
            if self.download is not None and self.download.dest is not None:
                return self.download.dest.expanduser()
            return self.root / RAW_STAGE
        st = self.stage(name)
        return (st.output_dir or (self.root / st.name)).expanduser()

    @property
    def state_dir(self) -> Path:
        return self.root / ".jwstflow"

    @property
    def log_dir(self) -> Path:
        return self.root / "logs"

    @property
    def asn_dir(self) -> Path:
        return self.root / "associations"


def _walk(obj: Any, prefix: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], Any]]:
    """Flatten nested dicts into (key-path, value) pairs."""
    out: list[tuple[tuple[str, ...], Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_walk(v, prefix + (str(k),)))
    else:
        out.append((prefix, obj))
    return out


ConfigLike = Annotated[Config, "jwstflow Config"]
