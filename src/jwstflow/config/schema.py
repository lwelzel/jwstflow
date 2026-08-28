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

import re

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
from ..steps.base import LEVEL_DIRS  # noqa: E402


def slugify(text: str) -> str:
    """Directory-safe label: lowercase, non-alphanumerics -> '-', e.g. 'ESO-Ha 569' -> 'eso-ha-569'."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "target"


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
        default=True,
        description="Sync all CRDS reference files for the inputs before the first stage "
        "(`crds bestrefs --sync-references`). On by default: it makes a plain `jwstflow run` "
        "self-sufficient and keeps parallel workers from racing to populate a shared (NFS) "
        "cache on demand. Costs seconds when the cache is warm; set false only if you manage "
        "the cache yourself.",
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
    """Fetch data from MAST. Files land in the target's raw directory (or `dest`)."""

    enabled: bool = True
    reference_products: bool = Field(
        default=False,
        description="Opt-in: also fetch MAST's own calibrated products (level-3 s3d/x1d/i2d and "
        "level-2 cal/s3d/x1d) of the same observations into <target>/mast_reference/<run>/, mirroring "
        "the run's stage/step layout, in the background while the run proceeds, so the `mast_compare` "
        "QA step can difference them against jwstflow's products. Headers (CAL_VER, CRDS_CTX) and "
        "provenance.json carry the archive provenance.",
    )
    reference_product_types: list[str] = Field(
        default_factory=lambda: ["S3D", "X1D", "I2D", "CAL"],
        description="MAST productSubGroupDescription values fetched when reference_products is on.",
    )
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
    run: str | None = Field(
        default=None,
        description="Take `stage` from a sibling run of the same target (e.g. run: nirspec_ifu) "
        "instead of this run: the way level-4 workflows combine instruments.",
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

    exptype: Literal["background", "imprint", "psf", "target_acquisition", "selfcal", "sourcecat"] = "background"
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
        default="jw{PROGRAM}-o{OBSERVTN}_{TARGID}_{INSTRUME}",
        description="(group mode) Product name template over header keywords, following the DMS "
        "level-3 convention jw<PPPPP>-o<OOO>_<tTTT>_<instrument>: {TARGID} is the MAST target id "
        "recorded at download time (falls back to a slug of TARGPROP), {INSTRUME} is lower-cased "
        "and cube_build appends the band / grating-filter itself. {OPTELEM} expands to "
        "'g395h-f290lp' / 'ch1-short' when a product name needs it (NIRSpec IFU: use _{GRATING}).",
    )
    relative_paths: bool = Field(
        default=False,
        description="Write member paths relative to the association file. Off by default: "
        "calwebb_spec2 opens level-2 members relative to the working directory, not the "
        "association, so absolute paths are the only form that works everywhere.",
    )
    official_rules: Literal["level2", "level3"] | None = Field(
        default=None,
        description="(official mode) which rule set to use. Defaults from `level`.",
    )


class StageConfig(_Base):
    """One unit of the workflow: a jwst pipeline/step or a user-defined step."""

    step: str = Field(
        description="What to run. Either a built-in alias (detector1, image2, spec2, image3, "
        "spec3, tso3, assign_wcs, ...), a name registered through the 'jwstflow.steps' entry "
        "point, or a dotted path 'my_pkg.module:MyStep' / 'my_pkg.module.my_function'. "
        "Any stpipe Step/Pipeline subclass works, so do subclasses of the official pipelines."
    )
    variant: str | None = Field(
        default=None,
        description="Short slug that distinguishes several stages using the same step "
        "(e.g. `pass1`); the stage is then named `<step name>-<variant>`.",
    )
    name: str = Field(
        default="",
        description="Derived: the step's canonical name (stpipe class_alias such as "
        "`calwebb_spec3`, or the jwstflow Step's `name`) plus `-<variant>`. Not settable in YAML.",
    )
    level: int | str = Field(
        default=4,
        description="Derived from the step: 1, 2, 3 (jwst calibration stages), 4 (derived "
        "products) or 'qa'; selects the stage directory (`stage1/`, ..., `qa/`). Not settable in YAML.",
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

    @field_validator("variant")
    @classmethod
    def _valid_variant(cls, v: str | None) -> str | None:
        if v is not None and not re.fullmatch(r"[a-z0-9][a-z0-9_]*", v):
            raise ValueError(f"variant {v!r} must be a lowercase slug (letters, digits, underscores)")
        return v

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if v and (any(c in v for c in " /\\:") or v == RAW_STAGE):
            raise ValueError(f"stage name {v!r} must not contain spaces or slashes, and not be {RAW_STAGE!r}")
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
    target: str = Field(
        description="Target label, e.g. 'ESO-Ha 569'. Names the target directory (slugified) "
        "and, when a step needs coordinates, is resolved by name (SIMBAD/Sesame) unless `target_coords` is set."
    )
    run: str = Field(description="Run label within the target, e.g. 'nirspec_ifu' or 'miri_mrs'. Slug.")
    target_coords: dict[str, Any] | None = Field(
        default=None,
        description="Optional explicit position {ra, dec} (deg or sexagesimal) when the target "
        "is not a catalogue object; otherwise the name is resolved.",
    )
    workspace: Path | None = Field(
        default=None,
        description="Directory holding all targets: <workspace>/<target>/<run>/. Default: "
        "<project root>/reductions, where the project root is the nearest ancestor of the YAML "
        "with .git, pyproject.toml, uv.lock or a .jwstflow-root marker.",
    )
    root: Path | None = Field(
        default=None,
        description="Explicit run directory (overrides workspace/target/run). Its parent is the target directory.",
    )
    description: str | None = None
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Extra environment variables exported to all workers.",
    )
    env_file: list[Path] = Field(
        default_factory=list,
        description=(
            "Extra dotenv file(s), relative to this YAML. Normally unnecessary: every `.env` and "
            "`.env.*` file in the project root (the nearest ancestor with .git/pyproject.toml/uv.lock "
            "or a .jwstflow-root marker, of the YAML and of the working directory) is loaded "
            "automatically, plus $JWSTFLOW_ENV_FILE."
        ),
    )
    env_file_override: bool = Field(
        default=True,
        description="Values from env files replace variables already set in the shell (default). "
        "Set false to let the shell win, as plain python-dotenv does.",
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

    @field_validator("root", "workspace", mode="before")
    @classmethod
    def _expand_root(cls, v: Any) -> Any:
        return Path(str(v)).expanduser() if v is not None else v

    @field_validator("run")
    @classmethod
    def _valid_run(cls, v: str) -> str:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_\-]*", v):
            raise ValueError(f"run {v!r} must be a lowercase slug (letters, digits, '_' or '-')")
        return v

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
        if any(not n for n in names):
            raise ValueError("stage names are derived from the step by the loader; use load_config/config_from_dict")
        known = set(names) | {RAW_STAGE}
        for st in self.stages:
            for inp in st.inputs:
                if inp.run is not None:
                    continue  # another run of the same target: checked when the stage runs
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

    @property
    def target_slug(self) -> str:
        return slugify(self.target)

    @property
    def name(self) -> str:
        """Label used in logs and manifests."""
        return f"{self.target_slug}/{self.run}"

    @property
    def run_dir(self) -> Path:
        if self.root is not None:
            return self.root.expanduser()
        if self.workspace is None:
            raise ValueError("workspace is unset (the loader fills it in); pass base_dir to config_from_dict")
        return self.workspace.expanduser() / self.target_slug / self.run

    @property
    def target_dir(self) -> Path:
        return self.run_dir.parent

    @property
    def reference_dir(self) -> Path:
        """MAST's own products for this run (opt-in download): <target>/mast_reference/<run>/,
        with the same stage/step sub-directories as the run itself."""
        return self.target_dir / "mast_reference" / self.run

    @property
    def raw_dir(self) -> Path:
        if self.download is not None and self.download.dest is not None:
            return self.download.dest.expanduser()
        return self.target_dir / RAW_STAGE

    def stage_dir(self, name: str) -> Path:
        if name == RAW_STAGE:
            return self.raw_dir
        st = self.stage(name)
        return self.run_dir / LEVEL_DIRS.get(st.level, "stage4") / st.name

    def sibling_stage_dir(self, run: str, stage: str) -> Path:
        """Output directory of a stage in another run of the same target (any level)."""
        base = self.target_dir / run
        if stage == RAW_STAGE:
            return self.raw_dir
        level_dirs = set(LEVEL_DIRS.values())
        hits = sorted(p for p in base.glob(f"*/{stage}") if p.is_dir() and p.parent.name in level_dirs)
        if not hits:
            raise FileNotFoundError(f"run {run!r} of target {self.target_slug!r} has no stage {stage!r} under {base}")
        return hits[0]

    @property
    def state_dir(self) -> Path:
        return self.run_dir / ".jwstflow"

    @property
    def log_dir(self) -> Path:
        return self.run_dir / "logs"

    @property
    def asn_dir(self) -> Path:
        return self.run_dir / "associations"


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
