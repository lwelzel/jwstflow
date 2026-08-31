"""Command line interface (``jwstflow --help``)."""

from __future__ import annotations

import os

import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from . import __version__
from .config import Config, ConfigError, list_presets, load_config, preset_path
from .engine.graph import select_stages
from .engine.runner import Runner
from .engine.state import StateStore
from .steps.base import activate_plugins, describe_target, registered_steps

app = typer.Typer(
    help="Lightweight YAML-driven orchestrator for the JWST calibration pipeline.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="markdown",
)
console = Console()
log = logging.getLogger("jwstflow")
err = Console(stderr=True)

ConfigArg = Annotated[Path, typer.Argument(help="YAML configuration file.", exists=True, dir_okay=False)]
SetOpt = Annotated[
    list[str] | None,
    typer.Option("--set", "-s", help="Override `key.path=value` (YAML values). Stages by name: stages.spec2.parameters.steps.cube_build.skip=true"),
]


def setup_logging(level: str, log_file: Path | None = None) -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(RichHandler(console=console, show_path=False, rich_tracebacks=False, markup=False))
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(fh)
    logging.captureWarnings(True)


def _load(config: Path, overrides: list[str] | None) -> Config:
    try:
        return load_config(config, overrides)
    except ConfigError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None


@app.callback()
def _main(version: Annotated[bool, typer.Option("--version", help="Show version and exit.")] = False) -> None:
    if version:
        console.print(f"jwstflow {__version__}")
        raise typer.Exit()


# ---------------------------------------------------------------------------


@app.command()
def run(
    config: ConfigArg,
    set_: SetOpt = None,
    only: Annotated[list[str] | None, typer.Option("--only", help="Run only these stage(s).")] = None,
    start: Annotated[str | None, typer.Option("--from", help="Start at this stage (skips download).")] = None,
    until: Annotated[str | None, typer.Option("--until", help="Stop after this stage.")] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag", help="Only stages carrying this tag.")] = None,
    force: Annotated[bool, typer.Option("--force", "-f", help="Ignore checkpoints.")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", "-n", help="Plan only, run nothing.")] = False,
    workers: Annotated[int | None, typer.Option("--workers", "-w", help="Override parallel.workers.")] = None,
    skip_download: Annotated[bool, typer.Option("--skip-download", help="Do not contact MAST.")] = False,
    task: Annotated[
        list[str] | None,
        typer.Option("--task", help="Only tasks whose label matches this glob (repeatable), e.g. '*g395h*'."),
    ] = None,
    log_level: Annotated[str | None, typer.Option("--log-level", help="DEBUG/INFO/WARNING")] = None,
) -> None:
    """Run the workflow (resumes automatically from checkpoints)."""
    setup_logging(log_level or "INFO")
    log.info("jwstflow run: loading %s", config)
    cfg = _load(config, set_)
    setup_logging(log_level or cfg.logging.level, cfg.log_dir / "jwstflow.log")
    log.info("target %r, run %r -> %s", cfg.target, cfg.run, cfg.run_dir)
    log.info("stages: %s", " -> ".join(st.name for st in cfg.stages if st.enabled))
    runner = Runner(
        cfg, force=force, dry_run=dry_run, only=only, start=start, until=until, tags=tag,
        workers=workers, skip_download=skip_download, tasks=task,
    )
    try:
        summary = runner.run()
    except (ValueError, RuntimeError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    table = Table(title=f"jwstflow run: {cfg.name}" + (" (dry run)" if dry_run else ""))
    for col in ("stage", "tasks", "cached", "ok", "failed", "seconds"):
        table.add_column(col, justify="right" if col != "stage" else "left")
    for s in summary.stages:
        table.add_row(s.stage, str(s.total), str(s.cached), str(s.success), str(s.failed), f"{s.seconds:.0f}")
    console.print(table)
    for s in summary.stages:
        for label, msg in s.failures:
            err.print(f"[red]{s.stage}/{label}[/red]: {msg}")
    raise typer.Exit(0 if summary.ok else 1)


@app.command()
def plan(config: ConfigArg, set_: SetOpt = None, only: Annotated[list[str] | None, typer.Option("--only")] = None) -> None:
    """Show the tasks each stage would run (and which are cached)."""
    cfg = _load(config, set_)
    setup_logging("WARNING")
    runner = Runner(cfg, only=only, dry_run=True)
    try:
        plan = runner.plan()
    except (ValueError, RuntimeError) as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from None
    for stage_name, tasks in plan.items():
        st = cfg.stage(stage_name)
        table = Table(title=f"{stage_name}  [{st.step}]  -> {cfg.stage_dir(stage_name)}")
        table.add_column("task")
        table.add_column("status")
        table.add_column("inputs")
        for t in tasks:
            inputs = ", ".join(p.name for p in t.inputs[:3]) + (" ..." if len(t.inputs) > 3 else "")
            table.add_row(t.label, "[green]cached[/green]" if t.cached else "[yellow]pending[/yellow]", inputs)
        if not tasks:
            table.add_row("-", "[red]no inputs[/red]", "")
        console.print(table)


@app.command()
def status(config: ConfigArg, set_: SetOpt = None, failed: Annotated[bool, typer.Option("--failed", help="List failed tasks.")] = False) -> None:
    """Summarise recorded task results per stage."""
    cfg = _load(config, set_)
    store = StateStore(cfg.state_dir / "state")
    table = Table(title=f"{cfg.name}  ({cfg.run_dir})")
    for col in ("stage", "success", "failed", "last finished"):
        table.add_column(col)
    for st in select_stages(cfg):
        recs = store.records(st.name)
        ok = [r for r in recs if r.status == "success"]
        bad = [r for r in recs if r.status == "failed"]
        last = max((r.finished or "" for r in recs), default="-")
        table.add_row(st.name, str(len(ok)), f"[red]{len(bad)}[/red]" if bad else "0", last)
        if failed:
            for r in bad:
                console.print(f"  [red]{st.name}/{r.label}[/red]: {r.error}\n    log: {r.log_file}")
    console.print(table)
    manifest = cfg.state_dir / "manifest.json"
    if manifest.exists():
        runs = json.loads(manifest.read_text()).get("runs", [])
        if runs:
            last_run = runs[-1]
            console.print(
                f"last run {last_run['time']}  jwst={last_run.get('jwst')}  "
                f"CRDS_CONTEXT={last_run.get('crds', {}).get('pinned_context') or last_run.get('crds', {}).get('CRDS_CONTEXT') or 'server default'}"
            )


@app.command()
def validate(config: ConfigArg, set_: SetOpt = None, resolve: Annotated[bool, typer.Option("--resolve", help="Also import every step (needs jwst installed).")] = False) -> None:
    """Validate a configuration file (schema, stage graph, optionally step imports)."""
    cfg = _load(config, set_)
    try:
        stages = select_stages(cfg)
    except ValueError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    console.print(f"[green]OK[/green] {config}: {len(stages)} stage(s): " + " -> ".join(s.name for s in stages))
    for f in cfg.env_file:
        console.print(f"  environment loaded from {f}")
    if cfg.download is not None:
        tok = os.environ.get(cfg.download.token_env, "").strip()
        state = f"set ({len(tok)} chars)" if tok else "not set -> anonymous (public data only)"
        console.print(f"  MAST token ${cfg.download.token_env}: {state}")
    console.print(f"  CRDS: path={cfg.crds.path}  server={cfg.crds.server_url}  context={cfg.crds.context}")
    if resolve:
        activate_plugins(cfg.plugins)
        bad = 0
        for st in stages:
            try:
                d = describe_target(st.step)
                shown = st.step if len(st.step) <= 40 else "..." + st.step[-37:]
                console.print(f"  {st.name:14s} {shown:40s} {d['kind']:8s} {d['object']}")
            except Exception as exc:
                bad += 1
                console.print(f"  {st.name:14s} {st.step:40s} [red]cannot resolve: {exc}[/red]")
        if bad:
            raise typer.Exit(2)


@app.command()
def download(config: ConfigArg, set_: SetOpt = None, dry_run: Annotated[bool, typer.Option("--dry-run", "-n")] = False) -> None:
    """Only download the data described in the `download:` section."""
    cfg = _load(config, set_)
    setup_logging(cfg.logging.level)
    if cfg.download is None:
        err.print("[red]no `download:` section in config[/red]")
        raise typer.Exit(2)
    runner = Runner(cfg, dry_run=dry_run)
    runner.prepare()
    files = runner.download()
    console.print(f"{len(files)} file(s) in {cfg.stage_dir('raw')}")


@app.command()
def asn(config: ConfigArg, stage: Annotated[str, typer.Argument(help="Stage with an `association:` section.")], set_: SetOpt = None) -> None:
    """Build (and print) the association files of one stage without running it."""
    cfg = _load(config, set_)
    setup_logging("WARNING")
    runner = Runner(cfg, dry_run=True)
    runner.prepare()
    try:
        st = cfg.stage(stage)
    except KeyError:
        err.print(f"[red]unknown stage {stage!r}[/red]")
        raise typer.Exit(2) from None
    if st.association is None:
        err.print(f"[red]stage {stage!r} has no association section[/red]")
        raise typer.Exit(2)
    from .associations import summarize

    tasks = runner.plan_stage(st)
    for t in tasks:
        data = json.loads(t.inputs[0].read_text())
        console.print(f"[bold]{t.inputs[0]}[/bold]\n  {summarize(data)}")
    console.print(f"{len(tasks)} association(s) in {cfg.asn_dir / stage}")


@app.command()
def schema(
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write JSON Schema to this file.")] = None,
) -> None:
    """Print the JSON Schema of the YAML format (for editor autocompletion).

    Add `# yaml-language-server: $schema=jwstflow.schema.json` at the top of a YAML file.
    """
    js = json.dumps(Config.model_json_schema(), indent=2)
    if output:
        output.write_text(js)
        console.print(f"wrote {output}")
    else:
        sys.stdout.write(js + "\n")


@app.command()
def init(
    preset: Annotated[str | None, typer.Argument(help="Preset name (omit to list).")] = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write here instead of stdout.")] = None,
    extend: Annotated[bool, typer.Option("--extend", help="Write a short file that `extends` the preset instead of a full copy.")] = False,
) -> None:
    """Start a new configuration from a bundled preset."""
    if preset is None:
        for name in list_presets():
            console.print(name)
        return
    try:
        src = preset_path(preset)
    except ConfigError as exc:
        err.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from None
    if extend:
        text = (
            f"# yaml-language-server: $schema=jwstflow.schema.json\n"
            f"extends: preset:{preset}\n"
            f"target: My Target          # resolved by name when a step needs coordinates; slug names the directory\n"
            f"run: {preset}              # outputs: <project root>/reductions/<target>/{preset}/\n"
            f"download:\n  program: 1234\n  observations: [1]\n"
            f"# Override anything from the preset below; stages are addressed by step (names are derived), e.g.:\n"
            f"# stages:\n#   - step: detector1\n#     parameters: {{steps: {{jump: {{rejection_threshold: 5}}}}}}\n"
        )
    else:
        text = src.read_text()
    if output:
        output.write_text(text)
        console.print(f"wrote {output}")
    else:
        sys.stdout.write(text)


@app.command()
def graph(
    config: Annotated[Path, typer.Argument(help="Workflow YAML.")],
    out: Annotated[Path | None, typer.Option("-o", "--out", help="Output directory (default: the run's qa/workflow_graph).")] = None,
    fmt: Annotated[str, typer.Option("--format", help="Comma-separated figure formats matplotlib can save, e.g. svg,pdf,png.")] = "svg,pdf",
    set_: Annotated[list[str] | None, typer.Option("--set", help="Override config values (key=value).")] = None,
) -> None:
    """Render the workflow DAG (data patterns -> steps -> products) without running anything."""
    from .dagviz import render
    from .steps.base import LEVEL_DIRS

    setup_logging("WARNING")
    cfg = _load(config, set_)
    out_dir = out or cfg.run_dir / LEVEL_DIRS["qa"] / "workflow_graph"
    files = render(cfg, out_dir, formats=tuple(f.strip() for f in fmt.split(",") if f.strip()))
    for f in files:
        console.print(str(f))


@app.command()
def prefetch(
    config: Annotated[Path, typer.Argument(help="Workflow YAML.")],
    set_: Annotated[list[str] | None, typer.Option("--set", help="Override config values (key=value).")] = None,
    log_level: Annotated[str | None, typer.Option("--log-level")] = None,
) -> None:
    """Download raw data, sync CRDS references and fetch MAST reference products -- no stages run.

    A later `jwstflow run` then starts computing immediately (handy before batch jobs or overnight).
    """
    setup_logging(log_level or "INFO")
    log.info("jwstflow prefetch: loading %s", config)
    cfg = _load(config, set_)
    setup_logging(log_level or cfg.logging.level, cfg.log_dir / "jwstflow.log")
    Runner(cfg).prefetch()


@app.command("check-step")
def check_step_cmd(
    spec: Annotated[list[str], typer.Argument(help="Step spec(s): name, pkg.module:Object or ./file.py:Object")],
    plugins: Annotated[list[Path] | None, typer.Option("--plugins", help="Files/directories with step code.")] = None,
) -> None:
    """Audit custom steps against the step contract without running them."""
    from .testing import check_step

    activate_plugins(plugins or [])
    bad = 0
    for one in spec:
        desc, problems = check_step(one)
        kind = desc.get("kind", "?")
        console.print(f"[bold]{one}[/bold]  ({kind}" + (f", stage '{desc['name']}', level {desc['level']}, batch {desc['batch']}" if kind == "step" else "") + ")")
        if kind == "step":
            console.print(f"  {desc.get('doc') or '(no docstring)'}")
            console.print(f"  inputs {desc['inputs'] or '(undeclared)'}  outputs {desc['outputs'] or '(undeclared)'}  version {desc['version']}")
            for prm in desc["params"]:
                console.print(f"  - {prm['name']}: {prm['type']} = {prm['default']!r}  {prm['description']}")
        for problem in problems:
            console.print(f"  [red]problem:[/red] {problem}")
            bad += 1
        if not problems:
            console.print("  [green]OK[/green]")
    if bad:
        raise typer.Exit(1)


@app.command("new-step")
def new_step(
    class_name: Annotated[str, typer.Argument(help="CamelCase class name, e.g. ExtractExtended")],
    directory: Annotated[Path, typer.Option("--dir", help="Where to write <snake>.py and test_<snake>.py")] = Path("."),
    level: Annotated[str, typer.Option("--level", help="1, 2, 3, 4 or qa")] = "4",
    inputs: Annotated[str, typer.Option("--inputs", help="Accepted input glob")] = "*_cal.fits",
    suffix: Annotated[str | None, typer.Option("--suffix", help="Product suffix written (lowercase alphanumeric)")] = None,
) -> None:
    """Scaffold a custom step (documented class + a passing test) following the step contract."""
    from .testing import scaffold

    lvl: int | str = int(level) if level.isdigit() else level
    for f in scaffold(class_name, directory, level=lvl, inputs=inputs, suffix=suffix):
        console.print(f"wrote {f}")
    console.print(f"next: edit the run() method, then `python -m pytest {directory}` and `jwstflow check-step {directory}/<module>.py:{class_name}`")


@app.command("new-package")
def new_package(
    name: Annotated[str, typer.Argument(help="Distribution name, e.g. jwstflow-mysteps (lowercase-with-dashes)")],
    directory: Annotated[Path, typer.Option("--dir", help="Parent directory; the package is written to <dir>/<name>")] = Path("."),
    steps: Annotated[str | None, typer.Option("--steps", help="Comma-separated snake_case step names (default: one stub named after the package)")] = None,
    private: Annotated[bool, typer.Option("--private", help="Proprietary flavour: restrictive LICENSE, 'Private :: Do Not Upload' classifier, private-repo instructions")] = False,
    git: Annotated[bool, typer.Option("--git/--no-git", help="Initialise a git repository with an initial commit")] = True,
) -> None:
    """Scaffold a contributed step package: pyproject with `jwstflow.steps` entry points, src layout,
    step stubs, a passing declaration test, README (with a publish-to-GitHub walkthrough), CI and git.
    Only scaffolding is generated -- each step's run() awaits your science."""
    from .scaffold import scaffold_package

    step_names = [s.strip() for s in steps.split(",") if s.strip()] if steps else None
    try:
        root, files, notes = scaffold_package(name, directory, steps=step_names, private=private, git=git)
    except (ValueError, FileExistsError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc
    for f in files:
        console.print(f"wrote {f.relative_to(directory) if f.is_relative_to(directory) else f}")
    for note in notes:
        console.print(f"[yellow]{note}[/yellow]")
    console.print(
        f"\nnext:\n"
        f"  cd {root}\n"
        f"  uv sync && uv run python -m pytest     # green before any science exists\n"
        f"  edit src/{name.replace('-', '_')}/steps.py   # implement each step's run()\n"
        f"  README.md walks through publishing to GitHub ({'private' if private else 'public'})"
    )


@app.command()
def steps() -> None:
    """List step names usable in `step:` (built-in aliases, entry points, registry)."""
    table = Table(title="registered steps")
    table.add_column("name")
    table.add_column("target")
    for name, target in registered_steps().items():
        table.add_row(name, str(target))
    console.print(table)
    console.print("Any dotted path `pkg.module:Object` works as well.")


@app.command()
def clean(
    config: ConfigArg,
    set_: SetOpt = None,
    stage: Annotated[list[str] | None, typer.Option("--stage", help="Only these stage(s).")] = None,
    outputs: Annotated[bool, typer.Option("--outputs", help="Also delete the stage output directories.")] = False,
    orphans: Annotated[
        bool, typer.Option("--orphans", help="Only delete files no current task produced; keep checkpoints.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Forget checkpoints (and optionally delete outputs) so stages rerun."""
    cfg = _load(config, set_)
    store = StateStore(cfg.state_dir / "state")
    targets = stage or [s.name for s in cfg.stages]
    if orphans:
        setup_logging("WARNING")
        runner = Runner(cfg, dry_run=True)
        runner.prepare()
        for name in targets:
            files = runner.orphans(cfg.stage(name))
            if files and not yes:
                console.print("\n".join(f"  {f}" for f in files))
                typer.confirm(f"{name}: delete these {len(files)} file(s)?", abort=True)
            for f in files:
                f.unlink()
            console.print(f"{name}: removed {len(files)} orphan file(s)")
        return
    if outputs and not yes:
        typer.confirm(f"delete outputs of {', '.join(targets)} under {cfg.run_dir}?", abort=True)
    for name in targets:
        n = store.clear(name)
        msg = f"{name}: forgot {n} record(s)"
        if outputs:
            d = cfg.stage_dir(name)
            if d.exists():
                shutil.rmtree(d)
                msg += f", removed {d}"
        console.print(msg)


if __name__ == "__main__":  # pragma: no cover
    app()
