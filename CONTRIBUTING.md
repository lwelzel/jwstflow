# Contributing to jwstflow

Thank you for considering a contribution to `jwstflow`.

`jwstflow` is a lightweight, YAML-driven orchestrator for reproducible reductions with the official JWST calibration pipeline. Contributions that improve correctness, reproducibility, documentation, testing, usability, or support for well-defined workflow patterns are welcome.

## Ways to contribute

Useful contributions include:

- bug reports with a minimal reproducible example;
- fixes for incorrect or fragile behaviour;
- tests that cover currently untested behaviour;
- documentation improvements;
- improvements to workflow validation, provenance, checkpointing, associations, or execution;
- support for additional observing modes when the behaviour is sufficiently general for the core package;
- improvements to the plugin/custom-step interface; and
- examples that demonstrate a scientifically realistic workflow without including proprietary data or credentials.

Instrument-, programme-, or science-case-specific processing steps will often fit better in a separate package that registers steps through the `jwstflow.steps` entry-point group. If you are unsure whether a contribution belongs in the core package, open an issue before doing substantial implementation work.

## Where code goes

The package layout is described in [`docs/guide.md`](docs/guide.md) ("How the package is laid out"). In short:

- a subsystem is a package (`config`, `engine`, `data`, `steps`, `qa`, `scaffold`, `testing`); code used by one subsystem only belongs inside it, not at the top level;
- a data step is defined with the feature it implements (a module, or `<feature>/steps.py` once the feature has a library of its own), a QA step in `jwstflow/qa/steps/`; `jwstflow/steps/` is the framework and defines no step;
- a new bundled step is registered twice, under the same name: as an entry point in `pyproject.toml` and in `BUNDLED_STEPS` (`jwstflow/steps/registry.py`); a test keeps the two lists equal;
- the toolkit modules step packages import (`naming`, `spectra`, `masks`, `apcorr`, `features`, `targets`, `stitching`) keep their paths;
- a step's checkpoints depend on the content of the file that defines it: moving such a file is free, editing it reruns the step in every existing reduction -- so keep refactoring of step modules separate from functional changes.

## Development setup

`jwstflow` requires Python 3.12 or later and uses `uv` for dependency management.

```bash
git clone https://github.com/lwelzel/jwstflow.git
cd jwstflow
uv sync --group dev
```

Run the test suite with:

```bash
uv run python -m pytest
```

For a faster development loop that excludes the mock-observation integration tests:

```bash
uv run python -m pytest -m "not integration"
```

Run the integration tests explicitly with:

```bash
uv run python -m pytest -m integration
```

Run the linter with:

```bash
uv run ruff check .
```

The testing strategy is described in [`docs/testing.md`](docs/testing.md).

## Before opening a pull request

Please make sure that:

1. the change has a clear purpose and does not mix unrelated refactoring with functional changes;
2. new behaviour is covered by tests where practical;
3. existing tests pass locally;
4. `ruff` reports no new issues;
5. user-facing behaviour is documented;
6. public interfaces remain backward compatible unless the change is intentionally breaking and clearly documented; and
7. no credentials, private data, proprietary observing products, or local machine paths are committed.

For numerical or scientific behaviour, tests should compare against independently constructed expectations or known physical/scientific properties rather than reproducing the implementation logic in the assertion.

For workflow or engine changes, prefer deterministic tests. The mock-observation utilities in `jwstflow.testing.mock` are intended to exercise realistic workflow structure while remaining offline and reproducible.

## Pull requests

A pull request should explain:

- what problem it solves;
- the approach taken;
- any user-visible or scientific behaviour that changes;
- how the change was tested; and
- any limitations or follow-up work that remains.

Small, focused pull requests are easier to review than large changes spanning unrelated parts of the package.

If a change affects configuration syntax, task planning, product naming, checkpoint semantics, associations, plugin contracts, or provenance, include a regression test that demonstrates the intended behaviour.

## Reporting bugs

When opening a bug report, include enough information to reproduce the problem where possible:

- `jwstflow` version or commit;
- Python version;
- `jwst` pipeline version;
- relevant CRDS context when applicable;
- the smallest workflow configuration that reproduces the issue;
- the command that was run; and
- the traceback or unexpected result.

Do not post MAST tokens, CRDS credentials, `.env` contents, proprietary observations, or other sensitive information in an issue.

## Security-sensitive reports

If a problem could expose credentials, private data, or another security-sensitive resource, do not open a public issue. Contact the maintainer directly at `welzel@strw.leidenuniv.nl` with enough information to investigate the problem.

## Documentation

Documentation changes are part of the software change. If a pull request adds or changes a command, configuration field, workflow behaviour, supported mode, or extension point, update the relevant README or document under `docs/` in the same pull request.

## Licensing

By contributing to this repository, you agree that your contribution will be distributed under the repository's BSD 3-Clause License.

## Conduct

Participation in the project is subject to [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
