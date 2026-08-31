"""Scaffolding for contributed step packages (``jwstflow new-package``).

Generates the complete boilerplate of a contributed package -- pyproject with
entry points, src layout, step stubs, a passing declaration test, README with
a publish-to-GitHub walkthrough, .gitignore, LICENSE, CI workflow -- and
initialises a git repository, so an author starts at "implement run()" instead
of at packaging. Only scaffolding is generated: each step's ``run()`` raises
NotImplementedError until the science is written.

The templates live here, inside the core package, so generated boilerplate
always matches the plugin contract of the installed jwstflow version.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from string import Template

from .naming import JWST_PRODUCT_SUFFIXES

log = logging.getLogger(__name__)

_NAME_RE = re.compile(r"^[a-z][a-z0-9]*(-[a-z0-9]+)*$")
_STEP_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def scaffold_package(name: str, directory: str | Path = ".", *, steps: list[str] | None = None,
                     private: bool = False, git: bool = True,
                     jwstflow_url: str = "https://github.com/lwelzel/jwstflow") -> tuple[Path, list[Path], list[str]]:
    """Write a contributed-package skeleton; returns ``(package_dir, files, notes)``.

    ``name`` is the distribution name (``jwstflow-mysteps``); the module name is
    derived from it. ``steps`` are snake_case step names (default: one stub named
    after the package); ``private`` selects the proprietary flavour (license,
    ``Private :: Do Not Upload`` classifier, private-repo instructions); ``git``
    initialises a repository with an initial commit when git is available.
    """
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"package name {name!r} must be lowercase-with-dashes, e.g. jwstflow-mysteps")
    notes: list[str] = []
    if not name.startswith("jwstflow-"):
        notes.append(f"note: the convention for contributed packages is a jwstflow- prefix (got {name!r})")
    module = name.replace("-", "_")
    short = module.removeprefix("jwstflow_") or module
    step_names = list(steps) if steps else [f"{short}_step"]
    for s in step_names:
        if not _STEP_RE.fullmatch(s):
            raise ValueError(f"step name {s!r} must be a lowercase slug (snake_case)")
    if len(set(step_names)) != len(step_names):
        raise ValueError(f"step names must be unique, got {step_names}")
    root = Path(directory).expanduser() / name
    if root.exists():
        raise FileExistsError(f"{root} already exists")

    defs = [_step_definition(s) for s in step_names]
    ctx = {
        "name": name, "module": module,
        "visibility": "private" if private else "public",
        "entry_points": "\n".join(f'{d["snake"]} = "{module}.steps:{d["cls"]}"' for d in defs),
        "classifiers_extra": '\n    "Private :: Do Not Upload",' if private else "",
        "step_list": "\n".join(f"    {d['snake']:24s}{d['cls']} (stub)" for d in defs),
        "step_rows": "\n".join(f"| `{d['snake']}` | stub — implement `run()` in `src/{module}/steps.py` |" for d in defs),
        "step_classes": ", ".join(d["cls"] for d in defs),
        "first_step": defs[0]["snake"],
        "jwstflow_url": jwstflow_url,
        "steps_code": "\n\n".join(_render(_STEP_TEMPLATE, {**d, "module": module}) for d in defs),
        "publish": _render(_PUBLISH_PRIVATE if private else _PUBLISH_PUBLIC, {"name": name}),
        "license_hint": "Proprietary (see LICENSE)" if private else "MIT (placeholder -- see LICENSE)",
    }
    files = {
        "pyproject.toml": _render(_PYPROJECT, ctx),
        "README.md": _render(_README, ctx),
        "LICENSE": _render(_LICENSE_PRIVATE if private else _LICENSE_PUBLIC, ctx),
        ".gitignore": _GITIGNORE,
        f"src/{module}/__init__.py": _render(_INIT, ctx),
        f"src/{module}/steps.py": _render(_STEPS_HEADER, ctx) + "\n\n" + ctx["steps_code"],
        "tests/test_declarations.py": _render(_TEST, ctx),
        ".github/workflows/test.yml": _WORKFLOW,
    }
    written: list[Path] = []
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content.rstrip() + "\n")
        written.append(path)
    if git:
        notes += _git_init(root)
    return root, written, notes


def _step_definition(snake: str) -> dict[str, str]:
    cls = "".join(part.capitalize() for part in snake.split("_"))
    suffix = re.sub(r"[^a-z0-9]", "", snake)[:8] or "custom"
    if suffix in JWST_PRODUCT_SUFFIXES:  # never collide with an official product type
        suffix = (suffix + "x")[:8] if len(suffix) < 8 else suffix[:7] + "x"
    title = snake.replace("_", " ")
    return {"snake": snake, "cls": cls, "suffix": suffix, "title": title}


def _git_init(root: Path) -> list[str]:
    """Initialise a git repository with one commit; degrade gracefully without git."""
    if shutil.which("git") is None:
        return ["git not found: skipped `git init` (the README shows the manual commands)"]

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)

    if run("init", "-b", "main").returncode != 0 and run("init").returncode != 0:
        return ["`git init` failed: initialise the repository yourself (see README)"]
    run("add", "-A")
    commit = run("commit", "-m", f"Initial jwstflow contributed-package scaffold ({root.name})")
    if commit.returncode != 0:  # usually: user.name/user.email not configured
        return ["git repository initialised; the initial commit needs your git identity "
                "(git config user.name/user.email), then: git add -A && git commit"]
    return ["git repository initialised on `main` with an initial commit"]


def _render(template: str, ctx: dict[str, str]) -> str:
    return Template(template).substitute(ctx)


# --------------------------------------------------------------------------- templates
_PYPROJECT = """\
[project]
name = "$name"
version = "0.1.0"
description = "Contributed steps for jwstflow: describe what this package adds."
authors = [
    { name = "Your Name", email = "you@example.org" }
]
requires-python = ">=3.12"
keywords = ["jwst", "jwstflow"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",$classifiers_extra
    "Programming Language :: Python :: 3",
    "Topic :: Scientific/Engineering :: Astronomy",
]
# Everything heavy (astropy, scipy, jwst, matplotlib) arrives through jwstflow itself;
# add your own extras (photutils, regions, ...) here.
dependencies = [
    "jwstflow>=0.1",
]

# The whole integration: jwstflow discovers these on install, `jwstflow steps`
# lists them, and workflows reference them by these names.
[project.entry-points."jwstflow.steps"]
$entry_points

[build-system]
requires = ["uv_build>=0.12.1,<0.13.0"]
build-backend = "uv_build"

# Until jwstflow is on an index, resolve it from git (or a local checkout while developing):
[tool.uv.sources]
jwstflow = { git = "$jwstflow_url" }
# jwstflow = { path = "../jwstflow", editable = true }

[dependency-groups]
dev = [
    "pytest>=9.1.1",
    "ruff>=0.16.4",
]
"""

_INIT = '''\
"""$name: contributed steps for jwstflow.

Steps (registered in the ``jwstflow.steps`` entry-point group, referenced by
these names in workflow YAMLs):

$step_list

Scaffolding generated by `jwstflow new-package`; implement each step's
``run()`` in ``steps.py`` and bump its ``version`` whenever results change.
"""

__version__ = "0.1.0"
'''

_STEPS_HEADER = '''\
"""Steps of $name. Implement run() per step; the declarations below are the contract.

The docstring of each class is the step's description in `jwstflow steps`:
its first paragraph is the short summary (the table), the rest the detailed
explanation (`jwstflow steps <name>`).
See jwstflow's docs (custom steps, `jwstflow.masks`, `jwstflow.stitching`) for
the helpers and product contracts available to build on.
"""

from __future__ import annotations

import logging
from pathlib import Path

from jwstflow import RunContext, Step

log = logging.getLogger(__name__)
'''

_STEP_TEMPLATE = '''\
class $cls(Step):
    """$title: one line on what this step produces (the `jwstflow steps` table).

    Detailed description (`jwstflow steps $snake`): what run() computes, in
    order, and what the output product contains.
    """

    name = "$snake"                # canonical stage name == the entry-point name in pyproject.toml
    level = 4                      # 1/2/3 jwst stages, 4 derived products, "qa" plots
    batch = "per_file"             # "all": one task receives every input at once
    inputs = ("*_cal.fits",)       # accepted input files (glob) -- adjust to your product type
    outputs = ("$suffix",)         # product suffix(es) written; never an official jwst one
    version = "1"                  # bump whenever results change -> cached tasks rerun

    # Typed parameters (the workflow's `parameters:` block), validated at config load:
    #
    #     from pydantic import Field
    #     from jwstflow import StepParams
    #
    #     class Params(StepParams):
    #         threshold: float = Field(0.05, gt=0, description="what it does")
    #
    # and mirror them as keyword-only arguments of run().

    def run(self, inputs: list[Path], ctx: RunContext, **params) -> list[Path]:
        # YOUR SCIENCE HERE. Typical shape:
        #     (src,) = inputs
        #     out = ctx.derived_path(src, "$suffix")     # <input base>_$suffix.fits in the stage dir
        #     ... read src, compute, write out ...
        #     return [out]                               # return every file written (checkpointing)
        raise NotImplementedError("implement $snake ($module.steps:$cls)")
'''

_TEST = '''\
"""Every step's declaration is valid (checkable before the science is implemented)."""

from __future__ import annotations

from jwstflow.testing import check_step

from $module.steps import $step_classes


def test_declarations_are_valid():
    for cls in ($step_classes,):
        _, problems = check_step(cls)
        assert problems == [], f"{cls.__name__}: {problems}"


# Once run() is implemented, test it on synthetic data (no CRDS or network needed):
#
#     from jwstflow.testing import run_step, synthetic_cube, synthetic_image, synthetic_x1d
#
#     def test_${first_step}_runs(tmp_path):
#         src = synthetic_image(tmp_path / "jw001_nrs1_cal.fits")
#         outputs = run_step(<StepClass>, [src], tmp_path, params={})
#         assert outputs
'''

_README = """\
# $name

Contributed steps for [jwstflow]($jwstflow_url) ($visibility). License: $license_hint.

## Steps

Registered in the `jwstflow.steps` entry-point group on install; reference
them by these names in workflow YAMLs (`jwstflow steps` lists them):

| name | status |
|---|---|
$step_rows

## Develop

```bash
uv sync                       # env + jwstflow (see [tool.uv.sources]) + dev tools
uv run python -m pytest       # declaration tests are green before any science exists
uv run jwstflow check-step src/$module/steps.py:<Class>
```

Implement each step's `run()` in `src/$module/steps.py` (the scaffold raises
NotImplementedError), add typed `Params`, and test on synthetic data with
`jwstflow.testing.run_step` -- no CRDS or network needed. Bump a step's
`version` attribute whenever its results change: that is what invalidates
users' cached tasks.

While developing against a local jwstflow checkout, switch `[tool.uv.sources]`
to the commented path line.

$publish

## Use in a workflow

Users install this package next to jwstflow and reference the steps by name:

```yaml
stages:
  - step: $first_step
    inputs: [{stage: ..., pattern: "..."}]
    parameters: {}
```
"""

_PUBLISH_PRIVATE = """\
## Publish to GitHub (private)

This package is proprietary: keep the repository **private** (the
`Private :: Do Not Upload` classifier also guards against an accidental PyPI
upload). The scaffold already initialised git with an initial commit, so:

```bash
gh repo create <org>/$name --private --source=. --push     # with the GitHub CLI
```

or by hand: create an empty **private** repository on github.com, then

```bash
git remote add origin git@github.com:<org>/$name.git
git push -u origin main
```

Collaborators with access install it with

```bash
uv pip install git+ssh://git@github.com/<org>/$name
```

(and pin it in their project's `[tool.uv.sources]`). Nothing in jwstflow
references this package, so people without access are unaffected."""

_PUBLISH_PUBLIC = """\
## Publish to GitHub (public)

The scaffold already initialised git with an initial commit, so:

```bash
gh repo create <you>/$name --public --source=. --push      # with the GitHub CLI
```

or by hand: create an empty repository on github.com, then

```bash
git remote add origin git@github.com:<you>/$name.git
git push -u origin main
```

Users install it with

```bash
uv pip install git+https://github.com/<you>/$name
```

(and pin it in their project's `[tool.uv.sources]`). Once released on PyPI
(`uv build && uv publish`), plain `uv pip install $name` works too."""

_LICENSE_PUBLIC = """\
MIT License (placeholder -- replace holder/year, or choose another license)

Copyright (c) 2026 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

_LICENSE_PRIVATE = """\
Proprietary -- $name.

All rights reserved. This software is for use by authorised collaborators
only; do not redistribute, publish, or upload to package indexes. (Replace
with your collaboration's actual license text.)
"""

_GITIGNORE = """\
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
.pytest_cache/
.ruff_cache/
uv.lock
"""

_WORKFLOW = """\
name: tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
      - run: uv sync
      - run: uv run python -m pytest
"""
