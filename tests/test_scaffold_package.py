"""Tests of the contributed-package generator (jwstflow new-package)."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from jwstflow.scaffold import scaffold_package
from jwstflow.steps.base import import_file
from jwstflow.testing import check_step


def test_generated_package_is_complete_and_valid(tmp_path: Path):
    root, files, notes = scaffold_package("jwstflow-demo", tmp_path, steps=["defringe", "stitch_bands"], git=False)
    assert root == tmp_path / "jwstflow-demo"
    rel = {str(f.relative_to(root)) for f in files}
    assert rel == {"pyproject.toml", "README.md", "LICENSE", ".gitignore",
                   "src/jwstflow_demo/__init__.py", "src/jwstflow_demo/steps.py",
                   "tests/test_declarations.py", ".github/workflows/test.yml"}
    # pyproject parses, depends on jwstflow, and wires the entry points to the generated classes
    meta = tomllib.loads((root / "pyproject.toml").read_text())
    assert any(d.startswith("jwstflow") for d in meta["project"]["dependencies"])
    eps = meta["project"]["entry-points"]["jwstflow.steps"]
    assert eps == {"defringe": "jwstflow_demo.steps:Defringe", "stitch_bands": "jwstflow_demo.steps:StitchBands"}
    # the generated steps import and pass the step-contract audit before any science exists
    module = import_file(root / "src/jwstflow_demo/steps.py")
    for ep, dotted in eps.items():
        cls = getattr(module, dotted.rpartition(":")[2])
        assert cls.name == ep  # canonical stage name == entry-point name
        _, problems = check_step(cls)
        assert problems == [], f"{cls.__name__}: {problems}"
        with pytest.raises(NotImplementedError):
            cls().run([], None)
    # public flavour: no do-not-upload classifier, MIT placeholder license
    assert "Private :: Do Not Upload" not in meta["project"]["classifiers"]
    assert "MIT" in (root / "LICENSE").read_text()


def test_private_flavour(tmp_path: Path):
    root, _, _ = scaffold_package("jwstflow-secret", tmp_path, private=True, git=False)
    meta = tomllib.loads((root / "pyproject.toml").read_text())
    assert "Private :: Do Not Upload" in meta["project"]["classifiers"]
    assert "Proprietary" in (root / "LICENSE").read_text()
    assert "--private" in (root / "README.md").read_text()
    # default step is named after the package
    assert meta["project"]["entry-points"]["jwstflow.steps"] == {"secret_step": "jwstflow_secret.steps:SecretStep"}


def test_reserved_suffixes_are_avoided(tmp_path: Path):
    root, _, _ = scaffold_package("jwstflow-x", tmp_path, steps=["cal"], git=False)  # 'cal' is an official suffix
    module = import_file(root / "src/jwstflow_x/steps.py")
    assert module.Cal.outputs == ("calx",)


def test_bad_inputs_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="lowercase-with-dashes"):
        scaffold_package("Jwstflow_Bad", tmp_path, git=False)
    with pytest.raises(ValueError, match="lowercase slug"):
        scaffold_package("jwstflow-ok", tmp_path, steps=["Bad-Name"], git=False)
    scaffold_package("jwstflow-dup", tmp_path, git=False)
    with pytest.raises(FileExistsError):
        scaffold_package("jwstflow-dup", tmp_path, git=False)
