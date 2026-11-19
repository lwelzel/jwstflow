"""`jwstflow steps` descriptions: the built-in table and docstring extraction."""

from __future__ import annotations

import textwrap
from pathlib import Path

from jwstflow.steps import (
    BUILTIN_ALIASES,
    Step,
    register_step,
    split_description,
    step_description,
)
from jwstflow.steps.descriptions import BUILTIN_DESCRIPTIONS
from jwstflow.steps.registry import _REGISTRY


def test_every_builtin_alias_is_described():
    for alias in BUILTIN_ALIASES:
        short, long = BUILTIN_DESCRIPTIONS.get(alias, ("", ""))
        assert short, f"{alias}: missing short description"
        assert long, f"{alias}: missing detailed description"
        # a description, not the name again
        assert short.lower() != alias.lower()
        assert len(short.split()) >= 5, f"{alias}: short description too thin"


def test_builtin_lookup():
    short, long = step_description("detector1")
    assert "ramp" in short.lower()
    assert "\n" in long  # multi-line detail


def own_steps() -> dict[str, str]:
    """jwstflow's own steps as pyproject.toml registers them: name -> module:object (the one list)."""
    import tomllib

    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())
    return pyproject["project"]["entry-points"]["jwstflow.steps"]


def test_own_steps_have_short_and_detailed_descriptions():
    for name in own_steps():
        short, long = step_description(name)
        assert short, f"{name}: no short description (docstring missing?)"
        assert long, f"{name}: no detailed description (single-paragraph docstring)"
        assert "\n" not in short  # collapsed to one line


def test_own_steps_are_installed_and_registered_from_their_defining_modules():
    """pyproject.toml is the one list of jwstflow's own steps. The installed metadata must agree with
    it (an editable install made before a step moved is stale: reinstall, e.g. `uv sync`), and each
    entry must name the module that *defines* the step -- its file is fingerprinted into the task ids
    and parsed for the description."""
    from importlib.metadata import distribution

    from jwstflow.steps import ENTRY_POINT_GROUP, resolve_target

    installed = {ep.name: ep.value for ep in distribution("jwstflow").entry_points if ep.group == ENTRY_POINT_GROUP}
    assert installed == own_steps(), "the installed entry points differ from pyproject.toml: reinstall jwstflow"
    for name, target in own_steps().items():
        obj = resolve_target(name)
        assert f"{obj.__module__}:{obj.__qualname__}" == target, name


def test_registered_object_is_asked_directly():
    @register_step("desc_test_step")
    def desc_test_step(inputs, ctx, **params):
        """Copy the inputs somewhere.

        The detailed story.
        """

    try:
        assert step_description("desc_test_step") == ("Copy the inputs somewhere.", "The detailed story.")
    finally:
        _REGISTRY.pop("desc_test_step", None)


def test_path_spec_docstring_without_import(tmp_path: Path):
    # the module would explode on import; the docstring must still be found
    (tmp_path / "boomstep.py").write_text(
        textwrap.dedent(
            '''
            raise RuntimeError("must never be imported for a description")


            class BoomStep:
                """Short line.

                Long part
                over two lines.
                """
            '''
        )
    )
    short, long = step_description(f"{tmp_path}/boomstep.py:BoomStep")
    assert short == "Short line."
    assert long == "Long part\nover two lines."


def test_unknown_spec_is_empty_not_an_error():
    assert step_description("no.such.module:Nothing") == ("", "")


def test_split_description():
    assert split_description(None) == ("", "")
    assert split_description("One\nparagraph only.") == ("One paragraph only.", "")
    short, long = split_description("First\nparagraph.\n\nSecond.\n\nThird.")
    assert short == "First paragraph."
    assert long == "Second.\n\nThird."


def test_step_describe_carries_both():
    class Documented(Step):
        """Sum the flux.

        Every plane is summed.
        """

        def run(self, inputs, ctx, **params):  # pragma: no cover
            return []

    d = Documented.describe()
    assert d["doc"] == "Sum the flux."
    assert d["description"] == "Every plane is summed."


def test_check_step_tells_when_an_address_is_not_the_defining_module(tmp_path: Path):
    """An entry point (or dotted path) through a re-exporting module would fingerprint the wrong file."""
    from jwstflow.testing import check_step

    for name in own_steps():                                    # jwstflow's own registrations are clean
        _, problems = check_step(name)
        assert problems == [], (name, problems)
    _, problems = check_step("jwstflow.cubes.psf:PsfCube")            # the package re-exports the class from psf/steps.py
    assert len(problems) == 1 and "defined in" in problems[0] and "steps.py" in problems[0]
    _, problems = check_step("jwstflow.cubes.psf.steps:PsfCube")
    assert problems == []
