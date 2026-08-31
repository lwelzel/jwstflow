"""`jwstflow steps` descriptions: the built-in table and docstring extraction."""

from __future__ import annotations

import textwrap
from pathlib import Path

from jwstflow.steps.base import (
    BUILTIN_ALIASES,
    CONTRIB_ALIASES,
    Step,
    _REGISTRY,
    register_step,
    split_description,
    step_description,
)
from jwstflow.steps.descriptions import BUILTIN_DESCRIPTIONS


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


def test_contrib_docstrings_have_short_and_detail():
    for name in CONTRIB_ALIASES:
        short, long = step_description(name)
        assert short, f"{name}: no short description (docstring missing?)"
        assert long, f"{name}: no detailed description (single-paragraph docstring)"
        assert "\n" not in short  # collapsed to one line


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
