"""Path fields of a recipe that are fixed before interpolation (root, workspace, download.dest)."""

from __future__ import annotations


def test_root_may_come_from_an_environment_variable(tmp_path, monkeypatch):
    """${env:VAR,default} in root resolves before the path is anchored at the recipe's directory."""
    from jwstflow.config.loader import load_config

    recipe = tmp_path / "project" / "recipes" / "t.yaml"
    recipe.parent.mkdir(parents=True)
    (tmp_path / "project" / ".jwstflow-root").write_text("")
    data = tmp_path / "somewhere" / "data"
    recipe.write_text(f"version: 1\ntarget: Some Disk\nrun: nirspec_ifu\n"
                      f"root: ${{env:JWSTFLOW_TEST_DATA,{data}}}/some_disk/nirspec_ifu\nstages: []\n")
    monkeypatch.delenv("JWSTFLOW_TEST_DATA", raising=False)
    assert load_config(recipe).run_dir == data / "some_disk" / "nirspec_ifu"          # the default, absolute
    other = tmp_path / "other"
    monkeypatch.setenv("JWSTFLOW_TEST_DATA", str(other))
    assert load_config(recipe).run_dir == other / "some_disk" / "nirspec_ifu"         # the variable
    monkeypatch.setenv("JWSTFLOW_TEST_DATA", "../../rel")
    assert load_config(recipe).run_dir == (recipe.parent / "../../rel/some_disk/nirspec_ifu").resolve()  # anchored
