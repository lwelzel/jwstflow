"""The architecture, as tests: who may import whom, and where steps are defined.

docs/guide.md ("How the package is laid out") states the rules in words. Here they are checked
on the source itself -- statically, imports inside functions included -- so a change that bends
the layout fails with the rule it broke rather than eroding it quietly. Nothing is imported
for the import rules: they hold for every module, also the ones a test run never loads.
"""

from __future__ import annotations

import ast
from pathlib import Path

import jwstflow
from jwstflow.steps import BUNDLED_STEPS, resolve_target

PACKAGE = Path(jwstflow.__file__).parent

#: The units of the package. A domain's features (the subpackages / modules that define its steps)
#: count on their own, because the toolkit of a domain must not depend on what is built with it.
CONTRACT = {"steps"}                                     # the step framework
ORCHESTRATION = {"config", "data", "engine"}             # the YAML, the input files, the run model
TOOLKIT = {"spectra", "cubes", "qa"}                     # what steps are built from, by domain
FEATURES = {"spectra.stitching", "cubes.psf", "cubes.clusters", "qa.steps"}   # where the bundled steps are
TOOLS = {"cli", "testing", "scaffold", "__init__"}
SCIENCE = TOOLKIT | FEATURES

#: Modules in which a step of the package may be defined (docs: "a data step lives with the
#: feature it implements, every QA step in jwstflow.qa.steps").
DATA_STEP_MODULES = {"jwstflow.spectra.stitching", "jwstflow.cubes.psf.steps", "jwstflow.cubes.clusters.steps",
                     "jwstflow.data.msa"}
QA_STEP_PACKAGE = "jwstflow.qa.steps"
#: Step subclasses that are not steps of the package: the framework's own, and the test stand-ins.
NOT_BUNDLED = {"jwstflow.steps.interface", "jwstflow.steps.adapters", "jwstflow.testing.mock.stubs"}


def unit_of(module: str) -> str | None:
    parts = module.split(".")
    if parts[0] != "jwstflow":
        return None
    if len(parts) == 1:
        return "__init__"
    two = ".".join(parts[1:3])
    return two if two in FEATURES else parts[1]


def module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE.parent).with_suffix("")
    return ".".join(rel.parts[:-1] if rel.name == "__init__" else rel.parts)


def imports_of(path: Path) -> set[str]:
    """Every jwstflow module ``path`` imports, anywhere in the file (relative imports resolved)."""
    package = module_name(path).split(".") if path.name == "__init__.py" else module_name(path).split(".")[:-1]
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - (node.level - 1)] if node.level else []
            module = ".".join(base + ([node.module] if node.module else []))
            found.add(module)
            found.update(f"{module}.{a.name}" for a in node.names)   # `from . import qa` names a module
    return {m for m in found if m.split(".")[0] == "jwstflow"}


def dependencies() -> dict[str, set[str]]:
    """unit -> the other units it imports."""
    graph: dict[str, set[str]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        me = unit_of(module_name(path))
        for target in imports_of(path):
            other = unit_of(target)
            if other and other != me and other != "__init__":
                graph.setdefault(me, set()).add(other)
    return graph


def test_the_top_level_holds_subpackages_only():
    """Code lives in a subsystem or a domain; the package directory itself holds the CLI and nothing else."""
    assert sorted(p.name for p in PACKAGE.glob("*.py")) == ["__init__.py", "cli.py"]
    units = {unit_of(module_name(p)) for p in PACKAGE.rglob("*.py")}
    known = CONTRACT | ORCHESTRATION | TOOLKIT | FEATURES | TOOLS
    assert units <= known, f"units without a place in tests/test_architecture.py: {sorted(units - known)}"


def test_the_step_contract_stands_on_its_own():
    assert dependencies().get("steps", set()) == set(), "jwstflow.steps is the framework: it imports no other unit"


def test_orchestration_knows_no_science():
    """config, data and engine run *any* workflow: they never import a domain, a feature or a QA step."""
    graph = dependencies()
    for unit in ORCHESTRATION:
        assert not graph.get(unit, set()) & (SCIENCE | TOOLS), (unit, sorted(graph[unit] & (SCIENCE | TOOLS)))
    assert not graph.get("config", set()) & {"data", "engine"}
    assert "engine" not in graph.get("data", set())


def test_science_never_reaches_into_the_engine():
    """Steps get everything through their RunContext; the domains import neither the engine nor the
    tools around it. (`config` and `data` are below them: the project root, headers, target positions.)"""
    graph = dependencies()
    for unit in SCIENCE:
        forbidden = graph.get(unit, set()) & ({"engine"} | TOOLS)
        assert not forbidden, (unit, sorted(forbidden))


def test_a_toolkit_does_not_depend_on_the_features_built_with_it():
    graph = dependencies()
    for unit in TOOLKIT:
        assert not graph.get(unit, set()) & FEATURES, (unit, sorted(graph[unit] & FEATURES))


def _step_classes() -> dict[str, tuple[str, bool]]:
    """Every class of the package that derives from Step: name -> (module, is a QA step).

    Found by name through the class hierarchy (Step, then every class based on a found one), so
    nothing has to be imported; a QA step is one whose class body says ``level = "qa"``."""
    classes: dict[str, tuple[str, list[str], bool]] = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                bases = [b.id if isinstance(b, ast.Name) else b.attr for b in node.bases if isinstance(b, (ast.Name, ast.Attribute))]
                qa_level = any(isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "level" for t in s.targets)
                               and isinstance(s.value, ast.Constant) and s.value.value == "qa" for s in node.body)
                classes[node.name] = (module_name(path), bases, qa_level)
    steps = {"Step"}
    while True:
        more = {name for name, (_, bases, _) in classes.items() if set(bases) & steps} - steps
        if not more:
            break
        steps |= more
    return {name: (classes[name][0], classes[name][2]) for name in steps if name in classes and name != "Step"}


def test_steps_are_defined_where_the_layout_says():
    """A data step lives with its feature, a QA step in jwstflow.qa.steps -- and nowhere else."""
    for name, (module, is_qa) in sorted(_step_classes().items()):
        if module in NOT_BUNDLED:
            continue
        if is_qa:
            assert module.startswith(QA_STEP_PACKAGE + "."), f"QA step {name} is defined in {module}, not in {QA_STEP_PACKAGE}"
        else:
            assert module in DATA_STEP_MODULES, (
                f"step {name} is defined in {module}: file it with its feature (and list the module in "
                f"DATA_STEP_MODULES), or in {QA_STEP_PACKAGE} if it draws")


def test_every_bundled_step_is_registered_from_its_defining_module():
    """BUNDLED_STEPS covers every step class of the package, and each name leads to an allowed module."""
    registered = {target.rpartition(":")[2]: target.partition(":")[0] for target in BUNDLED_STEPS.values()}
    for name, (module, _) in _step_classes().items():
        if module not in NOT_BUNDLED:
            assert registered.get(name) == module, f"{name} ({module}) is missing from BUNDLED_STEPS / pyproject.toml"
    for name, target in BUNDLED_STEPS.items():
        module = target.partition(":")[0]
        is_qa = getattr(resolve_target(name), "level", 4) == "qa"
        assert module.startswith(QA_STEP_PACKAGE + ".") if is_qa else module in DATA_STEP_MODULES, (name, target)
