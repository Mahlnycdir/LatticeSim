"""Release consistency: version numbers and the example in the README."""
import re
import tomllib
from pathlib import Path

import latsimpkg

ROOT = Path(__file__).resolve().parent.parent


def test_version_matches_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert latsimpkg.__version__ == project["version"]


def test_changelog_has_an_entry_for_this_version():
    assert f"## {latsimpkg.__version__}" in (ROOT / "CHANGELOG.md").read_text()


def test_readme_example_runs_and_shows_the_stated_result():
    """The first python block of the README is executed; its commented results must hold."""
    readme = (ROOT / "README.md").read_text()
    code = re.search(r"```python\n(.*?)```", readme, re.S).group(1)
    namespace = {}
    exec(compile(code, "README.md", "exec"), namespace)
    scan = namespace["scan"]
    assert scan.agrees
    assert [s.content for s in scan.bound_states] == ["psi_3L^-2 psi_4L^-1 psi_5R^2", "psi_3L^2 psi_4L psi_5R^-2"]
