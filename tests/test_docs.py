"""Documentation generator (tools/build_docs.py) and completeness of the documentation."""
import importlib
import inspect
import json
import pkgutil
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bd():
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        yield importlib.import_module("build_docs")
    finally:
        sys.path.remove(str(ROOT / "tools"))


# ----------------------------------------------------------------------
# LaTeX conversion
# ----------------------------------------------------------------------
def test_escape_special_characters(bd):
    assert bd.escape("a_b ^ 50% & {x} $ # ~ \\") == \
        r"a\_b \^{} 50\% \& \{x\} \$ \# \~{} \textbackslash{}"


def test_inline_code_spans(bd):
    assert bd.inline("see `M_W` here") == r"see \texttt{M\_\allowbreak{}W} here"


DOC = '''
    Summary line with `code`.

    A formula block:
        E = (1/2) c^T G c

    Args:
        x (int): First argument
            continued here.
        y: Second.

    Returns:
        float: The value.

    Raises:
        ValueError: If bad.

    Example:
        f(1, 2)
'''


def test_parse_docstring_blocks(bd):
    blocks = bd.parse_docstring(DOC)
    kinds = [b[0] for b in blocks]
    assert kinds == ["text", "text", "verbatim", "section", "section", "section", "section"]
    assert blocks[2][1] == ["E = (1/2) c^T G c"]
    args = blocks[3]
    assert args[1] == "Arguments"
    assert args[2] == [("x (int)", "First argument continued here."), ("y", "Second.")]
    assert blocks[4][2] == [("", "float: The value.")]
    assert blocks[6][1] == "Example" and blocks[6][2] == ["f(1, 2)"]


def test_render_docstring(bd):
    tex = bd.render_docstring(DOC)
    assert r"\docsection{Arguments}" in tex and r"\item[{\texttt{x (int)}}]" in tex
    assert "\\begin{docverbatim}\nE = (1/2) c^T G c\n\\end{docverbatim}" in tex


def test_unicode_guard(bd):
    assert bd.unicode_declarations("plain") == ""
    assert "2229" in bd.unicode_declarations("K \u2229 W")
    with pytest.raises(ValueError, match="U\\+2603"):
        bd.unicode_declarations("snowman \u2603")


# ----------------------------------------------------------------------
# completeness
# ----------------------------------------------------------------------
def public_modules():
    import latsimpkg
    return {m.name for m in pkgutil.walk_packages(latsimpkg.__path__, "latsimpkg.") if not m.ispkg}


def test_every_module_is_in_the_api_reference(bd):
    documented = {mod for _, mod in bd.API_MODULES}
    assert public_modules() <= documented


def test_every_public_object_has_a_docstring(bd):
    missing = []
    for _, modname in bd.API_MODULES:
        module = importlib.import_module(modname)
        if not module.__doc__:
            missing.append(modname)
        for name, obj in vars(module).items():
            if name.startswith("_") or getattr(obj, "__module__", None) != modname:
                continue
            if (inspect.isclass(obj) or inspect.isfunction(obj)) and not obj.__doc__:
                missing.append(f"{modname}.{name}")
            if inspect.isclass(obj):
                missing += [f"{modname}.{name}.{n}" for n, kind, v in bd._public_members(obj)
                            if not bd._member_doc(kind, v)]
    assert missing == []


def test_api_reference_lists_all_public_objects(bd, tmp_path):
    """Every public class and function gets its own entry, labelled api:<module>:<name> by \\docclass/\\docfunction."""
    text = bd.write_api(tmp_path / "api.tex").read_text()
    for _, modname in bd.API_MODULES:
        module = importlib.import_module(modname)
        for name, obj in vars(module).items():
            if not name.startswith("_") and getattr(obj, "__module__", None) == modname and \
                    (inspect.isclass(obj) or inspect.isfunction(obj)):
                assert "}{" + f"{modname}.{name}".replace(".", ":") + "}" in text, name


def test_chapters_only_use_known_figures():
    sys.path.insert(0, str(ROOT))
    try:
        figures = importlib.import_module("studies.figures")
    finally:
        sys.path.remove(str(ROOT))
    known = {spec.name for spec in figures.FIGURES}
    used = set()
    for tex in (ROOT / "docs" / "chapters").glob("*.tex"):
        used |= set(re.findall(r"\\latsimfigure\{([^}]+)\}", tex.read_text()))
    assert used <= known
    assert used == known, f"figures not shown in the documentation: {sorted(known - used)}"


def test_chapter_citations_exist():
    refs = (ROOT / "docs" / "chapters" / "references.tex").read_text()
    keys = set(re.findall(r"\\bibitem\{([^}]+)\}", refs))
    cited = set()
    for tex in (ROOT / "docs" / "chapters").glob("*.tex"):
        for group in re.findall(r"\\cite\{([^}]+)\}", tex.read_text()):
            cited |= {k.strip() for k in group.split(",")}
    assert cited <= keys
    assert keys <= cited, f"references never cited: {sorted(keys - cited)}"


# ----------------------------------------------------------------------
# figures and build
# ----------------------------------------------------------------------
def test_write_figures_with_present_and_missing_pdf(bd, tmp_path):
    figdir = tmp_path / "figures"
    figdir.mkdir()
    (figdir / "a.pdf").write_bytes(b"%PDF-1.4")
    manifest = [{"name": "a", "study": "s", "caption": "Caption A.", "pdf": "a.pdf", "png": "a.png"},
                {"name": "b", "study": "s", "caption": "Caption B.", "pdf": "b.pdf", "png": "b.png"}]
    (figdir / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "gen"
    bd.write_figures(figdir, out)
    a, b = (out / "a.tex").read_text(), (out / "b.tex").read_text()
    assert r"\includegraphics" in a and r"\caption{Caption A.}" in a and r"\label{fig:a}" in a
    assert not Path(re.search(r"\{([^{}]+\.pdf)\}", a).group(1)).is_absolute()
    assert r"\missingfigure{b}" in b and "python -m studies s" in b


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk not installed")
def test_documentation_builds_without_warnings():
    subprocess.run([sys.executable, str(ROOT / "tools" / "build_docs.py"), "--studies"], check=True, cwd=ROOT,
                   capture_output=True)
    log = (ROOT / "docs" / "build" / "latticesim.log").read_text(errors="replace")
    assert (ROOT / "docs" / "build" / "latticesim.pdf").exists()
    assert "undefined" not in log and "Overfull" not in log
