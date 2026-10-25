"""
Build the LatticeSim documentation (LaTeX -> PDF).

    python tools/build_docs.py                 # figures from existing results, API reference, PDF
    python tools/build_docs.py --studies       # first run the studies that have no results yet
    python tools/build_docs.py --no-pdf        # only generate the .tex files

Steps:
  1. figures: `studies.figures.render` (incremental) -> figures/*.pdf + manifest.json; with --studies the
     studies without results are run first;
  2. docs/generated/figures/<name>.tex: one figure environment per manifest entry (caption from the manifest,
     a placeholder box if the figure is missing);
  3. docs/generated/api.tex: API reference from the docstrings (Google style: Args/Returns/Raises/Attributes);
     the ASCII math of the docstrings is kept verbatim, the typeset mathematics lives in the chapters;
  4. latexmk -> docs/build/latticesim.pdf.
"""
import argparse
import importlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import fields, is_dataclass
from functools import cached_property
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
GENERATED = DOCS / "generated"
BUILD = DOCS / "build"
MAIN = "latticesim"

# Modules of the API reference, in reading order (layer by layer, see ROADMAP).
API_MODULES = [
    ("Conventions", "latsimpkg.conventions"),
    ("Theories", "latsimpkg.theory"),
    ("Lattice", "latsimpkg.lattice"),
    ("Phase space", "latsimpkg.phasespace"),
    ("Quadratic Hamiltonians", "latsimpkg.quadratic"),
    ("Gauge models", "latsimpkg.models.gauge"),
    ("Result types", "latsimpkg.analysis.results"),
    ("Spectra", "latsimpkg.analysis.spectrum"),
    ("Correlators", "latsimpkg.analysis.correlators"),
    ("Potentials", "latsimpkg.analysis.potentials"),
    ("Charge sectors", "latsimpkg.analysis.sectors"),
    ("Charge lattice", "latsimpkg.analysis.charge_lattice"),
    ("User API", "latsimpkg.api"),
    ("Plotting", "latsimpkg.plotting"),
]

SECTIONS = {"args": "Arguments", "arguments": "Arguments", "attributes": "Attributes", "returns": "Returns",
            "raises": "Raises", "example": "Example", "examples": "Examples", "note": "Note", "yields": "Yields"}


# ----------------------------------------------------------------------
# LaTeX helpers
# ----------------------------------------------------------------------
_ESCAPES = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&", "#": r"\#",
            "^": r"\^{}", "_": r"\_", "%": r"\%", "~": r"\~{}"}


def escape(text):
    """Escape LaTeX special characters of plain text."""
    return "".join(_ESCAPES.get(c, c) for c in text)


def inline(text):
    """Escape text and typeset `code` spans in typewriter font (with break points after _ . ,)."""
    parts = re.split(r"`([^`]+)`", text)
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            code = escape(part).replace(r"\_", r"\_\allowbreak{}").replace(".", r".\allowbreak{}")
            code = re.sub(r"(?<=[a-z])(?=[A-Z])", r"\\allowbreak{}", code)        # CamelCase break points
            out.append(r"\texttt{" + code + "}")
        else:
            out.append(escape(part))
    return "".join(out)


def verbatim(lines):
    """A block of lines kept exactly (ASCII formulas, examples)."""
    return "\\begin{docverbatim}\n" + "\n".join(lines) + "\n\\end{docverbatim}"


# ----------------------------------------------------------------------
# docstring parsing (Google style)
# ----------------------------------------------------------------------
def _indent(line):
    return len(line) - len(line.lstrip())


def parse_docstring(doc):
    """
    Split a docstring into blocks: ("text", [lines]), ("verbatim", [lines]) and
    ("section", title, [(name, description), ...] or [lines]).
    Indented lines that do not belong to a section are kept verbatim.
    """
    if not doc:
        return []
    lines = inspect.cleandoc(doc).splitlines()
    blocks, i = [], 0
    while i < len(lines):
        line = lines[i]
        header = line.strip().rstrip(":").lower()
        if line.strip().endswith(":") and header in SECTIONS and _indent(line) == 0:
            i += 1
            body = []
            while i < len(lines) and (not lines[i].strip() or _indent(lines[i]) > 0):
                body.append(lines[i])
                i += 1
            blocks.append(("section", SECTIONS[header], _section_items(body, header)))
            continue
        if not line.strip():
            i += 1
            continue
        if _indent(line) > 0:
            body = []
            while i < len(lines) and (not lines[i].strip() or _indent(lines[i]) > 0):
                body.append(lines[i])
                i += 1
            while body and not body[-1].strip():
                body.pop()
            base = min(_indent(l) for l in body if l.strip())
            blocks.append(("verbatim", [l[base:] for l in body]))
            continue
        para = []
        while i < len(lines) and lines[i].strip() and _indent(lines[i]) == 0 and not (
                lines[i].strip().endswith(":") and lines[i].strip().rstrip(":").lower() in SECTIONS):
            para.append(lines[i].strip())
            i += 1
        blocks.append(("text", para))
    return blocks


def _section_items(body, header):
    """Items 'name (type): description' with indented continuation lines; examples stay verbatim."""
    body = [l for l in body]
    while body and not body[-1].strip():
        body.pop()
    if not body:
        return []
    base = min(_indent(l) for l in body if l.strip())
    if header in ("example", "examples"):
        return [l[base:] for l in body]
    items = []
    for line in body:
        if not line.strip():
            continue
        if _indent(line) == base:
            name, sep, desc = line.strip().partition(":")
            if not sep or header in ("returns", "yields"):
                items.append(("", line.strip()))
            else:
                items.append((name.strip(), desc.strip()))
        elif items:
            name, desc = items[-1]
            items[-1] = (name, desc + " " + line.strip())
    return items


def render_docstring(doc):
    """LaTeX for a docstring."""
    out = []
    for block in parse_docstring(doc):
        if block[0] == "text":
            out.append(inline(" ".join(block[1])) + "\n")
        elif block[0] == "verbatim":
            out.append(verbatim(block[1]))
        else:
            _, title, items = block
            out.append(r"\docsection{" + title + "}")
            if items and isinstance(items[0], str):
                out.append(verbatim(items))
            elif items:
                out.append("\\begin{docitems}")
                for name, desc in items:
                    label = r"\item[{\texttt{" + escape(name) + "}}]" if name else r"\item[]"
                    out.append(label + " " + inline(desc))
                out.append("\\end{docitems}")
    return "\n".join(out)


# ----------------------------------------------------------------------
# API reference
# ----------------------------------------------------------------------
def _signature(name, obj, kind=None):
    """Signature line; methods carry their kind (method, classmethod, ...) at the end of the line."""
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(...)"
    sig = sig.replace("(self, ", "(").replace("(self)", "()").replace("(cls, ", "(").replace("(cls)", "()")
    head = r"\docsig{" + escape(name).replace(r"\_", r"\_\allowbreak{}") + "}{" + \
        escape(sig).replace(", ", r", \allowbreak{}") + "}"
    return head if kind is None else head[:-1] + r"\enspace{\normalfont\small\itshape " + kind + "}}"


def _public_members(cls):
    """Public methods and properties defined in the class itself, in source order."""
    members = []
    for name, value in cls.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, (property, cached_property)):
            members.append((name, "property", value))
        elif isinstance(value, (staticmethod, classmethod)):
            members.append((name, "classmethod" if isinstance(value, classmethod) else "staticmethod",
                            value.__func__))
        elif inspect.isfunction(value):
            members.append((name, "method", value))
    return members


def _member_doc(kind, value):
    if kind == "property":
        func = value.fget if isinstance(value, property) else value.func
        return inspect.getdoc(func)
    return inspect.getdoc(value)


def render_class(qualname, cls):
    out = [r"\docclass{" + escape(qualname) + "}{" + qualname.replace(".", ":") + "}"]
    if is_dataclass(cls):
        out.append(r"\docnote{dataclass with fields " +
                   ", ".join(r"\texttt{" + escape(f.name) + "}" for f in fields(cls)) + "}")
    init = cls.__dict__.get("__init__")
    if init is not None and inspect.isfunction(init):
        out.append(_signature(cls.__name__, init))
    out.append(render_docstring(inspect.getdoc(cls) if cls.__doc__ else ""))
    if init is not None and inspect.isfunction(init) and init.__doc__:
        out.append(render_docstring(init.__doc__))
    for name, kind, value in _public_members(cls):
        if kind == "property":
            out.append(r"\docmember{" + escape(name) + "}{" + kind + "}")
        else:
            out.append(_signature(name, value, kind))
        out.append(render_docstring(_member_doc(kind, value) or ""))
    return "\n".join(out)


def render_module(title, modname):
    module = importlib.import_module(modname)
    out = [r"\section{" + escape(title) + r": \texttt{" + escape(modname).replace(".", r".\allowbreak{}") + "}}",
           r"\label{api:" + modname + "}"]
    out.append(render_docstring(module.__doc__ or ""))
    for name, obj in vars(module).items():
        if name.startswith("_") or getattr(obj, "__module__", None) != modname:
            continue
        if inspect.isclass(obj):
            out.append(render_class(f"{modname}.{name}", obj))
        elif inspect.isfunction(obj):
            out.append(r"\docfunction{" + escape(name) + "}{" + f"{modname}.{name}".replace(".", ":") + "}")
            out.append(_signature(name, obj))
            out.append(render_docstring(inspect.getdoc(obj) or ""))
    return "\n\n".join(out)


# Non-ASCII characters that may appear in docstrings, and their LaTeX form (also valid inside verbatim blocks).
UNICODE = {"\u2229": r"\ensuremath{\cap}", "\u222a": r"\ensuremath{\cup}", "\u2264": r"\ensuremath{\leq}",
           "\u2265": r"\ensuremath{\geq}", "\u2260": r"\ensuremath{\neq}", "\u2248": r"\ensuremath{\approx}",
           "\u00b1": r"\ensuremath{\pm}", "\u00d7": r"\ensuremath{\times}", "\u00b7": r"\ensuremath{\cdot}",
           "\u2208": r"\ensuremath{\in}", "\u221e": r"\ensuremath{\infty}", "\u2192": r"\ensuremath{\to}",
           "\u22a5": r"\ensuremath{\perp}", "\u2261": r"\ensuremath{\equiv}", "\u03c0": r"\ensuremath{\pi}",
           "\u03bc": r"\ensuremath{\mu}", "\u03c9": r"\ensuremath{\omega}", "\u03c3": r"\ensuremath{\sigma}",
           "\u03b8": r"\ensuremath{\theta}", "\u03c6": r"\ensuremath{\varphi}", "\u0394": r"\ensuremath{\Delta}",
           "\u2014": "---", "\u2013": "--"}


def unicode_declarations(text):
    """
    LaTeX declarations for the non-ASCII characters in `text`. Docstrings should stay ASCII; this keeps the
    build working if one slips in, and names the character instead of failing deep inside LaTeX.

    Raises:
        ValueError: For a non-ASCII character without a known LaTeX form.
    """
    chars = sorted({c for c in text if ord(c) > 127})
    unknown = [c for c in chars if c not in UNICODE]
    if unknown:
        raise ValueError("Docstrings contain characters without LaTeX form: " +
                         ", ".join(f"{c!r} (U+{ord(c):04X})" for c in unknown) + "; use ASCII instead.")
    return "\n".join(r"\DeclareUnicodeCharacter{%04X}{%s}" % (ord(c), UNICODE[c]) for c in chars)


def write_api(path=GENERATED / "api.tex"):
    path.parent.mkdir(parents=True, exist_ok=True)
    body = ["% generated by tools/build_docs.py from the docstrings -- do not edit"]
    body += [render_module(title, mod) for title, mod in API_MODULES]
    text = "\n\n".join(body) + "\n"
    (path.parent / "unicode.tex").write_text(unicode_declarations(text) + "\n")
    path.write_text(text)
    return path


# ----------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------
def write_figures(figures_dir, out_dir=GENERATED / "figures"):
    """One .tex file per figure of the manifest; a placeholder if the PDF is missing."""
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = figures_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else []
    written = []
    for entry in manifest:
        pdf = figures_dir / entry["pdf"]
        if pdf.exists():
            graphic = r"\includegraphics[width=0.82\linewidth]{" + Path(os.path.relpath(pdf, DOCS)).as_posix() + "}"
        else:
            graphic = (r"\missingfigure{" + escape(entry["name"]) + "}{" +
                       escape("python -m studies " + entry["study"] + " && python -m studies.figures") + "}")
        tex = "\n".join([r"\begin{figure}[tbp]", r"\centering", graphic,
                         r"\caption{" + entry["caption"] + "}", r"\label{fig:" + entry["name"] + "}",
                         r"\end{figure}", ""])
        target = out_dir / f"{entry['name']}.tex"
        target.write_text(tex)
        written.append(target)
    return written


def ensure_figures(run_studies):
    sys.path.insert(0, str(ROOT))
    from studies import common
    from studies import figures as figs
    if run_studies:
        from studies.__main__ import STUDIES
        needed = {spec.study for spec in figs.FIGURES
                  if not all((common.RESULTS / p).exists() for p in spec.inputs)}
        for name in STUDIES:
            if name in needed:
                print(f"  running study {name} ...")
                STUDIES[name].run()
    return figs.render(), figs.FIGURES_DIR


# ----------------------------------------------------------------------
# build
# ----------------------------------------------------------------------
AUX_SUFFIXES = (".aux", ".out", ".toc", ".fdb_latexmk", ".fls")


def remove_aux_files(build=None):
    """Delete LaTeX auxiliary files; a truncated .out/.aux from an interrupted run breaks the next run."""
    build = BUILD if build is None else build
    for suffix in AUX_SUFFIXES:
        (build / f"{MAIN}{suffix}").unlink(missing_ok=True)


def build_pdf(run=subprocess.run):
    """
    latexmk -> docs/build/latticesim.pdf. If LaTeX fails, the auxiliary files are removed and the build is
    retried once (stale or truncated auxiliary files, e.g. the bookmark file of an interrupted run, are a known
    cause of spurious failures); a second failure is reported with the end of the log.
    """
    if shutil.which("latexmk") is None:
        sys.exit("latexmk not found (install MacTeX or TeX Live); the .tex files are in docs/.")
    BUILD.mkdir(parents=True, exist_ok=True)
    cmd = ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"-outdir={BUILD}", f"{MAIN}.tex"]
    for attempt in (1, 2):
        result = run(cmd, cwd=DOCS, capture_output=True, text=True)
        if not result.returncode:
            return BUILD / f"{MAIN}.pdf"
        if attempt == 1:
            print("  LaTeX failed, retrying once with fresh auxiliary files ...")
            remove_aux_files()
    log = BUILD / f"{MAIN}.log"
    tail = log.read_text(errors="replace").splitlines()[-40:] if log.exists() else result.stdout.splitlines()[-40:]
    sys.exit("LaTeX failed:\n" + "\n".join(tail))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--studies", action="store_true", help="run studies whose results are missing")
    parser.add_argument("--no-pdf", action="store_true", help="only generate the .tex files")
    args = parser.parse_args(argv)
    manifest, figures_dir = ensure_figures(args.studies)
    missing = [e["name"] for e in manifest if not (figures_dir / e["pdf"]).exists()]
    print(f"  figures: {len(manifest) - len(missing)} available" +
          (f", missing (placeholders): {', '.join(missing)}" if missing else ""))
    write_figures(figures_dir)
    print(f"  API reference: {write_api().relative_to(ROOT)}")
    if not args.no_pdf:
        print(f"  PDF: {build_pdf().relative_to(ROOT)}")


if __name__ == "__main__":
    main()
