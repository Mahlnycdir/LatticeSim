"""Shared helpers of the studies: models, output directory and report formatting."""
from pathlib import Path

from latsimpkg import axial_schwinger, model_3450, schwinger

RESULTS = Path(__file__).resolve().parent.parent / "results"

# The theories every study looks at (name, factory g -> ChiralTheory).
MODELS = [("Schwinger (vector)", schwinger),
          ("Schwinger (axial)", axial_schwinger),
          ("3-4-5-0", model_3450)]


def outdir(study, root=None):
    """Directory results/<study>, created if needed."""
    path = Path(root or RESULTS) / study
    path.mkdir(parents=True, exist_ok=True)
    return path


def slug(name):
    """File-name friendly version of a model name."""
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_").replace("__", "_")


def section(title):
    print()
    print(title)
    print("-" * len(title))
