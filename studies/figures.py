"""
Figures of the studies: rendered from the saved results, with captions for the documentation.

    python -m studies.figures            # render what is missing or out of date
    python -m studies.figures --force    # render everything

A figure is only redrawn if one of its result files is newer than the figure (make-like), so
rebuilding the documentation does not recompute or redraw anything unnecessarily. Every run writes
figures/manifest.json with file names and captions; the documentation includes the figures from there.
"""
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from latsimpkg.analysis.results import FitResult, Potential, SectorScan, Series, Spectrum
from latsimpkg.plotting import (plot_fits, plot_potentials, plot_sector_diagram, plot_series, plot_spectrum,
                                save_figure)
from studies.common import RESULTS

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
MODEL_FILES = [("Schwinger (vector)", "schwinger_vector"), ("Schwinger (axial)", "schwinger_axial"),
               ("3-4-5-0", "3_4_5_0")]


@dataclass(frozen=True)
class FigureSpec:
    name: str          # file stem in figures/
    study: str         # study that produced the inputs
    inputs: tuple      # result files relative to results/
    draw: callable     # (list of loaded results) -> Figure
    caption: str       # LaTeX caption for the documentation


def _volume_series(results):
    """E L/(2 pi) against N for the sectors of the volume study."""
    out = []
    for s in results:
        y = np.array(s.meta["Delta"])
        meta = dict(s.meta, ylabel=r"$E L / 2\pi$")
        out.append(Series(s.x, y, np.array([]), meta))
    labels = [rf"$N_L = {tuple(s.meta['NL'])},\ N_R = {tuple(s.meta['NR'])},\ q = {s.meta['q'][0]:g}$" for s in results]
    return plot_series(out, labels=labels, residuals=False, logy=True)


def _continuum(results):
    """|omega_min - sqrt(mu^2 + (pi/L)^2)| against a (log-log) with an a^2 reference line."""
    s = results[0]
    dev = np.abs(s.y - s.prediction)
    ref = dev[-1] * (s.x / s.x[-1]) ** 2
    series = Series(s.x, dev, ref, dict(s.meta))
    return plot_series(series, labels=[rf"Schwinger, $L = {s.meta['L']:g}$ (dashed: $\propto a^2$)"], residuals=False,
                       logx=True, logy=True, xlabel=r"lattice spacing $a$",
                       ylabel=r"$|\omega_{\min} - \sqrt{\mu^2 + (\pi/L)^2}|$")


FIGURES = [
    FigureSpec("free_chain_spectrum", "free_field", ("free_field/spectrum.npz",),
               lambda r: plot_spectrum(r, labels=["free Dirichlet chain"]),
               r"Normal-mode frequencies of the free scalar field on the Dirichlet chain ($N = 50$, $a = 0.001$) "
               r"compared with $\omega_n = (2/a)\sin(\pi n / 2(N+1))$ (dashed). The residuals are at round-off level."),
    FigureSpec("spectra_open_chain", "spectra", tuple(f"spectra/spectrum_{f}.npz" for _, f in MODEL_FILES),
               lambda r: plot_spectrum(r, labels=[m for m, _ in MODEL_FILES]),
               r"Spectra of the gauged theories on the open chain ($N = 300$, $g = 0.04$). Dashed: one branch "
               r"$\sqrt{\mu^2 + k^2}$ plus $F - 1$ massless branches. Vector couplings agree exactly; axial "
               r"couplings shift the boundary momenta, visible as residuals at small $n$."),
    FigureSpec("finite_size_scaling", "spectra", tuple(f"spectra/scaling_{f}.npz" for _, f in MODEL_FILES),
               lambda r: plot_series(r, labels=[m for m, _ in MODEL_FILES], logx=True, logy=True,
                                     xlabel=r"$N$", ylabel=r"$\omega_{\min}$"),
               r"Lowest frequency against the chain length. With only a massive branch it approaches the meson mass "
               r"$\mu$; with massless flavors (3-4-5-0) it falls like $1/N$. Dashed: Dirichlet prediction."),
    FigureSpec("continuum_limit", "spectra", ("spectra/continuum_schwinger.npz",),
               _continuum,
               r"Continuum limit at fixed length $L = (N+1)a = 300$ (Schwinger model, $g = 0.04$): the deviation from "
               r"$\sqrt{\mu^2 + (\pi/L)^2}$ falls like $a^2$ (dashed reference line)."),
    FigureSpec("correlator_fits", "correlators",
               ("correlators/fit_vector.npz", "correlators/fit_axial.npz", "correlators/fit_3_4_5_0.npz"),
               lambda r: plot_fits(r, labels=["vector", "axial", "3-4-5-0"]),
               r"Equal-time correlator of the electric field in the bulk of the open chain ($N = 300$) with fits "
               r"$A K_0(\mu r) + B$. The fitted masses agree with $\mu$ to about $10^{-3}$."),
    FigureSpec("anomalous_commutator", "correlators", ("correlators/anomalous_commutator.npz",),
               lambda r: plot_series(r, labels=["lattice"], residuals=False,
                                     xlabel=r"$|\sum Q_L^2 - \sum Q_R^2| / 4$", ylabel=r"$\max_{j,k} |[E_j, E_k]|$"),
               r"Commutator of the electric field for anomalous charge assignments against the anomaly coefficient "
               r"$|\sum Q_L^2 - \sum Q_R^2|/4$. Dashed: $g^2/\pi$ times the coefficient; it vanishes exactly for "
               r"anomaly-free theories (3-4-5-0 at the origin)."),
    FigureSpec("sectors_3450", "sectors", ("sectors/scan_3_4_5_0.npz",),
               lambda r: plot_sector_diagram(r[0]),
               r"Gauge-invariant massless sectors of the 3-4-5-0 model in the $(h, \bar h)$ plane. Filled: lattice "
               r"(ring, $N = 60$); open circles: charge-lattice prediction. The bound states "
               r"$\psi_{3L}^2\psi_{4L}\psi_{5R}^{-2}$ and its antiparticle are left-moving fermions with "
               r"$(h, \bar h) = (1/2, 0)$; $\psi_{0R}$ is the elementary neutral right mover."),
    FigureSpec("sector_volume_dependence", "sectors",
               ("sectors/volume_2_1_m2_0.npz", "sectors/volume_0_0_0_1.npz", "sectors/volume_2_1_m2_1.npz",
                "sectors/volume_1_0_0_0.npz", "sectors/volume_0_0_1_0.npz"),
               _volume_series,
               r"$E L / 2\pi$ of charge sectors against the ring size (3-4-5-0, $g = 0.3$). Gauge-neutral massless "
               r"sectors stay at $h + \bar h$; charged sectors (forbidden by the Gauss law on the ring) are screened, "
               r"so $E$ stays finite and $E L$ grows."),
    FigureSpec("probe_potentials", "potentials",
               ("potentials/screening_q1.npz", "potentials/screening_q05.npz", "potentials/confinement_q1.npz"),
               lambda r: plot_potentials(r, labels=["Schwinger, $q = 1$", "Schwinger, $q = 1/2$",
                                                    "neutral matter, $q = 1$"]),
               r"Static potential of probe charges $\pm q$ ($N = 400$, $g = 0.1$). Massless charged matter screens "
               r"every charge, $V = (g^2q^2/2\mu)(1 - e^{-\mu d})$; without it the potential rises linearly "
               r"(confinement), $V = g^2 q^2 d/2$."),
]

LOADERS = {"SectorScan": SectorScan, "Spectrum": Spectrum, "Series": Series, "FitResult": FitResult,
           "Potential": Potential}


def _load(path):
    import json as _json
    with np.load(path, allow_pickle=False) as data:
        kind = _json.loads(str(data["__meta__"]))["__type__"]
    return LOADERS[kind].load(path)


def render(results_root=None, figures_root=None, force=False, names=None):
    """
    Render the figures whose inputs exist. Returns the manifest (list of dicts). A figure is skipped
    if it is newer than all its inputs, unless `force`.
    """
    results_root = Path(results_root or RESULTS)
    figures_root = Path(figures_root or FIGURES_DIR)
    figures_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in FIGURES:
        if names and spec.name not in names:
            continue
        inputs = [results_root / p for p in spec.inputs]
        missing = [str(p) for p in inputs if not p.exists()]
        entry = {"name": spec.name, "study": spec.study, "caption": spec.caption,
                 "pdf": f"{spec.name}.pdf", "png": f"{spec.name}.png"}
        if missing:
            entry["status"] = "missing inputs: " + ", ".join(missing)
            manifest.append(entry)
            continue
        target = figures_root / f"{spec.name}.pdf"
        newest = max(p.stat().st_mtime for p in inputs)
        if not force and target.exists() and target.stat().st_mtime >= newest:
            entry["status"] = "up to date"
        else:
            save_figure(spec.draw([_load(p) for p in inputs]), figures_root / spec.name)
            entry["status"] = "rendered"
        manifest.append(entry)
    (figures_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    start = time.perf_counter()
    for entry in render(force="--force" in sys.argv):
        print(f"  {entry['name']:28s} {entry['status']}")
    print(f"  [{time.perf_counter() - start:.1f} s, figures in figures/]")
