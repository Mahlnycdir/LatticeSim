"""
Plots of saved results.

Every function takes result objects (`Spectrum`, `Series`, `FitResult`, `Potential`, `SectorScan`) and
returns a matplotlib `Figure`; nothing is recomputed. The figures are built with the object-oriented
API (no pyplot, no global state, no backend switching), so plotting is fast, works headless and does
not interfere with interactive sessions.

    fig = plot_sector_diagram(SectorScan.load("results/sectors/scan_3_4_5_0.npz"))
    save_figure(fig, "figures/sectors_3450")          # writes .pdf (vector, for LaTeX) and .png
"""
import re
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from latsimpkg.analysis.results import FitResult, Potential, SectorScan, Series, Spectrum

# Okabe-Ito palette (colour-blind safe).
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
KIND_COLORS = {"bound state": "#D55E00", "elementary": "#0072B2", "multi-particle": "#999999"}

STYLE = {
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "serif", "mathtext.fontset": "cm",
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5, "lines.linewidth": 1.2,
    "lines.markersize": 3.5, "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
}
SIZE = (5.0, 3.3)


# ----------------------------------------------------------------------
# infrastructure
# ----------------------------------------------------------------------
def new_figure(residuals=False, size=SIZE):
    """Figure with one axis, or a main axis plus a smaller residual axis below it."""
    with matplotlib.rc_context(STYLE):
        fig = Figure(figsize=size)
        FigureCanvasAgg(fig)
        if not residuals:
            return fig, fig.add_subplot(1, 1, 1), None
        ax, res = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08})
        return fig, ax, res


def save_figure(fig, stem, formats=("pdf", "png"), dpi=200):
    """Write `stem`.pdf / `stem`.png (directories are created). Returns the written paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    with matplotlib.rc_context(STYLE):
        for fmt in formats:
            path = stem.with_suffix(f".{fmt}")
            fig.savefig(path, dpi=dpi, metadata={"CreationDate": None} if fmt == "pdf" else None)
            paths.append(path)
    return paths


def content_to_tex(content):
    """'psi_3L^2 psi_4L psi_5R^-2' -> mathtext '$\\psi_{3L}^{2}\\,\\psi_{4L}\\,\\psi_{5R}^{-2}$'."""
    if not content or content == "1":
        return r"$1$"
    factors = []
    for factor in content.split():
        name, _, power = factor.partition("^")
        base, _, sub = name.partition("_")
        base = r"\psi" if base == "psi" else r"\mathrm{%s}" % base.replace("_", r"\_")
        tex = base + (r"_{\mathrm{%s}}" % sub.replace("_", r"\_") if sub else "")
        factors.append(tex + (f"^{{{power}}}" if power else ""))
    return "$" + r"\,".join(factors) + "$"


def theory_label(meta):
    """Short label of a theory from result meta, e.g. 'Q_L = (3, 4), Q_R = (5, 0)'."""
    def fmt(rows):
        return "; ".join("(" + ", ".join(f"{q:g}" for q in row) + ")" for row in rows)
    return rf"$Q_L = {fmt(meta['QL'])},\ Q_R = {fmt(meta['QR'])}$"


def _listify(x):
    return list(x) if isinstance(x, (list, tuple)) else [x]


def _labels(results, labels):
    return labels if labels is not None else [theory_label(r.meta) if "QL" in r.meta else None for r in results]


def _rescale_residuals(res):
    """
    Very small or large residuals: divide the data by 10^e and put the factor into the axis label,
    instead of matplotlib's offset text (which collides with the main panel).
    """
    if res.get_yscale() != "linear":
        return
    values = np.concatenate([np.abs(line.get_ydata()) for line in res.get_lines()] or [np.zeros(1)])
    vmax = values[np.isfinite(values)].max(initial=0.0)
    if vmax == 0 or 1e-2 <= vmax < 1e3:
        return
    e = int(np.floor(np.log10(vmax)))
    for line in res.get_lines():
        line.set_ydata(np.asarray(line.get_ydata()) / 10.0**e)
    res.relim()
    res.autoscale_view()
    res.set_ylabel(res.get_ylabel() + rf" $[10^{{{e}}}]$")


def _finish(fig, ax, res, legend=True):
    with matplotlib.rc_context(STYLE):
        if res is not None:
            _rescale_residuals(res)
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best")
        fig.align_ylabels([a for a in (ax, res) if a is not None])
    return fig


# ----------------------------------------------------------------------
# spectra and series
# ----------------------------------------------------------------------
def plot_spectrum(spectra, labels=None, residuals=True):
    """Normal-mode frequencies omega_n against the mode index n, with the prediction (dashed)."""
    spectra = _listify(spectra)
    has_pred = all(s.prediction.size == s.frequencies.size for s in spectra)
    fig, ax, res = new_figure(residuals and has_pred)
    with matplotlib.rc_context(STYLE):
        for i, (s, label) in enumerate(zip(spectra, _labels(spectra, labels))):
            n = np.arange(1, s.frequencies.size + 1)
            c, mk = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
            ax.plot(n, s.frequencies, mk, color=c, label=label, markersize=2.2 - 0.4 * (i % 3),
                    markeredgewidth=0, alpha=0.9)
            if s.prediction.size == s.frequencies.size:
                ax.plot(n, s.prediction, "--", color=c, linewidth=0.8)
                if res is not None:
                    res.plot(n, s.frequencies - s.prediction, mk, color=c, markersize=1.8, markeredgewidth=0)
        ax.set_ylabel(r"$\omega_n$")
        (res or ax).set_xlabel(r"mode index $n$")
        if res is not None:
            res.set_ylabel(r"$\omega - \omega_{\rm pred}$")
    return _finish(fig, ax, res)


def plot_series(series, labels=None, residuals=True, logx=False, logy=False, abs_residuals=False,
                xlabel=None, ylabel=None):
    """y(x) with the prediction (dashed); optional residual panel (|y - prediction| on a log scale if abs_residuals)."""
    series = _listify(series)
    has_pred = all(s.prediction.size == s.y.size for s in series)
    fig, ax, res = new_figure(residuals and has_pred)
    with matplotlib.rc_context(STYLE):
        for i, (s, label) in enumerate(zip(series, _labels(series, labels))):
            c, mk = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
            ax.plot(s.x, s.y, "-", marker=mk, color=c, label=label, markersize=4 - 0.5 * (i % 3), linewidth=0.8)
            if s.prediction.size == s.y.size:
                ax.plot(s.x, s.prediction, "--", color=c, linewidth=0.8)
                if res is not None:
                    dev = s.y - s.prediction
                    res.plot(s.x, np.abs(dev) if abs_residuals else dev, mk, color=c, markersize=3)
        ax.set_xscale("log" if logx else "linear")
        ax.set_yscale("log" if logy else "linear")
        meta = series[0].meta
        ax.set_ylabel(ylabel or meta.get("ylabel", "y"))
        (res or ax).set_xlabel(xlabel or meta.get("xlabel", "x"))
        if res is not None:
            if abs_residuals:
                res.set_yscale("log")
            res.set_ylabel(r"$|y - y_{\rm pred}|$" if abs_residuals else r"$y - y_{\rm pred}$")
    return _finish(fig, ax, res)


# ----------------------------------------------------------------------
# fits and potentials
# ----------------------------------------------------------------------
def plot_fits(fits, labels=None, residuals=True):
    """Electric-field correlators (normalized to the first point) with their fits A K_0(mu r) + B."""
    fits = _listify(fits)
    fig, ax, res = new_figure(residuals)
    with matplotlib.rc_context(STYLE):
        for i, (f, label) in enumerate(zip(fits, labels or [None] * len(fits))):
            c = COLORS[i % len(COLORS)]
            scale = f.y[0]
            text = rf"$\mu_{{\rm fit}} = {f.mu:.4f}$, $\mu = {f.mu_expected:.4f}$"
            ax.plot(f.r, f.y / scale, ".", color=c, markersize=3, label=f"{label}: {text}" if label else text)
            ax.plot(f.r, f.model(f.r) / scale, "-", color=c, linewidth=0.8)
            if res is not None:
                res.plot(f.r, (f.y - f.model(f.r)) / scale, ".", color=c, markersize=2)
        ax.set_ylabel(r"$\langle E_{j_0} E_{j_0+r}\rangle$ (normalized)")
        (res or ax).set_xlabel(r"distance $r$")
        if res is not None:
            res.set_ylabel("residual")
    return _finish(fig, ax, res)


def plot_potentials(potentials, labels=None, residuals=True):
    """Static potentials V(d) of probe charges with the continuum prediction; residual panel V/V_pred - 1."""
    potentials = _listify(potentials)
    has_pred = all(p.prediction.size == p.V.size for p in potentials)
    fig, ax, res = new_figure(residuals and has_pred)
    with matplotlib.rc_context(STYLE):
        for i, (p, label) in enumerate(zip(potentials, labels or [None] * len(potentials))):
            c, mk = COLORS[i % len(COLORS)], MARKERS[i % len(MARKERS)]
            text = label or f"q = {p.meta['q']:g} ({p.meta['regime']})"
            ax.plot(p.d, p.V, mk, color=c, markersize=3.5 - 0.5 * (i % 3), label=text)
            if p.prediction.size == p.V.size:
                ax.plot(p.d, p.prediction, "-", color=c, linewidth=0.8)
                if res is not None:
                    res.plot(p.d, p.V / p.prediction - 1, mk, color=c, markersize=2.5)
        ax.set_ylabel(r"$V(d)$")
        (res or ax).set_xlabel(r"distance $d$")
        if res is not None:
            res.set_ylabel(r"$V/V_{\rm pred} - 1$")
    return _finish(fig, ax, res)


# ----------------------------------------------------------------------
# sectors
# ----------------------------------------------------------------------
def plot_sector_diagram(scan, annotate=("bound state", "elementary"), size=(4.6, 4.2)):
    """
    Massless sectors in the (h, hb) plane: filled points = numerics (coloured by kind), open circles =
    charge-lattice prediction, dotted lines = constant Delta = h + hb. Sectors of the given kinds are
    labelled with their fermion content.
    """
    fig, ax, _ = new_figure(False, size)
    with matplotlib.rc_context(STYLE):
        hmax = max([max(s.h, s.hb) for s in scan.numeric + scan.analytic], default=1.0) + 0.4
        for Delta in np.arange(0.5, 2 * hmax, 0.5):
            ax.plot([0, Delta], [Delta, 0], ":", color="0.7", linewidth=0.6, zorder=0)
        ax.plot([s.h for s in scan.analytic], [s.hb for s in scan.analytic], "o", markersize=9,
                markerfacecolor="none", markeredgecolor="black", markeredgewidth=0.8, label="charge lattice")
        for kind, color in KIND_COLORS.items():
            pts = [s for s in scan.numeric if s.kind == kind]
            if pts:
                ax.plot([s.h for s in pts], [s.hb for s in pts], "o", color=color, markersize=5, label=kind)
        labelled = {}
        for s in scan.numeric:
            if s.kind in annotate:
                labelled.setdefault((round(s.h, 6), round(s.hb, 6)), []).append(content_to_tex(s.content))
        for (h, hb), texts in labelled.items():
            ax.annotate("\n".join(texts), (h, hb), xytext=(6, 6), textcoords="offset points", fontsize=7)
        ax.set_xlim(-0.1, hmax)
        ax.set_ylim(-0.1, hmax)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$h$")
        ax.set_ylabel(r"$\bar h$")
        if "QL" in scan.meta:
            ax.set_title(theory_label(scan.meta))
    return _finish(fig, ax, None)


# ----------------------------------------------------------------------
# dispatch
# ----------------------------------------------------------------------
def plot_result(result, **kwargs):
    """Default plot for any result type."""
    if isinstance(result, SectorScan):
        return plot_sector_diagram(result, **kwargs)
    if isinstance(result, Spectrum):
        return plot_spectrum(result, **kwargs)
    if isinstance(result, Series):
        return plot_series(result, **kwargs)
    if isinstance(result, FitResult):
        return plot_fits(result, **kwargs)
    if isinstance(result, Potential):
        return plot_potentials(result, **kwargs)
    raise TypeError(f"No plot for {type(result).__name__}.")
