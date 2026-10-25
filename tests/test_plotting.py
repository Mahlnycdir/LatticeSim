"""Plotting of saved results and the figure pipeline for the documentation."""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from matplotlib.figure import Figure

from latsimpkg import (ChiralTheory, Lattice, Series, build_gauge_model, find_bound_states, model_3450, schwinger,
                       screening_potential, spectrum)
from latsimpkg.analysis.correlators import meson_mass_from_E
from latsimpkg.plotting import (content_to_tex, plot_fits, plot_potentials, plot_result, plot_sector_diagram,
                                plot_series, plot_spectrum, save_figure)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def results():
    return {
        "scan": find_bound_states(model_3450(0.3), N=20),
        "spectrum": spectrum(schwinger(0.3), 30),
        "ring_spectrum": spectrum(model_3450(0.3), 20, boundary="ring"),
        "series": Series(np.arange(1.0, 4.0), np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.1]),
                         {"xlabel": "x", "ylabel": "y"}),
        "fit": meson_mass_from_E(build_gauge_model(schwinger(0.1), Lattice(80, 1.0, "open"))),
        "potential": screening_potential(schwinger(0.1), 60, [5, 10, 20]),
    }


@pytest.mark.parametrize("content, tex", [
    ("psi_3L^2 psi_4L psi_5R^-2", r"$\psi_{\mathrm{3L}}^{2}\,\psi_{\mathrm{4L}}\,\psi_{\mathrm{5R}}^{-2}$"),
    ("psi_0R", r"$\psi_{\mathrm{0R}}$"),
    ("1", r"$1$"),
    ("L0^2 R1^-1", r"$\mathrm{L0}^{2}\,\mathrm{R1}^{-1}$"),
    ("spectator_R1", r"$\mathrm{spectator}_{\mathrm{R1}}$"),
])
def test_content_to_tex(content, tex):
    assert content_to_tex(content) == tex


def test_spectrum_plot_has_residual_panel(results):
    fig = plot_spectrum(results["spectrum"])
    assert isinstance(fig, Figure) and len(fig.axes) == 2
    assert len(fig.axes[0].get_lines()) == 2                     # data + prediction


def test_spectrum_without_prediction_has_no_residual_panel(results):
    assert len(plot_spectrum(results["ring_spectrum"]).axes) == 1


def test_several_spectra_get_a_legend(results):
    fig = plot_spectrum([results["spectrum"], results["spectrum"]], labels=["a", "b"])
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["a", "b"]


def test_series_plot_and_residuals(results):
    fig = plot_series(results["series"])
    res = fig.axes[1].get_lines()[0].get_ydata()
    assert np.allclose(res, [0, 0, -0.1])


def test_tiny_residuals_are_rescaled_into_the_label():
    s = Series(np.arange(3.0), np.ones(3), np.ones(3) + 1e-12, {})
    fig = plot_series(s)
    assert "10^{-12}" in fig.axes[1].get_ylabel()
    assert np.allclose(fig.axes[1].get_lines()[0].get_ydata(), -1.0, rtol=1e-3)   # 1 + 1e-12 is not exact in floats


def test_fit_plot(results):
    fig = plot_fits(results["fit"], labels=["vector"])
    assert len(fig.axes) == 2 and "mu" in fig.axes[0].get_legend().get_texts()[0].get_text()


def test_potential_plot(results):
    fig = plot_potentials(results["potential"])
    assert len(fig.axes) == 2 and "screening" in fig.axes[0].get_legend().get_texts()[0].get_text()


def test_sector_diagram_marks_bound_states(results):
    fig = plot_sector_diagram(results["scan"])
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert {"charge lattice", "bound state", "elementary", "multi-particle"} <= set(labels)
    annotations = " ".join(t.get_text() for t in ax.texts)
    assert r"\psi_{\mathrm{3L}}^{2}" in annotations


def test_dispatch(results):
    for key in ("scan", "spectrum", "series", "fit", "potential"):
        assert isinstance(plot_result(results[key]), Figure)
    with pytest.raises(TypeError):
        plot_result(42)


def test_save_figure_writes_pdf_and_png(tmp_path, results):
    paths = save_figure(plot_spectrum(results["spectrum"]), tmp_path / "sub" / "spec")
    assert [p.suffix for p in paths] == [".pdf", ".png"] and all(p.stat().st_size > 1000 for p in paths)


def test_plotting_does_not_use_pyplot():
    """No global pyplot state: importing and plotting must not import matplotlib.pyplot."""
    code = ("import sys, latsimpkg.plotting as p, latsimpkg as l; "
            "p.plot_spectrum(l.spectrum(l.schwinger(0.3), 10)); print('matplotlib.pyplot' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


# ----------------------------------------------------------------------
# figure pipeline
# ----------------------------------------------------------------------
def test_render_is_incremental(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    free_field = importlib.import_module("studies.free_field")
    sectors = importlib.import_module("studies.sectors")
    figures = importlib.import_module("studies.figures")
    free_field.run(tmp_path / "results")
    sectors.run(tmp_path / "results")
    names = ["free_chain_spectrum", "sectors_3450", "probe_potentials"]

    first = {e["name"]: e["status"] for e in figures.render(tmp_path / "results", tmp_path / "fig", names=names)}
    assert first["free_chain_spectrum"] == first["sectors_3450"] == "rendered"
    assert first["probe_potentials"].startswith("missing inputs")
    assert (tmp_path / "fig" / "sectors_3450.pdf").exists()

    second = {e["name"]: e["status"] for e in figures.render(tmp_path / "results", tmp_path / "fig", names=names)}
    assert second["sectors_3450"] == "up to date"

    forced = {e["name"]: e["status"] for e in figures.render(tmp_path / "results", tmp_path / "fig", force=True,
                                                             names=names)}
    assert forced["sectors_3450"] == "rendered"

    manifest = json.loads((tmp_path / "fig" / "manifest.json").read_text())
    assert {e["name"] for e in manifest} == set(names)
    assert all(e["caption"] for e in manifest)


def test_every_figure_has_inputs_from_a_known_study(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT))
    figures = importlib.import_module("studies.figures")
    studies_main = importlib.import_module("studies.__main__")
    for spec in figures.FIGURES:
        assert spec.study in studies_main.STUDIES
        assert all(p.startswith(spec.study + "/") for p in spec.inputs)
