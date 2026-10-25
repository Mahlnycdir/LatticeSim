"""Phase 4: result types, saving/loading, the user-facing API and the studies."""
import importlib
from pathlib import Path

import numpy as np
import pytest

import latsimpkg
from latsimpkg import (ChiralTheory, FitResult, FitWindow, Lattice, Potential, Sector, SectorScan, Series, Spectrum,
                       build_gauge_model, find_bound_states, model_3450, schwinger, screening_potential, spectrum)
from latsimpkg.analysis.charge_lattice import fermion_content
from latsimpkg.analysis.correlators import meson_mass_from_E
from latsimpkg.analysis.results import SectorEnergy, sector_signature
from latsimpkg.analysis.sectors import sector_energy
from latsimpkg.analysis.spectrum import continuum_scaling, lowest_frequency_scaling


# ----------------------------------------------------------------------
# find_bound_states
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def scan_3450():
    return find_bound_states(model_3450(0.3), N=30)


def test_3450_bound_states(scan_3450):
    assert scan_3450.agrees and scan_3450.max_deviation < 1e-8
    assert {s.content for s in scan_3450.bound_states} == {"psi_3L^2 psi_4L psi_5R^-2", "psi_3L^-2 psi_4L^-1 psi_5R^2"}
    for s in scan_3450.bound_states:
        assert (s.h, s.hb) == pytest.approx((0.5, 0.0), abs=1e-8)


def test_3450_elementary_and_multi_particle(scan_3450):
    assert {s.content for s in scan_3450.elementary} == {"psi_0R", "psi_0R^-1"}
    assert len(scan_3450.multi_particle) == 8
    assert len(scan_3450.numeric) == len(scan_3450.analytic) == 12


def test_scan_meta(scan_3450):
    assert scan_3450.meta["N"] == 30 and scan_3450.meta["boundary"] == "ring"
    assert scan_3450.meta["n_conserved"] == 3


def test_schwinger_has_nothing_to_find():
    scan = find_bound_states(schwinger(0.3), N=20)
    assert scan.numeric == () and scan.analytic == () and scan.agrees


@pytest.mark.parametrize("g, N", [(1.0, 30), (0.3, 50), (0.05, 24)])
def test_massless_sectors_do_not_depend_on_g_or_N(scan_3450, g, N):
    assert sector_signature(find_bound_states(model_3450(g), N=N).numeric) == sector_signature(scan_3450.numeric)


def test_spectator_theory_names_the_added_fermion():
    """(3, 4 | 5): the pairing adds psi_0R as neutral spectator; same bound states as 3-4-5-0."""
    scan = find_bound_states(ChiralTheory([3, 4], [5], 0.3), N=30)
    assert {s.content for s in scan.elementary} == {"spectator_R1", "spectator_R1^-1"}
    assert len(scan.bound_states) == 2 and scan.agrees


# ----------------------------------------------------------------------
# spectra, scaling, fits, potentials
# ----------------------------------------------------------------------
def test_spectrum_schwinger_is_exact():
    sp = spectrum(schwinger(0.04), 120)
    assert sp.max_deviation < 1e-12
    assert sp.count_below(sp.meta["meson_masses"][0]) == 0


def test_spectrum_on_ring_has_no_prediction():
    sp = spectrum(model_3450(0.3), 20, boundary="ring")
    assert sp.prediction.size == 0 and np.isnan(sp.max_deviation)
    assert sp.meta["n_free"] == 3 and len(sp.frequencies) == 2 * 21 - 3


def test_lowest_frequency_scaling_schwinger():
    series = lowest_frequency_scaling(schwinger(0.04), [50, 100])
    assert np.array_equal(series.x, [50, 100]) and series.max_deviation < 1e-12


def test_continuum_scaling_falls_like_a_squared():
    series = continuum_scaling(schwinger(0.04), 120.0, [2.0, 1.0, 0.5])
    dev = np.abs(series.y - series.prediction)
    assert dev[0] / dev[1] == pytest.approx(4, rel=0.1) and dev[1] / dev[2] == pytest.approx(4, rel=0.1)


def test_fit_uses_explicit_window():
    model = build_gauge_model(schwinger(0.1), Lattice(200, 1.0, "open"))
    fit = meson_mass_from_E(model, window=FitWindow(60, 5, 80))
    assert fit.meta["window"] == [60, 5, 80] and len(fit.r) == 75
    assert fit.rel_deviation < 5e-3
    assert np.max(np.abs(fit.model(fit.r) - fit.y)) < 1e-3 * np.abs(fit.y).max()


def test_fit_window_must_fit():
    with pytest.raises(ValueError):
        FitWindow(150, 5, 80).check(200)


def test_fit_rejects_ring():
    with pytest.raises(ValueError):
        meson_mass_from_E(build_gauge_model(schwinger(0.1), Lattice(80, 1.0, "ring")))


def test_screening_and_confinement_regimes():
    s = screening_potential(schwinger(0.1), 300, [10, 50], q=0.5)
    c = screening_potential(ChiralTheory([0], [0], 0.1), 100, [5, 20])
    assert s.meta["regime"] == "screening" and s.max_rel_deviation < 1e-3
    assert c.meta["regime"] == "confinement" and c.max_rel_deviation < 1e-12


def test_probes_must_fit_on_the_lattice():
    with pytest.raises(ValueError):
        screening_potential(schwinger(0.1), 50, [10], j0=45)


def test_sector_energy_type():
    e = sector_energy(build_gauge_model(model_3450(0.3), Lattice(30, 1.0, "ring")), (2, 1), (-2, 0))
    assert isinstance(e, SectorEnergy) and e.q.shape == (1,) and e.Delta == pytest.approx(0.5)


# ----------------------------------------------------------------------
# result types and saving/loading
# ----------------------------------------------------------------------
def test_sector_properties():
    s = Sector((2, 1), (-2, 0), 0.5, 0.5, "bound state", "x")
    assert (s.h, s.hb, s.key) == (0.5, 0.0, ((2, 1), (-2, 0)))
    assert Sector.from_json(s.to_json()) == s
    assert s.as_dict()["h"] == 0.5


def test_fermion_content_without_theory():
    assert fermion_content((2, 0), (0, -1)) == "L0^2 R1^-1"
    assert fermion_content((0,), (0,)) == "1"


def results_for_roundtrip(scan):
    return [
        scan,
        spectrum(schwinger(0.3), 20),
        lowest_frequency_scaling(schwinger(0.3), [10, 20]),
        meson_mass_from_E(build_gauge_model(schwinger(0.1), Lattice(80, 1.0, "open"))),
        screening_potential(schwinger(0.1), 60, [5, 10]),
    ]


def assert_same(a, b):
    assert type(a) is type(b)
    for name in a.__dataclass_fields__:
        x, y = getattr(a, name), getattr(b, name)
        if isinstance(x, np.ndarray):
            assert np.array_equal(x, y)
        elif name == "meta":
            assert set(x) == set(y)
        else:
            assert x == y


def test_save_and_load_roundtrip(tmp_path, scan_3450):
    for i, result in enumerate(results_for_roundtrip(scan_3450)):
        path = tmp_path / f"r{i}.npz"
        result.save(path)
        assert_same(result, type(result).load(path))


def test_load_checks_the_type(tmp_path):
    spectrum(schwinger(0.3), 10).save(tmp_path / "s.npz")
    with pytest.raises(TypeError):
        Potential.load(tmp_path / "s.npz")


def test_saved_meta_is_plain_json(tmp_path, scan_3450):
    """numpy scalars in meta must not break saving."""
    s = Series(np.arange(3.0), np.arange(3.0), np.array([]), {"n": np.int64(3), "v": np.float32(1.5)})
    s.save(tmp_path / "s.npz")
    assert Series.load(tmp_path / "s.npz").meta == {"n": 3, "v": 1.5}


def test_public_api_is_importable():
    for name in latsimpkg.__all__:
        assert hasattr(latsimpkg, name)


# ----------------------------------------------------------------------
# studies (smoke tests)
# ----------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent


def load_study(name, monkeypatch):
    """The studies are scripts in the project, not part of the package: put the project root on the path."""
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module(f"studies.{name}")


@pytest.mark.parametrize("name", ["free_field", "sectors"])
def test_fast_studies_run_and_save(tmp_path, capsys, monkeypatch, name):
    study = load_study(name, monkeypatch)
    study.report(study.run(tmp_path))
    assert any((tmp_path / name).glob("*.npz"))
    assert capsys.readouterr().out.strip()


@pytest.mark.slow
@pytest.mark.parametrize("name", ["spectra", "correlators", "potentials"])
def test_slow_studies_run_and_save(tmp_path, capsys, monkeypatch, name):
    study = load_study(name, monkeypatch)
    study.report(study.run(tmp_path))
    assert any((tmp_path / name).glob("*.npz"))
