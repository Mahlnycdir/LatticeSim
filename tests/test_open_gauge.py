"""Gauged theories on the open chain: spectrum, Gauss law, meson mass."""
import numpy as np
import pytest

from builders import (THEORIES, electric_field_commutator, meson_mass, meson_mass_fit, normal_modes,
                      open_gauge_model)
from physics import dirichlet_momenta, gauge_anomaly, gauged_spectrum
from physics import meson_mass as meson_mass_formula

N, G = 120, 0.04


@pytest.mark.parametrize("name", THEORIES)
def test_meson_mass_formula(name):
    model = open_gauge_model(**THEORIES[name], N=10, g=G)
    assert meson_mass(model) == pytest.approx(meson_mass_formula(**THEORIES[name], g=G), rel=1e-14)


def test_schwinger_spectrum_is_exact():
    """The vector coupling acts locally on varphi: omega = sqrt(mu^2 + k^2) exactly on the lattice."""
    om = normal_modes(open_gauge_model(**THEORIES["schwinger"], N=N, g=G))
    assert np.max(np.abs(om - gauged_spectrum(**THEORIES["schwinger"], N=N, g=G))) < 1e-12


@pytest.mark.parametrize("name", ["axial_schwinger", "3450"])
def test_axial_spectrum_close_to_prediction(name):
    """Axial charges shift the boundary momenta (phase shift), so only approximate agreement."""
    th = THEORIES[name]
    om = normal_modes(open_gauge_model(**th, N=N, g=G))
    assert len(om) == N * len(th["QL"])
    assert np.max(np.abs(om - gauged_spectrum(**th, N=N, g=G))) < 2e-2


@pytest.mark.parametrize("name", THEORIES)
def test_gauss_law_consistent_for_anomaly_free(name):
    th = THEORIES[name]
    assert gauge_anomaly(**th) == 0
    assert np.abs(electric_field_commutator(open_gauge_model(**th, N=30, g=G))).max() < 1e-14


@pytest.mark.parametrize("QL, QR", [([2, 1], [0, 1]), ([2, 2], [0, 0]), ([3, 1], [1, 1])])
def test_anomalous_commutator(QL, QR):
    """[E_j, E_k] = i (g^2/pi) (Q^V.Q^A) sign(k - j), Q^V.Q^A = (sum Q_L^2 - sum Q_R^2)/4."""
    comm = np.abs(electric_field_commutator(open_gauge_model(QL, QR, N=30, g=G, allow_anomalous=True))).max()
    assert comm == pytest.approx(G**2 / np.pi * abs(gauge_anomaly(QL, QR)) / 4, rel=1e-12)


def test_schwinger_finite_size_lowest_mode():
    n = 100
    om = normal_modes(open_gauge_model(**THEORIES["schwinger"], N=n, g=G))
    k1 = dirichlet_momenta(n)[0]
    assert om[0] == pytest.approx(np.sqrt(meson_mass_formula(**THEORIES["schwinger"], g=G)**2 + k1**2), rel=1e-12)


def test_schwinger_no_mode_below_mu():
    model = open_gauge_model(**THEORIES["schwinger"], N=N, g=G)
    assert np.sum(normal_modes(model) < meson_mass(model)) == 0


def test_continuum_limit_at_fixed_length():
    """
    At fixed L = (N+1) a, omega_min -> sqrt(mu^2 + (pi/L)^2). The lattice dispersion
    k_lat = (2/a) sin(k a/2) deviates by O((k a)^2), so halving a must reduce the deviation by ~4.
    """
    L = 120.0
    mu = meson_mass_formula(**THEORIES["schwinger"], g=G)
    dev = []
    for a in [2.0, 1.0, 0.5]:
        n = int(round(L / a))
        om = normal_modes(open_gauge_model(**THEORIES["schwinger"], N=n, g=G, a=a))
        dev.append(abs(om[0] - np.sqrt(mu**2 + (np.pi / ((n + 1) * a))**2)))
    assert dev[0] / dev[1] == pytest.approx(4, rel=0.1)
    assert dev[1] / dev[2] == pytest.approx(4, rel=0.1)
    assert dev[2] < 1e-5


@pytest.mark.slow
@pytest.mark.parametrize("name, g", [("schwinger", 0.1), ("axial_schwinger", 0.1), ("3450", 0.04)])
def test_meson_mass_from_correlator(name, g):
    model = open_gauge_model(**THEORIES[name], N=300, g=g)
    mu_fit, _ = meson_mass_fit(model)
    assert mu_fit == pytest.approx(meson_mass(model), rel=2e-3)
