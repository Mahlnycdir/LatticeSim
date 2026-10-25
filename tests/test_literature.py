"""
Comparison with published results. Every test names its source; the bibliography of the documentation
(docs/chapters/references.tex) lists the full references, and the cross-check table in its appendix
summarizes the outcome.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import k0

from latsimpkg import ChiralTheory, find_bound_states, model_3450, schwinger, screening_potential, spectrum


# ----------------------------------------------------------------------
# Schwinger model
# ----------------------------------------------------------------------
@pytest.mark.parametrize("g", [0.04, 0.3, 1.0])
def test_schwinger_mass(g):
    """Schwinger, Phys. Rev. 128, 2425 (1962): the photon of 2d QED with massless fermions has mass e/sqrt(pi)."""
    assert schwinger(g).meson_masses[0] == pytest.approx(g / np.sqrt(np.pi), rel=1e-14)
    sp = spectrum(schwinger(g), 200)
    assert sp.max_deviation < 1e-12                       # sqrt(mu^2 + k^2) with mu = g / sqrt(pi)


@pytest.mark.parametrize("q", [1.0, 0.5])
def test_massless_schwinger_static_potential(q):
    """
    Bharadwaj, Isen, Kang, arXiv:2607.19468, eq. (3.88) (flat space):
        V(d) = (sqrt(pi)/2) e q^2 (1 - exp(-m d)),  m = e / sqrt(pi).
    Screening of every charge in the massless model: Coleman, Jackiw, Susskind, Ann. Phys. 93, 267 (1975).
    """
    g = 0.1
    pot = screening_potential(schwinger(g), 400, [5, 20, 60, 110], q=q)
    literature = np.sqrt(np.pi) / 2 * g * q**2 * (1 - np.exp(-g / np.sqrt(np.pi) * pot.d))
    assert np.max(np.abs(pot.V / literature - 1)) < 1e-3


def test_maxwell_string_tension():
    """Bharadwaj, Isen, Kang, arXiv:2607.19468, eq. (3.2): without dynamical charges V = (e^2 q^2 / 2) d."""
    g, q = 0.1, 1.0
    pot = screening_potential(ChiralTheory([0], [0], g), 100, [5, 20, 60], q=q)
    assert np.allclose(pot.V, g**2 * q**2 / 2 * pot.d, rtol=1e-12)


def test_equal_time_propagator_normalization():
    """
    The correlator fit assumes <phi(x) phi(0)> = int dk/2pi e^{ikx} / (2 omega_k) = K_0(mu |x|) / (2 pi)
    for a free boson of mass mu in 1+1 dimensions (checked here by direct integration).
    """
    mu, x = 0.7, 1.9
    # oscillatory Fourier integral: QUADPACK's QAWF via weight="cos"
    integral = quad(lambda k: 1 / (2 * np.sqrt(k**2 + mu**2)), 0, np.inf, weight="cos", wvar=x)[0] / np.pi
    assert integral == pytest.approx(k0(mu * x) / (2 * np.pi), rel=1e-6)


# ----------------------------------------------------------------------
# the 3-4-5-0 model
# ----------------------------------------------------------------------
G_3450 = np.array([3.0, 4.0, 5.0, 0.0])        # (psi1, psi2 | psit1, psit2) in Tong's notation
H_TONG = np.array([5.0, 0.0, 3.0, -4.0])


def test_3450_anomaly_free():
    """Wang, Wen, Phys. Rev. D 99, 111501(R) (2019) and Tong, JHEP 07 (2022) 001, sec. 3: 3^2 + 4^2 = 5^2 + 0^2."""
    assert model_3450().is_anomaly_free


def test_tong_second_symmetry_has_no_mixed_anomaly():
    """
    Tong, JHEP 07 (2022) 001, sec. 3: the 3-4-5-0 fermions have a second non-anomalous U(1), H = (5, 0 | 3, -4),
    with no mixed anomaly with G = (3, 4 | 5, 0). It is one of the surviving global symmetries of the gauged theory.
    """
    H = ChiralTheory([5, 0], [3, -4])
    assert H.is_anomaly_free
    assert G_3450[:2] @ H.QL[0] == G_3450[2:] @ H.QR[0]
    T = model_3450().global_symmetries()
    assert np.linalg.norm(H_TONG - T @ T.T @ H_TONG) < 1e-10


def test_tong_gauged_h_low_energy_dirac_fermion():
    """
    Tong, JHEP 07 (2022) 001, sec. 3 ("Gapping with a gauge interaction"): gauging H = (5, 0 | 3, -4) leaves at low
    energies a single massless Dirac fermion, the left mover Psi = psi2 and the right mover
    Psit = psi1^2 (psit1^dag)^2 psit2, with G[Psi] = 4 and G[Psit] = -4.
    """
    theory = ChiralTheory([5, 0], [3, -4], g=0.3, names=(["psi1", "psi2"], ["psit1", "psit2"]))
    scan = find_bound_states(theory)
    assert scan.agrees

    def G(s):
        return G_3450[:2] @ np.array(s.NL) + G_3450[2:] @ np.array(s.NR)

    psi = {s.content: s for s in scan.elementary}
    assert set(psi) == {"psi2", "psi2^-1"}
    assert (psi["psi2"].h, psi["psi2"].hb) == pytest.approx((0.5, 0.0)) and G(psi["psi2"]) == 4

    psit = {s.content: s for s in scan.bound_states}
    assert set(psit) == {"psi1^2 psit1^-2 psit2", "psi1^-2 psit1^2 psit2^-1"}
    composite = psit["psi1^2 psit1^-2 psit2"]
    assert (composite.h, composite.hb) == pytest.approx((0.0, 0.5)) and G(composite) == -4


def test_3450_is_the_parity_mirror_of_tongs_gauged_theory():
    """
    Gauging G = (3, 4 | 5, 0) (our convention) is, up to parity L <-> R and charge conjugation of one fermion,
    Tong's theory with H = (5, 0 | 3, -4) gauged: the spectra of massless sectors coincide with h <-> hb exchanged.
    """
    ours = find_bound_states(model_3450(0.3))
    tong = find_bound_states(ChiralTheory([5, 0], [3, -4], g=0.3))
    mirrored = sorted((round(s.hb, 6), round(s.h, 6), s.kind) for s in tong.numeric)
    assert sorted((round(s.h, 6), round(s.hb, 6), s.kind) for s in ours.numeric) == mirrored
