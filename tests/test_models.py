"""Model builder: U(1)^k, pairing choice, theta angle and probe charges."""
import numpy as np
import pytest

from latsimpkg.analysis.charge_lattice import charge_lattice_spectrum
from latsimpkg.analysis.sectors import charge_map, sector_scan
from latsimpkg.lattice import Lattice
from latsimpkg.models import build_gauge_model
from latsimpkg.theory import ChiralTheory, model_3450, schwinger
from physics import dirichlet_momenta, sector_signature


def build(theory, N=40, boundary="open", **kwargs):
    return build_gauge_model(theory, Lattice(N, 1.0, boundary), **kwargs)


# ----------------------------------------------------------------------
# input and structure
# ----------------------------------------------------------------------
def test_rejects_wrong_input_types():
    with pytest.raises(TypeError):
        build_gauge_model({"QL": [1], "QR": [1]}, Lattice(10))
    with pytest.raises(TypeError):
        build_gauge_model(schwinger(), 10)


@pytest.mark.parametrize("boundary, n, free", [("open", 80, 0), ("ring", 82, 3)])
def test_phase_space_and_zero_modes(boundary, n, free):
    """3-4-5-0: ring has the extra zero modes and 2F - k = 3 free particles (conserved charges)."""
    m = build(model_3450(0.3), boundary=boundary)
    assert m.H.n == n and m.H.n_free == free


def test_model_arrays_are_immutable():
    m = build(schwinger(0.3))
    with pytest.raises(ValueError):
        m.electric_field[0, 0, 0] = 1.0


# ----------------------------------------------------------------------
# U(1)^k
# ----------------------------------------------------------------------
def vector_like_spectrum(theory, N):
    """Vector-like charges act locally on varphi: branches sqrt(mu_a^2 + k^2) and F - k massless ones."""
    k = dirichlet_momenta(N)
    F = theory.n_L
    branches = [np.sqrt(mu**2 + k**2) for mu in theory.meson_masses] + [k] * (F - theory.k)
    return np.sort(np.concatenate(branches))


@pytest.mark.parametrize("QL, g", [(np.eye(2), [0.3, 0.5]),                 # two decoupled Schwinger copies
                                   ([[1, 1], [1, -1]], [0.3, 0.3]),          # rotated charges
                                   ([[1, 2, 0], [0, 1, 1]], [0.2, 0.3])])    # mixing, one massless flavor
def test_vector_like_u1_squared_spectrum(QL, g):
    t = ChiralTheory(QL, QL, g)
    om = build(t, N=40).H.frequencies
    assert np.max(np.abs(om - vector_like_spectrum(t, 40))) < 1e-12


def test_decoupled_copies_have_their_own_masses():
    t = ChiralTheory(np.eye(2), np.eye(2), [0.3, 0.5])
    assert t.meson_masses == pytest.approx(np.array([0.3, 0.5]) / np.sqrt(np.pi), rel=1e-14)


MIXED = ChiralTheory([[1, 1], [1, -1]], [[1, -1], [-1, -1]], g=[0.3, 0.7])   # chiral U(1)^2, anomaly free


def test_mixed_theory_is_anomaly_free():
    assert MIXED.is_anomaly_free


def test_pairing_is_chosen_such_that_electric_fields_commute():
    m = build(MIXED, N=20)
    assert m.pairing.pairs == ((0, 1), (1, 0))
    for a in range(2):
        for b in range(2):
            assert np.abs(m.electric_field_commutator(a, b)).max() < 1e-14


def test_bad_pairing_gives_constant_commutator():
    """[E_a,j, E_b,k] = i (g_a g_b/pi) x_ab (1 + delta_jk), x_ab = Q^V_a . Q^A_b (regularization artefact)."""
    m = build(MIXED, N=20, pairing_order=[0, 1])
    x = m.QV[0] @ m.QA[1]
    assert x != 0
    expected = MIXED.g[0] * MIXED.g[1] / np.pi * x * (np.ones((20, 20)) + np.eye(20))
    assert np.abs(m.electric_field_commutator(0, 1) - expected).max() < 1e-14


def test_mixed_theory_conserved_charges_on_the_ring():
    m = build(MIXED, N=20, boundary="ring")
    assert m.H.n_free == 2 * m.F - m.k
    T = charge_map(m)
    for a in range(m.k):                                  # every gauge charge is conserved
        gauge = np.concatenate([m.QV[a], m.QA[a]])
        assert np.linalg.norm(gauge - np.linalg.pinv(T) @ T @ gauge) < 1e-10 * np.linalg.norm(gauge)


def test_u1_squared_sectors_match_charge_lattice():
    """3-4-5-0 plus a vector-like U(1) on an extra flavor: same massless sectors as 3-4-5-0."""
    t = ChiralTheory([[3, 4, 0], [0, 0, 1]], [[5, 0, 0], [0, 0, 1]], g=[0.3, 0.3])
    m = build(t, N=30, boundary="ring")
    num = sector_scan(m, nmax=2, Delta_max=1.0)
    ana = charge_lattice_spectrum(m.pairing, nmax=2, Delta_max=1.0)
    assert sector_signature(num) == sector_signature(ana)
    assert sum(d["kind"] == "bound state" for d in num) == 2


# ----------------------------------------------------------------------
# theta angle and probe charges
# ----------------------------------------------------------------------
def test_background_does_not_change_the_spectrum():
    a = build(schwinger(0.3))
    b = build(schwinger(0.3), theta=1.3, external_charges=np.r_[1.0, np.zeros(38), -1.0])
    assert np.array_equal(a.M, b.M)


@pytest.mark.parametrize("theory", [schwinger(0.3), model_3450(0.3)])
def test_theta_is_unphysical_on_the_ring(theory):
    """Massless fermions: the theta background is absorbed by the zero modes, no energy cost."""
    assert build(theory, boundary="ring", theta=1.0).background_energy == pytest.approx(0.0, abs=1e-12)


def probe_pair(N, j0, d, q):
    ext = np.zeros(N)
    ext[j0] += q
    ext[j0 + d] -= q
    return ext


@pytest.mark.parametrize("q", [1.0, 0.5])
@pytest.mark.parametrize("d", [10, 40, 80])
def test_screening_potential(q, d):
    """Schwinger: V(d) = (g^2 q^2 / 2 mu) (1 - exp(-mu d)); every charge (also fractional) is screened."""
    g, N = 0.1, 400
    t = schwinger(g)
    mu = t.meson_masses[0]
    V = build(t, N=N, external_charges=probe_pair(N, 150, d, q)).background_energy
    assert V == pytest.approx(g**2 * q**2 / (2 * mu) * (1 - np.exp(-mu * d)), rel=1e-3)


@pytest.mark.parametrize("d", [5, 20, 60])
def test_confinement_without_charged_matter(d):
    """Only neutral fermions: no screening, V(d) = sigma d with string tension sigma = g^2 q^2 / 2."""
    g, q, N = 0.1, 1.0, 100
    V = build(ChiralTheory([0], [0], g), N=N, external_charges=probe_pair(N, 20, d, q)).background_energy
    assert V == pytest.approx(g**2 * q**2 * d / 2, rel=1e-12)


def test_ring_requires_neutral_probe_charges():
    with pytest.raises(ValueError, match="Gauss"):
        build(schwinger(0.3), boundary="ring", external_charges=np.r_[1.0, np.zeros(39)])


def test_external_charges_shape_is_checked():
    with pytest.raises(ValueError):
        build(schwinger(0.3), external_charges=np.zeros(7))
