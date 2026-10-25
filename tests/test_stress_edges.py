"""Edge cases and numerical limits: zero-mode classification, tiny/huge couplings, minimal and large systems."""
import numpy as np
import pytest

from latsimpkg import ChiralTheory, Lattice, build_gauge_model, find_bound_states, model_3450, schwinger
from latsimpkg.analysis.sectors import sector_energy
from latsimpkg.quadratic import QuadraticHamiltonian


# ----------------------------------------------------------------------
# zero-mode classification in the core
# ----------------------------------------------------------------------
def free_plus_oscillator(eps):
    """q1: oscillator with M_q = eps (tiny), q2: free particle; p's with unit mass."""
    return np.diag([eps, 0.0, 1.0, 1.0])


def test_clear_zero_modes_need_no_hint():
    H = QuadraticHamiltonian(free_plus_oscillator(1e-3))
    assert H.n_free == 1 and H.frequencies == pytest.approx([np.sqrt(1e-3)])


def test_ambiguous_eigenvalue_is_not_guessed():
    with pytest.raises(ValueError, match="n_zero"):
        QuadraticHamiltonian(free_plus_oscillator(1e-12))


def test_ambiguous_eigenvalue_with_n_zero():
    H = QuadraticHamiltonian(free_plus_oscillator(1e-12), n_zero=1)
    assert H.n_free == 1 and H.frequencies == pytest.approx([1e-6], rel=1e-6)


@pytest.mark.parametrize("n_zero", [0, 2])
def test_wrong_n_zero_is_detected(n_zero):
    with pytest.raises(ValueError, match="double precision"):
        QuadraticHamiltonian(free_plus_oscillator(1e-3), n_zero=n_zero)


def test_n_zero_out_of_range():
    with pytest.raises(ValueError):
        QuadraticHamiltonian(np.eye(2), n_zero=3)


def test_condition_and_precision():
    H = QuadraticHamiltonian(np.diag([1e-4, 0.0, 1.0, 1.0]))
    assert H.condition == pytest.approx(1e4)
    assert H.precision == pytest.approx(1e4 * np.finfo(float).eps)


# ----------------------------------------------------------------------
# couplings at the limits of double precision
# ----------------------------------------------------------------------
@pytest.mark.parametrize("g", [1e-4, 1e-2, 1.0, 1e2])
def test_3450_bound_states_over_six_orders_of_magnitude(g):
    scan = find_bound_states(model_3450(g), N=40)
    assert scan.agrees and len(scan.bound_states) == 2


@pytest.mark.parametrize("g, match", [(1e-6, "error of"), (1e-8, "double precision"), (1e3, "error of")])
def test_unresolvable_couplings_raise(g, match):
    with pytest.raises(ValueError, match=match):
        find_bound_states(model_3450(g), N=40)


def test_sector_energy_is_guarded_as_well():
    with pytest.raises(ValueError, match="error of"):
        sector_energy(build_gauge_model(model_3450(1e-6), Lattice(40, 1.0, "ring")), (2, 1), (-2, 0))


def test_zero_coupling_is_the_free_theory():
    m = build_gauge_model(ChiralTheory([3, 4], [5, 0], g=0.0), Lattice(20, 1.0, "ring"))
    assert m.H.n_free == 4                                      # all fermion numbers conserved


# ----------------------------------------------------------------------
# sizes and spacings
# ----------------------------------------------------------------------
@pytest.mark.parametrize("boundary", ["open", "ring"])
def test_minimal_lattice(boundary):
    """N = 2: open chain k = 2 sin(pi n / 6), n = 1, 2; ring k = 2 sin(pi n / 2), n = 0, 1 (exact for Schwinger)."""
    g = 0.5
    mu = g / np.sqrt(np.pi)
    m = build_gauge_model(schwinger(g), Lattice(2, 1.0, boundary))
    n = np.arange(1, 3) if boundary == "open" else np.arange(0, 2)
    k = 2 * np.sin(np.pi * n / (6 if boundary == "open" else 2))
    assert np.allclose(m.H.frequencies, np.sqrt(mu**2 + k**2), atol=1e-13)


def test_minimal_ring_sector_energies_are_exact():
    """Zero-mode energies are exact on any ring, even with N = 2."""
    scan = find_bound_states(model_3450(0.3), N=2)
    assert scan.agrees and len(scan.bound_states) == 2


@pytest.mark.parametrize("a", [1e-3, 0.1, 10.0])
def test_lattice_spacing_scaling(a):
    """Physical results depend on a only through mu a: the sector spectrum is exact for any a."""
    scan = find_bound_states(ChiralTheory([3, 4], [5, 0], g=0.3 / a), N=30, a=a)
    assert scan.agrees and len(scan.bound_states) == 2


def test_large_charges():
    """5-12-13-0: 25 + 144 = 169; composites need |N| up to 5."""
    t = ChiralTheory([5, 12], [13, 0], g=0.1)
    scan = find_bound_states(t, N=30, nmax=5, Delta_max=1.0)
    assert scan.agrees and len(scan.bound_states) == 2


@pytest.mark.slow
def test_three_flavors_with_two_chiral_blocks():
    t = ChiralTheory([3, 4, 1], [5, 0, -1], g=0.3)
    scan = find_bound_states(t, N=24, nmax=2, Delta_max=1.5)
    assert scan.agrees and scan.meta["n_conserved"] == 5
