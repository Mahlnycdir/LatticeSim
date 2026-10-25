"""Gaussian machinery: oscillator, free chain, ground state."""
import numpy as np
import pytest

from builders import THEORIES, free_chain, ground_state_covariance, normal_modes, open_gauge_model, symplectic_matrix
from physics import dirichlet_momenta


def test_single_oscillator_variance():
    w = 1.7
    sig = ground_state_covariance(free_chain(2), np.diag([w**2, 1.0]))
    assert sig[0, 0] == pytest.approx(1 / (2 * w), abs=1e-12)


def test_single_oscillator_momentum_variance():
    w = 1.7
    sig = ground_state_covariance(free_chain(2), np.diag([w**2, 1.0]))
    assert sig[1, 1] == pytest.approx(w / 2, abs=1e-12)
    assert sig[0, 1] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("a", [1.0, 0.001])
def test_dirichlet_chain_spectrum(a):
    N = 50
    om = normal_modes(free_chain(N, a))
    exact = dirichlet_momenta(N, a)
    assert np.max(np.abs(om - exact) / exact) < 1e-12


def test_number_of_modes_equals_degrees_of_freedom():
    assert len(normal_modes(free_chain(37))) == 37


@pytest.mark.parametrize("name", list(THEORIES) + ["free"])
def test_ground_state_is_pure(name):
    """sigma + iJ/2 >= 0 with equality (smallest eigenvalue 0) for a pure Gaussian state."""
    model = free_chain(40) if name == "free" else open_gauge_model(**THEORIES[name], N=40, g=0.3)
    sig = ground_state_covariance(model)
    lam = np.linalg.eigvalsh(sig + 0.5j * symplectic_matrix(model, sig.shape[0] // 2))
    assert lam.min() == pytest.approx(0.0, abs=1e-10)
    assert lam.min() > -1e-10


def test_zero_point_energy():
    N, a = 50, 0.001
    model = free_chain(N, a)
    assert model.ground_state_energy() == pytest.approx(0.5 * dirichlet_momenta(N, a).sum(), rel=1e-12)
