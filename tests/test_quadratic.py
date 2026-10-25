"""The core: PhaseSpace and QuadraticHamiltonian (model independent)."""
import numpy as np
import pytest
import scipy.linalg as sl

from latsimpkg.phasespace import PhaseSpace
from latsimpkg.quadratic import QuadraticHamiltonian


# ----------------------------------------------------------------------
# PhaseSpace
# ----------------------------------------------------------------------
def test_phasespace_layout():
    ps = PhaseSpace([("phi", "P", (2, 3)), ("theta_bar", "W", (2,))])
    assert ps.n == 8 and ps.dim == 16
    assert ps.index("phi", 1, 2) == 5
    assert ps.index("theta_bar", 1) == 7
    assert ps.index("P", 0, 0) == 8
    assert ps.index("W", 1) == 15
    assert ps.slice("phi", 1) == slice(3, 6)
    assert ps.slice("P") == slice(8, 14)


def test_phasespace_symplectic_matrix():
    J = PhaseSpace.flat(4).J
    assert np.array_equal(J.T, -J)
    assert np.array_equal(J @ J, -np.eye(8))


def test_phasespace_conjugate_pairs():
    """J pairs every coordinate with its own momentum: J[index(q), index(p)] = 1."""
    ps = PhaseSpace([("phi", "P", (2, 3)), ("theta_bar", "W", (2,))])
    assert ps.J[ps.index("phi", 1, 2), ps.index("P", 1, 2)] == 1
    assert ps.J[ps.index("theta_bar", 0), ps.index("W", 0)] == 1


@pytest.mark.parametrize("blocks", [[("a", "b", 3), ("a", "c", 2)], [("a", "b", 3), ("c", "b", 2)],
                                    [("a", "b", (0,))], [("a", "b", (2.5,))]])
def test_phasespace_rejects_bad_blocks(blocks):
    with pytest.raises(ValueError):
        PhaseSpace(blocks)


def test_phasespace_rejects_bad_indices():
    ps = PhaseSpace([("phi", "P", (2, 3))])
    with pytest.raises(IndexError):
        ps.index("phi", 1)
    with pytest.raises(KeyError):
        ps.index("psi", 0, 0)


# ----------------------------------------------------------------------
# validation
# ----------------------------------------------------------------------
@pytest.mark.parametrize("M", [np.eye(3), np.ones((2, 3)), np.array([[1.0, 2.0], [0.0, 1.0]]),
                               -np.eye(2), np.array([[np.nan, 0], [0, 1]])])
def test_rejects_invalid_matrices(M):
    with pytest.raises(ValueError):
        QuadraticHamiltonian(M)


def test_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        QuadraticHamiltonian(np.eye(4), PhaseSpace.flat(3))


def test_rejects_non_isotropic_kernel():
    """M = 0 for one degree of freedom: q and p both flat, no normal form."""
    with pytest.raises(ValueError, match="isotropic"):
        QuadraticHamiltonian(np.zeros((2, 2)))


def test_matrix_is_immutable():
    H = QuadraticHamiltonian(np.eye(2))
    with pytest.raises(ValueError):
        H.M[0, 0] = 2.0


# ----------------------------------------------------------------------
# Williamson normal form
# ----------------------------------------------------------------------
def random_symplectic(n, seed, scale=0.3):
    rng = np.random.default_rng(seed)
    H = rng.normal(size=(2 * n, 2 * n))
    return sl.expm(PhaseSpace.flat(n).J @ (H + H.T) * scale)


def model(omega, masses, seed):
    """
    M = S0^(-T) N S0^(-1) with a random symplectic S0: oscillators with frequencies omega and
    free particles with H = (1/2) masses_f p_f^2 (masses_f plays the role of 1/m). Returns M and S0^(-1).
    """
    n = len(omega) + len(masses)
    S0 = random_symplectic(n, seed)
    diag = np.concatenate([omega, np.zeros(len(masses)), omega, masses])
    Si = np.linalg.inv(S0)
    return Si.T @ np.diag(diag) @ Si, Si


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_frequencies_without_zero_modes(seed):
    omega = np.array([0.4, 1.1, 1.1, 2.7])
    M, _ = model(omega, [], seed)
    assert np.max(np.abs(QuadraticHamiltonian(M).frequencies - omega)) < 1e-10


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_frequencies_with_zero_modes(seed):
    omega = np.array([0.7, 1.3, 2.0])
    M, _ = model(omega, [1.0, 2.5], seed)
    H = QuadraticHamiltonian(M)
    assert H.n_free == 2 and H.n_oscillators == 3
    assert np.max(np.abs(H.frequencies - omega)) < 1e-10


@pytest.mark.parametrize("masses", [[], [1.0, 2.5]])
def test_williamson_diagonalizes(masses):
    M, _ = model(np.array([0.7, 1.3, 2.0]), masses, seed=3)
    omega, S = QuadraticHamiltonian(M).williamson
    assert np.abs(S.T @ M @ S - np.diag(np.concatenate([omega, omega]))).max() < 1e-10


def test_williamson_is_symplectic_without_zero_modes():
    M, _ = model(np.array([0.5, 0.9, 1.7]), [], seed=4)
    H = QuadraticHamiltonian(M)
    S = H.williamson[1]
    assert np.abs(S @ H.J @ S.T - H.J).max() < 1e-10


def test_oscillator_coordinates_commute_with_conserved_charges():
    M, _ = model(np.array([0.7, 1.3]), [1.0, 2.5], seed=5)
    H = QuadraticHamiltonian(M)
    S = H.williamson[1]
    assert np.abs(H.conserved_charges.T @ H.J @ S).max() < 1e-10


def test_conserved_charges_commute_with_hamiltonian():
    """M J A = 0, i.e. d/dt (A^T X) = A^T J M X = 0."""
    M, _ = model(np.array([0.7, 1.3]), [1.0, 2.5], seed=6)
    H = QuadraticHamiltonian(M)
    assert H.conserved_charges.shape[1] == 2
    assert np.abs(M @ H.J @ H.conserved_charges).max() < 1e-10


def test_free_particle_energy():
    """Free momenta p_f with H = (1/2) sum masses_f p_f^2: fixing p_f = 1 costs masses_f / 2."""
    masses = [1.0, 2.5]
    M, Si = model(np.array([0.7, 1.3]), masses, seed=7)
    H = QuadraticHamiltonian(M)
    n = H.n
    for f, mass in enumerate(masses):
        a = Si[n + 2 + f]                      # row of p_f = (S0^(-1) X)_(p_f)
        X, E = H.constrained_minimum(a[:, None], [1.0])
        assert E == pytest.approx(mass / 2, rel=1e-10)
        assert a @ X == pytest.approx(1.0, rel=1e-10)


def test_free_energy_matches_constrained_minimum():
    M, _ = model(np.array([0.7, 1.3]), [1.0, 2.5], seed=8)
    H = QuadraticHamiltonian(M)
    c = np.array([0.3, -1.2])
    X, E = H.constrained_minimum(H.conserved_charges, c)
    assert H.free_energy(c) == pytest.approx(E, rel=1e-12)
    assert np.allclose(H.free_configuration(c), X)


def test_constraint_along_kernel_is_rejected():
    M, _ = model(np.array([0.7]), [1.0], seed=9)
    H = QuadraticHamiltonian(M)
    with pytest.raises(ValueError):
        H.constrained_minimum(H.kernel, [1.0])


# ----------------------------------------------------------------------
# ground state
# ----------------------------------------------------------------------
def test_covariance_matches_closed_formula():
    """sigma = (1/2) M^(-1/2) |M^(1/2) J M^(1/2)| M^(-1/2) for positive definite M."""
    M, _ = model(np.array([0.5, 0.9, 1.7]), [], seed=10)
    H = QuadraticHamiltonian(M)
    A = H.sqrt @ H.J @ H.sqrt
    Msi = np.linalg.inv(H.sqrt)
    closed = 0.5 * Msi @ sl.sqrtm(A.T @ A).real @ Msi
    assert np.abs(H.covariance - closed).max() < 1e-10


def test_covariance_is_pure_state():
    M, _ = model(np.array([0.5, 0.9, 1.7]), [], seed=11)
    H = QuadraticHamiltonian(M)
    lam = np.linalg.eigvalsh(H.covariance + 0.5j * H.J)
    assert lam.min() == pytest.approx(0.0, abs=1e-10)


def test_oscillator_covariance_with_zero_modes():
    """In normal-form coordinates Y the oscillator part has <q^2> = <p^2> = 1/2."""
    omega = np.array([0.7, 1.3])
    M, Si = model(omega, [1.0, 2.5], seed=12)
    H = QuadraticHamiltonian(M)
    n = H.n
    rows = np.r_[0:2, n:n + 2]                 # q_1, q_2, p_1, p_2 of the oscillators
    Y = Si[rows] @ H.covariance @ Si[rows].T
    assert np.abs(Y - 0.5 * np.eye(4)).max() < 1e-10


def test_ground_state_energy():
    omega = np.array([0.7, 1.3, 2.0])
    M, _ = model(omega, [1.0], seed=13)
    assert QuadraticHamiltonian(M).ground_state_energy == pytest.approx(omega.sum() / 2, rel=1e-12)
