"""
Property-based stress tests of the mathematical core (QuadraticHamiltonian).

Every test checks a theorem of the theory of quadratic Hamiltonians (docs, chapter 2) on randomly generated
Hamiltonians with known normal form M = S0^(-T) diag(omega, 0, omega, masses) S0^(-1): oscillators with
frequencies omega spread over several orders of magnitude, free particles, and random symplectic S0.
"""
import numpy as np
import scipy.linalg as sl
from hypothesis import given
from hypothesis import strategies as st

from latsimpkg.phasespace import PhaseSpace
from latsimpkg.quadratic import QuadraticHamiltonian


def random_symplectic(n, rng, scale):
    H = rng.normal(size=(2 * n, 2 * n))
    return sl.expm(PhaseSpace.flat(n).J @ (H + H.T) * scale)


@st.composite
def hamiltonians(draw, max_osc=5, max_free=2, allow_free=True):
    """(M, omega, masses, S0^(-1)) with known oscillators and free particles."""
    n_osc = draw(st.integers(1, max_osc))
    n_free = draw(st.integers(0, max_free)) if allow_free else 0
    omega = np.sort(np.array(draw(st.lists(st.floats(-2, 2), min_size=n_osc, max_size=n_osc))))
    omega = 10.0 ** omega                                       # 0.01 ... 100
    masses = 10.0 ** np.array(draw(st.lists(st.floats(-1, 1), min_size=n_free, max_size=n_free)))
    rng = np.random.default_rng(draw(st.integers(0, 2**31)))
    scale = draw(st.floats(0.02, 0.4))
    S0 = random_symplectic(n_osc + n_free, rng, scale)
    Si = np.linalg.inv(S0)
    diag = np.concatenate([omega, np.zeros(n_free), omega, masses])
    M = Si.T @ np.diag(diag) @ Si
    return 0.5 * (M + M.T), omega, masses, Si


def rel(a, b):
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


# ----------------------------------------------------------------------
# normal form
# ----------------------------------------------------------------------
@given(hamiltonians())
def test_frequencies_and_zero_modes_are_recovered(h):
    M, omega, masses, _ = h
    H = QuadraticHamiltonian(M)
    assert H.n_free == len(masses)
    assert rel(H.frequencies, omega) < 1e-7


@given(hamiltonians())
def test_williamson_diagonalizes(h):
    M, omega, _, _ = h
    w, S = QuadraticHamiltonian(M).williamson
    D = np.diag(np.concatenate([w, w]))
    assert np.max(np.abs(S.T @ M @ S - D)) < 1e-7 * max(1.0, w.max())


@given(hamiltonians(allow_free=False))
def test_williamson_is_symplectic(h):
    M = h[0]
    H = QuadraticHamiltonian(M)
    S = H.williamson[1]
    assert np.max(np.abs(S @ H.J @ S.T - H.J)) < 1e-6 * max(1.0, np.abs(S).max() ** 2)


@given(hamiltonians(allow_free=False, max_osc=4))
def test_frequencies_match_eigenvalues_of_JM(h):
    """The eigenvalues of J M are +- i omega_k (checked against a direct non-symmetric diagonalization)."""
    M, omega, _, _ = h
    H = QuadraticHamiltonian(M)
    ev = np.sort(np.abs(np.linalg.eigvals(H.J @ M).imag))[::2]
    assert rel(ev, H.frequencies) < 1e-6


# ----------------------------------------------------------------------
# ground state
# ----------------------------------------------------------------------
@given(hamiltonians())
def test_ground_state_energy_is_expectation_of_H(h):
    """<H> = (1/2) Tr(M sigma) = (1/2) sum omega (the i J/2 part drops out because Tr(M J) = 0)."""
    M, omega, _, _ = h
    H = QuadraticHamiltonian(M)
    assert abs(0.5 * np.trace(M @ H.covariance) / (0.5 * omega.sum()) - 1) < 1e-7


@given(hamiltonians(allow_free=False))
def test_ground_state_is_pure_and_physical(h):
    """sigma + iJ/2 >= 0 (uncertainty principle) with smallest eigenvalue 0 (pure state)."""
    H = QuadraticHamiltonian(h[0])
    sigma = H.covariance
    lam = np.linalg.eigvalsh(sigma + 0.5j * H.J)
    scale = max(1.0, np.abs(sigma).max())
    assert lam.min() > -1e-7 * scale
    assert abs(lam.min()) < 1e-7 * scale


@given(hamiltonians(allow_free=False, max_osc=4), st.floats(0.01, 3.0))
def test_ground_state_is_stationary(h, t):
    """Time evolution U = exp(J M t) is symplectic, conserves H (U^T M U = M) and leaves sigma invariant."""
    M = h[0]
    H = QuadraticHamiltonian(M)
    t = t / H.frequencies.max()                                  # at most a few periods of the fastest mode
    U = sl.expm(H.J @ M * t)
    tol = 1e-6 * max(1.0, np.abs(U).max() ** 2)
    assert np.max(np.abs(U @ H.J @ U.T - H.J)) < tol
    assert np.max(np.abs(U.T @ M @ U - M)) < tol * np.abs(M).max()
    sigma = H.covariance
    assert np.max(np.abs(U @ sigma @ U.T - sigma)) < tol * np.abs(sigma).max()


@given(hamiltonians(allow_free=False))
def test_covariance_matches_closed_formula(h):
    """sigma = (1/2) M^(-1/2) |M^(1/2) J M^(1/2)| M^(-1/2) (docs, eq. for sigma)."""
    H = QuadraticHamiltonian(h[0])
    A = H.sqrt @ H.J @ H.sqrt
    Msi = np.linalg.inv(H.sqrt)
    closed = 0.5 * Msi @ sl.sqrtm(A.T @ A).real @ Msi
    assert np.max(np.abs(H.covariance - closed)) < 1e-6 * np.abs(closed).max()


# ----------------------------------------------------------------------
# invariances
# ----------------------------------------------------------------------
@given(hamiltonians(), st.integers(0, 2**31), st.floats(0.02, 0.3))
def test_symplectic_change_of_variables(h, seed, scale):
    """X = S1 X': M' = S1^T M S1 has the same frequencies; sigma' = S1^(-1) sigma S1^(-T)."""
    M, omega, masses, _ = h
    n = M.shape[0] // 2
    S1 = random_symplectic(n, np.random.default_rng(seed), scale)
    H, H1 = QuadraticHamiltonian(M), QuadraticHamiltonian(S1.T @ M @ S1)
    assert H1.n_free == H.n_free
    assert rel(H1.frequencies, H.frequencies) < 1e-6
    if not len(masses):
        S1i = np.linalg.inv(S1)
        expected = S1i @ H.covariance @ S1i.T
        assert np.max(np.abs(H1.covariance - expected)) < 1e-5 * np.abs(expected).max()


@given(hamiltonians(), st.floats(-2, 2))
def test_scaling(h, log_lambda):
    """M -> lambda M: frequencies scale with lambda, the ground state does not change."""
    M = h[0]
    lam = 10.0 ** log_lambda
    H, Hs = QuadraticHamiltonian(M), QuadraticHamiltonian(lam * M)
    assert rel(Hs.frequencies, lam * H.frequencies) < 1e-7
    assert np.max(np.abs(Hs.covariance - H.covariance)) < 1e-6 * np.abs(H.covariance).max()


@given(hamiltonians(max_osc=3, max_free=1), hamiltonians(max_osc=3, max_free=1))
def test_direct_sum(h1, h2):
    """Two decoupled systems: frequencies are the union, zero-point energies and free particles add."""
    (M1, w1, m1, _), (M2, w2, m2, _) = h1, h2
    n1, n2 = M1.shape[0] // 2, M2.shape[0] // 2
    n = n1 + n2
    order = np.r_[0:n1, 2 * n1:2 * n1 + n2, n1:2 * n1, 2 * n1 + n2:2 * n]   # (q1, q2, p1, p2) from (q1, p1, q2, p2)
    M = sl.block_diag(M1, M2)[np.ix_(order, order)]
    H = QuadraticHamiltonian(M)
    assert H.n_free == len(m1) + len(m2)
    assert rel(H.frequencies, np.sort(np.concatenate([w1, w2]))) < 1e-7
    assert abs(H.ground_state_energy / (0.5 * (w1.sum() + w2.sum())) - 1) < 1e-7


# ----------------------------------------------------------------------
# free particles
# ----------------------------------------------------------------------
@given(hamiltonians(max_free=3).filter(lambda h: len(h[2]) > 0), st.integers(0, 2**31))
def test_conserved_charges_and_constrained_minimum(h, seed):
    M, _, masses, _ = h
    H = QuadraticHamiltonian(M)
    A = H.conserved_charges
    assert A.shape[1] == len(masses)
    assert np.max(np.abs(M @ H.J @ A)) < 1e-7 * np.abs(M).max()
    rng = np.random.default_rng(seed)
    c = rng.normal(size=A.shape[1])
    X = H.free_configuration(c)
    E = H.free_energy(c)
    assert np.allclose(A.T @ X, c, atol=1e-8 * max(1.0, np.abs(c).max()))
    assert abs(0.5 * X @ M @ X - E) < 1e-7 * max(1.0, abs(E))
    # every feasible perturbation (A^T dX = 0) raises the energy
    P = np.eye(M.shape[0]) - A @ A.T
    for _ in range(5):
        dX = P @ rng.normal(size=M.shape[0])
        assert 0.5 * (X + dX) @ M @ (X + dX) >= E - 1e-8 * max(1.0, abs(E))


@given(hamiltonians(max_free=2).filter(lambda h: len(h[2]) > 0))
def test_free_masses_are_recovered(h):
    """In normal-form coordinates the free particles have H = (1/2) mass_f p_f^2."""
    M, omega, masses, Si = h
    H = QuadraticHamiltonian(M)
    n = M.shape[0] // 2
    for f, mass in enumerate(masses):
        a = Si[n + len(omega) + f]
        _, E = H.constrained_minimum(a[:, None], [1.0])
        assert abs(E / (mass / 2) - 1) < 1e-7
