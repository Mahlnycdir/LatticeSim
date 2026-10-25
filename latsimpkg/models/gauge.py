"""
Construction kit: ChiralTheory + Lattice -> quadratic lattice Hamiltonian of the bosonized gauge theory.

Fields (after pairing into F Dirac bosons): varphi_{i,j}, P_{i,j} = a Pi_i(x_j), vartheta_{i,j} = sum_{k<=j} P_{i,k}.
Gauss law (temporal gauge), one electric field per U(1)_a:
    E_{a,j} = (g_a/sqrt(pi)) sum_i [ Q^V_{a,i} varphi_{i,j} + Q^A_{a,i} vartheta_{i,j} ] + e_{a,j},
with a c-number background e_{a,j} = g_a theta_a/(2 pi) + g_a sum_{x<=j} q^ext_{a,x} (theta angle and
static probe charges, E_j - E_(j-1) = g q^ext_j).
Hamiltonian:
    H = sum_{i,links} [ P^2 + l^2 ] / (2a) + (a/2) sum_{a,j} E_{a,j}^2 = (1/2) X^T M X + b^T X + const.

Boundaries:
    "open": Dirichlet chain varphi_{-1} = varphi_N = 0, N + 1 links.
    "ring": periodic, with the zero modes vartheta-bar_i and W_i = sqrt(pi) n_i (see `build_gauge_model`).

Commutator of the electric fields: with [varphi_j, vartheta_k] = i 1(j <= k),
    [E_{a,j}, E_{b,k}] = i (g_a g_b / pi) ( x_ab 1(j <= k) - x_ba 1(k <= j) ),   x_ab = Q^V_a . Q^A_b.
Gauge-anomaly freedom gives x_ab + x_ba = A_ab / 2 = 0, so for a = b the commutator vanishes. For a != b a
constant i x_ab remains; x_ab = (Q_R,a . Q_L,b - Q_L,a . Q_R,b)/4 depends on the pairing and is a regularization
artefact. It vanishes for a pairing with Q_L Q_R^T symmetric, which `build_gauge_model` searches for.
"""
import itertools
from dataclasses import dataclass

import numpy as np

from latsimpkg.conventions import SQRT_PI
from latsimpkg.lattice import Lattice
from latsimpkg.phasespace import PhaseSpace
from latsimpkg.quadratic import QuadraticHamiltonian
from latsimpkg.theory import ChiralTheory, Pairing


@dataclass(frozen=True, eq=False)
class GaugeModel:
    """
    A built lattice model. All matrices act on the phase-space vector X of `phasespace`.

    Attributes:
        theory (ChiralTheory): The continuum input.
        lattice (Lattice): Geometry and boundary condition.
        pairing (Pairing): Dirac bosons used for the bosonization.
        phasespace (PhaseSpace): Layout of X.
        H (QuadraticHamiltonian): Quadratic part (1/2) X^T M X.
        electric_field (numpy.ndarray): C of shape (k, N, dim) with E_a = C_a X + e_a.
        links (numpy.ndarray): B of shape (F, n_links, dim) with the links l_i = B_i X.
        background (numpy.ndarray): e of shape (k, N) (theta angle and probe charges).
        linear (numpy.ndarray): b of the linear term b^T X = a sum_a e_a^T C_a X.
        constant (float): (a/2) sum e^2.
    """
    theory: ChiralTheory
    lattice: Lattice
    pairing: Pairing
    phasespace: PhaseSpace
    H: QuadraticHamiltonian
    electric_field: np.ndarray
    links: np.ndarray
    background: np.ndarray
    linear: np.ndarray
    constant: float

    @property
    def M(self):
        return self.H.M

    @property
    def F(self):
        return self.pairing.F

    @property
    def k(self):
        return self.theory.k

    @property
    def QV(self):
        return self.pairing.QV

    @property
    def QA(self):
        return self.pairing.QA


    def gauge_charges(self, n, m):
        """Gauge charges q_a = Q^V_a . n + Q^A_a . m of the sector with windings n and momenta m."""
        return self.QV @ np.asarray(n, dtype=float) + self.QA @ np.asarray(m, dtype=float)


    def electric_field_commutator(self, a=0, b=0):
        """K with [E_{a,j}, E_{b,k}] = i K_jk."""
        C = self.electric_field
        return C[a] @ self.phasespace.J @ C[b].T


    def momentum(self, X):
        """Classical lattice momentum P = -(1/a) sum_i P_i^T l_i of the configuration X."""
        ps = self.phasespace
        return -sum(X[ps.slice("P", i)] @ (self.links[i] @ X) for i in range(self.F)) / self.lattice.a


    def _check_linear(self):
        K = self.H.kernel
        if K.shape[1] and np.abs(K.T @ self.linear).max() > 1e-9 * max(np.abs(self.linear).max(), 1.0):
            raise ValueError("The background couples to a flat direction: the energy is unbounded.")

    @property
    def mean_configuration(self):
        """Shift <X> = -M^+ b of the ground state caused by the background (0 without background)."""
        self._check_linear()
        return -self.H.pinv @ self.linear

    @property
    def background_energy(self):
        """Classical energy of the background, min_X [ (1/2) X^T M X + b^T X ] + const = const - (1/2) b^T M^+ b."""
        self._check_linear()
        return self.constant - 0.5 * self.linear @ self.H.pinv @ self.linear

    @property
    def ground_state_energy(self):
        """Zero-point energy of the oscillators plus the background energy."""
        return self.H.ground_state_energy + self.background_energy


# ----------------------------------------------------------------------
# building blocks
# ----------------------------------------------------------------------
def _commutation_defect(pairing):
    """max |x_ab| with x_ab = Q^V_a . Q^A_b: zero iff all electric fields commute (anomaly-free theory)."""
    return np.abs(pairing.QV @ pairing.QA.T).max()


def choose_pairing(theory, order=None):
    """
    Pairing for the lattice model. With `order` given it is used as is. Otherwise the given order is
    tried first, then all permutations of the right movers, until the electric fields of different U(1)s
    commute (x_ab = 0, see module docstring). For k = 1 every pairing works.

    Raises:
        ValueError: If the theory is anomaly free but no pairing gives commuting electric fields.
    """
    if order is not None:
        return theory.pairing(order)
    pairing = theory.pairing()
    if not theory.is_anomaly_free or _commutation_defect(pairing) < 1e-12:
        return pairing
    for perm in itertools.permutations(range(theory.n_R)):
        candidate = theory.pairing(perm)
        if _commutation_defect(candidate) < 1e-12:
            return candidate
    raise ValueError("No pairing of left and right movers gives commuting electric fields "
                     "(Q_L Q_R^T is not symmetric for any order of the right movers).")


def _phasespace(F, lattice):
    blocks = [("phi", "P", (F, lattice.N))]
    if lattice.is_ring:
        blocks.append(("theta_bar", "W", (F,)))
    return PhaseSpace(blocks)


def _link_matrices(F, lattice, ps):
    """Links l_i = B_i X: open chain with Dirichlet ends (N + 1 links), ring with winding (N links)."""
    N = lattice.N
    if lattice.is_ring:
        D = np.roll(np.eye(N), 1, axis=1) - np.eye(N)           # (D x)_j = x_(j+1) - x_j, 1^T D = 0
    else:
        D = (np.eye(N + 1, N) - np.eye(N + 1, N, k=-1))          # l_j = varphi_j - varphi_(j-1), varphi_(-1) = varphi_N = 0
    B = np.zeros((F, D.shape[0], ps.dim))
    for i in range(F):
        B[i][:, ps.slice("phi", i)] = D
        if lattice.is_ring:
            B[i][:, ps.index("W", i)] = 1.0 / N
    return B


def _electric_field_matrices(pairing, g, lattice, ps):
    N = lattice.N
    cumulative = np.tril(np.ones((N, N)))                         # vartheta = cumulative sum of P
    j = np.arange(N)
    C = np.zeros((len(g), N, ps.dim))
    for a in range(len(g)):
        for i in range(pairing.F):
            QV, QA = pairing.QV[a, i], pairing.QA[a, i]
            C[a][:, ps.slice("phi", i)] += QV * np.eye(N)
            C[a][:, ps.slice("P", i)] += QA * cumulative
            if lattice.is_ring:
                C[a][:, ps.index("W", i)] += QV * j / N          # winding part of varphi
                C[a][:, ps.index("theta_bar", i)] += QA        # dual zero mode
        C[a] *= g[a] / SQRT_PI
    return C


def _background(theory, lattice, theta, external_charges):
    k, N = theory.k, lattice.N
    e = np.zeros((k, N))
    if theta is not None:
        theta = np.broadcast_to(np.asarray(theta, dtype=float), (k,))
        e += (theory.g * theta / (2 * np.pi))[:, None]
    if external_charges is not None:
        q = np.asarray(external_charges, dtype=float)
        q = q.reshape(1, -1) if q.ndim == 1 else q
        if q.shape != (k, N):
            raise ValueError(f"external_charges must have shape ({k}, {N}), got {q.shape}.")
        if lattice.is_ring and np.abs(q.sum(axis=1)).max() > 1e-12:
            raise ValueError("On the ring the total external charge of every U(1) must vanish (Gauss law).")
        e += theory.g[:, None] * np.cumsum(q, axis=1)
    return e


def build_gauge_model(theory, lattice, theta=None, external_charges=None, pairing_order=None, tol=None):
    """
    Build the lattice Hamiltonian of a bosonized U(1)^k gauge theory.

    Ring: besides varphi~_{i,j}, P_{i,j} every flavor gets the dual zero mode vartheta-bar_i with conjugate
    W_i = sqrt(pi) n_i ([vartheta-bar_i, W_i] = i); varphi_{i,j} = varphi~_{i,j} + W_i j/N and
    vartheta_{i,j} = vartheta-bar_i + sum_{k<=j} P_{i,k}, giving [varphi_j, vartheta_k] = i (1(j <= k) - j/N).
    The winding and the total momentum are the fermion numbers (see `conventions`).

    Args:
        theory (ChiralTheory): Charges and couplings.
        lattice (Lattice): N, a and the boundary condition.
        theta (float | array_like, optional): Theta angle(s), constant background field g theta/(2 pi).
        external_charges (array_like, optional): Static probe charges per site, shape (k, N) or (N,) for k = 1.
        pairing_order (sequence, optional): Fixed pairing of the right movers (default: automatic).
        tol (Tolerances, optional): Passed to `QuadraticHamiltonian`.

    Returns:
        GaugeModel: The built model.
    """
    if not isinstance(theory, ChiralTheory):
        raise TypeError("theory must be a ChiralTheory.")
    if not isinstance(lattice, Lattice):
        raise TypeError("lattice must be a Lattice.")
    pairing = choose_pairing(theory, pairing_order)
    F, a = pairing.F, lattice.a
    ps = _phasespace(F, lattice)

    B = _link_matrices(F, lattice, ps)
    M = np.zeros((ps.dim, ps.dim))
    for i in range(F):
        M += B[i].T @ B[i] / a
        P = ps.slice("P", i)
        M[P, P] += np.eye(lattice.N) / a
    C = _electric_field_matrices(pairing, theory.g, lattice, ps)
    for Ca in C:
        M += a * Ca.T @ Ca

    e = _background(theory, lattice, theta, external_charges)
    linear = a * np.einsum("aj,ajd->d", e, C)
    constant = 0.5 * a * float(np.sum(e**2))

    H = QuadraticHamiltonian(M, ps) if tol is None else QuadraticHamiltonian(M, ps, tol)
    for arr in (C, B, e, linear):
        arr.setflags(write=False)
    return GaugeModel(theory, lattice, pairing, ps, H, C, B, e, linear, constant)
