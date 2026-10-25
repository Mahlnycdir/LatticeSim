from dataclasses import dataclass

import numpy as np
import scipy.linalg as sl

from latsimpkg.conventions import vector_axial


def _readonly(x):
    x = np.array(x, dtype=float)
    x.setflags(write=False)
    return x


@dataclass(frozen=True)
class Pairing:
    """
    Pairing of the chiral fermions into Dirac bosons.

    Attributes:
        QV (numpy.ndarray): Vector charges, shape (k, F).
        QA (numpy.ndarray): Axial charges, shape (k, F).
        pairs (tuple): For every Dirac boson the pair (L index, R index); None marks an added
            neutral spectator.
        spectators (tuple): Indices of the Dirac bosons that contain a spectator.
    """
    QV: np.ndarray
    QA: np.ndarray
    pairs: tuple
    spectators: tuple

    @property
    def F(self):
        return self.QV.shape[1]


class ChiralTheory:
    """
    A (1+1)-dimensional U(1)^k gauge theory of chiral fermions, given by their charges.

    Charges: QL[a, i] is the charge of the left mover L_i under U(1)_a, QR[a, j] the one of the
    right mover R_j; g[a] is the coupling of U(1)_a.

    Gauge anomaly: the two-point function of the gauge currents gives the k x k matrix
        A_ab = Q_L,a . Q_L,b - Q_R,a . Q_R,b,
    which must vanish for a consistent gauge theory. In vector-axial form (after pairing)
    A_ab = 2 (Q^V_a . Q^A_b + Q^A_a . Q^V_b), the coefficient of [E_a, E_b] on the lattice.

    Meson masses: the electric energy (a/2) sum_a E_a^2 with E_a = (g_a/sqrt(pi)) (Q^V_a . varphi + Q^A_a . vartheta)
    gives at zero momentum the mass matrix
        mu^2_ab = (g_a g_b / pi) (Q^V_a . Q^V_b + Q^A_a . Q^A_b) = (g_a g_b / 2pi) (Q_L,a . Q_L,b + Q_R,a . Q_R,b).

    Global symmetries: a U(1) with charges t = (t_L, t_R) survives the gauging if it has no mixed
    anomaly with any gauge U(1), t_L . Q_L,a = t_R . Q_R,a for all a, i.e. t in the null space of
    [Q_L, -Q_R]. For an anomaly-free theory the gauge charges themselves are among them.

    Args:
        QL (array_like): Left charges, shape (k, n_L) or (n_L,) for k = 1.
        QR (array_like): Right charges, shape (k, n_R) or (n_R,) for k = 1.
        g (float | array_like): Couplings, one per U(1) (a scalar is used for all).
        names (tuple, optional): (names of the left movers, names of the right movers).
        allow_anomalous (bool, optional): Accept a gauge anomaly. Defaults to False.
        allow_fractional (bool, optional): Accept non-integer charges. Defaults to False.

    Raises:
        ValueError: For inconsistent shapes, wrong number or negative couplings, fractional
            charges (unless allowed) or a gauge anomaly (unless allowed).
    """

    def __init__(self, QL, QR, g=1.0, names=None, allow_anomalous=False, allow_fractional=False):
        QL = np.array(QL, dtype=float)
        QR = np.array(QR, dtype=float)
        QL = QL.reshape(1, -1) if QL.ndim <= 1 else QL
        QR = QR.reshape(1, -1) if QR.ndim <= 1 else QR
        if QL.ndim != 2 or QR.ndim != 2:
            raise ValueError("Charges must be given as (k, n) matrices.")
        if QR.size == 0 and QL.shape[0] > 1:
            QR = np.zeros((QL.shape[0], 0))
        if QL.size == 0 and QR.shape[0] > 1:
            QL = np.zeros((QR.shape[0], 0))
        if QL.shape[0] != QR.shape[0]:
            raise ValueError(f"Q_L has {QL.shape[0]} gauge rows, Q_R has {QR.shape[0]}.")
        if QL.shape[1] + QR.shape[1] == 0:
            raise ValueError("The theory needs at least one fermion.")
        if not (np.all(np.isfinite(QL)) and np.all(np.isfinite(QR))):
            raise ValueError("Charges must be finite.")
        k = QL.shape[0]

        g = np.atleast_1d(np.array(g, dtype=float))
        if g.size == 1 and k > 1:
            g = np.full(k, g[0])
        if g.shape != (k,):
            raise ValueError(f"Need {k} couplings (one per U(1)), got {g.size}.")
        if np.any(~np.isfinite(g)) or np.any(g < 0):
            raise ValueError(f"Couplings must be finite and non-negative, got {g}.")

        if not allow_fractional:
            for Q in (QL, QR):
                if not np.allclose(Q, np.round(Q), atol=1e-12):
                    raise ValueError("Fractional charges; pass allow_fractional=True if intended.")

        self.QL, self.QR, self.g = _readonly(QL), _readonly(QR), _readonly(g)
        if names is None:
            names = ([f"L{i}" for i in range(self.n_L)], [f"R{j}" for j in range(self.n_R)])
        names_L, names_R = (tuple(str(x) for x in part) for part in names)
        if len(names_L) != self.n_L or len(names_R) != self.n_R:
            raise ValueError("names must contain one name per left and per right mover.")
        self.names_L, self.names_R = names_L, names_R

        if not allow_anomalous and not self.is_anomaly_free:
            raise ValueError(f"Gauge anomaly Q_L Q_L^T - Q_R Q_R^T = {self.anomaly_matrix.tolist()} != 0; "
                             "pass allow_anomalous=True if intended.")


    @property
    def k(self):
        """Number of U(1) gauge fields."""
        return self.QL.shape[0]

    @property
    def n_L(self):
        return self.QL.shape[1]

    @property
    def n_R(self):
        return self.QR.shape[1]

    @property
    def anomaly_matrix(self):
        """A_ab = Q_L,a . Q_L,b - Q_R,a . Q_R,b."""
        return self.QL @ self.QL.T - self.QR @ self.QR.T

    @property
    def is_anomaly_free(self):
        scale = max(np.abs(self.QL).max(initial=0.0), np.abs(self.QR).max(initial=0.0), 1.0) ** 2
        return bool(np.abs(self.anomaly_matrix).max() <= 1e-12 * scale)


    def pairing(self, order=None):
        """
        Pair L_i with R_order[i] into Dirac bosons (default: in the given order).

        If n_L != n_R the missing partners are added as neutral fermions (charge 0 under every
        U(1)). A neutral fermion is free and decoupled: it changes no anomaly and only adds an
        elementary sector.

        Args:
            order (sequence, optional): Permutation of the right movers.

        Returns:
            Pairing: Q^V, Q^A of shape (k, F) with F = max(n_L, n_R), the pairs and the spectators.
        """
        if order is None:
            order = range(self.n_R)
        order = list(order)
        if sorted(order) != list(range(self.n_R)):
            raise ValueError(f"order must be a permutation of range({self.n_R}), got {order}.")
        F = max(self.n_L, self.n_R)
        QL = np.zeros((self.k, F)); QR = np.zeros((self.k, F))
        QL[:, :self.n_L] = self.QL
        QR[:, :self.n_R] = self.QR[:, order]
        pairs = tuple((i if i < self.n_L else None, order[i] if i < self.n_R else None) for i in range(F))
        spectators = tuple(i for i, (l, r) in enumerate(pairs) if l is None or r is None)
        QV, QA = vector_axial(QL, QR)
        return Pairing(_readonly(QV), _readonly(QA), pairs, spectators)


    @property
    def meson_mass_matrix(self):
        """mu^2_ab = (g_a g_b / 2pi) (Q_L,a . Q_L,b + Q_R,a . Q_R,b)."""
        Q2 = self.QL @ self.QL.T + self.QR @ self.QR.T
        return np.outer(self.g, self.g) * Q2 / (2 * np.pi)

    @property
    def meson_masses(self):
        """Sorted meson masses (square roots of the eigenvalues of mu^2)."""
        return np.sqrt(np.clip(np.linalg.eigvalsh(self.meson_mass_matrix), 0, None))


    def global_symmetries(self):
        """
        Basis of the U(1) symmetries without mixed anomaly with the gauge group: columns
        t = (t_L, t_R) (length n_L + n_R) with t_L . Q_L,a = t_R . Q_R,a for all a.
        """
        constraint = np.hstack([self.QL, -self.QR])
        return sl.null_space(constraint) if np.any(constraint) else np.eye(self.n_L + self.n_R)


    def __repr__(self):
        return (f"ChiralTheory(QL = {self.QL.tolist()}, QR = {self.QR.tolist()}, g = {self.g.tolist()})")


# ----------------------------------------------------------------------
# known theories
# ----------------------------------------------------------------------
def schwinger(g=1.0):
    """Schwinger model: one Dirac fermion with vector charge 1 (Q_L = Q_R = 1)."""
    return ChiralTheory([1], [1], g, names=(["psi_L"], ["psi_R"]))


def axial_schwinger(g=1.0):
    """Axially gauged Schwinger model (Q_L = 1, Q_R = -1); equivalent to the vector one by charge conjugation of psi_R."""
    return ChiralTheory([1], [-1], g, names=(["psi_L"], ["psi_R"]))


def model_3450(g=1.0):
    """3-4-5-0 model: left movers with charges 3, 4, right movers with 5, 0 (3^2 + 4^2 = 5^2 + 0^2)."""
    return ChiralTheory([3, 4], [5, 0], g, names=(["psi_3L", "psi_4L"], ["psi_5R", "psi_0R"]))
