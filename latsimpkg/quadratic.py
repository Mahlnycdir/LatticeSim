from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.linalg as sl

from latsimpkg.phasespace import PhaseSpace


@dataclass(frozen=True)
class Tolerances:
    """
    All numerical thresholds in one place. Values are relative to the scale of M
    (its largest eigenvalue) unless stated otherwise.

    Attributes:
        symmetry (float): Allowed max|M - M^T| / max|M|.
        psd (float): Allowed negative eigenvalue of M, relative.
        kernel (float): Eigenvalues of M below this (relative) count as zero modes.
        isotropy (float): Allowed max|K^T J K| for the kernel K (absolute, K orthonormal).
    """
    symmetry: float = 1e-10
    psd: float = 1e-10
    kernel: float = 1e-10
    isotropy: float = 1e-8


class QuadraticHamiltonian:
    """
    A quadratic Hamiltonian H = (1/2) X^T M X with [X_a, X_b] = i J_ab, M real symmetric
    and positive semidefinite. Immutable; expensive decompositions are computed once.

    Decomposition (Williamson normal form including zero modes):
        K = ker M (columns orthonormal) are the flat directions. They must be isotropic,
        K^T J K = 0; then every zero mode is a free particle.
        Equations of motion X' = J M X, so a linear charge w^T X is conserved iff M J w = 0,
        i.e. the conserved charges are A = J K (orthonormal because J is orthogonal).
        The oscillators live on W = K^omega ∩ K^perp = {X : K^T X = 0, K^T J X = 0}, with basis B.
        There M_W = B^T M B is positive definite and Omega_W = B^T J B is the (non-standard)
        symplectic form of the coordinates xi = B^T X. Every X splits uniquely as
            X = M^+ A G c + B xi + K kappa,   c = A^T X,   G = (A^T M^+ A)^(-1),
        and the energy separates exactly,
            H = (1/2) c^T G c + (1/2) xi^T M_W xi.
        The first term is the kinetic energy of the free particles (conserved momenta c),
        the second a set of oscillators.

    Williamson for (M_W, Omega_W): with L = M_W^(1/2) the matrix L Omega_W L is antisymmetric;
    the eigenvectors of the Hermitian i L Omega_W L give an orthogonal O and frequencies D with
    O^T (L Omega_W L) O = [[0, D], [-D, 0]]. Then S_W = L^(-1) O diag(D, D)^(1/2) satisfies
        S_W^T M_W S_W = diag(D, D),   S_W J S_W^T = Omega_W,
    so xi = S_W Y with canonical Y and H_W = sum_k omega_k (q_k^2 + p_k^2)/2.

    Args:
        M (array_like): The (2n x 2n) Hamiltonian matrix.
        phasespace (PhaseSpace, optional): Layout of X. Defaults to `PhaseSpace.flat(n)`.
        tol (Tolerances, optional): Numerical thresholds.

    Raises:
        ValueError: If M is not square, has odd dimension, is not finite, not symmetric,
            not positive semidefinite, does not match the phase space, or has a non-isotropic kernel.
    """

    def __init__(self, M, phasespace=None, tol=Tolerances()):
        M = np.array(M, dtype=float)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError(f"M must be a square matrix, got shape {M.shape}.")
        if M.shape[0] % 2:
            raise ValueError(f"M must have even dimension 2n, got {M.shape[0]}.")
        if not np.all(np.isfinite(M)):
            raise ValueError("M contains non-finite entries.")
        scale = np.abs(M).max()
        if scale > 0 and np.abs(M - M.T).max() > tol.symmetry * scale:
            raise ValueError(f"M is not symmetric (max|M - M^T| = {np.abs(M - M.T).max():.2e}).")
        if phasespace is None:
            phasespace = PhaseSpace.flat(M.shape[0] // 2)
        if phasespace.dim != M.shape[0]:
            raise ValueError(f"M has dimension {M.shape[0]}, the phase space {phasespace.dim}.")
        M = 0.5 * (M + M.T)
        M.setflags(write=False)
        self.M = M
        self.phasespace = phasespace
        self.tol = tol
        self.n = M.shape[0] // 2
        w = self._eigh[0]
        if w.size and w.min() < -tol.psd * max(w.max(), 1.0):
            raise ValueError(f"M is not positive semidefinite (smallest eigenvalue {w.min():.2e}).")
        K = self.kernel
        if K.shape[1] and np.abs(K.T @ self.J @ K).max() > tol.isotropy:
            raise ValueError("The kernel of M is not isotropic (K^T J K != 0): the zero modes are not "
                             "free particles and the Hamiltonian has no Williamson normal form.")


    # ------------------------------------------------------------------
    # basic decompositions
    # ------------------------------------------------------------------
    @property
    def J(self):
        return self.phasespace.J

    @cached_property
    def _eigh(self):
        return np.linalg.eigh(self.M)

    @cached_property
    def _zero(self):
        """Mask of the eigenvalues of M that count as zero."""
        w = self._eigh[0]
        return w <= self.tol.kernel * max(w.max(initial=0.0), 1e-300)

    @cached_property
    def kernel(self):
        """Orthonormal basis K of ker M (flat directions), shape (2n, r)."""
        return self._eigh[1][:, self._zero]

    @property
    def n_free(self):
        """Number of free particles (zero modes) r."""
        return self.kernel.shape[1]

    @property
    def n_oscillators(self):
        return self.n - self.n_free

    @cached_property
    def pinv(self):
        """Moore-Penrose pseudo-inverse M^+."""
        w, U = self._eigh
        inv = np.where(self._zero, 0.0, 1.0 / np.where(self._zero, 1.0, w))
        return (U * inv) @ U.T

    @cached_property
    def sqrt(self):
        """M^(1/2)."""
        w, U = self._eigh
        return (U * np.sqrt(np.clip(w, 0, None))) @ U.T


    # ------------------------------------------------------------------
    # free particles
    # ------------------------------------------------------------------
    @cached_property
    def conserved_charges(self):
        """
        Orthonormal basis A (columns) of the conserved linear charges A^T X: M J A = 0, A = J K.
        """
        return self.J @ self.kernel

    @cached_property
    def free_kinetic_matrix(self):
        """G = (A^T M^+ A)^(-1): energy (1/2) c^T G c of the free particles with conserved charges c."""
        A = self.conserved_charges
        return np.linalg.inv(A.T @ self.pinv @ A)

    def free_energy(self, c):
        """Lowest energy (1/2) c^T G c in the sector with conserved charges c = A^T X."""
        c = np.asarray(c, dtype=float)
        return 0.5 * c @ self.free_kinetic_matrix @ c

    def free_configuration(self, c):
        """Minimum-energy configuration X* = M^+ A G c with A^T X* = c (oscillators at rest)."""
        return self.pinv @ self.conserved_charges @ self.free_kinetic_matrix @ np.asarray(c, dtype=float)

    def constrained_minimum(self, A, c):
        """
        Minimum of (1/2) X^T M X under the linear constraints A^T X = c (Lagrange multipliers):
            M X = A lambda  =>  X* = M^+ A (A^T M^+ A)^(-1) c,  E = (1/2) c^T (A^T M^+ A)^(-1) c.

        Args:
            A (array_like): Constraint vectors as columns, shape (2n, m). Must be orthogonal to ker M.
            c (array_like): Constraint values, shape (m,).

        Returns:
            tuple: (X*, E).

        Raises:
            ValueError: If a constraint has a component along ker M (the minimum is then not attained
                in this form: the constraint can be satisfied at no cost along a flat direction).
        """
        A = np.asarray(A, dtype=float).reshape(self.M.shape[0], -1)
        c = np.atleast_1d(np.asarray(c, dtype=float))
        if self.n_free and np.abs(self.kernel.T @ A).max() > self.tol.isotropy * max(np.abs(A).max(), 1.0):
            raise ValueError("A constraint has a component along ker M.")
        G = np.linalg.inv(A.T @ self.pinv @ A)
        X = self.pinv @ A @ G @ c
        return X, 0.5 * c @ G @ c


    # ------------------------------------------------------------------
    # oscillators
    # ------------------------------------------------------------------
    @cached_property
    def _reduced(self):
        """Basis B of W, M_W^(1/2) and the antisymmetric L Omega_W L."""
        K = self.kernel
        if K.shape[1] == 0:
            B = np.eye(self.M.shape[0])
        else:
            B = sl.null_space(np.vstack([K.T, K.T @ self.J]))
        MW = B.T @ self.M @ B
        wW, UW = np.linalg.eigh(0.5 * (MW + MW.T))
        L = (UW * np.sqrt(wW)) @ UW.T
        Linv = (UW / np.sqrt(wW)) @ UW.T
        OmegaW = B.T @ self.J @ B
        return B, L, Linv, L @ OmegaW @ L

    @cached_property
    def frequencies(self):
        """Sorted oscillator frequencies omega_k > 0 (n - r values; free particles excluded)."""
        A = self._reduced[3]
        m = A.shape[0] // 2
        return np.linalg.eigvalsh(1j * A)[m:]

    @cached_property
    def williamson(self):
        """
        Symplectic normal form of the oscillators.

        Returns:
            tuple: (omega, S) with the sorted frequencies omega (length n - r) and S of shape
                (2n, 2(n - r)) such that X_osc = S Y with canonical Y = (q_1.., p_1..),
                S^T M S = diag(omega, omega) and S J S^T = B Omega_W B^T.
        """
        B, L, Linv, A = self._reduced
        m = A.shape[0] // 2
        lam, V = np.linalg.eigh(1j * A)
        omega, V = lam[m:], V[:, m:]
        x = np.sqrt(2) * V.real                # A x = omega y,  A y = -omega x
        y = np.sqrt(2) * V.imag
        O = np.hstack([y, x])                  # O^T A O = [[0, D], [-D, 0]]
        S = B @ Linv @ O * np.sqrt(np.concatenate([omega, omega]))
        return omega, S


    # ------------------------------------------------------------------
    # ground state
    # ------------------------------------------------------------------
    @property
    def ground_state_energy(self):
        """Zero-point energy (1/2) sum_k omega_k of the oscillators."""
        return 0.5 * np.sum(self.frequencies)

    @cached_property
    def covariance(self):
        """
        Symmetrised ground-state covariance sigma_ab = (1/2) <{X_a, X_b}> = (1/2) S S^T.

        For positive definite M this is the full ground state. With zero modes it is the
        covariance of the oscillator part in a sharp-momentum state of the free particles:
        it is exact for every observable w^T X with K^T w = 0, while the flat directions K
        themselves have infinite variance (they are not contained in sigma).
        """
        S = self.williamson[1]
        return 0.5 * S @ S.T


    def __repr__(self):
        return (f"QuadraticHamiltonian(n = {self.n}, oscillators = {self.n_oscillators}, "
                f"free = {self.n_free}, {self.phasespace!r})")
