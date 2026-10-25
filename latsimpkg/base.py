from abc import ABC, abstractmethod

import numpy as np


class LatticeHamiltonian(ABC):
    """
    Base class for all lattice simulations in the Hamiltonian formulation.

    Provides the infrastructure shared by every simulation: the lattice
    geometry (number of sites and lattice spacing), storage for the
    Hamiltonian matrix and generic routines for quadratic Hamiltonians
        H = (1/2) X^T M X,   [X_a, X_b] = i J_ab,
    namely the symplectic matrix J, the normal-mode frequencies and the
    ground-state covariance. These routines only use M and therefore work
    for every Gaussian (free or Gaussian-interacting) model.

    The model-specific physics (field content, discretization, couplings,
    boundary conditions) is defined in subclasses by overriding
    `free_hamiltonian_matrix` and `add_interacting_hamiltonian_matrix`.

    Attributes:
        N (int): Number of lattice sites.
        a (float): Lattice spacing.
        hamiltonian (numpy.ndarray | None): Matrix M of H = (1/2) X^T M X.
            None until it has been built by a subclass.
    """

    def __init__(self, N = 100, a = 1.0):
        """
        Initialize the lattice.

        Args:
            N (int, optional): Number of lattice sites, must be an integer
                               of at least 2. Defaults to 100.
            a (float, optional): Lattice spacing. Defaults to 1.0.

        Raises:
            TypeError: If N is not an integer.
            ValueError: If N is smaller than 2.
        """
        if not isinstance(N, int):
            raise TypeError(f"A simulation based on non-integer ({N}) is not possible!")
        elif N < 2:
            raise ValueError(f"A simulation on less then two ({N}) lattice points is not possible!")

        self.N = N
        self.a = a
        self.hamiltonian = None


    @abstractmethod
    def free_hamiltonian_matrix(self):
        """
        Build the free (non-interacting) part of the Hamiltonian.

        The Hamiltonian of a field theory splits into a free and an
        interacting part. The free part depends on the specific model
        (kinetic term, mass terms, boundary conditions) and must therefore
        be implemented by every subclass. The resulting matrix is stored
        in `self.hamiltonian`.
        """


    def add_interacting_hamiltonian_matrix(self, *args):
        """
        Build the interacting part of the Hamiltonian.

        The interaction term is model-specific (e.g. coupling to a gauge
        field) and must be implemented by every subclass that contains
        interactions. The result is added to the free part stored in
        `self.hamiltonian`.
        """


    def _matrix(self, M):
        """Return M, or the stored Hamiltonian if M is None."""
        if M is not None:
            return M
        if self.hamiltonian is None:
            raise RuntimeError("No Hamiltonian built yet, call free_hamiltonian_matrix() first.")
        return self.hamiltonian


    def symplectic_matrix(self, n):
        """
        The symplectic matrix J, defined by the commutator [X_a, X_b] = i J_ab
        for X = (q_0, ..., q_(n-1), p_0, ..., p_(n-1)).

        Args:
            n (int): Number of degrees of freedom (half the phase-space dimension).

        Returns:
            numpy.ndarray: The (2n x 2n) matrix [[0, I], [-I, 0]].
        """
        Z = np.zeros((n, n))
        I = np.eye(n)
        return np.block([[Z, I], [-I, Z]])


    def normal_modes(self, M = None):
        """
        Normal-mode frequencies omega_k > 0 of H = (1/2) X^T M X.

        The eigenvalues of J M come in pairs +- i omega_k. J M is similar to
        M^(1/2) J M^(1/2), so the omega_k are the eigenvalues of the Hermitian
        matrix i M^(1/2) J M^(1/2), which is faster and more stable than a
        non-symmetric eigenvalue problem (requires positive definite M).

        Args:
            M (numpy.ndarray, optional): Hamiltonian matrix. Defaults to `self.hamiltonian`.

        Returns:
            numpy.ndarray: The sorted frequencies omega_k (one per degree of freedom).
        """
        M = self._matrix(M)
        n = M.shape[0] // 2
        Ms = self.sqrtm_psd(M)
        ev = np.linalg.eigvalsh(1j * (Ms @ self.symplectic_matrix(n) @ Ms))
        return ev[n:]


    def ground_state_energy(self, M = None):
        """
        Ground-state (zero-point) energy E_0 = (1/2) sum_k omega_k.

        Args:
            M (numpy.ndarray, optional): Hamiltonian matrix. Defaults to `self.hamiltonian`.

        Returns:
            float: The zero-point energy.
        """
        return 0.5 * np.sum(self.normal_modes(M))


    def ground_state_covariance(self, M = None):
        """
        Symmetrised ground-state covariance sigma_ab = (1/2) <{X_a, X_b}>.

        Uses sigma = (1/2) M^(-1/2) |M^(1/2) J M^(1/2)| M^(-1/2), valid for
        positive definite M.

        Args:
            M (numpy.ndarray, optional): Hamiltonian matrix. Defaults to `self.hamiltonian`.

        Returns:
            numpy.ndarray: The (2n x 2n) covariance matrix of the ground state.
        """
        M = self._matrix(M)
        Ms = self.sqrtm_psd(M)
        Msi = np.linalg.inv(Ms)
        A = Ms @ self.symplectic_matrix(M.shape[0] // 2) @ Ms
        return 0.5 * Msi @ self.sqrtm_psd(A.T @ A) @ Msi


    def sqrtm_psd(self, symmetric_matrix):
        """Square root of a real symmetric positive semi-definite matrix."""
        w, U = np.linalg.eigh(symmetric_matrix)
        return (U * np.sqrt(np.clip(w, 0, None))) @ U.T
