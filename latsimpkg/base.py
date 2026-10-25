from abc import ABC, abstractmethod

import numpy as np

from latsimpkg.phasespace import PhaseSpace
from latsimpkg.quadratic import QuadraticHamiltonian


class LatticeHamiltonian(ABC):
    """
    Base class for all lattice simulations in the Hamiltonian formulation.

    Provides the lattice geometry (number of sites and lattice spacing), the storage
    for the Hamiltonian matrix M of H = (1/2) X^T M X and the connection to the
    model-independent core `QuadraticHamiltonian`, which does all the linear algebra
    (normal modes, ground state, zero modes, conserved charges).

    The model-specific physics (field content, discretization, couplings,
    boundary conditions) is defined in subclasses by overriding
    `free_hamiltonian_matrix`, `add_interacting_hamiltonian_matrix` and `phasespace`.

    Note: `normal_modes`, `ground_state_energy`, `ground_state_covariance` and
    `symplectic_matrix` are thin compatibility wrappers around `quadratic()`; they are
    removed together with this class in phase 3 of the roadmap.

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
        if not isinstance(N, (int, np.integer)) or isinstance(N, bool):
            raise TypeError(f"A simulation based on non-integer ({N}) is not possible!")
        elif N < 2:
            raise ValueError(f"A simulation on less then two ({N}) lattice points is not possible!")

        self.N = int(N)
        self.a = a
        self.hamiltonian = None


    @property
    def hamiltonian(self):
        return self._hamiltonian

    @hamiltonian.setter
    def hamiltonian(self, M):
        """Every assignment (also `+=`) invalidates the cached `QuadraticHamiltonian`."""
        self._hamiltonian = M
        self._quadratic = None


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


    @property
    def phasespace(self):
        """Layout of X. Subclasses with named fields override this; default: flat."""
        return PhaseSpace.flat(self.hamiltonian.shape[0] // 2)


    def quadratic(self, M = None):
        """
        The `QuadraticHamiltonian` of M (default: the stored Hamiltonian, cached until it changes).

        Raises:
            RuntimeError: If no Hamiltonian has been built yet.
        """
        if M is not None:
            return QuadraticHamiltonian(M)
        if self.hamiltonian is None:
            raise RuntimeError("No Hamiltonian built yet, call free_hamiltonian_matrix() first.")
        if self._quadratic is None:
            self._quadratic = QuadraticHamiltonian(self.hamiltonian, self.phasespace)
        return self._quadratic


    def symplectic_matrix(self, n):
        """Symplectic matrix J of n degrees of freedom (compatibility wrapper)."""
        return PhaseSpace.flat(n).J


    def normal_modes(self, M = None):
        """Sorted oscillator frequencies omega_k > 0 (compatibility wrapper)."""
        return self.quadratic(M).frequencies


    def ground_state_energy(self, M = None):
        """Zero-point energy (1/2) sum_k omega_k (compatibility wrapper)."""
        return self.quadratic(M).ground_state_energy


    def ground_state_covariance(self, M = None):
        """Symmetrised ground-state covariance (compatibility wrapper)."""
        return self.quadratic(M).covariance
