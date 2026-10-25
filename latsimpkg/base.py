from abc import ABC, abstractmethod

import numpy as np 


class LatticeHamiltonian(ABC):
    """
    Base class for all lattice simulations in the Hamiltonian formulation.

    Provides the infrastructure shared by every simulation: the lattice
    geometry (number of sites and lattice spacing), storage for the
    Hamiltonian matrix and generic routines for its diagonalization
    (energy spectrum, normal modes, ground state).

    The model-specific physics (field content, discretization, couplings,
    boundary conditions) is defined in subclasses by overriding
    `free_hamiltonian_matrix` and `interacting_hamiltonian_matrix`.

    Attributes:
        N (int): Number of lattice sites.
        a (float): Lattice spacing.
        hamiltonian (numpy.ndarray | None): Matrix representation of the
            Hamiltonian. None until it has been built by a subclass.
    """
    
    def __init__(self, N = 100, a = 1.0):
        """
        Initialize the lattice.

        Args:
            N (int, optional): Number of lattice sites, must be an integer
                               of at least 2. Defaults to 100. Defaults to 100.
            a (float, optional): Lattice spacing. Defaults to 1.0.

        Raises:
            ValueError: If N is not an integer or smaller than 2.
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
        
        
    def normal_modes(self):
        """
        Compute the normal modes of the system.

        The normal modes are the eigenvectors of `self.hamiltonian`. In the
        basis they span, the Hamiltonian is diagonal with the energy
        eigenvalues on the diagonal. They are the starting point for further
        calculations (energy spectrum, ground state, correlation functions).

        Returns:
            numpy.ndarray: Matrix whose columns are the normalized
                eigenvectors of the Hamiltonian.
        """        
        
        
    def energy_eigenvalues(self):
        """
        Compute the energy spectrum of a given Hamiltonian.

        Returns:
            numpy.ndarray: Energy eigenvalues of the Hamiltonian.
        """
        return np.linalg.eigvals(self.hamiltonian)
            
            
    def ground_state_covariance(self):
        """
        Determine the ground state covariance, i.e. the state of lowest energy, of a
        given Hamiltonian.

        Returns:
            numpy.ndarray: Ground state of the system.
        """
        
        
    def sqrtm_psd(self, symmetric_matrix):
        """Square root of a real symmetric positive semi-definite matrix."""
        w, U = np.linalg.eigh(symmetric_matrix)
        return (U * np.sqrt(np.clip(w, 0, None))) @ U.T
    