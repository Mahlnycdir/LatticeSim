import numpy as np

from latsimpkg.base import LatticeHamiltonian

class ScalarFieldTheory(LatticeHamiltonian):
    """
    We consider a scalar field theory, more precisley a gaussian one. 

    Args:
        LatticeHamiltonian (_type_): _description_
    """
    
    def __init__(self, N=100, a=1.0):
         super().__init__(N, a)


    def free_hamiltonian_matrix(self):
        """
        This represents the Hamiltonian matrix of our 
        discretized free scalar theory, namley
        H = 1/(2a) sum_j {P_j^2 + (phi_(j+1) - phi_j)^2}
        = (1/2) X^T M X,
        where we defined 
        X = (phi_0,...,phi_(N-1),P_0,...,P_(N-1))^T.
        We would find the Matrix as 
        [[ 2 -1  0  0  0 ... 0],
        [-1  2 -1  0  0 ... 0], 
        [ 0 -1  2 -1  0 ... 0],
        [ 0  0 -1  2 -1 ... 0],
                . 
                .  N - Times 
                .
        [ 0^(NxN),    I^(NxN)]]
        We dont make the 0|0th and N-1|N-1th entry to 1, because we want Dirichlet boundary conditions. 
        Free boundaries what be equal to set them to 1, but this results in non-invertability.
        Args:
                N (int)  : Number of lattice points
                a (float): Lattice spacing
        """
        N = self.N
        a = self.a 
        
        phi_matrix = ((2 * np.eye(N)) + -np.eye(N, k=1) - np.eye(N, k=-1)) / a
        p_matrix   = np.eye(N) / a 
        z = np.zeros((N,N))
        
        self.hamiltonian = np.block([[phi_matrix,   z   ],
                                     [     z,   p_matrix]]) 


    def normal_modes(self):
        """
        Normal-mode frequencies omega_k > 0 of H = 1/2 X^T M X (eigenvalues of J M are +- i omega_k).
        """
        
        ev = np.linalg.eigvals(self.symplectic_matrix() @ self.hamiltonian)
        return np.sort(ev.imag[ev.imag > 0])
    
    
    def ground_state_covariance(self, M = None):
        """Symmetrised ground-state covariance sigma_ab = 1/2 <{X_a, X_b}>."""
        if M is None:
            M = self.hamiltonian
            
        Ms = self.sqrtm_psd(M)
        Msi = np.linalg.inv(Ms)
        A = Ms @ self.symplectic_matrix(M.shape[0] // 2) @ Ms
        return 0.5 * Msi @ self.sqrtm_psd(A.T @ A) @ Msi
    
    
    def symplectic_matrix(self, N=None):
        """
        The symplectic matrix, defined by the commutator
        [X_a,X_b] = i J_ab,

        Args:
            N (int, optional): Number of degrees of freedom. Defaults to self.N.

        Returns:
            numpy.ndarray: The (2N x 2N) matrix [[0, I], [-I, 0]].
        """
        if N is None:
            N = self.N
        Z = np.zeros((N,N))
        I = np.eye(N)
        return np.block([[Z, I], [-I, Z]])