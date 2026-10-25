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
        H = 1/(2a) sum_j {P_j^2 + (phi_(j+1) - phi_j)}
        = (1/2) X^T M X,
        where we defined 
        X = (phi_0,...,phi_(N-1),P_0,...,P_(N-1))^T.
        We would find the Matrix as 
        [[ 1 -1  0  0  0 ... 0],
        [-1  2 -1  0  0 ... 0], 
        [ 0 -1  2 -1  0 ... 0],
        [ 0  0 -1  2 -1 ... 0],
                . 
                .  N - Times 
                .
        [ 0^(NxN),    I^(NxN)]]
        Args:
                N (int)  : Number of lattice points
                a (float): Lattice spacing
        """
        N = self.N
        a = self.a 
        
        phi_matrix = ((2 * np.eye(N)) + -np.eye(N, k=1) - np.eye(N, k=-1)) / a
        phi_matrix[0][0] *= 0.5 
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
        A = Ms @ self.symplectic_matrix() @ Ms
        return 0.5 * Msi @ self.sqrtm_psd(A.T @ A) @ Msi
    
    
    def symplectic_matrix(self):
        """
        The symplectic matrix, defined by the commutator
        [X_a,X_b] = i J_ab,

        Args:
            N (_type_): _description_

        Returns:
            _type_: _description_
        """
        N = self.N
        Z = np.zeros((N,N))
        I = np.eye(N)
        return np.block([[Z, I], [-I, Z]])