import numpy as np

from scipy.special import k0
from scipy.optimize import curve_fit

from latsimpkg.base import LatticeHamiltonian

class ScalarFieldTheory(LatticeHamiltonian):
    """
    The bosonized Q gauge theory on a spatial lattice (vector-axial form).
    Here we consider Q as a charge vector. In our case we want to keep this
    as general as possible to analize other theories too.
    
    Lattice:  varphi_{i,j}, P_{i,j} = a Pi_i(x_j),  vartheta_{i,j} = sum_{k<=j} P_{i,k}.
              Gauss law (temporal gauge, no background field, open chain):
                  E_j = g/sqrt(pi) * sum_i ( Q^V_i varphi_{i,j} + Q^A_i vartheta_{i,j} )
              H = sum_{i,j} [ P_{i,j}^2/(2a) + (varphi_{i,j+1} - varphi_{i,j})^2/(2a) ] + (a/2) sum_j E_j^2.
              Continuum prediction: one massive mode with mu^2 = (g^2/pi) sum_i [(Q^V_i)^2 + (Q^A_i)^2].
              Phase-space ordering: X = (all varphi_{0,.}, ..., varphi_{F-1,.}, all P_{0,.}, ..., P_{F-1,.}).

    Args:
        LatticeHamiltonian (_type_): _description_
    """
    
    def __init__(self, N=100, a=1.0, g= 1.0, QV = [1], QA = [0], F = 1):
         super().__init__(N, a)
         self.QV = QV
         self.QA = QA
         self.F = F
         self.g = g
         
         
    def free_hamiltonian_matrix(self):
        """
        This represents the Hamiltonian matrix of a discretized free scalar theory for one field. 
        Namley,
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
        

    def add_interacting_hamiltonian_matrix(self):
        """
        We implement the electric field term. 
        """
        a = self.a
        if self.hamiltonian is None:
            self.free_hamiltonian_matrix()
        
        C = self.electric_field_matrix()
        self.hamiltonian += a * C.T @ C, C
        


    def normal_modes(self):
        """
        Normal-mode frequencies omega_k > 0 of H = 1/2 X^T M X (eigenvalues of J M are +- i omega_k).
        """
        ev = np.linalg.eigvals(self.symplectic_matrix() @ self.hamiltonian)
        return np.sort(ev.imag[ev.imag > 0])
    
    
    def ground_state_covariance(self, M = None):
        """
        Symmetrised ground-state covariance sigma_ab = 1/2 <{X_a, X_b}>.
        """
        if M is None:
            M = self.hamiltonian
            
        Ms = self.sqrtm_psd(M)
        Msi = np.linalg.inv(Ms)
        A = Ms @ self.symplectic_matrix(M.shape[0] // 2) @ Ms
        return 0.5 * Msi @ self.sqrtm_psd(A.T @ A) @ Msi 
    
    
    def electric_field_matrix(self):
        """
        Matrix C with E_j = (C X)_j.
        """
        QV = self.QV
        QA = self.QA
        N  = self.N
        g = self.g
        
        F = len(QV); n = F * N
        L = np.tril(np.ones((N, N)))                       # vartheta = L @ P  (cumulative sum)
        C = np.zeros((N, 2 * n))
        for i in range(F):
            C[:, i*N:(i+1)*N] += QV[i] * np.eye(N)         # varphi part: vector charge to the left
            C[:, n+i*N:n+(i+1)*N] += QA[i] * L             # vartheta part: axial charge to the left
        return g / np.sqrt(np.pi) * C
    
    
    def meson_mass(self, g):
        return g * np.sqrt(sum(v * v + w * w for v, w in zip(self.QV, self.QA)) / np.pi)
    
    
    def meson_mass_from_E(self, mu_guess):
        """
        Fit <E_j E_k> = (C sigma C^T)_jk in the bulk by A K_0(mu r) + B.
        (B != 0 for axial charges: uniform background-field fluctuation from the open boundary.)
        """
        C = self.electric_field_matrix()
        N = self.N
        G = C @ self.ground_state_covariance() @ C.T
        j0 = N // 2 - 60; r = np.arange(5, 110); y = G[j0, j0 + r]
        p, _ = curve_fit(lambda r, A, m, B: A * k0(m * r) + B, r, y,
                        p0=[y[0] / k0(mu_guess * 5), mu_guess, y[-1]], maxfev=20000)
        return p[1], p[2]
    
    
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