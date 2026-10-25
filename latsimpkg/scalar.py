import numpy as np

from scipy.special import k0
from scipy.optimize import curve_fit

from latsimpkg.base import LatticeHamiltonian

class ScalarFieldGaugeTheory(LatticeHamiltonian):
    """
    The bosonized U(1) gauge theory on a spatial lattice (vector-axial form).
    Every Dirac fermion i is replaced by a compact boson varphi_i that carries
    the vector charge Q^V_i and the axial charge Q^A_i. The charges are kept
    general so that other theories (Schwinger model, 3-4-5-0 model, ...) can be
    analysed with the same code.

    Lattice:  varphi_{i,j}, P_{i,j} = a Pi_i(x_j),  vartheta_{i,j} = sum_{k<=j} P_{i,k}.
              Gauss law (temporal gauge, no background field, open chain):
                  E_j = g/sqrt(pi) * sum_i ( Q^V_i varphi_{i,j} + Q^A_i vartheta_{i,j} )
              H = sum_{i,j} [ P_{i,j}^2/(2a) + (varphi_{i,j+1} - varphi_{i,j})^2/(2a) ] + (a/2) sum_j E_j^2.
              Continuum prediction: one massive mode with mu^2 = (g^2/pi) sum_i [(Q^V_i)^2 + (Q^A_i)^2],
              the remaining F - 1 modes stay massless.
              Gauge-anomaly freedom: Q^V . Q^A = 0, otherwise [E_j, E_k] != 0.
              Phase-space ordering: X = (all varphi_{0,.}, ..., varphi_{F-1,.}, all P_{0,.}, ..., P_{F-1,.}).

    Attributes:
        g (float): Gauge coupling.
        QV (numpy.ndarray): Vector charges, one per flavor.
        QA (numpy.ndarray): Axial charges, one per flavor.
        F (int): Number of flavors, F = len(QV).
    """

    def __init__(self, N=100, a=1.0, g=1.0, QV=(1,), QA=(0,)):
        """
        Args:
            N (int, optional): Number of lattice sites. Defaults to 100.
            a (float, optional): Lattice spacing. Defaults to 1.0.
            g (float, optional): Gauge coupling. Defaults to 1.0.
            QV (sequence, optional): Vector charges. Defaults to (1,) (Schwinger model).
            QA (sequence, optional): Axial charges. Defaults to (0,).

        Raises:
            ValueError: If QV and QA do not have the same length.
        """
        super().__init__(N, a)
        self.QV = np.asarray(QV, dtype=float)
        self.QA = np.asarray(QA, dtype=float)
        if self.QV.shape != self.QA.shape:
            raise ValueError(f"QV ({len(self.QV)}) and QA ({len(self.QA)}) need one entry per flavor!")
        self.F = len(self.QV)
        self.g = g


    def free_hamiltonian_matrix(self):
        """
        This represents the Hamiltonian matrix of the discretized free scalar
        theory for F decoupled fields. For one field,
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
                [ 0^(NxN),    I^(NxN)]] / a
        We dont make the 0|0th and N-1|N-1th entry to 1, because we want Dirichlet boundary conditions.
        Free boundaries would be equal to set them to 1, but this results in non-invertability.
        For F flavors both blocks become block diagonal, kron(I_F, .), matching
        the ordering of X given in the class docstring.
        """
        N = self.N
        a = self.a

        phi_matrix = ((2 * np.eye(N)) + -np.eye(N, k=1) - np.eye(N, k=-1)) / a
        p_matrix   = np.eye(N) / a
        phi_matrix = np.kron(np.eye(self.F), phi_matrix)
        p_matrix   = np.kron(np.eye(self.F), p_matrix)
        z = np.zeros_like(phi_matrix)

        self.hamiltonian = np.block([[phi_matrix,   z   ],
                                     [     z,   p_matrix]])


    def add_interacting_hamiltonian_matrix(self):
        """
        We implement the electric field term (a/2) sum_j E_j^2 = (1/2) X^T (a C^T C) X
        with E = C X. The free part is rebuilt first, so calling this method twice
        does not add the interaction twice.
        """
        self.free_hamiltonian_matrix()
        C = self.electric_field_matrix()
        self.hamiltonian += self.a * C.T @ C


    def electric_field_matrix(self):
        """
        Matrix C with E_j = (C X)_j, solving the Gauss law on the open chain.

        Returns:
            numpy.ndarray: The (N x 2FN) matrix C.
        """
        N = self.N
        n = self.F * N
        L = np.tril(np.ones((N, N)))                       # vartheta = L @ P  (cumulative sum)
        C = np.zeros((N, 2 * n))
        for i in range(self.F):
            C[:, i*N:(i+1)*N] += self.QV[i] * np.eye(N)         # varphi part: vector charge to the left
            C[:, n+i*N:n+(i+1)*N] += self.QA[i] * L             # vartheta part: axial charge to the left
        return self.g / np.sqrt(np.pi) * C


    def electric_field_commutator(self):
        """
        The matrix K with [E_j, E_k] = i K_jk. It vanishes exactly if the
        charges are anomaly free (Q^V . Q^A = 0); otherwise max|K| = (g^2/pi) |Q^V . Q^A|.
        """
        C = self.electric_field_matrix()
        return C @ self.symplectic_matrix(C.shape[1] // 2) @ C.T


    def meson_mass(self):
        """Continuum meson mass mu = g sqrt( sum_i [(Q^V_i)^2 + (Q^A_i)^2] / pi )."""
        return self.g * np.sqrt(np.sum(self.QV**2 + self.QA**2) / np.pi)


    def meson_mass_from_E(self, mu_guess=None):
        """
        Fit the ground-state correlator <E_j E_k> = (C sigma C^T)_jk in the bulk by A K_0(mu r) + B,
        with r = a |j - k|. The reference site j0 and the distances are chosen relative to N,
        so the window stays away from the open boundaries.
        (B != 0 for axial charges: uniform background-field fluctuation from the open boundary.)

        Args:
            mu_guess (float, optional): Start value of the fit. Defaults to `meson_mass()`.

        Returns:
            tuple: (mu_fit, B).
        """
        if mu_guess is None:
            mu_guess = self.meson_mass()
        if self.hamiltonian is None:
            self.add_interacting_hamiltonian_matrix()
        N = self.N
        if N < 60:
            raise ValueError(f"N = {N} is too small for a bulk fit of the correlator (need N >= 60).")
        C = self.electric_field_matrix()
        G = C @ self.ground_state_covariance() @ C.T
        j0 = N // 2 - N // 5
        r = np.arange(5, int(0.37 * N))
        y = G[j0, j0 + r]
        d = self.a * r
        p, _ = curve_fit(lambda d, A, m, B: A * k0(m * d) + B, d, y,
                         p0=[y[0] / k0(mu_guess * d[0]), mu_guess, y[-1]], maxfev=20000)
        return p[1], p[2]
