import itertools

import numpy as np
import scipy.linalg as sl

from latsimpkg.charge_lattice import classify_sector
from latsimpkg.scalar import ScalarFieldGaugeTheory


class RingGaugeTheory(ScalarFieldGaugeTheory):
    """
    The bosonized U(1) gauge theory of `ScalarFieldGaugeTheory` on a periodic chain (ring),
    including the zero modes of the compact bosons. These zero modes carry the fermion
    numbers, so on the ring the charge sectors (and with them the bound states) become visible.

    Charge sectors of Dirac boson i (varphi_i ~ varphi_i + sqrt(pi)):
        winding   varphi_i(j + N) = varphi_i(j) + sqrt(pi) n_i     (vector number n_i)
        momentum  sum_j P_{i,j}   = sqrt(pi) m_i                    (axial number m_i)
        chiral numbers N_L = (n + m)/2, N_R = (n - m)/2, hence n_i = m_i mod 2.

    Extended phase space: besides the periodic fields varphi~_{i,j}, P_{i,j} every flavor gets
    the dual zero mode vartheta-bar_i and its conjugate W_i = sqrt(pi) n_i, [vartheta-bar_i, W_i] = i.
        varphi_{i,j}   = varphi~_{i,j} + W_i j/N
        vartheta_{i,j} = vartheta-bar_i + sum_{k<=j} P_{i,k}
    which gives [varphi_j, vartheta_k] = i (theta(j <= k) - j/N), the commutator on the ring.

    Ordering: X = (varphi~_{0,.}, ..., varphi~_{F-1,.}, vartheta-bar_0..F-1,
                   P_{0,.}, ..., P_{F-1,.}, W_0..F-1),  dimension 2F(N+1).

    Gauss law on the ring: E_{j+N} - E_j = g q with the gauge charge q = Q^V.n + Q^A.m,
    so only sectors with q = 0 are physical.

    Note: M has zero modes on the ring (flat directions conjugate to the conserved charges),
    so `ground_state_covariance` and `meson_mass_from_E` are not available here.
    """

    def _phi(self, i):
        """Slice of varphi~_{i,.} in X."""
        return slice(i * self.N, (i + 1) * self.N)

    def _theta(self, i):
        """Index of vartheta-bar_i in X."""
        return self.F * self.N + i

    def _P(self, i):
        """Slice of P_{i,.} in X."""
        n = self.F * (self.N + 1)
        return slice(n + i * self.N, n + (i + 1) * self.N)

    def _W(self, i):
        """Index of W_i in X."""
        return 2 * self.F * (self.N + 1) - self.F + i

    def _difference_matrix(self):
        """Periodic difference (D x)_j = x_(j+1) - x_j with x_N = x_0, so 1^T D = 0."""
        N = self.N
        return np.roll(np.eye(N), 1, axis=1) - np.eye(N)

    def link_matrix(self, i):
        """
        Matrix B_i with l_{i,.} = B_i X: the links varphi_{i,j+1} - varphi_{i,j}
        including the winding, l = D varphi~ + (W/N) 1.
        """
        B = np.zeros((self.N, 2 * self.F * (self.N + 1)))
        B[:, self._phi(i)] = self._difference_matrix()
        B[:, self._W(i)] = 1.0 / self.N
        return B


    def free_hamiltonian_matrix(self):
        """
        Free Hamiltonian on the ring,
            H = 1/(2a) sum_{i,j} [P_{i,j}^2 + l_{i,j}^2],
        with the links l of `link_matrix`. Since 1^T D = 0 the winding decouples,
        sum_j l_j^2 = |D varphi~|^2 + W^2/N, i.e. the winding energy is W^2/(2L) = pi n^2/(2L)
        and the momentum zero mode gives P_tot^2/(2L) = pi m^2/(2L).
        vartheta-bar does not appear (flat direction) until the gauge field is switched on.
        """
        dim = 2 * self.F * (self.N + 1)
        M = np.zeros((dim, dim))
        for i in range(self.F):
            B = self.link_matrix(i)
            M += B.T @ B / self.a
            P = self._P(i)
            M[P, P] += np.eye(self.N) / self.a
        self.hamiltonian = M


    def electric_field_matrix(self):
        """
        Matrix C with E_j = (C X)_j on the ring,
            E_j = g/sqrt(pi) sum_i [ Q^V_i (varphi~_{i,j} + W_i j/N) + Q^A_i (vartheta-bar_i + sum_{k<=j} P_{i,k}) ].

        Returns:
            numpy.ndarray: The (N x 2F(N+1)) matrix C.
        """
        N = self.N
        C = np.zeros((N, 2 * self.F * (N + 1)))
        L = np.tril(np.ones((N, N)))                       # cumulative sum
        j = np.arange(N)
        for i in range(self.F):
            C[:, self._phi(i)] += self.QV[i] * np.eye(N)
            C[:, self._W(i)]   += self.QV[i] * j / N
            C[:, self._theta(i)] += self.QA[i]
            C[:, self._P(i)]   += self.QA[i] * L
        return self.g / np.sqrt(np.pi) * C


    def ring_momentum(self, X):
        """Classical lattice momentum P = -(1/a) sum_{i,j} P_{i,j} l_{i,j} of the configuration X."""
        return -sum(X[self._P(i)] @ (self.link_matrix(i) @ X) for i in range(self.F)) / self.a


    def conserved_charges(self, rcond=1e-9):
        """
        Linear conserved charges w^T X. From X' = J M X one finds
            d/dt (w^T X) = w^T J M X = 0 for all X   <=>   M J w = 0,
        so they span the null space of M J. They are not put in by hand; their number and form
        follow from the Hamiltonian alone.

        Returns:
            numpy.ndarray: Orthonormal columns A spanning the conserved charges.
        """
        if self.hamiltonian is None:
            self.add_interacting_hamiltonian_matrix()
        M = self.hamiltonian
        return sl.null_space(M @ self.symplectic_matrix(M.shape[0] // 2), rcond=rcond)


    def sector_state(self, n, m):
        """Configuration X_v of the sector (n, m): W_i = sqrt(pi) n_i, P_{i,j} = sqrt(pi) m_i / N."""
        X = np.zeros(2 * self.F * (self.N + 1))
        for i in range(self.F):
            X[self._W(i)] = np.sqrt(np.pi) * n[i]
            X[self._P(i)] = np.sqrt(np.pi) * m[i] / self.N
        return X


    def charge_map(self, A=None):
        """
        Matrix T with c = T (n, m): the values of the conserved charges A^T X_v in the sector (n, m).
        Its rows (t_n, t_m) are the conserved combinations of the fermion numbers; for an
        anomaly-free gauge theory they satisfy Q^A.t_n + Q^V.t_m = 0 (no mixed anomaly with the gauge U(1)).
        """
        if A is None:
            A = self.conserved_charges()
        F = self.F
        units = np.eye(2 * F)
        return np.array([A.T @ self.sector_state(u[:F], u[F:]) for u in units]).T


    def _sector_solver(self, A):
        """Pseudo-inverse of M and G = (A^T M^+ A)^(-1) for the constrained minimum."""
        Mp = np.linalg.pinv(self.hamiltonian, rcond=1e-10)
        return Mp, np.linalg.inv(A.T @ Mp @ A)


    def sector_energy(self, NL, NR, A=None):
        """
        Energy of the lowest state in the sector with chiral fermion numbers (N_L, N_R).

        Williamson normal form: H = sum_k omega_k (oscillators) + (1/2) p_c^T K p_c, with the
        conserved momenta p_c. So the energy above the vacuum is the classical minimum of
        (1/2) X^T M X under A^T X = c, c = A^T X_v. Only the conserved charges are fixed; the
        non-conserved direction relaxes into the massive mode. Lagrange multipliers give
            X* = M^+ A G c,   E = (1/2) c^T G c,   G = (A^T M^+ A)^(-1).

        Args:
            NL, NR (sequence): Chiral fermion numbers per Dirac pair.
            A (numpy.ndarray, optional): Conserved charges. Defaults to `conserved_charges()`.

        Returns:
            dict: q (gauge charge), E, Delta = E L/(2 pi), s = -P L/(2 pi).
        """
        if A is None:
            A = self.conserved_charges()
        NL = np.asarray(NL); NR = np.asarray(NR)
        n, m = NL + NR, NL - NR
        Mp, G = self._sector_solver(A)
        c = A.T @ self.sector_state(n, m)
        X = Mp @ A @ G @ c
        L = self.N * self.a
        E = 0.5 * c @ G @ c
        return {"q": float(self.QV @ n + self.QA @ m), "E": E,
                "Delta": E * L / (2 * np.pi), "s": -self.ring_momentum(X) * L / (2 * np.pi)}


    def sector_scan(self, nmax=3, Delta_max=2.0):
        """
        Search all gauge-neutral charge sectors with |N_L|, |N_R| <= nmax per component.

        Vectors v with the same conserved charges c describe the same sector; the one with the
        smallest |N_L|^2 + |N_R|^2 is kept as representative. A sector whose representative is a
        single fermion (norm 1) is elementary, otherwise it is a composite (bound) state.

        Args:
            nmax (int, optional): Range of the fermion numbers. Defaults to 3.
            Delta_max (float, optional): Keep sectors with E L/(2 pi) <= Delta_max. Defaults to 2.0.

        Returns:
            list[dict]: Sectors sorted by Delta, see `classify_sector`.
        """
        A = self.conserved_charges()
        T = self.charge_map(A)
        Mp, G = self._sector_solver(A)
        L = self.N * self.a
        scale = np.abs(T).max()
        F = self.F
        classes = {}
        for v in itertools.product(range(-nmax, nmax + 1), repeat=2 * F):
            NL, NR = np.array(v[:F]), np.array(v[F:])
            n, m = NL + NR, NL - NR
            if abs(self.QV @ n + self.QA @ m) > 1e-9 or not np.any(v):
                continue
            c = T @ np.concatenate([n, m])
            key = tuple(np.round(c / scale, 6))
            norm = int(NL @ NL + NR @ NR)
            if key not in classes or norm < classes[key][0]:
                classes[key] = (norm, NL, NR, c)
        sectors = []
        for norm, NL, NR, c in classes.values():
            E = 0.5 * c @ G @ c
            Delta = E * L / (2 * np.pi)
            if Delta > Delta_max + 1e-6 or Delta < 1e-9:
                continue
            s = -self.ring_momentum(Mp @ A @ G @ c) * L / (2 * np.pi)
            sectors.append(classify_sector(NL, NR, Delta, s))
        return sorted(sectors, key=lambda d: (round(d["Delta"], 6), -round(d["s"], 6), d["NL"]))
