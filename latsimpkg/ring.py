import itertools

import numpy as np
from latsimpkg.charge_lattice import classify_sector
from latsimpkg.phasespace import PhaseSpace
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

    Note: M has zero modes on the ring (free particles conjugate to the conserved charges).
    `ground_state_covariance` is the oscillator covariance at sharp conserved charges (see
    `QuadraticHamiltonian.covariance`); `meson_mass_from_E` assumes the open chain.
    """

    @property
    def phasespace(self):
        """Fields varphi~/P of shape (F, N) plus the zero modes vartheta-bar/W of shape (F,)."""
        return PhaseSpace([("phi", "P", (self.F, self.N)), ("theta_bar", "W", (self.F,))])

    def _difference_matrix(self):
        """Periodic difference (D x)_j = x_(j+1) - x_j with x_N = x_0, so 1^T D = 0."""
        N = self.N
        return np.roll(np.eye(N), 1, axis=1) - np.eye(N)

    def link_matrix(self, i):
        """
        Matrix B_i with l_{i,.} = B_i X: the links varphi_{i,j+1} - varphi_{i,j}
        including the winding, l = D varphi~ + (W/N) 1.
        """
        ps = self.phasespace
        B = np.zeros((self.N, ps.dim))
        B[:, ps.slice("phi", i)] = self._difference_matrix()
        B[:, ps.index("W", i)] = 1.0 / self.N
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
        ps = self.phasespace
        M = np.zeros((ps.dim, ps.dim))
        for i in range(self.F):
            B = self.link_matrix(i)
            M += B.T @ B / self.a
            P = ps.slice("P", i)
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
        ps = self.phasespace
        C = np.zeros((N, ps.dim))
        L = np.tril(np.ones((N, N)))                       # cumulative sum
        j = np.arange(N)
        for i in range(self.F):
            C[:, ps.slice("phi", i)]       += self.QV[i] * np.eye(N)
            C[:, ps.index("W", i)]         += self.QV[i] * j / N
            C[:, ps.index("theta_bar", i)] += self.QA[i]
            C[:, ps.slice("P", i)]         += self.QA[i] * L
        return self.g / np.sqrt(np.pi) * C


    def ring_momentum(self, X):
        """Classical lattice momentum P = -(1/a) sum_{i,j} P_{i,j} l_{i,j} of the configuration X."""
        ps = self.phasespace
        return -sum(X[ps.slice("P", i)] @ (self.link_matrix(i) @ X) for i in range(self.F)) / self.a


    def conserved_charges(self):
        """
        Linear conserved charges w^T X. From X' = J M X one finds
            d/dt (w^T X) = w^T J M X = 0 for all X   <=>   M J w = 0,
        so they are A = J K with the kernel K of M (see `QuadraticHamiltonian`). They are not put
        in by hand; their number and form follow from the Hamiltonian alone.

        Returns:
            numpy.ndarray: Orthonormal columns A spanning the conserved charges.
        """
        if self.hamiltonian is None:
            self.add_interacting_hamiltonian_matrix()
        return self.quadratic().conserved_charges


    def sector_state(self, n, m):
        """Configuration X_v of the sector (n, m): W_i = sqrt(pi) n_i, P_{i,j} = sqrt(pi) m_i / N."""
        ps = self.phasespace
        X = ps.zeros()
        for i in range(self.F):
            X[ps.index("W", i)] = np.sqrt(np.pi) * n[i]
            X[ps.slice("P", i)] = np.sqrt(np.pi) * m[i] / self.N
        return X


    def charge_map(self):
        """
        Matrix T with c = T (n, m): the values of the conserved charges A^T X_v in the sector (n, m).
        Its rows (t_n, t_m) are the conserved combinations of the fermion numbers; for an
        anomaly-free gauge theory they satisfy Q^A.t_n + Q^V.t_m = 0 (no mixed anomaly with the gauge U(1)).
        """
        A = self.conserved_charges()
        F = self.F
        units = np.eye(2 * F)
        return np.array([A.T @ self.sector_state(u[:F], u[F:]) for u in units]).T


    def _sector(self, c):
        """Energy data of the sector with conserved charges c (free-particle part of the normal form)."""
        H = self.quadratic()
        L = self.N * self.a
        E = H.free_energy(c)
        s = -self.ring_momentum(H.free_configuration(c)) * L / (2 * np.pi)
        return E, E * L / (2 * np.pi), s


    def sector_energy(self, NL, NR):
        """
        Energy of the lowest state in the sector with chiral fermion numbers (N_L, N_R).

        Williamson normal form (`QuadraticHamiltonian`): H = sum_k omega_k (oscillators)
        + (1/2) c^T G c with the conserved momenta c = A^T X. So the energy above the vacuum is
        the minimum of (1/2) X^T M X under A^T X = c, c = A^T X_v. Only the conserved charges are
        fixed; the non-conserved direction relaxes into the massive mode.

        Args:
            NL, NR (sequence): Chiral fermion numbers per Dirac pair.

        Returns:
            dict: q (gauge charge), E, Delta = E L/(2 pi), s = -P L/(2 pi).
        """
        if self.hamiltonian is None:
            self.add_interacting_hamiltonian_matrix()
        NL = np.asarray(NL); NR = np.asarray(NR)
        n, m = NL + NR, NL - NR
        c = self.conserved_charges().T @ self.sector_state(n, m)
        E, Delta, s = self._sector(c)
        return {"q": float(self.QV @ n + self.QA @ m), "E": E, "Delta": Delta, "s": s}


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
        T = self.charge_map()
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
            E, Delta, s = self._sector(c)
            if Delta > Delta_max + 1e-6 or Delta < 1e-9:
                continue
            sectors.append(classify_sector(NL, NR, Delta, s))
        return sorted(sectors, key=lambda d: (round(d["Delta"], 6), -round(d["s"], 6), d["NL"]))
