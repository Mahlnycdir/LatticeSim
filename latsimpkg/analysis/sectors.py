"""
Charge sectors of a gauge model on the ring.

The fermion numbers of Dirac boson i are its zero modes: winding W_i = sqrt(pi) n_i and total momentum
sum_j P_{i,j} = sqrt(pi) m_i, with N_L = (n + m)/2, N_R = (n - m)/2 (see `conventions`). Only the conserved
combinations c = A^T X_v are good quantum numbers (A from the kernel of M, `QuadraticHamiltonian`); the
lowest energy in a sector is the free-particle energy (1/2) c^T G c of the Williamson normal form. Massless
sectors have E L/(2 pi) = h + hb independent of L, the momentum gives the spin s = h - hb.
"""
import itertools

import numpy as np

from latsimpkg.analysis.charge_lattice import classify_sector, fermion_content, sort_sectors
from latsimpkg.analysis.results import SectorEnergy
from latsimpkg.conventions import SQRT_PI, winding_momentum


def _require_ring(model):
    if not model.lattice.is_ring:
        raise ValueError("Charge sectors need the ring (the open chain has no windings).")


def sector_state(model, n, m):
    """Configuration X_v of the sector (n, m): W_i = sqrt(pi) n_i, P_{i,j} = sqrt(pi) m_i / N."""
    _require_ring(model)
    ps, N = model.phasespace, model.lattice.N
    X = ps.zeros()
    for i in range(model.F):
        X[ps.index("W", i)] = SQRT_PI * n[i]
        X[ps.slice("P", i)] = SQRT_PI * m[i] / N
    return X


def charge_map(model):
    """
    Matrix T with c = T (n, m): the values of the conserved charges A^T X_v in the sector (n, m).
    Its rows (t_n, t_m) are the conserved combinations of the fermion numbers; for an anomaly-free
    gauge theory they satisfy Q^A_a . t_n + Q^V_a . t_m = 0 (no mixed anomaly with any gauge U(1)).
    """
    A = model.H.conserved_charges
    F = model.F
    units = np.eye(2 * F)
    return np.array([A.T @ sector_state(model, u[:F], u[F:]) for u in units]).T


def _sector(model, c):
    """E, E L/(2 pi) and s of the sector with conserved charges c."""
    H, L = model.H, model.lattice.L
    E = H.free_energy(c)
    s = -model.momentum(H.free_configuration(c)) * L / (2 * np.pi)
    return E, E * L / (2 * np.pi), s


def sector_energy(model, NL, NR):
    """
    Energy of the lowest state in the sector with chiral fermion numbers (N_L, N_R).

    Returns:
        SectorEnergy: gauge charges q (shape (k,)), E, Delta = E L/(2 pi), s = -P L/(2 pi).
    """
    n, m = winding_momentum(NL, NR)
    c = model.H.conserved_charges.T @ sector_state(model, n, m)
    E, Delta, s = _sector(model, c)
    return SectorEnergy(model.gauge_charges(n, m), E, Delta, s)


def sector_scan(model, nmax=3, Delta_max=2.0):
    """
    Search all gauge-neutral charge sectors with |N_L|, |N_R| <= nmax per component.

    Vectors v with the same conserved charges c describe the same sector; the one with the
    smallest |N_L|^2 + |N_R|^2 is kept as representative (see `classify_sector`).

    Args:
        model (GaugeModel): A model on the ring.
        nmax (int, optional): Range of the fermion numbers. Defaults to 3.
        Delta_max (float, optional): Keep sectors with E L/(2 pi) <= Delta_max. Defaults to 2.0.

    Returns:
        list[Sector]: Sorted by Delta, then by decreasing s.
    """
    T = charge_map(model)
    scale = np.abs(T).max() if T.size else 1.0
    F = model.F
    classes = {}
    for v in itertools.product(range(-nmax, nmax + 1), repeat=2 * F):
        NL, NR = np.array(v[:F]), np.array(v[F:])
        n, m = NL + NR, NL - NR
        if np.abs(model.gauge_charges(n, m)).max() > 1e-9 or not np.any(v):
            continue
        c = T @ np.concatenate([n, m])
        key = tuple(np.round(c / scale, 6))
        norm = int(NL @ NL + NR @ NR)
        if key not in classes or norm < classes[key][0]:
            classes[key] = (norm, NL, NR, c)
    sectors = []
    for norm, NL, NR, c in classes.values():
        E, Delta, s = _sector(model, c)
        if Delta > Delta_max + 1e-6 or Delta < 1e-9:
            continue
        sectors.append(classify_sector(NL, NR, Delta, s, fermion_content(NL, NR, model.theory, model.pairing)))
    return sort_sectors(sectors)
