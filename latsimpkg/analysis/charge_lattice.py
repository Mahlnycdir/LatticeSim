"""Analytic charge-lattice analysis of the gauge-invariant massless sectors (continuum prediction)."""
import itertools

import numpy as np
import scipy.linalg as sl

from latsimpkg.analysis.results import Sector
from latsimpkg.conventions import chiral


def _perp_projector(Q):
    """Projector onto the orthogonal complement of the row space of Q (shape (k, F))."""
    F = Q.shape[1]
    if not np.any(Q):
        return np.eye(F)
    basis = sl.orth(Q.T)
    return np.eye(F) - basis @ basis.T


def fermion_content(NL, NR, theory=None, pairing=None):
    """
    Write a sector as product of fermion operators, e.g. "psi_3L^2 psi_4L psi_5R^-2".

    The exponent is the fermion number (negative: antiparticles). With `theory` and `pairing` the
    names of the theory are used; added spectators are called "spectator_L<i>" / "spectator_R<i>".
    """
    parts = []
    for chi, numbers in (("L", NL), ("R", NR)):
        for i, n in enumerate(numbers):
            if not n:
                continue
            name = f"{chi}{i}"
            if theory is not None and pairing is not None:
                idx = pairing.pairs[i][0 if chi == "L" else 1]
                names = theory.names_L if chi == "L" else theory.names_R
                name = names[idx] if idx is not None else f"spectator_{chi}{i}"
            parts.append(name if n == 1 else f"{name}^{int(n)}")
    return " ".join(parts) if parts else "1"


def classify_sector(NL, NR, Delta, s, content="", tol=1e-6):
    """
    Collect the data of a sector and classify it.

    A sector with h + hb = 1/2 and |s| = 1/2 is a single massless chiral fermion. It is
    "elementary" if its lightest representative is one of the original fermions
    (|N_L|^2 + |N_R|^2 = 1), otherwise a "bound state" (gauge-invariant composite).
    All other sectors are "multi-particle" states built from these fermions.
    (This criterion assumes that the massless sector is fermionic, h in k^2/2, as for 3-4-5-0.)

    Returns:
        Sector
    """
    NL, NR = tuple(int(x) for x in NL), tuple(int(x) for x in NR)
    norm = sum(x * x for x in NL + NR)
    if abs(Delta - 0.5) < tol and abs(abs(s) - 0.5) < tol:
        kind = "elementary" if norm == 1 else "bound state"
    else:
        kind = "multi-particle"
    clean = lambda x: 0.0 if abs(x) < 1e-9 else float(x)   # numerical zeros, avoids printing -0.000
    return Sector(NL, NR, clean(Delta), clean(s), kind, content)


def charge_lattice_spectrum(pairing, nmax=3, Delta_max=2.0, theory=None):
    """
    Continuum prediction for the gauge-invariant massless sectors.

    With the chiral charges Q_L = Q^V + Q^A, Q_R = Q^V - Q^A (rows a = 1..k) a state with fermion numbers
    (N_L, N_R) is gauge neutral if Q_L,a . N_L + Q_R,a . N_R = 0 for all a. The gauge fields give a mass to
    the combinations Q_L,a . varphi_L and Q_R,a . varphi_R, so only the parts orthogonal to the row spaces
    of Q_L and Q_R stay massless:
        h = |N_L^perp|^2 / 2,   hb = |N_R^perp|^2 / 2,   s = h - hb = (|N_L|^2 - |N_R|^2) / 2.
    Vectors with the same (N_L^perp, N_R^perp) describe the same sector. The fermion numbers refer to the
    Dirac bosons of `pairing` (spectators included), exactly as in the numerical sector scan.

    Args:
        pairing (Pairing): Charges of the Dirac bosons, e.g. `theory.pairing()` or `model.pairing`.
        nmax (int, optional): Range of the fermion numbers. Defaults to 3.
        Delta_max (float, optional): Keep sectors with h + hb <= Delta_max. Defaults to 2.0.
        theory (ChiralTheory, optional): Only used to name the fermions in `Sector.content`.

    Returns:
        list[Sector]: Sorted by Delta, then by decreasing s.
    """
    QL, QR = chiral(pairing.QV, pairing.QA)
    F = QL.shape[1]
    PL, PR = _perp_projector(QL), _perp_projector(QR)
    classes = {}
    for v in itertools.product(range(-nmax, nmax + 1), repeat=2 * F):
        NL, NR = np.array(v[:F]), np.array(v[F:])
        if np.abs(QL @ NL + QR @ NR).max() > 1e-9:
            continue
        pL, pR = PL @ NL, PR @ NR
        key = tuple(np.round(np.concatenate([pL, pR]), 6))
        norm = int(NL @ NL + NR @ NR)
        if key not in classes or norm < classes[key][0]:
            classes[key] = (norm, NL, NR, pL @ pL / 2, pR @ pR / 2)
    sectors = []
    for norm, NL, NR, h, hb in classes.values():
        if 1e-9 < h + hb <= Delta_max + 1e-6:
            sectors.append(classify_sector(NL, NR, h + hb, h - hb, fermion_content(NL, NR, theory, pairing)))
    return sort_sectors(sectors)


def sort_sectors(sectors):
    """Canonical order: increasing Delta, decreasing s, then by representative."""
    return sorted(sectors, key=lambda x: (round(x.Delta, 6), -round(x.s, 6), x.NL, x.NR))
