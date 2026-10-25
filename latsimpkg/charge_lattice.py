import itertools

import numpy as np


def _perp(x, q):
    """Component of x orthogonal to q (x itself if q = 0)."""
    qq = q @ q
    return x - (x @ q) / qq * q if qq > 0 else x


def classify_sector(NL, NR, Delta, s, tol=1e-6):
    """
    Collect the data of a sector and classify it.

    A sector with h + hb = 1/2 and |s| = 1/2 is a single massless chiral fermion. It is
    "elementary" if its lightest representative is one of the original fermions
    (|N_L|^2 + |N_R|^2 = 1), otherwise a "bound state" (gauge-invariant composite).
    All other sectors are "multi-particle" states built from these fermions.
    (This criterion assumes that the massless sector is fermionic, h in k^2/2, as for 3-4-5-0.)

    Returns:
        dict: NL, NR, Delta, s, h, hb, kind.
    """
    NL, NR = tuple(int(x) for x in NL), tuple(int(x) for x in NR)
    norm = sum(x * x for x in NL + NR)
    if abs(Delta - 0.5) < tol and abs(abs(s) - 0.5) < tol:
        kind = "elementary" if norm == 1 else "bound state"
    else:
        kind = "multi-particle"
    clean = lambda x: float(np.round(x, 12)) + 0.0         # avoid printing -0.000
    return {"NL": NL, "NR": NR, "Delta": clean(Delta), "s": clean(s),
            "h": clean((Delta + s) / 2), "hb": clean((Delta - s) / 2), "kind": kind}


def charge_lattice_spectrum(QV, QA, nmax=3, Delta_max=2.0):
    """
    Continuum prediction for the gauge-invariant massless sectors (analytic charge-lattice analysis).

    With the chiral charges Q_L = Q^V + Q^A, Q_R = Q^V - Q^A a state with fermion numbers
    (N_L, N_R) is gauge neutral if Q_L.N_L + Q_R.N_R = 0. The gauge field gives a mass to the
    combinations Q_L.varphi_L and Q_R.varphi_R, so only the orthogonal parts stay massless:
        h = |N_L^perp|^2 / 2,   hb = |N_R^perp|^2 / 2,   s = h - hb = (|N_L|^2 - |N_R|^2) / 2.
    Vectors with the same (N_L^perp, N_R^perp) describe the same sector.

    Args:
        QV, QA (sequence): Vector and axial charges.
        nmax (int, optional): Range of the fermion numbers. Defaults to 3.
        Delta_max (float, optional): Keep sectors with h + hb <= Delta_max. Defaults to 2.0.

    Returns:
        list[dict]: Sectors sorted by Delta, see `classify_sector`.
    """
    QV = np.asarray(QV, dtype=float); QA = np.asarray(QA, dtype=float)
    QL, QR = QV + QA, QV - QA
    F = len(QV)
    classes = {}
    for v in itertools.product(range(-nmax, nmax + 1), repeat=2 * F):
        NL, NR = np.array(v[:F]), np.array(v[F:])
        if abs(QL @ NL + QR @ NR) > 1e-9:
            continue
        pL, pR = _perp(NL, QL), _perp(NR, QR)
        key = tuple(np.round(np.concatenate([pL, pR]), 6))
        norm = int(NL @ NL + NR @ NR)
        if key not in classes or norm < classes[key][0]:
            classes[key] = (norm, NL, NR, pL @ pL / 2, pR @ pR / 2)
    sectors = []
    for norm, NL, NR, h, hb in classes.values():
        if 1e-9 < h + hb <= Delta_max + 1e-6:
            sectors.append(classify_sector(NL, NR, h + hb, h - hb))
    return sorted(sectors, key=lambda d: (round(d["Delta"], 6), -round(d["s"], 6), d["NL"]))
