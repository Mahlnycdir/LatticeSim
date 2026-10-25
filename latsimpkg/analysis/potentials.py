"""Static potentials between probe charges: screening and confinement."""
import numpy as np

from latsimpkg.analysis.results import Potential
from latsimpkg.lattice import Lattice
from latsimpkg.models import build_gauge_model


def potential_prediction(theory, q, d, u1=0):
    """
    Continuum potential of probe charges +q, -q at distance d (one U(1)):
        mu > 0:  V(d) = (g^2 q^2 / 2 mu) (1 - exp(-mu d))   (screening by the massless fermions)
        mu = 0:  V(d) = g^2 q^2 d / 2                        (confinement, string tension g^2 q^2 / 2)
    Derivation: minimizing (1/2) int [(d varphi)^2 + (mu varphi + e)^2] for the probe field e = g q 1_[0,d]
    gives V = (g^2 q^2) int dk/2pi (1 - cos k d) / (k^2 + mu^2).
    Returns None for k > 1 (no closed formula implemented).
    """
    if theory.k != 1:
        return None
    g, mu = theory.g[u1], theory.meson_masses[0]
    d = np.asarray(d, dtype=float)
    if mu > 1e-12:
        return g**2 * q**2 / (2 * mu) * (1 - np.exp(-mu * d)), "screening"
    return g**2 * q**2 * d / 2, "confinement"


def screening_potential(theory, N, distances, q=1.0, a=1.0, j0=None, u1=0):
    """
    V(d) on the open chain: probe charges +q at site j0 and -q at site j0 + d, energy of the background
    (ground-state energy shift, see `GaugeModel.background_energy`).

    Args:
        theory (ChiralTheory): The theory.
        N (int): Number of sites; keep j0 and j0 + d far (>> 1/mu) from the boundaries.
        distances (sequence[int]): Distances in lattice units.
        q (float, optional): Probe charge (fractional values allowed). Defaults to 1.
        a (float, optional): Lattice spacing. Defaults to 1.
        j0 (int, optional): Site of +q. Defaults to N // 2 - max(distances) // 2.
        u1 (int, optional): Which U(1) the probes are charged under. Defaults to 0.

    Returns:
        Potential
    """
    distances = np.asarray(distances, dtype=int)
    j0 = N // 2 - int(distances.max()) // 2 if j0 is None else j0
    if j0 < 0 or j0 + distances.max() >= N:
        raise ValueError("The probe charges do not fit on the lattice.")
    lattice = Lattice(N, a, "open")
    V = []
    for dist in distances:
        ext = np.zeros((theory.k, N))
        ext[u1, j0] += q
        ext[u1, j0 + dist] -= q
        V.append(build_gauge_model(theory, lattice, external_charges=ext).background_energy)
    d = a * distances
    pred = potential_prediction(theory, q, d, u1)
    prediction, regime = (np.array([]), "unknown") if pred is None else pred
    meta = {"q": q, "j0": j0, "u1": u1, "regime": regime, "N": N, "a": a,
            "QL": theory.QL.tolist(), "QR": theory.QR.tolist(), "g": theory.g.tolist()}
    return Potential(d, np.array(V), np.asarray(prediction, dtype=float), meta)
