"""
User-facing functions: one call from a theory to a result.

    from latsimpkg import model_3450, find_bound_states
    scan = find_bound_states(model_3450(g=0.3))
    for s in scan.bound_states:
        print(s.content, s.h, s.hb)
"""
from latsimpkg.analysis.charge_lattice import charge_lattice_spectrum
from latsimpkg.analysis.potentials import screening_potential as _screening_potential
from latsimpkg.analysis.results import SectorScan
from latsimpkg.analysis.sectors import sector_scan
from latsimpkg.analysis.spectrum import mode_spectrum
from latsimpkg.lattice import Lattice
from latsimpkg.models import build_gauge_model


def find_bound_states(theory, N=40, a=1.0, nmax=3, Delta_max=2.0):
    """
    Search the gauge-invariant massless sectors of a theory and identify its bound states.

    Builds the theory on a ring of N sites (zero modes = fermion numbers), scans all gauge-neutral
    fermion numbers |N_L|, |N_R| <= nmax, groups them by their conserved charges, reads off
    Delta = h + hb and the spin s, and classifies every sector as elementary fermion, bound state
    (gauge-invariant composite fermion) or multi-particle state. The analytic charge-lattice
    prediction is computed alongside as an independent cross-check (`SectorScan.agrees`).

    The massless sectors do not depend on g > 0 or on N (E L/(2 pi) is exact on the lattice), so
    small N suffices; nmax limits which composites can be found.

    Args:
        theory (ChiralTheory): The theory (must be anomaly free).
        N (int, optional): Ring size. Defaults to 40.
        a (float, optional): Lattice spacing. Defaults to 1.
        nmax (int, optional): Range of the fermion numbers. Defaults to 3.
        Delta_max (float, optional): Largest Delta kept. Defaults to 2.

    Returns:
        SectorScan: numerical and analytic sectors; `.bound_states`, `.elementary`, `.agrees`.
    """
    model = build_gauge_model(theory, Lattice(N, a, "ring"))
    numeric = sector_scan(model, nmax=nmax, Delta_max=Delta_max)
    analytic = charge_lattice_spectrum(model.pairing, nmax=nmax, Delta_max=Delta_max, theory=theory)
    meta = model.describe()
    meta.update({"nmax": nmax, "Delta_max": Delta_max, "n_conserved": model.H.n_free})
    return SectorScan(tuple(numeric), tuple(analytic), meta)


def spectrum(theory, N, a=1.0, boundary="open", **build_kwargs):
    """
    Normal-mode spectrum of a theory on a lattice, with the continuum-motivated prediction
    (open chain; see `analysis.spectrum.predicted_spectrum`).

    Returns:
        Spectrum
    """
    return mode_spectrum(build_gauge_model(theory, Lattice(N, a, boundary), **build_kwargs))


def screening_potential(theory, N, distances, q=1.0, a=1.0, j0=None, u1=0):
    """
    Static potential V(d) of probe charges +q, -q (open chain) with the continuum prediction:
    screening (1 - exp(-mu d)) with charged massless matter, linear confinement without.

    Returns:
        Potential
    """
    return _screening_potential(theory, N, distances, q=q, a=a, j0=j0, u1=u1)
