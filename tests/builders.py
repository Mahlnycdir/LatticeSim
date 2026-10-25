"""
Adapters between the tests and the library.

The tests never construct models directly but only through the functions in this file.
When the API changes (ROADMAP phases 1-3), only these adapters are rewritten; the tests,
i.e. the physics they check, stay untouched.

Theories are given in chiral form: left charges QL, right charges QR. Every model is built as
ChiralTheory (validation, anomaly check) + Lattice -> build_gauge_model.
"""
import numpy as np

from latsimpkg.analysis import correlators, sectors
from latsimpkg.analysis.charge_lattice import charge_lattice_spectrum
from latsimpkg.conventions import vector_axial
from latsimpkg.lattice import Lattice
from latsimpkg.models import build_gauge_model
from latsimpkg.phasespace import PhaseSpace
from latsimpkg.quadratic import QuadraticHamiltonian
from latsimpkg.theory import ChiralTheory


# Known theories in chiral form.
THEORIES = {
    "schwinger":       {"QL": [1],    "QR": [1]},
    "axial_schwinger": {"QL": [1],    "QR": [-1]},
    "3450":            {"QL": [3, 4], "QR": [5, 0]},
}


def free_chain(N, a=1.0):
    """Single free scalar on the open Dirichlet chain (a neutral Dirac fermion)."""
    return build_gauge_model(ChiralTheory([0], [0], g=0.0), Lattice(N, a, "open"))


def open_gauge_model(QL, QR, N, g, a=1.0, allow_anomalous=False):
    """Gauged theory on the open chain."""
    return build_gauge_model(ChiralTheory(QL, QR, g, allow_anomalous=allow_anomalous), Lattice(N, a, "open"))


def free_ring(N, F=1, a=1.0):
    """F free compact bosons on the ring (zero modes included)."""
    return build_gauge_model(ChiralTheory(np.zeros(F), np.zeros(F), g=0.0), Lattice(N, a, "ring"))


def ring_model(QL, QR, N, g, a=1.0):
    """Gauged theory on the ring."""
    return build_gauge_model(ChiralTheory(QL, QR, g), Lattice(N, a, "ring"))


def quadratic(model, M=None):
    """The model-independent core for the model's Hamiltonian (or an explicit matrix M)."""
    return model.H if M is None else QuadraticHamiltonian(M)


def normal_modes(model):
    return quadratic(model).frequencies


def ground_state_energy(model):
    return quadratic(model).ground_state_energy


def ground_state_covariance(model, M=None):
    return quadratic(model, M).covariance


def symplectic_matrix(model, n):
    return PhaseSpace.flat(n).J


def electric_field_commutator(model):
    return model.electric_field_commutator()


def meson_mass(model):
    """Continuum meson mass (one U(1): a single meson)."""
    return model.theory.meson_masses[0]


def meson_mass_fit(model):
    """(mu_fit, B) from the electric-field correlator."""
    fit = correlators.meson_mass_from_E(model)
    return fit.mu, fit.B


def conserved_charges(model):
    """Basis of the conserved linear charges (columns)."""
    return model.H.conserved_charges


def charge_map(model):
    """T with c = T (n, m); rows are the conserved combinations (t_n, t_m)."""
    return sectors.charge_map(model)


def sector_energy(model, NL, NR):
    """Dict with q, E, Delta = E L/2pi, s for the sector (N_L, N_R)."""
    e = sectors.sector_energy(model, NL, NR)
    return {"q": e.q, "E": e.E, "Delta": e.Delta, "s": e.s}


def sector_scan(model, nmax=3, Delta_max=2.0):
    """Numerical sectors as list of dicts with NL, NR, Delta, s, h, hb, kind."""
    return [s.as_dict() for s in sectors.sector_scan(model, nmax=nmax, Delta_max=Delta_max)]


def analytic_sectors(QL, QR, nmax=3, Delta_max=2.0):
    """Charge-lattice prediction, same format as `sector_scan`."""
    return [s.as_dict() for s in charge_lattice_spectrum(ChiralTheory(QL, QR).pairing(), nmax=nmax, Delta_max=Delta_max)]
