"""
Adapters between the tests and the library.

The tests never construct models directly but only through the functions in this file.
When the API changes (ROADMAP phases 1-3), only these adapters are rewritten; the tests,
i.e. the physics they check, stay untouched.

Theories are given in chiral form: left charges QL, right charges QR. They are turned into a
`ChiralTheory` (validation, anomaly check) and paired into Dirac bosons with
Q^V = (Q_L + Q_R)/2, Q^A = (Q_L - Q_R)/2 by `ChiralTheory.pairing`.
"""
import numpy as np

from latsimpkg.charge_lattice import charge_lattice_spectrum
from latsimpkg.conventions import vector_axial
from latsimpkg.phasespace import PhaseSpace
from latsimpkg.quadratic import QuadraticHamiltonian
from latsimpkg.ring import RingGaugeTheory
from latsimpkg.scalar import ScalarFieldGaugeTheory
from latsimpkg.theory import ChiralTheory


# Known theories in chiral form.
THEORIES = {
    "schwinger":       {"QL": [1],    "QR": [1]},
    "axial_schwinger": {"QL": [1],    "QR": [-1]},
    "3450":            {"QL": [3, 4], "QR": [5, 0]},
}


def paired_charges(QL, QR, g=1.0, allow_anomalous=False):
    """(Q^V, Q^A) of a single-U(1) ChiralTheory; U(1)^k models follow in phase 3."""
    theory = ChiralTheory(QL, QR, g, allow_anomalous=allow_anomalous)
    if theory.k != 1:
        raise NotImplementedError("The current lattice models support one U(1) only (phase 3).")
    p = theory.pairing()
    return p.QV[0], p.QA[0]


def free_chain(N, a=1.0):
    """Single free scalar on the open Dirichlet chain."""
    model = ScalarFieldGaugeTheory(N, a)
    model.free_hamiltonian_matrix()
    return model


def open_gauge_model(QL, QR, N, g, a=1.0, allow_anomalous=False):
    """Gauged theory on the open chain with the full Hamiltonian built."""
    QV, QA = paired_charges(QL, QR, g, allow_anomalous)
    model = ScalarFieldGaugeTheory(N, a, g, QV, QA)
    model.add_interacting_hamiltonian_matrix()
    return model


def free_ring(N, F=1, a=1.0):
    """F free compact bosons on the ring (zero modes included)."""
    model = RingGaugeTheory(N, a, 0.0, np.ones(F), np.zeros(F))
    model.free_hamiltonian_matrix()
    return model


def ring_model(QL, QR, N, g, a=1.0):
    """Gauged theory on the ring with the full Hamiltonian built."""
    QV, QA = paired_charges(QL, QR, g)
    model = RingGaugeTheory(N, a, g, QV, QA)
    model.add_interacting_hamiltonian_matrix()
    return model


def quadratic(model, M=None):
    """The model-independent core for the model's Hamiltonian (or an explicit matrix M)."""
    return QuadraticHamiltonian(model.hamiltonian if M is None else M)


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
    return model.meson_mass()


def meson_mass_fit(model):
    """(mu_fit, B) from the electric-field correlator."""
    return model.meson_mass_from_E()


def conserved_charges(model):
    """Basis of the conserved linear charges (columns)."""
    return quadratic(model).conserved_charges


def charge_map(model):
    """T with c = T (n, m); rows are the conserved combinations (t_n, t_m)."""
    return model.charge_map()


def sector_energy(model, NL, NR):
    """Dict with q, E, Delta = E L/2pi, s for the sector (N_L, N_R)."""
    return model.sector_energy(NL, NR)


def sector_scan(model, nmax=3, Delta_max=2.0):
    """Numerical sectors as list of dicts with NL, NR, Delta, s, h, hb, kind."""
    return model.sector_scan(nmax=nmax, Delta_max=Delta_max)


def analytic_sectors(QL, QR, nmax=3, Delta_max=2.0):
    """Charge-lattice prediction, same format as `sector_scan`."""
    QV, QA = paired_charges(QL, QR)
    return charge_lattice_spectrum(QV, QA, nmax=nmax, Delta_max=Delta_max)
