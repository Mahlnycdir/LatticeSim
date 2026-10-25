"""Normal-mode spectra, their continuum prediction and scaling with the lattice size."""
import numpy as np

from latsimpkg.analysis.results import Series, Spectrum
from latsimpkg.lattice import Lattice
from latsimpkg.models import build_gauge_model


def dirichlet_momenta(N, a=1.0):
    """Free frequencies of the Dirichlet chain, (2/a) sin(pi n / (2(N+1))), n = 1..N (k_n = pi n / L)."""
    return 2 / a * np.sin(np.pi * np.arange(1, N + 1) / (2 * (N + 1)))


def predicted_spectrum(model):
    """
    Continuum-motivated spectrum on the open chain: for every meson mass mu_a a branch sqrt(mu_a^2 + k^2),
    and F - (number of mesons) massless branches k, with the Dirichlet lattice momenta k.

    Exact for vector-like charges (they act locally on varphi). Axial charges act on vartheta = cumulative P,
    whose boundary conditions differ: their momenta are k_n = (n + delta) pi / L with a phase shift delta,
    so the prediction is only correct in the bulk (deviation ~ 1/N). Returns an empty array on the ring.
    """
    if model.lattice.is_ring:
        return np.array([])
    k = dirichlet_momenta(model.lattice.N, model.lattice.a)
    masses = model.theory.meson_masses
    masses = masses[masses > 1e-12]
    branches = [np.sqrt(mu**2 + k**2) for mu in masses] + [k] * (model.F - len(masses))
    return np.sort(np.concatenate(branches))


def mode_spectrum(model):
    """Normal-mode frequencies of `model` with the prediction of `predicted_spectrum`."""
    meta = model.describe()
    meta.update({"n_free": model.H.n_free, "meson_masses": model.theory.meson_masses.tolist()})
    return Spectrum(model.H.frequencies.copy(), predicted_spectrum(model), meta)


def lowest_frequency_scaling(theory, Ns, a=1.0):
    """
    omega_min against N on the open chain.

    Prediction: with a meson branch only (F = number of mesons) omega_min = sqrt(mu^2 + k_1^2) -> mu,
    otherwise the massless branch gives omega_min = k_1 = (2/a) sin(pi/(2(N+1))) -> 0.
    """
    y, pred = [], []
    for N in Ns:
        model = build_gauge_model(theory, Lattice(int(N), a, "open"))
        y.append(model.H.frequencies[0])
        pred.append(predicted_spectrum(model)[0])
    meta = {"name": "lowest frequency", "xlabel": "N", "ylabel": "omega_min", "a": a,
            "QL": theory.QL.tolist(), "QR": theory.QR.tolist(), "g": theory.g.tolist()}
    return Series(np.asarray(Ns, dtype=float), np.array(y), np.array(pred), meta)


def continuum_scaling(theory, L, spacings):
    """
    omega_min at fixed physical length L = (N + 1) a for decreasing a (open chain).

    Prediction: sqrt(mu^2 + (pi/L)^2) for the lightest meson branch; the lattice dispersion
    (2/a) sin(k a / 2) deviates by O((k a)^2), so the deviation falls like a^2.
    """
    y, pred, Ns = [], [], []
    mu = theory.meson_masses[0]
    for a in spacings:
        N = int(round(L / a)) - 1
        model = build_gauge_model(theory, Lattice(N, a, "open"))
        y.append(model.H.frequencies[0])
        pred.append(np.sqrt(mu**2 + (np.pi / model.lattice.L)**2))
        Ns.append(N)
    meta = {"name": "continuum limit", "xlabel": "a", "ylabel": "omega_min", "L": L, "N": Ns,
            "QL": theory.QL.tolist(), "QR": theory.QR.tolist(), "g": theory.g.tolist()}
    return Series(np.asarray(spacings, dtype=float), np.array(y), np.array(pred), meta)
