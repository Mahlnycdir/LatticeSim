"""
Analytic reference formulas used by the tests. Independent of the library on purpose:
a test compares the library against these, never against the library itself.
"""
import numpy as np


def dirichlet_momenta(N, a=1.0):
    """Frequencies of the free Dirichlet chain, omega_n = (2/a) sin(pi n / (2(N+1)))."""
    return 2 / a * np.sin(np.pi * np.arange(1, N + 1) / (2 * (N + 1)))


def meson_mass(QL, QR, g):
    """mu^2 = (g^2 / 2pi) sum (Q_L^2 + Q_R^2) = (g^2/pi) sum ((Q^V)^2 + (Q^A)^2)."""
    return g * np.sqrt((np.sum(np.square(QL)) + np.sum(np.square(QR))) / (2 * np.pi))


def gauge_anomaly(QL, QR):
    """Gauge anomaly coefficient sum Q_L^2 - sum Q_R^2 (zero for a consistent theory)."""
    return np.sum(np.square(QL)) - np.sum(np.square(QR))


def gauged_spectrum(QL, QR, N, g, a=1.0):
    """Continuum-motivated open-chain spectrum: one branch sqrt(mu^2 + k^2), F - 1 branches k."""
    k = dirichlet_momenta(N, a)
    mu = meson_mass(QL, QR, g)
    return np.sort(np.concatenate([np.sqrt(mu**2 + k**2)] + [k] * (len(QL) - 1)))


def sector_signature(sectors, digits=6):
    """Representation-independent fingerprint of a sector list: sorted (Delta, s, kind)."""
    return sorted((round(d["Delta"], digits) + 0.0, round(d["s"], digits) + 0.0, d["kind"])
                  for d in sectors)
