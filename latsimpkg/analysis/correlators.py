"""Correlation functions of the electric field."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import k0

from latsimpkg.analysis.results import FitResult, FitWindow


def electric_field_correlator(model, a=0, b=0):
    """Ground-state correlator <E_{a,j} E_{b,k}> (connected, symmetrised) = (C_a sigma C_b^T)_jk."""
    C = model.electric_field
    return C[a] @ model.H.covariance @ C[b].T


def meson_mass_from_E(model, window=None, mu_guess=None, u1=0):
    """
    Fit the ground-state correlator <E_{j0} E_{j0+r}> by A K_0(mu r a) + B.

    Physics: the electric field is proportional to the massive boson, and the equal-time propagator of a
    free massive boson in 1+1 dimensions is K_0(mu |x|) / (2 pi), so the decay of <E E> measures mu.
    The offset B is a uniform fluctuation of the background field caused by the open boundary
    (non-zero for axial charges).

    Args:
        model (GaugeModel): Model on the open chain.
        window (FitWindow, optional): Reference site and distances. Defaults to `FitWindow.bulk(N)`.
        mu_guess (float, optional): Start value. Defaults to the lightest continuum meson mass.
        u1 (int, optional): Which U(1). Defaults to 0.

    Returns:
        FitResult
    """
    if model.lattice.is_ring:
        raise ValueError("The correlator fit assumes the open chain.")
    N, a = model.lattice.N, model.lattice.a
    window = FitWindow.bulk(N) if window is None else window
    window.check(N)
    mu_expected = float(model.theory.meson_masses[0])
    mu_guess = mu_expected if mu_guess is None else mu_guess
    G = electric_field_correlator(model, u1, u1)
    r = np.arange(window.r_min, window.r_max)
    y = G[window.j0, window.j0 + r]
    d = a * r
    p, _ = curve_fit(lambda d, A, m, B: A * k0(m * d) + B, d, y,
                     p0=[y[0] / k0(mu_guess * d[0]), mu_guess, y[-1]], maxfev=20000)
    meta = model.describe()
    meta.update({"window": [window.j0, window.r_min, window.r_max], "u1": u1})
    return FitResult(float(p[1]), float(p[0]), float(p[2]), mu_expected, d, y, meta)
