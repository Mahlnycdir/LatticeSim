"""Correlation functions of the electric field."""
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import k0


def electric_field_correlator(model, a=0, b=0):
    """Ground-state correlator <E_{a,j} E_{b,k}> (connected, symmetrised) = (C_a sigma C_b^T)_jk."""
    C = model.electric_field
    return C[a] @ model.H.covariance @ C[b].T


def meson_mass_from_E(model, mu_guess=None, u1=0):
    """
    Fit the ground-state correlator <E_j E_k> in the bulk by A K_0(mu r) + B, with r = a |j - k|.
    The reference site j0 and the distances are chosen relative to N, so the window stays away
    from the open boundaries. (B != 0 for axial charges: uniform background-field fluctuation from
    the open boundary.)

    Args:
        model (GaugeModel): Model on the open chain.
        mu_guess (float, optional): Start value of the fit. Defaults to the lightest continuum meson mass.
        u1 (int, optional): Which U(1) to use. Defaults to 0.

    Returns:
        tuple: (mu_fit, B).
    """
    if model.lattice.is_ring:
        raise ValueError("The correlator fit assumes the open chain.")
    N = model.lattice.N
    if N < 60:
        raise ValueError(f"N = {N} is too small for a bulk fit of the correlator (need N >= 60).")
    if mu_guess is None:
        mu_guess = model.theory.meson_masses[0]
    G = electric_field_correlator(model, u1, u1)
    j0 = N // 2 - N // 5
    r = np.arange(5, int(0.37 * N))
    y = G[j0, j0 + r]
    d = model.lattice.a * r
    p, _ = curve_fit(lambda d, A, m, B: A * k0(m * d) + B, d, y,
                     p0=[y[0] / k0(mu_guess * d[0]), mu_guess, y[-1]], maxfev=20000)
    return p[1], p[2]
