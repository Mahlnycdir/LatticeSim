"""Study 1: the Gaussian machinery on the free scalar field."""
import numpy as np

from latsimpkg import ChiralTheory, Lattice, build_gauge_model
from latsimpkg.analysis.spectrum import dirichlet_momenta, mode_spectrum
from latsimpkg.quadratic import QuadraticHamiltonian
from studies.common import outdir, section

N, A = 50, 0.001
W = 1.7


def run(root=None):
    out = outdir("free_field", root)
    model = build_gauge_model(ChiralTheory([0], [0], 0.0), Lattice(N, A, "open"))
    spec = mode_spectrum(model)
    spec.save(out / "spectrum.npz")
    sig = model.H.covariance
    return {"spectrum": spec,
            "oscillator_q2": QuadraticHamiltonian(np.diag([W**2, 1.0])).covariance[0, 0],
            "purity": np.linalg.eigvalsh(sig + 0.5j * model.H.J).min(),
            "E0": model.H.ground_state_energy}


def report(r):
    section(f"1. Free scalar field (N = {N}, a = {A})")
    print(f"  single oscillator:  <q^2> = {r['oscillator_q2']:.12f}   expected 1/(2w) = {1 / (2 * W):.12f}")
    spec = r["spectrum"]
    rel = np.max(np.abs(spec.frequencies - spec.prediction) / spec.prediction)
    print(f"  Dirichlet chain:    max|omega - (2/a) sin(pi n/(2(N+1)))| = {spec.max_deviation:.1e}   (relative {rel:.1e})")
    print(f"  uncertainty principle sigma + iJ/2 >= 0:  smallest eigenvalue = {r['purity']:.1e}   (0 = pure state)")
    print(f"  zero-point energy:  E_0 = {r['E0']:.6f}   expected sum_n omega_n/2 = {0.5 * dirichlet_momenta(N, A).sum():.6f}")
