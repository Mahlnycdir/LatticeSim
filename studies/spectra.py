"""Study 2: spectra of the gauged theories on the open chain, finite-size scaling, continuum limit."""
import numpy as np

from latsimpkg import Lattice, build_gauge_model, schwinger
from latsimpkg.analysis.correlators import electric_field_correlator
from latsimpkg.analysis.spectrum import continuum_scaling, lowest_frequency_scaling, mode_spectrum
from studies.common import MODELS, outdir, section, slug

N, G = 300, 0.04
NS = [100, 200, 300, 450, 600]
L_FIXED, SPACINGS = 300.0, [2.0, 1.0, 0.5]


def run(root=None):
    out = outdir("spectra", root)
    r = {"models": {}}
    E0_free = None
    for name, factory in MODELS:
        theory = factory(G)
        model = build_gauge_model(theory, Lattice(N, 1.0, "open"))
        spec = mode_spectrum(model)
        spec.save(out / f"spectrum_{slug(name)}.npz")
        scaling = lowest_frequency_scaling(theory, NS)
        scaling.save(out / f"scaling_{slug(name)}.npz")
        sig = model.H.covariance
        E2 = np.diag(electric_field_correlator(model))
        r["models"][name] = {
            "theory": theory, "spectrum": spec, "scaling": scaling,
            "commutator": np.abs(model.electric_field_commutator()).max(),
            "E0": model.H.ground_state_energy,
            "purity": np.linalg.eigvalsh(sig + 0.5j * model.H.J).min(),
            "E2_bulk": E2[N // 2], "E2_boundary": E2[0], "F": model.F}
    E0_free = 0.5 * np.sum(2 * np.sin(np.pi * np.arange(1, N + 1) / (2 * (N + 1))))
    r["E0_free"] = E0_free
    cont = continuum_scaling(schwinger(G), L_FIXED, SPACINGS)
    cont.save(out / "continuum_schwinger.npz")
    r["continuum"] = cont
    return r


def report(r):
    section(f"2. Models (N = {N}, g = {G}, a = 1)")
    print(f"  {'model':20s} {'Q_L':>10s} {'Q_R':>10s} {'anomaly':>8s} {'mu':>8s}")
    for name, m in r["models"].items():
        t = m["theory"]
        print(f"  {name:20s} {str(t.QL[0].astype(int).tolist()):>10s} {str(t.QR[0].astype(int).tolist()):>10s} "
              f"{t.anomaly_matrix[0, 0]:8.1f} {t.meson_masses[0]:8.4f}")

    section(f"3. Spectrum and Gauss-law consistency (N = {N}, g = {G})")
    print("  prediction: omega = sqrt(mu^2 + k^2) for one mode, plus F - 1 free branches k (Dirichlet momenta k)")
    for name, m in r["models"].items():
        s = m["spectrum"]
        print(f"  {name:20s} modes = {len(s.frequencies)}  mu = {s.meta['meson_masses'][0]:.4f}  "
              f"omega_min = {s.omega_min:.4f}  omega_max = {s.frequencies[-1]:.4f}  "
              f"max|omega - prediction| = {s.max_deviation:.1e}  max|[E,E]| = {m['commutator']:.1e}")
    print("  (vector couplings are exact on the lattice; axial couplings act on vartheta = cumulative P and shift the\n"
          "   boundary momenta to k_n = (n + delta) pi / L, so their prediction holds in the bulk only, see 6.)")

    section(f"4. Ground state (N = {N}, g = {G})")
    for name, m in r["models"].items():
        print(f"  {name:20s} E_0 = {m['E0']:10.4f}  E_0 - F E_0(free) = {m['E0'] - m['F'] * r['E0_free']:8.4f}  "
              f"<E^2> bulk = {m['E2_bulk']:.3e}  boundary = {m['E2_boundary']:.3e}  "
              f"min eig(sigma + iJ/2) = {m['purity']:.1e}")

    section(f"5. Mode counting below mu (N = {N}, g = {G})")
    print("  prediction: the massive branch has no mode below mu, every massless branch has #(k < mu)")
    for name, m in r["models"].items():
        s = m["spectrum"]
        mu = s.meta["meson_masses"][0]
        print(f"  {name:20s} modes below mu: {s.count_below(mu):3d}   prediction: {int(np.sum(s.prediction < mu)):3d}")

    section(f"6. Finite-size scaling of the lowest frequency (g = {G})")
    print("  n_eff = k_1 (N+1)/pi with k_1 from omega_min; Dirichlet: n_eff = 1, n_eff = 1 + delta = boundary phase shift")
    for name, m in r["models"].items():
        sc = m["scaling"]
        mu = m["theory"].meson_masses[0]
        k = np.sqrt(np.clip(sc.y**2 - mu**2, 0, None)) if m["F"] == 1 else sc.y
        neff = 2 * (sc.x + 1) / np.pi * np.arcsin(k / 2)
        print(f"  {name:20s} N           " + "  ".join(f"{n:6.0f}" for n in sc.x))
        print(f"  {'':20s} omega_min   " + "  ".join(f"{v:.4f}" for v in sc.y))
        print(f"  {'':20s} prediction  " + "  ".join(f"{p:.4f}" for p in sc.prediction))
        print(f"  {'':20s} n_eff       " + "  ".join(f"{e:6.3f}" for e in neff))

    c = r["continuum"]
    section(f"7. Continuum limit at fixed length L = (N+1) a = {L_FIXED:.0f} (Schwinger, g = {G})")
    print("  prediction: omega_min -> sqrt(mu^2 + (pi/L)^2); the deviation falls like a^2")
    for a, n, y, p in zip(c.x, c.meta["N"], c.y, c.prediction):
        print(f"  a = {a:4.2f}  N = {n:4d}  omega_min = {y:.6f}   continuum = {p:.6f}   deviation = {abs(y - p):.1e}")
