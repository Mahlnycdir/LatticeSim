from latsimpkg.scalar import ScalarFieldGaugeTheory
import numpy as np


# 3-4-5-0 model: left-movers with charges (3, 4), right-movers with (5, 0).
# Pairing (3, 5) and (4, 0) into Dirac bosons gives Q^V = (Q_L + Q_R)/2, Q^A = (Q_L - Q_R)/2.
# Anomaly freedom 3^2 + 4^2 = 5^2 + 0^2 is equivalent to Q^V . Q^A = 0.
QV_3450 = [4, 2]
QA_3450 = [-1, 2]

MODELS = [("Schwinger (vector)", [1], [0]),
          ("Schwinger (axial)",  [0], [1]),
          ("3-4-5-0",            QV_3450, QA_3450)]


def gauged_model(QV, QA, N, g, a=1.0):
    """Gauged model with the full Hamiltonian already built."""
    model = ScalarFieldGaugeTheory(N, a, g, QV, QA)
    model.add_interacting_hamiltonian_matrix()
    return model


def free_modes(N, a=1.0):
    """Frequencies of a single free Dirichlet chain."""
    free = ScalarFieldGaugeTheory(N, a)
    free.free_hamiltonian_matrix()
    return free.normal_modes()


def section(title):
    print()
    print(title)
    print("-" * len(title))


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # 1. Free scalar field (consistency of the Gaussian machinery)
    # ------------------------------------------------------------------
    w = 1.7
    N = 50
    a = 0.001

    sc = ScalarFieldGaugeTheory(N, a)
    sc.free_hamiltonian_matrix()

    section(f"1. Free scalar field (N = {N}, a = {a})")
    sig = sc.ground_state_covariance(np.diag([w**2, 1.0]))
    print(f"  Test 1  single oscillator:  <q^2> = {sig[0, 0]:.12f}   expected 1/(2w) = {1/(2*w):.12f}")

    om = sc.normal_modes()
    exact = (2 / sc.a) * np.sin(np.pi * np.arange(1, N + 1) / (2 * (N + 1)))
    print(f"  Test 2  Dirichlet chain:    max|omega - (2/a) sin(pi n/(2(N+1)))| = {np.max(np.abs(om - exact)):.1e}"
          f"   (relative {np.max(np.abs(om - exact) / exact):.1e})")

    sig = sc.ground_state_covariance()
    lam = np.linalg.eigvalsh(sig + 0.5j * sc.symplectic_matrix(N)).min()
    print(f"  Test 3  uncertainty principle sigma + iJ/2 >= 0:  smallest eigenvalue = {lam:.1e}   (0 = pure state)")

    E0 = sc.ground_state_energy()
    print(f"  Test 4  zero-point energy:  E_0 = {E0:.6f}   expected sum_n omega_n/2 = {0.5 * exact.sum():.6f}")

    # ------------------------------------------------------------------
    # 2. Gauged models: parameters
    # ------------------------------------------------------------------
    N, g = 300, 0.04
    kfree = free_modes(N)

    section(f"2. Models (N = {N}, g = {g}, a = 1)")
    print(f"  {'model':20s} {'Q^V':>10s} {'Q^A':>10s} {'Q^V.Q^A':>8s} {'sum Q^2':>8s} {'mu':>8s}")
    for name, QV, QA in MODELS:
        m = gauged_model(QV, QA, N, g)
        print(f"  {name:20s} {str(QV):>10s} {str(QA):>10s} {np.dot(QV, QA):8.1f} "
              f"{np.sum(np.square(QV) + np.square(QA)):8.1f} {m.meson_mass():8.4f}")

    # ------------------------------------------------------------------
    # 3. Spectrum and Gauss law
    # ------------------------------------------------------------------
    section(f"3. Spectrum and Gauss-law consistency (N = {N}, g = {g})")
    print("  prediction: omega = sqrt(mu^2 + k^2) for one mode, plus F - 1 free branches k,"
          " with the free Dirichlet momenta k")
    for name, QV, QA in MODELS:
        m = gauged_model(QV, QA, N, g)
        om = m.normal_modes(); mu = m.meson_mass()
        pred = np.sort(np.concatenate([np.sqrt(mu**2 + kfree**2)] + [kfree] * (m.F - 1)))
        comm = np.abs(m.electric_field_commutator()).max()
        print(f"  {name:20s} modes = {len(om)}  mu = {mu:.4f}  omega_min = {om[0]:.4f}  "
              f"omega_max = {om[-1]:.4f}  max|omega - prediction| = {np.abs(om - pred).max():.1e}  "
              f"max|[E,E]| = {comm:.1e}")
    print("  (the vector coupling is exact on the lattice. The axial coupling acts on vartheta = cumulative P;"
          " in dual variables the Dirichlet chain\n   becomes a chain with one Dirichlet and one free end plus a"
          " non-local rank-one term, so the Dirichlet momenta are only correct\n   in the bulk. The allowed"
          " momenta become k_n = (n + delta) pi / L with a boundary phase shift delta; the absolute deviation"
          " shrinks ~ 1/N, see section 8)")

    # ------------------------------------------------------------------
    # 4. Ground state of the gauged models
    # ------------------------------------------------------------------
    section(f"4. Ground state (N = {N}, g = {g})")
    E0_free = 0.5 * kfree.sum()
    for name, QV, QA in MODELS:
        m = gauged_model(QV, QA, N, g)
        sig = m.ground_state_covariance()
        lam = np.linalg.eigvalsh(sig + 0.5j * m.symplectic_matrix(sig.shape[0] // 2)).min()
        C = m.electric_field_matrix()
        E2 = np.diag(C @ sig @ C.T)
        E0 = m.ground_state_energy()
        print(f"  {name:20s} E_0 = {E0:10.4f}  E_0 - F E_0(free) = {E0 - m.F * E0_free:8.4f}  "
              f"<E^2> bulk = {E2[N // 2]:.3e}  boundary = {E2[0]:.3e}  min eig(sigma + iJ/2) = {lam:.1e}")

    # ------------------------------------------------------------------
    # 5. Meson mass from the electric-field correlator
    # ------------------------------------------------------------------
    section("5. Meson mass from the electric-field correlator (fit A K0(mu r) + B)")
    for name, QV, QA, gg in [("vector", [1], [0], 0.1), ("axial", [0], [1], 0.1),
                             ("3-4-5-0", QV_3450, QA_3450, 0.04)]:
        m = gauged_model(QV, QA, N, gg)
        mu = m.meson_mass()
        mfit, B = m.meson_mass_from_E()
        print(f"  {name:8s} g = {gg:.2f}  mu = {mu:.4f}  mu_fit = {mfit:.4f}  "
              f"rel. deviation = {abs(mfit - mu) / mu:.1e}  offset B = {B:.1e}")

    # ------------------------------------------------------------------
    # 6. Anomalous charges
    # ------------------------------------------------------------------
    section("6. Anomalous charges: [E_j, E_k] = i (g^2/pi) (Q^V.Q^A) sign(k - j)")
    for QV, QA in [([1, 1], [1, 0]), ([1, 1], [1, 1]), ([2, 1], [1, 0]), (QV_3450, QA_3450)]:
        m = ScalarFieldGaugeTheory(60, 1.0, g, QV, QA)
        comm = np.abs(m.electric_field_commutator()).max()
        pred = g**2 / np.pi * abs(np.dot(QV, QA))
        print(f"  Q^V = {str(QV):8s} Q^A = {str(QA):8s}  max|[E,E]| = {comm:.3e}   prediction = {pred:.3e}")

    # ------------------------------------------------------------------
    # 7. Mode counting below mu
    # ------------------------------------------------------------------
    section(f"7. Mode counting below mu (N = {N}, g = {g})")
    print("  prediction: the massive branch has no mode below mu, every massless branch has #(k < mu)")
    for name, QV, QA in MODELS:
        m = gauged_model(QV, QA, N, g)
        om = m.normal_modes(); mu = m.meson_mass()
        print(f"  {name:20s} modes below mu: {np.sum(om < mu):3d}   prediction: {(m.F - 1) * np.sum(kfree < mu):3d}")
    print("  (a difference of one mode for axial charges comes from the shifted boundary momenta of section 3)")

    # ------------------------------------------------------------------
    # 8. Finite-size scaling
    # ------------------------------------------------------------------
    Ns = [100, 200, 300, 450, 600]
    section(f"8. Finite-size scaling of the lowest frequency (g = {g})")
    print("  prediction: sqrt(mu^2 + k_1^2) -> mu for F = 1, k_1 = 2 sin(pi/(2(N+1))) -> 0 for F > 1")
    print("  n_eff = k_1 (N+1)/pi with k_1 from omega_min (massive branch for F = 1, massless branch for F > 1);"
          " Dirichlet means n_eff = 1, n_eff = 1 + delta measures the boundary phase shift")
    for name, QV, QA in MODELS:
        vals, preds, neff = [], [], []
        for n in Ns:
            m = gauged_model(QV, QA, n, g)
            mu = m.meson_mass()
            k1 = 2 * np.sin(np.pi / (2 * (n + 1)))
            om_min = m.normal_modes()[0]
            vals.append(om_min)
            preds.append(np.sqrt(mu**2 + k1**2) if m.F == 1 else k1)
            k_eff = np.sqrt(max(om_min**2 - mu**2, 0.0)) if m.F == 1 else om_min
            neff.append(2 * (n + 1) / np.pi * np.arcsin(k_eff / 2))
        print(f"  {name:20s} N           " + "  ".join(f"{n:6d}" for n in Ns))
        print(f"  {'':20s} omega_min   " + "  ".join(f"{v:.4f}" for v in vals))
        print(f"  {'':20s} prediction  " + "  ".join(f"{p:.4f}" for p in preds))
        print(f"  {'':20s} n_eff       " + "  ".join(f"{e:6.3f}" for e in neff))

    # ------------------------------------------------------------------
    # 9. Continuum limit at fixed physical length
    # ------------------------------------------------------------------
    L = 300.0
    section(f"9. Continuum limit at fixed length L = N a = {L:.0f} (Schwinger, g = {g})")
    print("  prediction: omega_min -> sqrt(mu^2 + (pi/L)^2), mu does not depend on a")
    for a in [2.0, 1.0, 0.5]:
        n = int(round(L / a))
        m = gauged_model([1], [0], n, g, a)
        mu = m.meson_mass()
        print(f"  a = {a:4.2f}  N = {n:4d}  omega_min = {m.normal_modes()[0]:.6f}   "
              f"continuum = {np.sqrt(mu**2 + (np.pi / ((n + 1) * a))**2):.6f}")
