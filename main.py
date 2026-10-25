from latsimpkg.analysis import correlators, sectors
from latsimpkg.analysis.charge_lattice import charge_lattice_spectrum
from latsimpkg.conventions import chiral
from latsimpkg.lattice import Lattice
from latsimpkg.models import build_gauge_model
from latsimpkg.quadratic import QuadraticHamiltonian
from latsimpkg.theory import ChiralTheory
import numpy as np


# 3-4-5-0 model: left-movers with charges (3, 4), right-movers with (5, 0).
# Pairing (3, 5) and (4, 0) into Dirac bosons gives Q^V = (Q_L + Q_R)/2, Q^A = (Q_L - Q_R)/2.
# Anomaly freedom 3^2 + 4^2 = 5^2 + 0^2 is equivalent to Q^V . Q^A = 0.
QV_3450 = [4, 2]
QA_3450 = [-1, 2]

MODELS = [("Schwinger (vector)", [1], [0]),
          ("Schwinger (axial)",  [0], [1]),
          ("3-4-5-0",            QV_3450, QA_3450)]


def theory_from(QV, QA, g, allow_anomalous=False):
    """ChiralTheory with the pairing L_i <-> R_i of the given vector/axial charges."""
    QL, QR = chiral(QV, QA)
    return ChiralTheory(QL, QR, g, allow_anomalous=allow_anomalous)


def gauged_model(QV, QA, N, g, a=1.0, allow_anomalous=False):
    """Gauged model on the open chain."""
    return build_gauge_model(theory_from(QV, QA, g, allow_anomalous), Lattice(N, a, "open"))


def free_chain(N, a=1.0):
    """A single free scalar on the Dirichlet chain."""
    return build_gauge_model(ChiralTheory([0], [0], 0.0), Lattice(N, a, "open"))


def free_modes(N, a=1.0):
    """Frequencies of a single free Dirichlet chain."""
    return free_chain(N, a).H.frequencies


def ring_model(QV, QA, N, g, a=1.0):
    """Gauged model on the ring."""
    return build_gauge_model(theory_from(QV, QA, g), Lattice(N, a, "ring"))


def meson_mass(model):
    return model.theory.meson_masses[0]


def fermion_content(NL, NR, QV, QA):
    """Write a sector as product of fermion operators, e.g. psi_3L^2 psi_4L psibar_5R^2."""
    QL = np.asarray(QV) + np.asarray(QA); QR = np.asarray(QV) - np.asarray(QA)
    parts = []
    for N_, Q, chi in [(NL, QL, "L"), (NR, QR, "R")]:
        for k, q in zip(N_, Q):
            if k:
                op = ("psi" if k > 0 else "psibar") + f"_{q:g}{chi}"
                parts.append(op + (f"^{abs(k)}" if abs(k) > 1 else ""))
    return " ".join(parts) if parts else "1"


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

    sc = free_chain(N, a)

    section(f"1. Free scalar field (N = {N}, a = {a})")
    sig = QuadraticHamiltonian(np.diag([w**2, 1.0])).covariance
    print(f"  Test 1  single oscillator:  <q^2> = {sig[0, 0]:.12f}   expected 1/(2w) = {1/(2*w):.12f}")

    om = sc.H.frequencies
    exact = (2 / sc.lattice.a) * np.sin(np.pi * np.arange(1, N + 1) / (2 * (N + 1)))
    print(f"  Test 2  Dirichlet chain:    max|omega - (2/a) sin(pi n/(2(N+1)))| = {np.max(np.abs(om - exact)):.1e}"
          f"   (relative {np.max(np.abs(om - exact) / exact):.1e})")

    sig = sc.H.covariance
    lam = np.linalg.eigvalsh(sig + 0.5j * sc.H.J).min()
    print(f"  Test 3  uncertainty principle sigma + iJ/2 >= 0:  smallest eigenvalue = {lam:.1e}   (0 = pure state)")

    E0 = sc.H.ground_state_energy
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
              f"{np.sum(np.square(QV) + np.square(QA)):8.1f} {meson_mass(m):8.4f}")

    # ------------------------------------------------------------------
    # 3. Spectrum and Gauss law
    # ------------------------------------------------------------------
    section(f"3. Spectrum and Gauss-law consistency (N = {N}, g = {g})")
    print("  prediction: omega = sqrt(mu^2 + k^2) for one mode, plus F - 1 free branches k,"
          " with the free Dirichlet momenta k")
    for name, QV, QA in MODELS:
        m = gauged_model(QV, QA, N, g)
        om = m.H.frequencies; mu = meson_mass(m)
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
        sig = m.H.covariance
        lam = np.linalg.eigvalsh(sig + 0.5j * m.H.J).min()
        E2 = np.diag(correlators.electric_field_correlator(m))
        E0 = m.H.ground_state_energy
        print(f"  {name:20s} E_0 = {E0:10.4f}  E_0 - F E_0(free) = {E0 - m.F * E0_free:8.4f}  "
              f"<E^2> bulk = {E2[N // 2]:.3e}  boundary = {E2[0]:.3e}  min eig(sigma + iJ/2) = {lam:.1e}")

    # ------------------------------------------------------------------
    # 5. Meson mass from the electric-field correlator
    # ------------------------------------------------------------------
    section("5. Meson mass from the electric-field correlator (fit A K0(mu r) + B)")
    for name, QV, QA, gg in [("vector", [1], [0], 0.1), ("axial", [0], [1], 0.1),
                             ("3-4-5-0", QV_3450, QA_3450, 0.04)]:
        m = gauged_model(QV, QA, N, gg)
        mu = meson_mass(m)
        mfit, B = correlators.meson_mass_from_E(m)
        print(f"  {name:8s} g = {gg:.2f}  mu = {mu:.4f}  mu_fit = {mfit:.4f}  "
              f"rel. deviation = {abs(mfit - mu) / mu:.1e}  offset B = {B:.1e}")

    # ------------------------------------------------------------------
    # 6. Anomalous charges
    # ------------------------------------------------------------------
    section("6. Anomalous charges: [E_j, E_k] = i (g^2/pi) (Q^V.Q^A) sign(k - j)")
    for QV, QA in [([1, 1], [1, 0]), ([1, 1], [1, 1]), ([2, 1], [1, 0]), (QV_3450, QA_3450)]:
        m = gauged_model(QV, QA, 60, g, allow_anomalous=True)
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
        om = m.H.frequencies; mu = meson_mass(m)
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
            mu = meson_mass(m)
            k1 = 2 * np.sin(np.pi / (2 * (n + 1)))
            om_min = m.H.frequencies[0]
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
        mu = meson_mass(m)
        print(f"  a = {a:4.2f}  N = {n:4d}  omega_min = {m.H.frequencies[0]:.6f}   "
              f"continuum = {np.sqrt(mu**2 + (np.pi / ((n + 1) * a))**2):.6f}")

    # ------------------------------------------------------------------
    # 10. Ring with zero modes: free compact boson
    # ------------------------------------------------------------------
    section("10. Ring with zero modes: free compact boson (F = 1, g = 0, N = 20)")
    print("  prediction: E L/(2 pi) = (N_L^2 + N_R^2)/2 = h + hb,  s = (N_L^2 - N_R^2)/2 = h - hb")
    r = build_gauge_model(ChiralTheory([0], [0], 0.0), Lattice(20, 1.0, "ring"))
    A = r.H.conserved_charges
    print(f"  conserved charges from ker(M J): {A.shape[1]}   (prediction 2: winding n and momentum m)")
    for NL, NR in [((1,), (0,)), ((0,), (1,)), ((1,), (1,)), ((2,), (0,)), ((2,), (1,))]:
        d = sectors.sector_energy(r, NL, NR)
        print(f"  N_L = {NL[0]:2d}  N_R = {NR[0]:2d}   E L/(2 pi) = {d['Delta']:.6f}  "
              f"(pred {(NL[0]**2 + NR[0]**2) / 2:.3f})   s = {d['s']:+.6f}  (pred {(NL[0]**2 - NR[0]**2) / 2:+.3f})")

    # ------------------------------------------------------------------
    # 11. Conserved charges of the gauged models
    # ------------------------------------------------------------------
    N_ring, g_ring = 60, 0.3
    section(f"11. Conserved charges from ker(M J) on the ring (N = {N_ring}, g = {g_ring})")
    print("  prediction: the conserved combinations t_n.n + t_m.m are the global U(1)s without mixed anomaly,")
    print("  Q^A.t_n + Q^V.t_m = 0, i.e. 2F - 1 of them, and the gauge charge q = Q^V.n + Q^A.m is one of them")
    for name, QV, QA in MODELS:
        r = ring_model(QV, QA, N_ring, g_ring)
        A = r.H.conserved_charges
        T = sectors.charge_map(r)
        anomaly = np.abs(T @ np.concatenate([r.QA[0], r.QV[0]])).max() / np.abs(T).max()
        gauge = np.concatenate([r.QV[0], r.QA[0]])
        resid = np.linalg.norm(gauge - np.linalg.pinv(T) @ T @ gauge) / np.linalg.norm(gauge)
        print(f"  {name:20s} number = {A.shape[1]} (pred {2 * r.F - 1})   "
              f"max|Q^A.t_n + Q^V.t_m| = {anomaly:.1e}   gauge charge outside span = {resid:.1e}   "
              f"max|[E,E]| = {np.abs(r.electric_field_commutator()).max():.1e}")

    # ------------------------------------------------------------------
    # 12. Sector search: numerics on the ring vs. charge-lattice analysis
    # ------------------------------------------------------------------
    section(f"12. Search for gauge-invariant massless sectors (ring N = {N_ring}, g = {g_ring}, |N| <= 3, Delta <= 2)")
    print("  numerics: E L/(2 pi) and momentum of the constrained minimum;  analytics: h = |N_L^perp|^2/2, hb = |N_R^perp|^2/2")
    for name, QV, QA in MODELS:
        r = ring_model(QV, QA, N_ring, g_ring)
        num = sectors.sector_scan(r, nmax=3, Delta_max=2.0)
        ana = {(d["NL"], d["NR"]): d for d in charge_lattice_spectrum(r.pairing, nmax=3, Delta_max=2.0)}
        print(f"  {name}: {len(num)} sectors numerically, {len(ana)} analytically")
        if not num and not ana:
            print("    none: the only conserved charge is the gauge charge, every neutral state lies in the vacuum sector")
        for d in num:
            a_ = ana.get((d["NL"], d["NR"]))
            cmp = (f"ana Delta = {a_['Delta']:.4f}  s = {a_['s']:+.4f}" if a_ else "ana: not found")
            print(f"    N_L = {str(d['NL']):9s} N_R = {str(d['NR']):9s}  Delta = {d['Delta']:.6f}  s = {d['s']:+.6f}  "
                  f"(h, hb) = ({d['h']:.3f}, {d['hb']:.3f})   {cmp}   {d['kind']:15s} {fermion_content(d['NL'], d['NR'], QV, QA)}")
        missing = set(ana) - {(d["NL"], d["NR"]) for d in num}
        for key in sorted(missing):
            print(f"    only analytic: N_L = {key[0]}  N_R = {key[1]}  Delta = {ana[key]['Delta']:.4f}")
        bound = [d for d in num if d["kind"] == "bound state"]
        print(f"    -> bound states: {len(bound)}" + "".join(
            f"\n       {fermion_content(d['NL'], d['NR'], QV, QA)}   (h, hb) = ({d['h']:.2f}, {d['hb']:.2f}),"
            f" gauge charge {np.dot(np.add(QV, QA), d['NL']) + np.dot(np.subtract(QV, QA), d['NR']):g}" for d in bound))

    # ------------------------------------------------------------------
    # 13. Volume dependence: massless sectors vs. charged sectors
    # ------------------------------------------------------------------
    section(f"13. Volume dependence of the sectors (3-4-5-0, g = {g_ring})")
    print("  massless sector: E L/(2 pi) independent of L.  Charged sector (q != 0, forbidden by Gauss on the ring):")
    print("  the seam acts as external counter charge, which is screened -> E stays finite (gapped), no 1/L state")
    Ns_ring = [40, 80, 160]
    tests = [((2, 1), (-2, 0)), ((0, 0), (0, 1)), ((2, 1), (-2, 1)), ((1, 0), (0, 0)), ((0, 0), (1, 0))]
    print(f"  {'N_L, N_R':20s} {'q':>4s}   " + "   ".join(f"E (N={n:3d})   E L/2pi" for n in Ns_ring))
    rings = [ring_model(QV_3450, QA_3450, n, g_ring) for n in Ns_ring]
    for NL, NR in tests:
        ds = [sectors.sector_energy(r, NL, NR) for r in rings]
        print(f"  {str(NL) + ' ' + str(NR):20s} {ds[0]['q'][0]:4.0f}   " + "   ".join(
            f"{d['E']:.6f}  {d['Delta']:9.4f}" for d in ds))

