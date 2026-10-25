"""Study 4: charge sectors on the ring, conserved charges and the bound-state search."""
import numpy as np

from latsimpkg import ChiralTheory, Lattice, build_gauge_model, find_bound_states, model_3450
from latsimpkg.analysis.results import Series
from latsimpkg.analysis.sectors import charge_map, sector_energy
from studies.common import MODELS, outdir, section, slug

N_RING, G_RING = 60, 0.3
FREE_SECTORS = [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]
VOLUMES = [40, 80, 160]
VOLUME_SECTORS = [((2, 1), (-2, 0)), ((0, 0), (0, 1)), ((2, 1), (-2, 1)), ((1, 0), (0, 0)), ((0, 0), (1, 0))]


def run(root=None):
    out = outdir("sectors", root)
    free = build_gauge_model(ChiralTheory([0], [0], 0.0), Lattice(20, 1.0, "ring"))
    r = {"free_conserved": free.H.n_free,
         "free": [(NL, NR, sector_energy(free, (NL,), (NR,))) for NL, NR in FREE_SECTORS],
         "charges": {}, "scans": {}}
    for name, factory in MODELS:
        theory = factory(G_RING)
        model = build_gauge_model(theory, Lattice(N_RING, 1.0, "ring"))
        T = charge_map(model)
        QV, QA = model.QV[0], model.QA[0]
        gauge = np.concatenate([QV, QA])
        r["charges"][name] = {
            "number": model.H.n_free, "expected": 2 * model.F - model.k,
            "anomaly": np.abs(T @ np.concatenate([QA, QV])).max() / np.abs(T).max(),
            "gauge_outside": np.linalg.norm(gauge - np.linalg.pinv(T) @ T @ gauge) / np.linalg.norm(gauge),
            "commutator": np.abs(model.electric_field_commutator()).max()}
        scan = find_bound_states(theory, N=N_RING)
        scan.save(out / f"scan_{slug(name)}.npz")
        r["scans"][name] = scan
    rings = [build_gauge_model(model_3450(G_RING), Lattice(n, 1.0, "ring")) for n in VOLUMES]
    r["volume"] = []
    for NL, NR in VOLUME_SECTORS:
        energies = [sector_energy(m, NL, NR) for m in rings]
        series = Series(np.array(VOLUMES, dtype=float), np.array([e.E for e in energies]), np.array([]),
                        {"name": f"sector {NL} {NR}", "xlabel": "N", "ylabel": "E", "NL": NL, "NR": NR,
                         "q": energies[0].q, "Delta": [e.Delta for e in energies]})
        series.save(out / f"volume_{'_'.join(map(str, NL + NR)).replace('-', 'm')}.npz")
        r["volume"].append(series)
    return r


def report(r):
    section("10. Ring with zero modes: free compact boson (F = 1, g = 0, N = 20)")
    print("  prediction: E L/(2 pi) = (N_L^2 + N_R^2)/2 = h + hb,  s = (N_L^2 - N_R^2)/2 = h - hb")
    print(f"  conserved charges from ker(M J): {r['free_conserved']}   (prediction 2: winding n and momentum m)")
    for NL, NR, e in r["free"]:
        print(f"  N_L = {NL:2d}  N_R = {NR:2d}   E L/(2 pi) = {e.Delta:.6f}  (pred {(NL**2 + NR**2) / 2:.3f})   "
              f"s = {e.s:+.6f}  (pred {(NL**2 - NR**2) / 2:+.3f})")

    section(f"11. Conserved charges from ker(M J) on the ring (N = {N_RING}, g = {G_RING})")
    print("  prediction: the global U(1)s without mixed anomaly, Q^A.t_n + Q^V.t_m = 0, i.e. 2F - k of them,")
    print("  and the gauge charge q = Q^V.n + Q^A.m is one of them")
    for name, c in r["charges"].items():
        print(f"  {name:20s} number = {c['number']} (pred {c['expected']})   max|Q^A.t_n + Q^V.t_m| = {c['anomaly']:.1e}   "
              f"gauge charge outside span = {c['gauge_outside']:.1e}   max|[E,E]| = {c['commutator']:.1e}")

    section(f"12. Search for gauge-invariant massless sectors (ring N = {N_RING}, g = {G_RING}, |N| <= 3, Delta <= 2)")
    print("  numerics: E L/(2 pi) and momentum of the constrained minimum;  analytics: h = |N_L^perp|^2/2, hb = |N_R^perp|^2/2")
    for name, scan in r["scans"].items():
        print(f"  {name}: {len(scan.numeric)} sectors numerically, {len(scan.analytic)} analytically, "
              f"agree: {scan.agrees}, max deviation {scan.max_deviation:.1e}")
        if not scan.numeric and not scan.analytic:
            print("    none: the only conserved charge is the gauge charge, every neutral state lies in the vacuum sector")
        ana = {x.key: x for x in scan.analytic}
        for x in scan.numeric:
            a = ana.get(x.key)
            cmp = f"ana Delta = {a.Delta:.4f}  s = {a.s:+.4f}" if a else "ana: other representative"
            print(f"    N_L = {str(x.NL):9s} N_R = {str(x.NR):9s}  Delta = {x.Delta:.6f}  s = {x.s:+.6f}  "
                  f"(h, hb) = ({x.h:.3f}, {x.hb:.3f})   {cmp}   {x.kind:15s} {x.content}")
        print(f"    -> bound states: {len(scan.bound_states)}" +
              "".join(f"\n       {x.content}   (h, hb) = ({x.h:.2f}, {x.hb:.2f})" for x in scan.bound_states))

    section(f"13. Volume dependence of the sectors (3-4-5-0, g = {G_RING})")
    print("  massless sector: E L/(2 pi) independent of L.  Charged sector (q != 0, forbidden by Gauss on the ring):")
    print("  the seam acts as external counter charge, which is screened -> E stays finite (gapped), no 1/L state")
    print(f"  {'N_L, N_R':20s} {'q':>4s}   " + "   ".join(f"E (N={n:3d})   E L/2pi" for n in VOLUMES))
    for s in r["volume"]:
        print(f"  {str(tuple(s.meta['NL'])) + ' ' + str(tuple(s.meta['NR'])):20s} {s.meta['q'][0]:4.0f}   " + "   ".join(
            f"{E:.6f}  {D:9.4f}" for E, D in zip(s.y, s.meta["Delta"])))
