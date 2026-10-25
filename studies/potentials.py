"""Study 5: probe charges (screening vs. confinement) and the theta angle."""
import numpy as np

from latsimpkg import ChiralTheory, Lattice, build_gauge_model, model_3450, schwinger, screening_potential
from studies.common import outdir, section

N, G = 400, 0.1
DISTANCES = list(range(2, 121, 6))
CASES = [("screening_q1", "Schwinger, q = 1", schwinger(G), 1.0),
         ("screening_q05", "Schwinger, q = 1/2", schwinger(G), 0.5),
         ("confinement_q1", "neutral matter, q = 1", ChiralTheory([0], [0], G), 1.0)]
THETA = 1.0


def run(root=None):
    out = outdir("potentials", root)
    r = {"potentials": {}, "theta": {}}
    for key, label, theory, q in CASES:
        pot = screening_potential(theory, N, DISTANCES, q=q)
        pot.save(out / f"{key}.npz")
        r["potentials"][label] = pot
    for name, theory in [("Schwinger", schwinger(0.3)), ("3-4-5-0", model_3450(0.3))]:
        r["theta"][name] = {b: build_gauge_model(theory, Lattice(40, 1.0, b), theta=THETA).background_energy
                            for b in ("ring", "open")}
    return r


def report(r):
    section(f"14. Static potential of probe charges +q, -q (open chain N = {N}, g = {G})")
    print("  screening: V(d) = (g^2 q^2 / 2 mu)(1 - exp(-mu d));  confinement: V(d) = g^2 q^2 d / 2")
    for label, pot in r["potentials"].items():
        picks = [0, len(pot.d) // 2, len(pot.d) - 1]
        values = "  ".join(f"V({pot.d[i]:.0f}) = {pot.V[i]:.5f}" for i in picks)
        print(f"  {label:24s} {pot.meta['regime']:12s} {values}   max rel. deviation = {pot.max_rel_deviation:.1e}")

    section(f"15. Theta angle (theta = {THETA}, N = 40, g = 0.3): background energy")
    print("  massless fermions: theta is absorbed by the zero modes on the ring (energy 0); the open chain fixes them")
    for name, e in r["theta"].items():
        print(f"  {name:10s} ring: {e['ring']:.1e}   open: {e['open']:.3e}")
