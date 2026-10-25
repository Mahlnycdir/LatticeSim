"""Study 3: meson mass from the electric-field correlator; anomalous charges."""
import numpy as np

from latsimpkg import ChiralTheory, Lattice, build_gauge_model, axial_schwinger, model_3450, schwinger
from latsimpkg.analysis.correlators import meson_mass_from_E
from latsimpkg.analysis.results import Series
from studies.common import outdir, section, slug

N = 300
FITS = [("vector", schwinger, 0.1), ("axial", axial_schwinger, 0.1), ("3-4-5-0", model_3450, 0.04)]
G_ANOMALY = 0.04
ANOMALOUS = [([2, 1], [0, 1]), ([2, 2], [0, 0]), ([3, 1], [1, 1]), ([3, 4], [5, 0])]


def run(root=None):
    out = outdir("correlators", root)
    fits = {}
    for name, factory, g in FITS:
        fit = meson_mass_from_E(build_gauge_model(factory(g), Lattice(N, 1.0, "open")))
        fit.save(out / f"fit_{slug(name)}.npz")
        fits[name] = (g, fit)
    x, y = [], []
    for QL, QR in ANOMALOUS:
        theory = ChiralTheory(QL, QR, G_ANOMALY, allow_anomalous=True)
        model = build_gauge_model(theory, Lattice(60, 1.0, "open"))
        x.append(abs(theory.anomaly_matrix[0, 0]) / 4)
        y.append(np.abs(model.electric_field_commutator()).max())
    x = np.array(x)
    anomaly = Series(x, np.array(y), G_ANOMALY**2 / np.pi * x,
                     {"name": "anomalous commutator", "xlabel": "|sum Q_L^2 - sum Q_R^2| / 4",
                      "ylabel": "max |[E_j, E_k]|", "g": G_ANOMALY, "charges": ANOMALOUS})
    anomaly.save(out / "anomalous_commutator.npz")
    return {"fits": fits, "anomaly": anomaly}


def report(r):
    section("8. Meson mass from the electric-field correlator (fit A K0(mu r) + B)")
    for name, (g, fit) in r["fits"].items():
        print(f"  {name:8s} g = {g:.2f}  mu = {fit.mu_expected:.4f}  mu_fit = {fit.mu:.4f}  "
              f"rel. deviation = {fit.rel_deviation:.1e}  offset B = {fit.B:.1e}  window = {fit.meta['window']}")

    a = r["anomaly"]
    section("9. Anomalous charges: [E_j, E_k] = i (g^2/pi) (Q^V.Q^A) sign(k - j),  Q^V.Q^A = (sum Q_L^2 - sum Q_R^2)/4")
    for (QL, QR), y, p in zip(a.meta["charges"], a.y, a.prediction):
        print(f"  Q_L = {str(QL):8s} Q_R = {str(QR):8s}  max|[E,E]| = {y:.3e}   prediction = {p:.3e}")
