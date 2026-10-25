# LatticeSim

Hamiltonian lattice simulations of bosonized (1+1)-dimensional gauge theories: from a set of chiral charges to
the gauge-invariant particle content of the theory, including its bound states.

```python
from latsimpkg import ChiralTheory, find_bound_states

theory = ChiralTheory(QL=[3, 4], QR=[5, 0], g=0.3)   # the 3-4-5-0 chiral gauge theory
scan = find_bound_states(theory)

scan.agrees                    # True: lattice and analytic charge lattice agree
[s.content for s in scan.bound_states]
# ['psi_3L^-2 psi_4L^-1 psi_5R^2', 'psi_3L^2 psi_4L psi_5R^-2']   massless composite fermion and antiparticle
```

Massless fermions coupled to abelian gauge fields U(1)^k become, after bosonization and elimination of the
gauge field, a quadratic Hamiltonian. LatticeSim solves it exactly on the lattice (Williamson normal form
including zero modes) and reads off the charge sectors, i.e. the massless gauge-invariant particles.

## Installation

Python >= 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage

```bash
make test        # fast tests (~10 s)
make test-all    # including the slow ones
make studies     # run all studies, results -> results/
make figures     # figures -> figures/ (only what is out of date)
make docs        # documentation -> docs/build/latticesim.pdf (needs latexmk, e.g. MacTeX)
```

More in the documentation (`make docs`): theory chapters from quantum mechanics to the bound-state search,
a worked 3-4-5-0 example, a recipe for testing your own theory, and the API reference.

| Function | Result |
|---|---|
| `find_bound_states(theory, N, nmax, Delta_max)` | `SectorScan`: massless sectors, bound states, analytic cross-check |
| `spectrum(theory, N, boundary)` | `Spectrum`: normal modes with continuum prediction |
| `screening_potential(theory, N, distances, q)` | `Potential`: screening or confinement of probe charges |
| `build_gauge_model(theory, lattice, theta, external_charges)` | `GaugeModel` with the `QuadraticHamiltonian` `model.H` |

All results can be saved (`.save("x.npz")`) and plotted (`latsimpkg.plotting`).

## Scope and limits

Quadratic Hamiltonians only (massless fermions, abelian gauge fields); fermion masses and multi-fermion
interactions need non-Gaussian methods. The coupling must resolve the meson in double precision
(mu L >~ 1e-2, mu a <~ 10); outside this window the library raises an error instead of returning wrong numbers.
See `ROADMAP.md` for the plans and `CHANGELOG.md` for the history.

## Validation

Every result is checked by tests against analytic results and the literature (Schwinger mass, screening
potential, Tong's 3-4-5-0 analysis), and the mathematical core by property-based stress tests.
