# LatticeSim – Roadmap

Goal: a library in which (1+1)-dimensional abelian gauge theories with chiral fermions can be tested with
**a single theory call**. Bosonization, lattice, Hamiltonian and analysis are generic and protected by tests.

**Current state: release 0.1** – the full path from a theory to its gauge-invariant massless spectrum and
bound states, with documentation, figures, literature cross-check and stress tests. Details in `CHANGELOG.md`.

## Decisions

| Topic | Decision |
|---|---|
| Input | chiral charges Q_L, Q_R (as in the literature); pairing into Dirac bosons is automatic |
| Gauge group | U(1)^k (charges are k x n matrices) |
| Scope | Gaussian core (massless fermions, abelian gauge fields); theory and lattice are separated so that a later non-Gaussian backend can use the same `ChiralTheory` |
| Docstrings | ASCII math, readable in an editor; typeset mathematics lives in `docs/chapters/` |
| Numerics | never return silently wrong numbers: validate inputs, estimate precision, raise with an explanation |

## Architecture

```
latsimpkg/
  conventions.py        normalizations (sqrt(pi)), L/R <-> V/A, fermion numbers <-> zero modes, signs
  theory.py             ChiralTheory, Pairing, schwinger / axial_schwinger / model_3450
  lattice.py            Lattice(N, a, boundary = "open" | "ring")
  phasespace.py         PhaseSpace: named blocks of X, symplectic matrix J
  quadratic.py          QuadraticHamiltonian: Williamson normal form incl. zero modes, ground state,
                        conserved charges, constrained minima, precision estimate
  models/gauge.py       build_gauge_model(theory, lattice, theta, external_charges) -> GaugeModel
  analysis/results.py   Sector, SectorScan, Spectrum, Series, FitResult, Potential (saveable)
  analysis/spectrum.py, correlators.py, potentials.py, sectors.py, charge_lattice.py
  api.py                find_bound_states, spectrum, screening_potential
  plotting.py           figures of saved results (object-oriented matplotlib, no pyplot)
studies/                studies (python -m studies) and documentation figures (python -m studies.figures)
tools/build_docs.py     documentation: docstrings -> API reference, figures, LaTeX -> PDF
docs/                   latticesim.tex, chapters/ (theory), generated/ and build/ (git-ignored)
tests/                  pytest + hypothesis; builders.py = the only place where tests construct models
```

Layering rule: every layer imports only from the layers below it in this list.

## History

| Step | Content | Key findings |
|---|---|---|
| Phase 0 | test safety net: `pyproject.toml`, `.venv`, adapters in `tests/builders.py`, reference values | – |
| Phase 1 | `PhaseSpace`, `QuadraticHamiltonian` (Williamson with zero modes) | `eigvals(J M)` is unreliable for free particles (Jordan blocks); the Hermitian form is exact |
| Phase 2 | `conventions`, `Lattice`, `ChiralTheory` (U(1)^k, anomaly matrix, pairing, global symmetries) | analytic global symmetries = conserved charges found numerically on the ring |
| Phase 3 | one model builder for open chain and ring, theta, probe charges | constant [E_a, E_b] for k > 1 is a pairing artefact; `choose_pairing` avoids it. Screening and confinement potentials reproduced |
| Phase 4 | result types, `find_bound_states`, `studies/` instead of `main.py` | – |
| 0.1 plotting | `plotting.py`, 9 documentation figures with captions, incremental rendering | – |
| 0.1 docs | 7 theory chapters from quantum mechanics to bound states, generated API reference, `make docs` | – |
| 0.1 literature | 12 references verified, `tests/test_literature.py` | gauging H = (5,0 \| 3,-4) reproduces Tong's IR Dirac fermion exactly; potential identical to arXiv:2607.19468 eq. (3.88) |
| 0.1 stress tests | property-based tests of the core and of random anomaly-free theories, edge cases | **two bugs fixed:** weak couplings were taken for zero modes (now exact `n_zero` + precision guard); the ring dual zero mode broke translation invariance for axial charges (now averages, exact phi <-> vartheta duality) |

## Known limitations

- Only quadratic Hamiltonians: massless fermions, abelian gauge fields. Fermion masses and multi-fermion
  interactions (cosine terms) need a non-Gaussian backend.
- The bound-state search finds massless sectors; the classification assumes a fermionic massless sector
  (h in k^2/2). Composites need a representative with |N| <= nmax.
- Double precision limits the coupling window: mu L >~ 1e-2 and mu a <~ 10 (otherwise a clear error).
- Dense linear algebra: cost ~ (F N)^3; N of a few hundred per flavor is comfortable.

## Next

In this order, each with its own tests and documentation:
1. **'t Hooft anomaly matching** of the surviving U(1)s between UV fermions and IR sectors (algebraic), and the
   IR central charges (c_L, c_R) from the charge lattice.
2. **Central charge from the lattice**: Casimir energy E_0(L) ~ -pi (c + cb) / (12 L) on the ring, and the
   entanglement entropy of Gaussian states (symplectic eigenvalues of reduced covariance matrices).
3. **Vertex correlators**: expectation values of exp(i beta . X) from the covariance; condensates (Schwinger
   chiral condensate, 't Hooft vertex of 3-4-5-0) and propagators of the bound states (values to be verified
   against the literature first).
4. **Non-Gaussian backend interface**: the same `ChiralTheory` with mass terms / cosine interactions (DMRG or
   Hamiltonian truncation); targets: theta dependence and fractional-charge confinement in the massive Schwinger
   model, symmetric mass generation in 3-4-5-0.
5. **Performance**: sparse matrices and iterative eigensolvers for large N.

## Workflow

- One branch per step; tests first, then refactor, then delete the old parts; every step ends green.
- For physics: derivation in the docstring or the documentation first, then the code; new reference values are
  checked against the literature.
- `make test` (fast), `make test-all`, `make docs`; `HYPOTHESIS_PROFILE=thorough make test-all` before a release.
