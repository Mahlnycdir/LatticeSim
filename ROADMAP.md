# LatticeSim – Roadmap to a robust base

Goal: a library in which new (1+1)-dimensional abelian gauge theories with chiral fermions can be
tested with **a single theory call**. Everything else (bosonization, lattice, Hamiltonian, analysis)
is generic and protected by tests.

## Decisions

| Topic | Decision |
|---|---|
| Input | chiral charges Q_L, Q_R (as in the literature); pairing into Dirac bosons is automatic |
| Gauge group | U(1)^k from the start (charges are k x n matrices) |
| API | may break; the tests from phase 0 protect the physics |
| Scope | Gaussian core, but theory and lattice are separated so that a later non-Gaussian backend (DMRG, truncation) can use the same theory |

## Target architecture

```
latsimpkg/
  conventions.py    normalizations (sqrt(pi)), L/R <-> V/A, signs of momentum and spin
  theory.py         ChiralTheory
  lattice.py        Lattice(N, a, boundary)
  phasespace.py     PhaseSpace
  quadratic.py      QuadraticHamiltonian
  models/gauge.py   build_gauge_hamiltonian(theory, lattice, ...) -> QuadraticHamiltonian
  analysis/
    spectrum.py     normal modes, predictions, mode counting
    sectors.py      charge sectors, scan, classification
    charge_lattice.py  analytic cross-check
    correlators.py  E correlator, fits
tests/              pytest
studies/            analysis scripts (today's main.py)
pyproject.toml
```

Layering rule: every layer imports only from the layers below.
`conventions` < `theory`, `lattice`, `phasespace` < `quadratic` < `models` < `analysis` < `studies`.

---

## Phase 0 – Safety net  ✅ done

**Goal:** the current state is pinned down by automated tests before anything is refactored.

- `pyproject.toml` with dependencies `numpy`, `scipy`, dev: `pytest`.
- Virtual environment `.venv` (Python 3.13), listed in `.gitignore`.
- `latsimpkg/__init__.py`.
- `tests/builders.py`: **the only place where tests construct models.** During the refactor only these
  adapters are rewritten; the tests (the physics) stay unchanged. Theories are already given in chiral
  form (`THEORIES`), so the tests survive phase 2 as well.
- `tests/physics.py`: analytic reference formulas, independent of the library.
- Fast tests (small N) by default, expensive ones marked `@pytest.mark.slow` (`pytest -m slow`).

Status: 60 fast + 3 slow tests, all green.

| File | Content |
|---|---|
| `test_linear_algebra.py` | oscillator, Dirichlet chain, pure ground state, zero-point energy |
| `test_open_gauge.py` | meson mass, Schwinger spectrum, Gauss law, anomalous commutator, finite size, continuum limit, correlator fit (slow) |
| `test_ring.py` | free compact boson, conserved charges, 3-4-5-0 sectors and bound states, volume scaling, screened charged sector |
| `test_invariances.py` | pairing, flavor permutation, charge conjugation |

### Reference values (all verified)

| Test | Expectation | Tolerance |
|---|---|---|
| Single oscillator, M = diag(w², 1) | <q²> = 1/(2w), <p²> = w/2 | 1e-12 |
| Dirichlet chain | ω_n = (2/a) sin(πn/(2(N+1))) | relative 1e-12 |
| Uncertainty of the ground state | min eig(σ + iJ/2) = 0 (pure state) | 1e-10 |
| Zero-point energy | E_0 = Σ ω/2 | relative 1e-12 |
| Vector Schwinger, open chain | spectrum = sqrt(μ² + k²), μ = g/sqrt(π) | 1e-12 |
| Anomalous commutator | max\|[E,E]\| = (g²/π)\|Σ Q_L² - Σ Q_R²\|/4 | relative 1e-12 |
| Continuum, fixed L | deviation from sqrt(μ² + (π/L)²) falls like a² | ratio 4 ± 10% |
| Meson mass from E correlator | relative deviation < 2e-3 | |
| Ring, free boson | E L/2π = (N_L² + N_R²)/2, s = (N_L² - N_R²)/2 | 1e-10 |
| Conserved charges | number = 2F - k; all anomaly free; gauge charge among them | 1e-10 |
| 3-4-5-0 sectors (Δ <= 2, \|N\| <= 3) | 12 sectors, numerics = analytics, 2 bound states | 1e-8 |
| Charged sector (regression) | 3-4-5-0, g = 0.3, q = 5: E = 0.612125, independent of N | 1e-6 |

Invariance tests (cover real classes of bugs):
- **Pairing:** 3-4-5-0 with pairing (3,5)(4,0) and (3,0)(4,5) gives the same sector spectrum.
- **Flavor permutation:** permuting the fermions changes nothing.
- **Charge conjugation:** Q -> -Q leaves the spectrum invariant.

---

## Phase 1 – Core: `PhaseSpace` and `QuadraticHamiltonian`  ✅ done

**Goal:** all linear algebra for H = ½ XᵀMX in one place, robust and model independent.

Status: `latsimpkg/phasespace.py`, `latsimpkg/quadratic.py`, 34 new tests in `tests/test_quadratic.py`
(random symplectic matrices with known oscillators and free particles). `base.py` only holds thin
compatibility wrappers around `quadratic()` and is removed in phase 3; `ring.py` uses `PhaseSpace` and the
core's conserved charges and free-particle energies. `main.py` output unchanged up to round-off (~1e-14).

`PhaseSpace`
- named coordinate blocks, e.g. `phi[i]` (N sites), `theta_bar[i]` (1); every block automatically
  gets its conjugate momentum block.
- `ps.index("phi", i, j)`, `ps.slice("P", i)`, `ps.dim`, `ps.J`.
- replaces all hand-written index helpers (`_phi`, `_W`, ...).

`QuadraticHamiltonian(M, phasespace, tol=Tolerances())`
- checks on construction: M symmetric, positive semidefinite, dimension matches.
- `williamson()`: symplectic decomposition **including zero modes**. Phase space = oscillators
  (ω > 0, symplectic matrix S) ⊕ free particles (kernel of M and its partners J·kernel).
  Requirement: the kernel is isotropic; otherwise a clear error instead of wrong numbers.
- derived from it: `normal_modes()`, `ground_state_energy()`, `covariance()` (also with zero modes: the free
  directions are treated as sharp momentum states), `conserved_charges()` (kernel of M J),
  `constrained_minimum(A, c)`.
- expensive decompositions are computed once and cached (M is immutable).
- `Tolerances`: one place for all thresholds (rank, pseudo-inverse, grouping).

**Done when:** the tests of the general linear algebra (oscillator, chain, uncertainty, free ring)
run through `QuadraticHamiltonian` and `base.py` is no longer needed.

---

## Phase 2 – Theory and lattice  ✅ done

Status: `conventions.py`, `lattice.py`, `theory.py` (`ChiralTheory`, `Pairing`, `schwinger`, `axial_schwinger`,
`model_3450`) with 56 new tests in `test_conventions_lattice.py` and `test_theory.py`. The test adapters build every
model through `ChiralTheory` (anomaly check included; anomalous tests pass `allow_anomalous=True`). Cross-check:
the analytic `global_symmetries()` span exactly the conserved charges found numerically on the ring.
`Lattice` is not yet used by the models; it is wired in by the model builder of phase 3.

`ChiralTheory(QL, QR, g, names=None)`
- `QL`: k x n_L, `QR`: k x n_R (one row per U(1)), `g`: k couplings.
- Validation:
  - gauge anomaly: QL QLᵀ - QR QRᵀ = 0 (k x k, all pairs a, b). Default: error; allowed with
    `allow_anomalous=True` (needed for the anomalous-commutator tests).
  - integer charges (or deliberately fractional ones with a flag).
- `pairing()`: automatic pairing L_i <-> R_i into Dirac bosons, Q^V = (Q_L + Q_R)/2, Q^A = (Q_L - Q_R)/2.
  If n_L != n_R, neutral spectator fermions are added and flagged.
- analytic: `global_symmetries()` (anomaly-free U(1)s), meson mass matrix
  μ²_ab = (g_a g_b / π)(Q^V_a·Q^V_b + Q^A_a·Q^A_b).
- library of known theories: `schwinger()`, `axial_schwinger()`, `model_3450()`.

`Lattice(N, a, boundary="open" | "ring")`
- validation (N integer including numpy types, N >= 2, a > 0), `L = N a`.

`conventions.py`
- sqrt(π) normalization, N_L = (n + m)/2, N_R = (n - m)/2, n ≡ m (mod 2), signs of P and s,
  each with its derivation in the docstring.

**Done when:** 3-4-5-0 is entered as `ChiralTheory([[3, 4]], [[5, 0]], g=[g])` and the
pairing-invariance tests are green.

---

## Phase 3 – Model building  ✅ done

Status: `latsimpkg/models/gauge.py` with `build_gauge_model(theory, lattice, theta, external_charges, pairing_order)
-> GaugeModel` (instead of returning a bare `QuadraticHamiltonian`: the model also carries the phase space, the
electric-field and link matrices, the pairing and the background, which the analysis needs; `model.H` is the
`QuadraticHamiltonian`). U(1)^k, open chain and ring in one builder; bit-identical M to the old classes.
The analysis code moved as is (generalized to U(1)^k) to `analysis/sectors.py`, `analysis/charge_lattice.py`,
`analysis/correlators.py`; `base.py`, `scalar.py`, `ring.py` and the old `charge_lattice.py` are deleted.
27 new tests in `tests/test_models.py`; `main.py` ported, output unchanged.

Findings:
- **Pairing and commuting electric fields (k > 1):** on the lattice
  [E_a,j, E_b,k] = i (g_a g_b/π)(x_ab 1(j<=k) - x_ba 1(k<=j)) with x_ab = Q^V_a·Q^A_b. Anomaly freedom only gives
  x_ab + x_ba = 0, so a constant commutator i x_ab can remain between different U(1)s. x_ab = (Q_R,a·Q_L,b - Q_L,a·Q_R,b)/4
  depends on the pairing (regularization artefact) and vanishes iff Q_L Q_Rᵀ is symmetric. `choose_pairing` searches
  for such a pairing and raises if none exists (none found among small charges; for k = 1 it always exists).
- **Probe charges:** Schwinger screening V(d) = (g²q²/2μ)(1 - e^{-μd}) reproduced to 5e-4 (also for q = 1/2);
  without charged matter V(d) = g²q²d/2 exactly (confinement).
- **θ angle:** absorbed by the zero modes on the ring (energy 0 to 1e-15, massless fermions); on the open chain a small
  boundary energy remains because the Dirichlet ends fix the zero mode.

`build_gauge_hamiltonian(theory, lattice, theta=None, external_charges=None) -> QuadraticHamiltonian`
- one construction kit instead of two classes: gradient, kinetic term, E field per U(1), boundary condition from `lattice`.
- open chain: Dirichlet; ring: zero modes θ̄_i, W_i (as in today's `ring.py`).
- `electric_field_operator(a)` for every U(1), `gauge_charges(n, m)`.
- extension points already in the signature (optional/empty at first): θ angle (constant in E),
  external probe charges (source term in the Gauss law).

**Done when:** all phase-0 tests run through the new construction kit and `scalar.py`/`ring.py` are deleted.

---

## Phase 4 – Analysis layer  ✅ done

Status: `analysis/results.py` (`Sector`, `SectorScan`, `SectorEnergy`, `Spectrum`, `Series`, `FitWindow`, `FitResult`,
`Potential`, all saveable as .npz with JSON meta), `analysis/spectrum.py`, `analysis/potentials.py`, explicit fit
windows in `analysis/correlators.py`, `latsimpkg/api.py` (`find_bound_states`, `spectrum`, `screening_potential`,
exported from `latsimpkg`). `studies/` (free_field, spectra, correlators, sectors, potentials; `python -m studies`)
replaces `main.py` and writes all results to `results/` (git-ignored). 32 new tests in `tests/test_api_results.py`
(3 of them slow). Sectors now name their fermion content from the theory, e.g. "psi_3L^2 psi_4L psi_5R^-2".

- results as `dataclass`es (`Spectrum`, `Sector`, `SectorScan`, `FitResult`, `Potential`) instead of dicts;
  no `print`s in the library; every result can be saved and loaded (`.npz`/JSON) without recomputation.
- numerical scan and charge-lattice analysis return the same `Sector` type -> `SectorScan` holds both plus the comparison.
- `correlators.meson_mass_from_E` with explicit fit windows instead of hidden heuristics.
- classification (elementary / bound / multi-particle) with its documented assumption.
- **first user-facing API:** `find_bound_states(theory, N, g, ...)` (ring model, sector scan, analytic cross-check,
  classification, report in one call), plus `spectrum(...)` and `screening_potential(...)`.
- `studies/`: scripts that compute, save data and figures; they replace `main.py`.

**Done when:** `main.py` is replaced by `studies/` and a new theory is fully analysed in ~5 lines.

---

## Release 0.1 – first usable version (after phase 4, then merge to `main`)

1. ✅ **Plotting** (`latsimpkg/plotting.py`, matplotlib, headless): plots take saved results only (no recomputation);
   vector PDF for LaTeX plus PNG. First plots: dispersion/spectrum, sector diagram in the (h, hb) plane with bound
   states highlighted, finite-size scaling, E correlator with fit, screening/confinement potential.
   Done: object-oriented matplotlib only (no pyplot, tested), Okabe-Ito colours, residual panels with the scale in the
   label; `studies/figures.py` lists the 9 documentation figures with LaTeX captions, renders incrementally
   (`python -m studies.figures`, `--force`) and writes `figures/manifest.json` for the documentation. 18 tests.
2. ✅ **Documentation** (`tools/build_docs.py`, `make docs`, LaTeX via latexmk), in English:
   - API reference generated from the docstrings (docstrings keep ASCII math);
   - hand-written theory chapters, prerequisite quantum mechanics and linear algebra: harmonic oscillator,
     Gaussian states, Williamson normal form, lattice scalar field, bosonization, Gauss law and Schwinger model,
     anomalies, compact bosons and charge lattice, charge sectors, worked example 3-4-5-0 (bound-state search);
   - figures from `studies/` included automatically.
   Done: `docs/latticesim.tex` + `docs/chapters/` (7 chapters, conventions appendix, references), generated
   `docs/generated/` (API reference, one .tex per figure, unicode guard), output `docs/build/latticesim.pdf`
   (53 pages, builds without LaTeX warnings). `Makefile` with test/test-all/studies/figures/docs/clean.
   11 tests in `tests/test_docs.py`, including: every module is in the API reference, every public object has a
   docstring (29 were added), every figure is shown, every reference is cited, the full build has no warnings (slow).
3. ✅ **Literature cross-check** of all reference values, with citations. Includes verifying the bibliography of
   `docs/chapters/references.tex` (written from memory) and the claims attributed to it.
   Done: all 12 references verified (publisher/arXiv); corrections: Wang-Wen is PRD 99, 111501(R), page ranges added;
   new reference Bharadwaj-Isen-Kang, arXiv:2607.19468. `tests/test_literature.py` (11 tests): Schwinger mass;
   massless-Schwinger static potential = (sqrt(pi)/2) e q^2 (1 - e^{-mu d}) (their eq. 3.88, identical to ours);
   string tension e^2 q^2 / 2 (eq. 3.2); K_0 propagator normalization (numerical integral); 3-4-5-0 anomaly;
   Tong's second symmetry H = (5,0|3,-4) is found among our conserved charges; gauging H reproduces Tong's IR Dirac
   fermion exactly (Psi = psi2, Psit = psi1^2 (psit1^dag)^2 psit2, G = +4 / -4); our 3-4-5-0 is its parity mirror.
   Documentation: comparison paragraph in chapter 7, potential in chapter 5, table in appendix A.
4. ✅ **Stress tests of the base**, above all the mathematics: property-based tests (random symplectic matrices,
   random anomaly-free theories), time evolution vs. normal modes, invariances.
   Done: `hypothesis` (dev dependency, deterministic default profile, `HYPOTHESIS_PROFILE=thorough` for 300 examples
   per property); `tests/test_stress_core.py` (13 theorems of the Gaussian core), `tests/test_stress_models.py`
   (15 properties of random anomaly-free theories), `tests/test_stress_edges.py` (24 edge cases).
   Two real bugs found and fixed:
   - zero-mode detection used a fixed relative threshold 1e-10, so weak couplings (g <~ 1e-5) were taken for extra
     zero modes and gave wrong sectors. Now: zero below 1e-13, physical above 1e-11, ambiguous -> error; the model
     builder passes the exact number of zero modes (`n_zero`, from the symmetries); `QuadraticHamiltonian.precision`
     (condition * eps) guards the sector search, which raises with mu a / mu L instead of returning wrong sectors.
   - on the ring the dual zero mode was the value of vartheta at site 0 instead of its average; this broke
     translation invariance and gave wrong ring spectra for axial charges (sectors were unaffected). Now both zero
     modes are averages, [varphi_j, vartheta_k] -> (i/2) sign(k - j), and the phi <-> vartheta duality is exact
     (axial = vector Schwinger spectrum on the ring to 1e-15; every meson mass lies exactly in the ring spectrum).
   Documentation updated (numerical limits, ring zero modes, test overview); 294 fast + 8 slow tests green.
5. **Roadmap clean-up** (finished phases condensed), merge to `main`.

---

## Phase 5 – Extensions (after the base)

In this order, each with its own tests:
1. 't Hooft anomaly matching of the surviving U(1)s (algebraic).
2. Vertex correlators: condensates (Schwinger: <ψ̄ψ> = -e^γ μ/(2π); 3-4-5-0: 't Hooft vertex), propagator of the bound states.
3. Central charge from the Casimir energy and the entanglement entropy.
4. θ angle and probe charges (screening vs. confinement).
5. Interface for non-Gaussian backends (mass terms, cos interactions).

---

## Workflow

- One branch per phase; you implement, I help and review; you commit.
- Every phase ends green: tests first, then refactor, then delete the old parts.
- For physics questions: derivation in the docstring first, then the code.
