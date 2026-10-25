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

## Phase 1 – Core: `PhaseSpace` and `QuadraticHamiltonian`

**Goal:** all linear algebra for H = ½ XᵀMX in one place, robust and model independent.

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

## Phase 2 – Theory and lattice

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

## Phase 3 – Model building

`build_gauge_hamiltonian(theory, lattice, theta=None, external_charges=None) -> QuadraticHamiltonian`
- one construction kit instead of two classes: gradient, kinetic term, E field per U(1), boundary condition from `lattice`.
- open chain: Dirichlet; ring: zero modes θ̄_i, W_i (as in today's `ring.py`).
- `electric_field_operator(a)` for every U(1), `gauge_charges(n, m)`.
- extension points already in the signature (optional/empty at first): θ angle (constant in E),
  external probe charges (source term in the Gauss law).

**Done when:** all phase-0 tests run through the new construction kit and `scalar.py`/`ring.py` are deleted.

---

## Phase 4 – Analysis layer

- results as `dataclass`es (`Sector`, `SpectrumReport`, `FitResult`) instead of dicts; no `print`s in the library.
- `sectors.scan(H, theory, nmax, Delta_max)` and `charge_lattice.spectrum(theory, ...)` return the same
  types -> direct comparison `compare(numeric, analytic)`.
- `correlators.meson_mass_from_E` with explicit fit windows instead of hidden heuristics.
- classification (elementary / bound / multi-particle) with its documented assumption.
- `studies/report.py` produces today's output of `main.py`.

**Done when:** `main.py` is replaced by `studies/` and a new theory is fully analysed in ~5 lines.

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
