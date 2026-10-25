# Changelog

## 0.1.0 – first usable release

### Library
- `ChiralTheory`: chiral charges for U(1)^k, anomaly matrix (including mixed anomalies), automatic pairing into
  Dirac bosons with neutral spectators, meson mass matrix, anomaly-free global symmetries; `schwinger`,
  `axial_schwinger`, `model_3450`. Fermions are named after their charges by default (`psi_3L`, `psi_5R`, ...).
- `Lattice` (open Dirichlet chain or ring), `PhaseSpace` (named blocks of the phase-space vector).
- `QuadraticHamiltonian`: Williamson normal form including zero modes (free particles), ground-state covariance,
  conserved charges, constrained minima, explicit zero-mode classification and a precision estimate.
- `build_gauge_model`: one builder for open chain and ring, one electric field per U(1), theta angle, static probe
  charges, automatic choice of a pairing with commuting electric fields.
- Analysis: spectra with predictions, finite-size and continuum scaling, electric-field correlator fits with explicit
  windows, static potentials, charge-sector scan and analytic charge lattice; saveable result types.
- User API: `find_bound_states`, `spectrum`, `screening_potential`.
- `plotting`: figures of saved results without pyplot (PDF and PNG).

### Studies, figures, documentation
- `python -m studies` (five studies), `python -m studies.figures` (nine figures, incremental, manifest).
- LaTeX documentation (`make docs`): seven theory chapters from quantum mechanics to the bound-state search,
  conventions, literature cross-check, generated API reference, figures included automatically.

### Validation
- 294 fast and 8 slow tests: analytic reference values, invariances (pairing, permutation, charge conjugation),
  literature values, property-based stress tests (hypothesis) of the mathematical core and of random anomaly-free
  theories, documentation completeness and a warning-free documentation build.
- Literature: Schwinger mass; massless-Schwinger static potential (arXiv:2607.19468, eq. 3.88); string tension;
  Tong's gauged-H analysis of the 3-4-5-0 fermions reproduced exactly (JHEP 07 (2022) 001, sec. 3).

### Fixed during the release (found by the stress tests)
- Weak couplings (g <~ 1e-5) were misclassified as extra zero modes and produced wrong sectors; the builder now
  passes the exact number of zero modes and the sector search refuses to run beyond double precision.
- On the ring the dual zero mode was the value of vartheta at the first site instead of its average, which broke
  translation invariance for axial charges; the phi <-> vartheta duality is now exact on the lattice.
