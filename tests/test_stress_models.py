"""
Property-based stress tests of theories, lattice models and the bound-state search.

Random anomaly-free theories are assembled from blocks that are anomaly free on their own (vector-like and
axial fermions, chiral "Pythagorean" blocks such as 3-4-5-0, neutral fermions), with random charge multiples,
couplings and independent orderings of the left and right movers (so the pairing changes as well).
"""
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from latsimpkg import ChiralTheory, Lattice, build_gauge_model, find_bound_states, schwinger, axial_schwinger
from latsimpkg.analysis.results import sector_signature
from latsimpkg.analysis.sectors import charge_map
from latsimpkg.conventions import vector_axial
from latsimpkg.models.gauge import expected_zero_modes

# blocks (left charges, right charges), each anomaly free: sum QL^2 = sum QR^2
BLOCKS = [((1,), (1,)), ((1,), (-1,)), ((2,), (2,)), ((0,), (0,)),
          ((3, 4), (5, 0)), ((5, 5), (1, 7)), ((2, 9), (6, 7)), ((6, 8), (10, 0))]


@st.composite
def theories(draw, max_flavors=2):
    """A random anomaly-free single-U(1) theory with at most `max_flavors` Dirac pairs and some charged fermion."""
    QL, QR = [], []
    while len(QL) < max_flavors:
        bl, br = draw(st.sampled_from(BLOCKS))
        if len(QL) + len(bl) > max_flavors:
            break
        s = draw(st.sampled_from([1, -1, 2]))
        QL += [s * q for q in bl]
        QR += [s * q for q in br]
        if not draw(st.booleans()):
            break
    assume(any(QL) or any(QR))
    QL = list(draw(st.permutations(QL)))
    QR = list(draw(st.permutations(QR)))
    g = 10.0 ** draw(st.floats(-1.3, 0.3))                      # 0.05 ... 2
    return ChiralTheory(QL, QR, g)


def ring(theory, N=24):
    return build_gauge_model(theory, Lattice(N, 1.0, "ring"))


# ----------------------------------------------------------------------
# theory layer
# ----------------------------------------------------------------------
@given(theories(max_flavors=3))
def test_random_theories_are_anomaly_free_and_count_symmetries(t):
    assert t.is_anomaly_free
    rank = np.linalg.matrix_rank(np.hstack([t.QL, -t.QR]))
    assert t.global_symmetries().shape[1] == t.n_L + t.n_R - rank


@given(theories())
def test_meson_mass_formula(t):
    p = t.pairing()
    assert t.meson_masses[0] ** 2 == pytest.approx(t.g[0] ** 2 / np.pi * np.sum(p.QV**2 + p.QA**2), rel=1e-12)


# ----------------------------------------------------------------------
# lattice models
# ----------------------------------------------------------------------
@given(theories(), st.sampled_from(["open", "ring"]))
def test_number_of_zero_modes_is_exact(t, boundary):
    m = build_gauge_model(t, Lattice(20, 1.0, boundary))
    assert m.H.n_free == expected_zero_modes(m.pairing, t.g, m.lattice)
    assert m.H.n_free == (0 if boundary == "open" else t.global_symmetries().shape[1])


@given(theories(), st.sampled_from(["open", "ring"]))
def test_electric_field_commutes(t, boundary):
    m = build_gauge_model(t, Lattice(20, 1.0, boundary))
    assert np.abs(m.electric_field_commutator()).max() < 1e-12 * max(1.0, t.g[0] ** 2 * np.abs(t.QL).max() ** 2)


@given(theories())
def test_meson_mass_is_in_the_ring_spectrum(t):
    """The zero-momentum mode of the massive combination has frequency exactly mu (translation invariance)."""
    om = ring(t).H.frequencies
    for mu in t.meson_masses[t.meson_masses > 0]:
        assert np.min(np.abs(om - mu)) < 1e-10 * max(1.0, mu)


@given(theories())
def test_vector_like_open_chain_is_exact(t):
    """Vector-like theories (Q_L = Q_R) on the open chain: branches sqrt(mu^2 + k^2) and massless ones, exactly."""
    vl = ChiralTheory(t.QL, t.QL, t.g)
    m = build_gauge_model(vl, Lattice(30, 1.0, "open"))
    k = 2 * np.sin(np.pi * np.arange(1, 31) / 62)
    masses = vl.meson_masses[vl.meson_masses > 0]
    pred = np.sort(np.concatenate([np.sqrt(mu**2 + k**2) for mu in masses] + [k] * (m.F - len(masses))))
    assert np.max(np.abs(m.H.frequencies - pred)) < 1e-11


@pytest.mark.parametrize("N", [20, 41, 60])
def test_axial_and_vector_schwinger_are_dual_on_the_ring(N):
    """Regression: phi <-> vartheta duality, both have exactly sqrt(mu^2 + k^2) on the ring (zero modes as averages)."""
    g = 0.3
    mu = g / np.sqrt(np.pi)
    pred = np.sort(np.sqrt(mu**2 + (2 * np.sin(np.pi * np.arange(N) / N)) ** 2))
    for theory in (schwinger(g), axial_schwinger(g)):
        assert np.max(np.abs(ring(theory, N).H.frequencies - pred)) < 1e-12


# ----------------------------------------------------------------------
# conserved charges and the bound-state search
# ----------------------------------------------------------------------
@given(theories())
def test_numerical_conserved_charges_are_the_anomaly_free_symmetries(t):
    """
    The kernel directions of M J on the ring span exactly the U(1)s without mixed anomaly, written in fermion
    numbers of the pairing: t_L.N_L + t_R.N_R = t_n.n + t_m.m with (t_n, t_m) = ((t_L + t_R)/2, (t_L - t_R)/2).
    """
    p = t.pairing()
    paired = ChiralTheory(p.QV + p.QA, p.QV - p.QA, t.g)          # chiral charges including spectators
    tL, tR = np.split(paired.global_symmetries(), [p.F], axis=0)
    analytic = np.vstack(vector_axial(tL, tR))
    numeric = charge_map(ring(t)).T
    P = numeric @ np.linalg.pinv(numeric)
    assert analytic.shape[1] == numeric.shape[1]
    assert np.abs(P @ analytic - analytic).max() < 1e-9


@given(theories())
def test_lattice_agrees_with_charge_lattice(t):
    assert find_bound_states(t, N=20, nmax=2, Delta_max=1.5).agrees


@given(theories(), st.integers(0, 3))
def test_sectors_do_not_depend_on_g_or_N(t, choice):
    reference = sector_signature(find_bound_states(t, N=20, nmax=2, Delta_max=1.5).numeric)
    g, N = [(0.1, 20), (1.0, 20), (0.3, 33), (0.5, 48)][choice]
    other = ChiralTheory(t.QL, t.QR, g)
    assert sector_signature(find_bound_states(other, N=N, nmax=2, Delta_max=1.5).numeric) == reference


@given(theories(), st.randoms(use_true_random=False))
def test_sectors_do_not_depend_on_pairing_or_order(t, rnd):
    reference = sector_signature(find_bound_states(t, N=20, nmax=2, Delta_max=1.5).numeric)
    QL, QR = list(t.QL[0]), list(t.QR[0])
    rnd.shuffle(QL)
    rnd.shuffle(QR)
    shuffled = ChiralTheory(QL, QR, t.g)
    assert sector_signature(find_bound_states(shuffled, N=20, nmax=2, Delta_max=1.5).numeric) == reference


@given(theories())
def test_sectors_are_invariant_under_charge_conjugation(t):
    a = find_bound_states(t, N=20, nmax=2, Delta_max=1.5).numeric
    b = find_bound_states(ChiralTheory(-t.QL, -t.QR, t.g), N=20, nmax=2, Delta_max=1.5).numeric
    assert sector_signature(a) == sector_signature(b)


# ----------------------------------------------------------------------
# extreme couplings: correct or a clear error, never silently wrong
# ----------------------------------------------------------------------
@given(theories(), st.floats(-8, 4), st.integers(16, 48))
def test_extreme_couplings_never_give_wrong_sectors(t, log_g, N):
    reference = sector_signature(find_bound_states(t, N=20, nmax=2, Delta_max=1.5).numeric)
    theory = ChiralTheory(t.QL, t.QR, 10.0**log_g)
    try:
        scan = find_bound_states(theory, N=N, nmax=2, Delta_max=1.5)
    except ValueError as err:
        assert "double precision" in str(err) or "error of" in str(err)
        return
    assert scan.agrees
    assert sector_signature(scan.numeric) == reference
