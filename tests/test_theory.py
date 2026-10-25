"""ChiralTheory: validation, pairing, anomalies, meson masses, global symmetries."""
import numpy as np
import pytest

from builders import THEORIES, charge_map, ring_model
from latsimpkg.conventions import vector_axial
from latsimpkg.theory import ChiralTheory, axial_schwinger, model_3450, schwinger


# ----------------------------------------------------------------------
# input and validation
# ----------------------------------------------------------------------
def test_one_dimensional_input_means_one_u1():
    t = ChiralTheory([3, 4], [5, 0], g=0.3)
    assert t.k == 1 and t.n_L == 2 and t.n_R == 2
    assert t.QL.shape == (1, 2) and t.g.shape == (1,)


def test_scalar_coupling_is_broadcast():
    t = ChiralTheory(np.eye(2), np.eye(2), g=0.5)
    assert np.array_equal(t.g, [0.5, 0.5])


def test_charges_are_immutable():
    t = model_3450()
    with pytest.raises(ValueError):
        t.QL[0, 0] = 7


@pytest.mark.parametrize("kwargs", [
    {"QL": [[1, 0], [0, 1]], "QR": [1, 0]},                 # different number of gauge rows
    {"QL": np.eye(2), "QR": np.eye(2), "g": [1, 2, 3]},      # wrong number of couplings
    {"QL": [1], "QR": [1], "g": -1.0},                       # negative coupling
    {"QL": [0.5], "QR": [0.5]},                              # fractional charges
    {"QL": [1], "QR": [1], "names": (["a", "b"], ["c"])},    # wrong number of names
])
def test_invalid_input(kwargs):
    with pytest.raises(ValueError):
        ChiralTheory(**kwargs)


def test_fractional_charges_with_flag():
    assert ChiralTheory([0.5], [0.5], allow_fractional=True).is_anomaly_free


# ----------------------------------------------------------------------
# anomalies
# ----------------------------------------------------------------------
@pytest.mark.parametrize("factory", [schwinger, axial_schwinger, model_3450])
def test_known_theories_are_anomaly_free(factory):
    t = factory()
    assert t.is_anomaly_free
    assert np.all(t.anomaly_matrix == 0)


def test_anomalous_theory_is_rejected():
    with pytest.raises(ValueError, match="anomaly"):
        ChiralTheory([2, 1], [0, 1])


def test_anomaly_matrix():
    t = ChiralTheory([2, 1], [0, 1], allow_anomalous=True)
    assert np.array_equal(t.anomaly_matrix, [[4]])


def test_anomaly_in_vector_axial_form():
    """A_ab = 2 (Q^V_a . Q^A_b + Q^A_a . Q^V_b)."""
    t = ChiralTheory([[2, 1], [1, 3]], [[0, 1], [2, 2]], allow_anomalous=True)
    p = t.pairing()
    assert np.allclose(t.anomaly_matrix, 2 * (p.QV @ p.QA.T + p.QA @ p.QV.T))


def test_vector_like_u1_squared_is_anomaly_free():
    t = ChiralTheory([[1, 1], [1, -1]], [[1, 1], [1, -1]])
    assert t.k == 2 and t.is_anomaly_free


def test_mixed_anomaly_between_two_u1s_is_detected():
    """Each U(1) alone is fine (diagonal zero), the off-diagonal mixed anomaly is not."""
    t = ChiralTheory([[1, 0], [1, 0]], [[1, 0], [0, 1]], allow_anomalous=True)
    A = t.anomaly_matrix
    assert A[0, 0] == 0 and A[1, 1] == 0 and A[0, 1] != 0
    with pytest.raises(ValueError):
        ChiralTheory([[1, 0], [1, 0]], [[1, 0], [0, 1]])


# ----------------------------------------------------------------------
# pairing
# ----------------------------------------------------------------------
def test_pairing_3450():
    p = model_3450().pairing()
    assert np.array_equal(p.QV, [[4, 2]]) and np.array_equal(p.QA, [[-1, 2]])
    assert p.pairs == ((0, 0), (1, 1)) and p.spectators == ()


def test_other_pairing_3450():
    p = model_3450().pairing(order=[1, 0])
    assert np.allclose(p.QV, [[1.5, 4.5]]) and np.allclose(p.QA, [[1.5, -0.5]])


def test_pairing_rejects_bad_order():
    with pytest.raises(ValueError):
        model_3450().pairing(order=[0, 0])


def test_spectator_completes_the_pairing():
    """(3, 4 | 5) is anomaly free; pairing adds a neutral right mover, which is exactly psi_0R of 3-4-5-0."""
    t = ChiralTheory([3, 4], [5])
    p = t.pairing()
    assert p.spectators == (1,) and p.pairs == ((0, 0), (1, None))
    q = model_3450().pairing()
    assert np.array_equal(p.QV, q.QV) and np.array_equal(p.QA, q.QA)


def test_left_spectator():
    p = ChiralTheory([5], [3, 4]).pairing()
    assert p.F == 2 and p.pairs == ((0, 0), (None, 1)) and p.spectators == (1,)


# ----------------------------------------------------------------------
# meson masses
# ----------------------------------------------------------------------
@pytest.mark.parametrize("factory, sumQ2", [(schwinger, 1), (axial_schwinger, 1), (model_3450, 25)])
def test_meson_mass(factory, sumQ2):
    g = 0.3
    assert factory(g).meson_masses == pytest.approx([g * np.sqrt(sumQ2 / np.pi)], rel=1e-14)


def test_meson_mass_matrix_in_vector_axial_form():
    t = ChiralTheory([[2, 1], [1, 3]], [[0, 1], [2, 2]], g=[0.3, 0.7], allow_anomalous=True)
    p = t.pairing()
    expected = np.outer(t.g, t.g) * (p.QV @ p.QV.T + p.QA @ p.QA.T) / np.pi
    assert np.allclose(t.meson_mass_matrix, expected)


def test_decoupled_schwinger_copies():
    t = ChiralTheory(np.eye(2), np.eye(2), g=0.4)
    assert np.allclose(t.meson_mass_matrix, np.diag([0.4**2 / np.pi] * 2))
    assert t.meson_masses == pytest.approx([0.4 / np.sqrt(np.pi)] * 2)


# ----------------------------------------------------------------------
# global symmetries
# ----------------------------------------------------------------------
@pytest.mark.parametrize("factory, count", [(schwinger, 1), (axial_schwinger, 1), (model_3450, 3)])
def test_number_of_global_symmetries(factory, count):
    assert factory().global_symmetries().shape[1] == count


def test_gauge_charge_is_a_global_symmetry():
    t = model_3450()
    T = t.global_symmetries()
    gauge = np.concatenate([t.QL[0], t.QR[0]])
    assert np.linalg.norm(gauge - T @ T.T @ gauge) < 1e-12


def test_free_theory_keeps_all_symmetries():
    assert ChiralTheory([0, 0], [0, 0]).global_symmetries().shape[1] == 4


@pytest.mark.parametrize("name", THEORIES)
def test_global_symmetries_match_ring_conserved_charges(name):
    """
    Analytic t = (t_L, t_R) vs. numerical conserved charges on the ring. A charge t_L.N_L + t_R.N_R
    is t_n.n + t_m.m with (t_n, t_m) = ((t_L + t_R)/2, (t_L - t_R)/2): both must span the same space.
    """
    th = THEORIES[name]
    t = ChiralTheory(th["QL"], th["QR"])
    F = t.n_L
    tn, tm = vector_axial(*np.split(t.global_symmetries(), [F], axis=0))
    analytic = np.vstack([tn, tm])                                   # (2F, d) in (n, m) coordinates
    numeric = charge_map(ring_model(**th, N=30, g=0.3)).T            # (2F, d)
    P = numeric @ np.linalg.pinv(numeric)                            # projector onto the numerical span
    assert analytic.shape[1] == numeric.shape[1]
    assert np.abs(P @ analytic - analytic).max() < 1e-10
