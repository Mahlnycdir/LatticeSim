"""Conventions and lattice geometry."""
import numpy as np
import pytest

from latsimpkg.conventions import chiral, chiral_numbers, vector_axial, winding_momentum
from latsimpkg.lattice import Lattice


def test_vector_axial_3450():
    QV, QA = vector_axial([3, 4], [5, 0])
    assert np.array_equal(QV, [4, 2]) and np.array_equal(QA, [-1, 2])


@pytest.mark.parametrize("QL, QR", [([3, 4], [5, 0]), ([1], [-1]), ([[1, 2], [0, 3]], [[2, 1], [3, 0]])])
def test_chiral_inverts_vector_axial(QL, QR):
    QL2, QR2 = chiral(*vector_axial(QL, QR))
    assert np.array_equal(QL2, QL) and np.array_equal(QR2, QR)


def test_vector_axial_rejects_unpaired():
    with pytest.raises(ValueError):
        vector_axial([3, 4], [5])


@pytest.mark.parametrize("NL, NR", [(1, 0), (0, 1), (2, -1), ((2, 1), (-2, 0))])
def test_fermion_numbers_round_trip(NL, NR):
    NL2, NR2 = chiral_numbers(*winding_momentum(NL, NR))
    assert np.array_equal(NL2, NL) and np.array_equal(NR2, NR)


@pytest.mark.parametrize("n, m", [(1, 0), (0, 1), ((1, 1), (1, 0))])
def test_parity_rule(n, m):
    """n = m (mod 2), otherwise the state is not fermionic."""
    with pytest.raises(ValueError):
        chiral_numbers(n, m)


def test_lattice_lengths():
    assert Lattice(10, 0.5, "open").L == pytest.approx(5.5)
    assert Lattice(10, 0.5, "ring").L == pytest.approx(5.0)


def test_lattice_accepts_numpy_integers():
    assert Lattice(np.int64(12)).N == 12


@pytest.mark.parametrize("kwargs, error", [({"N": 1}, ValueError), ({"N": 2.0}, TypeError), ({"N": True}, TypeError),
                                           ({"N": 4, "a": 0.0}, ValueError), ({"N": 4, "a": np.inf}, ValueError),
                                           ({"N": 4, "boundary": "periodic"}, ValueError)])
def test_lattice_validation(kwargs, error):
    with pytest.raises(error):
        Lattice(**kwargs)


def test_lattice_is_immutable():
    lat = Lattice(10)
    with pytest.raises(AttributeError):
        lat.N = 20
