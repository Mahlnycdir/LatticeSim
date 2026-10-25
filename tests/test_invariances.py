"""Physical results must not depend on bookkeeping choices."""
import numpy as np
import pytest

from builders import THEORIES, conserved_charges, normal_modes, open_gauge_model, ring_model, sector_scan
from physics import sector_signature

N, G = 40, 0.3
QL, QR = THEORIES["3450"]["QL"], THEORIES["3450"]["QR"]


def test_pairing_invariance_sectors():
    """Pairing (3,5)(4,0) vs (3,0)(4,5) is pure bookkeeping: same sectors."""
    a = sector_scan(ring_model(QL, QR, N=N, g=G))
    b = sector_scan(ring_model(QL, QR[::-1], N=N, g=G))
    assert sector_signature(a) == sector_signature(b)


def test_pairing_invariance_conserved_charges():
    assert conserved_charges(ring_model(QL, QR[::-1], N=N, g=G)).shape[1] == 3


def test_flavor_permutation_open_spectrum():
    a = normal_modes(open_gauge_model(QL, QR, N=N, g=G))
    b = normal_modes(open_gauge_model(QL[::-1], QR[::-1], N=N, g=G))
    assert np.max(np.abs(a - b)) < 1e-12


def test_flavor_permutation_sectors():
    a = sector_scan(ring_model(QL, QR, N=N, g=G))
    b = sector_scan(ring_model(QL[::-1], QR[::-1], N=N, g=G))
    assert sector_signature(a) == sector_signature(b)


@pytest.mark.parametrize("name", THEORIES)
def test_charge_conjugation_open_spectrum(name):
    th = THEORIES[name]
    a = normal_modes(open_gauge_model(**th, N=N, g=G))
    b = normal_modes(open_gauge_model([-q for q in th["QL"]], [-q for q in th["QR"]], N=N, g=G))
    assert np.max(np.abs(a - b)) < 1e-12


def test_charge_conjugation_sectors():
    a = sector_scan(ring_model(QL, QR, N=N, g=G))
    b = sector_scan(ring_model([-q for q in QL], [-q for q in QR], N=N, g=G))
    assert sector_signature(a) == sector_signature(b)
