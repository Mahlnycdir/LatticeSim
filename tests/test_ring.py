"""Ring with zero modes: conserved charges and charge sectors (bound states)."""
import numpy as np
import pytest

from builders import (THEORIES, analytic_sectors, charge_map, conserved_charges, free_ring, ring_model,
                      sector_energy, sector_scan, vector_axial)
from physics import sector_signature

N, G = 40, 0.3


@pytest.mark.parametrize("NL, NR", [(1, 0), (0, 1), (1, 1), (2, 0), (2, 1), (-1, 3)])
def test_free_boson_sector_energy_and_spin(NL, NR):
    """Free Dirac fermion: E L/2pi = (N_L^2 + N_R^2)/2, s = (N_L^2 - N_R^2)/2."""
    d = sector_energy(free_ring(20), (NL,), (NR,))
    assert d["Delta"] == pytest.approx((NL**2 + NR**2) / 2, abs=1e-10)
    assert d["s"] == pytest.approx((NL**2 - NR**2) / 2, abs=1e-10)


@pytest.mark.parametrize("F", [1, 2])
def test_free_ring_conserves_all_numbers(F):
    assert conserved_charges(free_ring(20, F)).shape[1] == 2 * F


@pytest.mark.parametrize("name", THEORIES)
def test_number_of_conserved_charges(name):
    """2F - 1: one combination is lost to the mixed anomaly with the gauge U(1)."""
    th = THEORIES[name]
    assert conserved_charges(ring_model(**th, N=N, g=G)).shape[1] == 2 * len(th["QL"]) - 1


@pytest.mark.parametrize("name", THEORIES)
def test_conserved_charges_are_anomaly_free(name):
    """Rows (t_n, t_m) satisfy Q^A.t_n + Q^V.t_m = 0 (no mixed anomaly with the gauge field)."""
    th = THEORIES[name]
    QV, QA = vector_axial(**th)
    T = charge_map(ring_model(**th, N=N, g=G))
    assert np.abs(T @ np.concatenate([QA, QV])).max() / np.abs(T).max() < 1e-10


@pytest.mark.parametrize("name", THEORIES)
def test_gauge_charge_is_conserved(name):
    th = THEORIES[name]
    QV, QA = vector_axial(**th)
    T = charge_map(ring_model(**th, N=N, g=G))
    gauge = np.concatenate([QV, QA])
    assert np.linalg.norm(gauge - np.linalg.pinv(T) @ T @ gauge) < 1e-10 * np.linalg.norm(gauge)


@pytest.mark.parametrize("name", ["schwinger", "axial_schwinger"])
def test_schwinger_has_no_massless_sectors(name):
    assert sector_scan(ring_model(**THEORIES[name], N=N, g=G)) == []
    assert analytic_sectors(**THEORIES[name]) == []


def test_3450_sectors_match_charge_lattice():
    num = sector_scan(ring_model(**THEORIES["3450"], N=N, g=G))
    ana = analytic_sectors(**THEORIES["3450"])
    assert len(num) == len(ana) == 12
    assert sector_signature(num) == sector_signature(ana)


def test_3450_sector_representatives_match_charge_lattice():
    num = {(d["NL"], d["NR"]): d for d in sector_scan(ring_model(**THEORIES["3450"], N=N, g=G))}
    ana = {(d["NL"], d["NR"]): d for d in analytic_sectors(**THEORIES["3450"])}
    assert num.keys() == ana.keys()
    for key in num:
        assert num[key]["Delta"] == pytest.approx(ana[key]["Delta"], abs=1e-8)
        assert num[key]["s"] == pytest.approx(ana[key]["s"], abs=1e-8)


def test_3450_has_two_bound_states():
    """psi_3L^2 psi_4L psibar_5R^2 and its antiparticle, both (h, hb) = (1/2, 0)."""
    bound = [d for d in sector_scan(ring_model(**THEORIES["3450"], N=N, g=G)) if d["kind"] == "bound state"]
    assert {(d["NL"], d["NR"]) for d in bound} == {((2, 1), (-2, 0)), ((-2, -1), (2, 0))}
    for d in bound:
        assert (d["h"], d["hb"]) == pytest.approx((0.5, 0.0), abs=1e-8)


def test_3450_elementary_neutral_fermion():
    elementary = [d for d in sector_scan(ring_model(**THEORIES["3450"], N=N, g=G)) if d["kind"] == "elementary"]
    assert {(d["NL"], d["NR"]) for d in elementary} == {((0, 0), (0, 1)), ((0, 0), (0, -1))}


@pytest.mark.parametrize("NL, NR, Delta", [((2, 1), (-2, 0), 0.5), ((0, 0), (0, 1), 0.5), ((2, 1), (-2, 1), 1.0)])
def test_massless_sectors_scale_as_one_over_L(NL, NR, Delta):
    for n in [40, 80]:
        d = sector_energy(ring_model(**THEORIES["3450"], N=n, g=G), NL, NR)
        assert d["q"] == 0
        assert d["Delta"] == pytest.approx(Delta, abs=1e-8)


def test_charged_sector_is_screened_regression():
    """q = 5 is forbidden on the ring; the seam charge is screened, so E stays finite (regression value)."""
    for n in [40, 80]:
        d = sector_energy(ring_model(**THEORIES["3450"], N=n, g=G), (0, 0), (1, 0))
        assert d["q"] == 5
        assert d["E"] == pytest.approx(0.612125, abs=1e-6)
