"""
Result types of the analysis layer.

Every result is an immutable dataclass that carries its data and the parameters it was computed with
(`meta`), and can be saved and loaded without recomputation:
    result.save("spectrum.npz");  Spectrum.load("spectrum.npz")
Arrays are stored as arrays in the .npz file, everything else as JSON under the key "__meta__".
"""
import json
from dataclasses import dataclass, field, fields

import numpy as np


# ----------------------------------------------------------------------
# saving and loading
# ----------------------------------------------------------------------
def to_jsonable(value):
    """Recursively convert numpy scalars/arrays and tuples into plain JSON types."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


class _Saveable:
    """Mixin: save/load a dataclass with numpy arrays and JSON-serializable fields to/from .npz."""

    def _encode(self):
        arrays, meta = {}, {}
        for f in fields(self):
            value = getattr(self, f.name)
            if isinstance(value, np.ndarray):
                arrays[f.name] = value
            elif f.name in self._nested:
                meta[f.name] = [item.to_json() for item in value]
            else:
                meta[f.name] = value
        return arrays, meta

    def save(self, path):
        """Save to `path` (.npz)."""
        arrays, meta = self._encode()
        meta = to_jsonable(meta)
        meta["__type__"] = type(self).__name__
        np.savez(path, __meta__=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path):
        """Load a result saved with `save`."""
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["__meta__"]))
            arrays = {k: data[k] for k in data.files if k != "__meta__"}
        if meta.pop("__type__", cls.__name__) != cls.__name__:
            raise TypeError(f"{path} does not contain a {cls.__name__}.")
        for name, item_cls in cls._nested.items():
            meta[name] = tuple(item_cls.from_json(item) for item in meta[name])
        return cls(**meta, **arrays)

    _nested = {}


# ----------------------------------------------------------------------
# sectors
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Sector:
    """
    A gauge-invariant charge sector, labelled by its lightest representative (N_L, N_R).

    Attributes:
        NL, NR (tuple): Chiral fermion numbers per Dirac boson of the pairing.
        Delta (float): Scaling dimension h + hb = E L / (2 pi).
        s (float): Spin h - hb.
        kind (str): "elementary", "bound state" or "multi-particle" (see `classify_sector`).
        content (str): The sector as product of fermion operators, e.g. "psi_3L^2 psi_4L psibar_5R^2".
    """
    NL: tuple
    NR: tuple
    Delta: float
    s: float
    kind: str
    content: str = ""

    @property
    def h(self):
        """Left-moving conformal weight h = (Delta + s)/2."""
        h = (self.Delta + self.s) / 2
        return 0.0 if abs(h) < 1e-9 else h

    @property
    def hb(self):
        """Right-moving conformal weight hb = (Delta - s)/2."""
        hb = (self.Delta - self.s) / 2
        return 0.0 if abs(hb) < 1e-9 else hb

    @property
    def key(self):
        """(NL, NR): the representative, used to match numerics and analytics."""
        return (self.NL, self.NR)

    def as_dict(self):
        """Plain dict including h and hb."""
        return {"NL": self.NL, "NR": self.NR, "Delta": self.Delta, "s": self.s,
                "h": self.h, "hb": self.hb, "kind": self.kind, "content": self.content}

    def to_json(self):
        """JSON-serializable dict (used by `SectorScan.save`)."""
        return {"NL": list(self.NL), "NR": list(self.NR), "Delta": self.Delta, "s": self.s,
                "kind": self.kind, "content": self.content}

    @classmethod
    def from_json(cls, d):
        """Inverse of `to_json`."""
        return cls(tuple(d["NL"]), tuple(d["NR"]), d["Delta"], d["s"], d["kind"], d.get("content", ""))


def sector_signature(sectors, digits=6):
    """Representation-independent fingerprint of a sector list: sorted (Delta, s, kind)."""
    return sorted((round(x.Delta, digits) + 0.0, round(x.s, digits) + 0.0, x.kind) for x in sectors)


@dataclass(frozen=True)
class SectorEnergy:
    """
    Lowest energy in the sector (N_L, N_R) of a ring model.

    Attributes:
        q (numpy.ndarray): Gauge charges, shape (k,); a physical sector on the ring needs q = 0.
        E (float): Energy above the vacuum.
        Delta (float): E L / (2 pi).
        s (float): Spin from the lattice momentum, -P L / (2 pi).
    """
    q: np.ndarray
    E: float
    Delta: float
    s: float


@dataclass(frozen=True)
class SectorScan(_Saveable):
    """
    Result of a bound-state search: numerical sectors on the ring and the analytic charge-lattice prediction.

    Attributes:
        numeric (tuple[Sector]): Sectors found from the lattice Hamiltonian.
        analytic (tuple[Sector]): Sectors predicted by the charge-lattice analysis.
        meta (dict): Theory, lattice and scan parameters.
    """
    numeric: tuple
    analytic: tuple
    meta: dict = field(default_factory=dict)

    _nested = {"numeric": Sector, "analytic": Sector}

    @property
    def bound_states(self):
        """Numerical sectors classified as bound states."""
        return tuple(x for x in self.numeric if x.kind == "bound state")

    @property
    def elementary(self):
        """Numerical sectors that are single elementary fermions."""
        return tuple(x for x in self.numeric if x.kind == "elementary")

    @property
    def multi_particle(self):
        """Numerical sectors that are multi-particle states."""
        return tuple(x for x in self.numeric if x.kind == "multi-particle")

    @property
    def agrees(self):
        """Numerics and analytics give the same sectors (same Delta, s and kind)."""
        return sector_signature(self.numeric) == sector_signature(self.analytic)

    @property
    def max_deviation(self):
        """Largest |Delta_num - Delta_ana| and |s_num - s_ana| over sectors with the same representative."""
        ana = {x.key: x for x in self.analytic}
        dev = [max(abs(x.Delta - ana[x.key].Delta), abs(x.s - ana[x.key].s)) for x in self.numeric if x.key in ana]
        return max(dev, default=0.0)


# ----------------------------------------------------------------------
# spectra and series
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Spectrum(_Saveable):
    """
    Normal-mode frequencies of a model.

    Attributes:
        frequencies (numpy.ndarray): Sorted oscillator frequencies.
        prediction (numpy.ndarray): Continuum-motivated prediction of the same length (empty if none).
        meta (dict): Theory and lattice parameters, number of free particles.
    """
    frequencies: np.ndarray
    prediction: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def omega_min(self):
        """Lowest frequency."""
        return float(self.frequencies[0])

    @property
    def max_deviation(self):
        """max |omega - prediction| (nan without prediction)."""
        if self.prediction.size != self.frequencies.size:
            return float("nan")
        return float(np.max(np.abs(self.frequencies - self.prediction)))

    def count_below(self, value):
        """Number of frequencies below `value`."""
        return int(np.sum(self.frequencies < value))


@dataclass(frozen=True)
class Series(_Saveable):
    """
    A measured quantity y(x) with an optional prediction, e.g. the lowest frequency against N.

    Attributes:
        x, y (numpy.ndarray): Data.
        prediction (numpy.ndarray): Prediction at x (empty if none).
        meta (dict): Labels ("name", "xlabel", "ylabel") and parameters.
    """
    x: np.ndarray
    y: np.ndarray
    prediction: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def max_deviation(self):
        """max |y - prediction| (nan without prediction)."""
        if self.prediction.size != self.y.size:
            return float("nan")
        return float(np.max(np.abs(self.y - self.prediction)))


# ----------------------------------------------------------------------
# fits and potentials
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class FitWindow:
    """
    Window of a correlator fit: reference site j0 and distances r_min <= r < r_max.

    `FitWindow.bulk(N)` keeps j0 = N/2 - N/5 and r < 0.37 N, i.e. all sites at least ~0.2 N away from
    the open boundaries.
    """
    j0: int
    r_min: int
    r_max: int

    @classmethod
    def bulk(cls, N):
        """Default window for N sites (see class docstring)."""
        return cls(N // 2 - N // 5, 5, int(0.37 * N))

    def check(self, N):
        """Raise ValueError if the window does not fit into N sites."""
        if not (0 <= self.j0 and 1 <= self.r_min < self.r_max and self.j0 + self.r_max <= N):
            raise ValueError(f"Fit window {self} does not fit into N = {N} sites.")


@dataclass(frozen=True)
class FitResult(_Saveable):
    """
    Fit of the electric-field correlator by A K_0(mu r) + B.

    Attributes:
        mu (float): Fitted meson mass.
        A, B (float): Amplitude and constant offset.
        mu_expected (float): Continuum meson mass.
        r (numpy.ndarray): Distances (physical units) used in the fit.
        y (numpy.ndarray): Correlator values at r.
        meta (dict): Fit window, theory and lattice parameters.
    """
    mu: float
    A: float
    B: float
    mu_expected: float
    r: np.ndarray
    y: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def rel_deviation(self):
        """|mu - mu_expected| / mu_expected."""
        return abs(self.mu - self.mu_expected) / self.mu_expected

    def model(self, r):
        """Fitted curve A K_0(mu r) + B."""
        from scipy.special import k0
        return self.A * k0(self.mu * np.asarray(r)) + self.B


@dataclass(frozen=True)
class Potential(_Saveable):
    """
    Static potential V(d) between probe charges +q and -q at distance d.

    Attributes:
        d (numpy.ndarray): Distances (physical units).
        V (numpy.ndarray): Energy of the background (ground-state energy shift).
        prediction (numpy.ndarray): Continuum prediction (empty if none).
        meta (dict): q, theory and lattice parameters, "regime" ("screening" or "confinement").
    """
    d: np.ndarray
    V: np.ndarray
    prediction: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def max_rel_deviation(self):
        """max |V / prediction - 1| (nan without prediction)."""
        if self.prediction.size != self.V.size:
            return float("nan")
        return float(np.max(np.abs(self.V / self.prediction - 1)))
