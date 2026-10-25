from dataclasses import dataclass

import numpy as np


BOUNDARIES = ("open", "ring")


@dataclass(frozen=True)
class Lattice:
    """
    A one-dimensional spatial lattice of N sites with spacing a.

    Boundary conditions:
        "open": Dirichlet chain, varphi_{-1} = varphi_N = 0. The fixed end points lie outside
                the N sites, so the physical length is L = (N + 1) a and the free momenta are
                k_n = pi n / L (lattice: (2/a) sin(k_n a / 2)).
        "ring": periodic chain, site N = site 0, so L = N a. The compact bosons keep their
                zero modes (windings and total momenta), i.e. the charge sectors.

    Attributes:
        N (int): Number of sites, N >= 2.
        a (float): Lattice spacing, a > 0.
        boundary (str): "open" or "ring".
    """
    N: int
    a: float = 1.0
    boundary: str = "open"

    def __post_init__(self):
        if isinstance(self.N, bool) or not isinstance(self.N, (int, np.integer)):
            raise TypeError(f"N must be an integer, got {self.N!r}.")
        if self.N < 2:
            raise ValueError(f"A lattice needs at least two sites, got N = {self.N}.")
        if isinstance(self.a, bool) or not np.isfinite(self.a) or self.a <= 0:
            raise ValueError(f"The lattice spacing must be finite and positive, got a = {self.a!r}.")
        if self.boundary not in BOUNDARIES:
            raise ValueError(f"Unknown boundary {self.boundary!r}, choose one of {BOUNDARIES}.")
        object.__setattr__(self, "N", int(self.N))
        object.__setattr__(self, "a", float(self.a))

    @property
    def L(self):
        """Physical length: (N + 1) a for the open chain, N a for the ring."""
        return (self.N + 1) * self.a if self.boundary == "open" else self.N * self.a

    @property
    def is_ring(self):
        return self.boundary == "ring"
