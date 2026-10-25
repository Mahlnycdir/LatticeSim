from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Block:
    """
    A named block of canonical coordinates and its conjugate momenta.

    Attributes:
        name (str): Name of the coordinates, e.g. "phi".
        conjugate (str): Name of the conjugate momenta, e.g. "P".
        shape (tuple): Shape of the block, e.g. (F, N) for F flavors on N sites.
        offset (int): Position of the first coordinate in X.
    """
    name: str
    conjugate: str
    shape: tuple
    offset: int

    @property
    def size(self):
        return int(np.prod(self.shape, dtype=int))


class PhaseSpace:
    """
    Layout of the phase-space vector X = (q, p) with named blocks.

    All coordinates come first, then all momenta in the same order, so that
    [X_a, X_b] = i J_ab with the standard J = [[0, I], [-I, 0]]. Every coordinate
    block automatically has a conjugate momentum block at offset + n.

    Example:
        ps = PhaseSpace([("phi", "P", (F, N)), ("theta_bar", "W", (F,))])
        ps.index("phi", i, j)     # flat index of varphi_{i,j}
        ps.slice("P", i)          # slice of P_{i,.}
        ps.index("W", i)          # flat index of W_i

    Attributes:
        n (int): Number of degrees of freedom (coordinates).
        dim (int): Phase-space dimension 2n.
    """

    def __init__(self, blocks):
        """
        Args:
            blocks (list): Tuples (name, conjugate, shape). Shapes are tuples of ints
                (an int is accepted for one-dimensional blocks).

        Raises:
            ValueError: If a name is used twice or a shape is empty or non-positive.
        """
        self._blocks = []
        self._by_name = {}
        offset = 0
        for name, conjugate, shape in blocks:
            shape = (shape,) if np.isscalar(shape) else tuple(shape)
            if len(shape) == 0 or any(int(s) != s or s <= 0 for s in shape):
                raise ValueError(f"Block {name!r} needs a shape of positive integers, got {shape}.")
            shape = tuple(int(s) for s in shape)
            for key in (name, conjugate):
                if key in self._by_name:
                    raise ValueError(f"The name {key!r} is used twice.")
            block = Block(name, conjugate, shape, offset)
            self._blocks.append(block)
            self._by_name[name] = (block, 0)
            self._by_name[conjugate] = (block, None)        # momentum offset is filled in below
            offset += block.size
        self.n = offset
        self.dim = 2 * offset
        for block in self._blocks:
            self._by_name[block.conjugate] = (block, self.n)


    @classmethod
    def flat(cls, n):
        """Unstructured phase space with coordinates q_0..q_(n-1) and momenta p_0..p_(n-1)."""
        return cls([("q", "p", (n,))])


    @property
    def blocks(self):
        return tuple(self._blocks)


    def _lookup(self, name):
        if name not in self._by_name:
            raise KeyError(f"Unknown block {name!r}; known: {sorted(self._by_name)}")
        return self._by_name[name]


    def index(self, name, *idx):
        """
        Flat index of one entry, e.g. index("phi", i, j).

        Raises:
            IndexError: If the number of indices or their values do not fit the block.
        """
        block, shift = self._lookup(name)
        if len(idx) != len(block.shape):
            raise IndexError(f"Block {name!r} has shape {block.shape}, got indices {idx}.")
        return shift + block.offset + int(np.ravel_multi_index(idx, block.shape))


    def slice(self, name, *leading):
        """
        Contiguous slice of a block, fixing the leading indices, e.g. slice("phi", i) for varphi_{i,.}
        or slice("phi") for the whole block.
        """
        block, shift = self._lookup(name)
        if len(leading) > len(block.shape):
            raise IndexError(f"Block {name!r} has shape {block.shape}, got indices {leading}.")
        rest = int(np.prod(block.shape[len(leading):], dtype=int))
        start = 0
        if leading:
            start = int(np.ravel_multi_index(leading, block.shape[:len(leading)])) * rest
        start += shift + block.offset
        return slice(start, start + rest)


    def zeros(self):
        """Zero phase-space vector."""
        return np.zeros(self.dim)


    @property
    def J(self):
        """Symplectic matrix [[0, I], [-I, 0]], [X_a, X_b] = i J_ab."""
        Z = np.zeros((self.n, self.n))
        I = np.eye(self.n)
        return np.block([[Z, I], [-I, Z]])


    def __repr__(self):
        parts = ", ".join(f"{b.name}/{b.conjugate}{b.shape}" for b in self._blocks)
        return f"PhaseSpace({parts}; n = {self.n})"
