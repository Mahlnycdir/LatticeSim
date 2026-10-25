"""
Conventions of the bosonized theories, defined exactly once.

Bosonization (one Dirac fermion psi = (psi_L, psi_R) <-> one compact boson varphi):
    rho_V = psi^dag psi       = d_x varphi / sqrt(pi)
    rho_A = psi^dag g5 psi    = Pi / sqrt(pi)
    varphi ~ varphi + sqrt(pi)   (compactness)
The factor sqrt(pi) is `SQRT_PI`.

Charges. A Dirac pair i combines the left mover L_i and the right mover R_i:
    vector charge  Q^V = (Q_L + Q_R)/2,   axial charge  Q^A = (Q_L - Q_R)/2,
so that the gauge current Q_L j_L + Q_R j_R = Q^V j_V + Q^A j_A with j_V = j_L + j_R, j_A = j_L - j_R.

Fermion numbers. On the ring the fermion numbers are the zero modes of the boson:
    winding   varphi(x + L) = varphi(x) + sqrt(pi) n,   n = integral of rho_V = N_L + N_R,
    momentum  integral of Pi = sqrt(pi) m,              m = integral of rho_A = N_L - N_R.
N_L and N_R are integers, hence n = m (mod 2): (n, m) = (1, 0) would be "half a left and half a
right fermion" and is not a state. This parity rule is what makes the compact boson fermionic.

Signs. Lattice momentum P = -(1/a) sum_j P_j l_j (continuum P = -int Pi d_x varphi) and spin
s = -P L/(2 pi) = h - hb, chosen such that a single left mover (N_L, N_R) = (1, 0) has s = +1/2.
"""
import numpy as np

SQRT_PI = np.sqrt(np.pi)


def vector_axial(QL, QR):
    """
    Q^V = (Q_L + Q_R)/2,  Q^A = (Q_L - Q_R)/2  (pairing L_i <-> R_i).
    """
    QL = np.asarray(QL, dtype=float); QR = np.asarray(QR, dtype=float)
    if QL.shape != QR.shape:
        raise ValueError(f"Q_L {QL.shape} and Q_R {QR.shape} must have the same shape to be paired.")
    return (QL + QR) / 2, (QL - QR) / 2


def chiral(QV, QA):
    """
    Inverse of vector_axial: Q_L = Q^V + Q^A,  Q_R = Q^V - Q^A.
    """
    QV = np.asarray(QV, dtype=float); QA = np.asarray(QA, dtype=float)
    if QV.shape != QA.shape:
        raise ValueError(f"Q^V {QV.shape} and Q^A {QA.shape} must have the same shape.")
    return QV + QA, QV - QA


def winding_momentum(NL, NR):
    """
    Fermion numbers -> winding n = N_L + N_R and zero-mode momentum m = N_L - N_R.
    """
    NL = np.asarray(NL); NR = np.asarray(NR)
    return NL + NR, NL - NR


def chiral_numbers(n, m):
    """
    Inverse: N_L = (n + m)/2, N_R = (n - m)/2. Raises ValueError unless n = m (mod 2).
    """
    n = np.asarray(n); m = np.asarray(m)
    if np.any(np.asarray(n - m) % 2 != 0):
        raise ValueError(f"(n, m) = ({n}, {m}) violates n = m (mod 2): not a fermionic state.")
    return (n + m) // 2, (n - m) // 2
