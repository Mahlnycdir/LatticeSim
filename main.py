import numpy as np 
from numpy.linalg import eigvals

def symplectic_matrix(N):
    """
    The symplectic matrix, defined by the commutator
    [X_a,X_b] = i J_ab,

    Args:
        N (_type_): _description_

    Returns:
        _type_: _description_
    """
    Z = np.zeros((N,N))
    I = np.eye(N)
    return np.block([[Z, I], [-I, Z]])

def hamiltonian_matrix(N, a = 1.0):
    """
    This represents the Hamiltonian matrix of our 
    discretized free scalar theory, namley
    H = 1/(2a) sum_j {P_j^2 + (phi_(j+1) - phi_j)}
      = (1/2) X^T M X,
    where we defined 
    X = (phi_0,...,phi_(N-1),P_0,...,P_(N-1))^T.
    We would find the Matrix as 
    [[ 1 -1  0  0  0 ... 0],
     [-1  2 -1  0  0 ... 0], 
     [ 0 -1  2 -1  0 ... 0],
     [ 0  0 -1  2 -1 ... 0],
            . 
            .  N - Times 
            .
     [ 0^(NxN),    I^(NxN)]]
    Args:
            N (int)  : Number of lattice points
            a (float): Lattice spacing
    """
    if N < 2 or not isinstance(N, int): 
        raise ValueError(f"A simulation on less then two ({N}) or non-integer lattice points is not possible!")
                       
    phi_matrix = ((2 * np.eye(N)) + -np.eye(N, k=1) - np.eye(N, k=-1)) / a
    phi_matrix[0][0] *= 0.5 
    p_matrix   = np.eye(N) / a 
    z = np.zeros((N,N))
     
    return np.block([[phi_matrix,   z   ],
                     [     z,   p_matrix]]) 
    

def normal_modes():
    """
    This method provides the eigenvectors of a given quantum system for further calculations to the 
    energy spectrum (basically the diagonalized Hamiltonian with the eigenvalues on the diagonal).
    """

def energy_eigenvalues(Hamiltonian):
    """_summary_

    Args:
        Hamiltonian (_type_): _description_
    """
    
def ground_state(Hamiltonian):
    """_summary_

    Args:
        Hamiltonian (_type_): _description_
    """
    

if __name__ == "__main__":
    N = 300                 # Needs to be mod(N,d) = 0, where d is the dimension,
                            # because N is the number of lattice point-entries!
                            # (always d entries per point)
    a = 0.001
    print(hamiltonian_matrix(N, a))
    # hamiltonian_matrix(3)
    