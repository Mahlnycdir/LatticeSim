import numpy as np 
from numpy.linalg import eigvals

def symplectic_matrix(N):
    Z = np.zeros((N,N))
    I = np.eye(N)
    return np.block([[Z, I], [-I, Z]])

def hamiltonian_matrix(N, a = 1.0):
    '''
    This represents the Hamiltonian matrix of our 
    discretized free scalar theory, namley
    H = 1/(2a) sum_j {P_j^2 + (phi_(j+1) - phi_j)}
      = X^T M X,
    where we defined 
    X = (phi_0,...,phi_(N-1),P_0,...,P_(N-1))^T.
    We would find the Matrix as 
    [[ 1 -1 0 0... 0],
     [-1  1 0 0 ... 0],
     [ 0  0 1 0 ... 0]]
     [ 0  0 0 1 ... 0]
            .  N - Times 
            .]
    The added Block is for N=2 
    '''
    phi_matrix = (np.eye(2) - np.fliplr(np.eye(2))) / 2*a
    p_matrix   = np.eye(2) / 2*a 
    z = np.zeros((2,2))
    h = np.block([[phi_matrix, z],
                  [z, p_matrix]])

    return np.kron(np.eye(int(N/2)),h)

def energy_eigenvalues(Hamiltonian):
    '''
    Extracted from [X_c,H]. 
    '''
    


if __name__ == "__main__":
    N = 300                 # Needs to be mod(N,d) = 0, where d is the dimension,
                            # because N is the number of lattice point-entries!
                            # (always d entries per point)
    a = 0.001
    
    