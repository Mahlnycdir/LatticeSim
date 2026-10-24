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
    '''
    phi_matrix = (np.eye(2) - np.fliplr(np.eye(2))) / a # Wrong!!!
    p_matrix   = np.eye(N) / a 
    z = np.zeros((N,N))
    h = np.block([[phi_matrix, z],
                  [z, p_matrix]])

    return np.kron(np.eye(int(N/2)),h)

def energy_eigenvalues(Hamiltonian):
    '''
    Extracted from [H,X_c]. 
    '''
    


if __name__ == "__main__":
    N = 300                 # Needs to be mod(N,d) = 0, where d is the dimension,
                            # because N is the number of lattice point-entries!
                            # (always d entries per point)
    a = 0.001
    
    