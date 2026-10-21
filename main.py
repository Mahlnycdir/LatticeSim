import numpy as np 

def symplectic_matrix(N):
    Z = np.zeros((N,N))
    I = np.eye(N)
    return np.block([[Z, I], [-I, Z]])

def evals(M):
    return np.linalg.eigvals(M)


if __name__ == "__main__":
    print(symplectic_matrix(2))