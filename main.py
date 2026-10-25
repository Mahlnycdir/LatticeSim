from latsimpkg.scalar import ScalarFieldTheory
import numpy as np
        

if __name__ == "__main__":
    w = 1.7
    N = 50
    
    sc =ScalarFieldTheory(N, 0.001)
    sc.free_hamiltonian_matrix()
    
    # sig = sc.ground_state_covariance(np.diag([w**2, 1.0]))
    # print(f"Test 1  single oscillator:  <q^2> = {sig[0, 0]:.12f}   expected 1/(2w) = {1/(2*w):.12f}")
        
    om = sc.normal_modes()
    exact = 2 * np.sin(np.pi * np.arange(1, N + 1) / (2 * (N + 1)))
    print(f"Test 2  Dirichlet chain:    max|omega - 2 sin(pi n/(2(N+1)))| = {np.max(np.abs(om - exact)):.1e}")
    
    sig = sc.ground_state_covariance()
    lam = np.linalg.eigvalsh(sig + 0.5j * sc.symplectic_matrix()).min()
    print(f"Test 3  uncertainty principle sigma + iJ/2 >= 0:  smallest eigenvalue = {lam:.1e}")
    