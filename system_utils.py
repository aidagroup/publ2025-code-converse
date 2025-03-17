import sympy as sp
import numpy as np
from sympy import Matrix, symbols

def generate_g_matrix(dim_state, dim_inputs, N, Per_mag):
    """Generates a symbolic matrix 'g' with elements depending on state variables."""
    state_vars = Matrix(sp.symbols(f's1:{dim_state + 1}'))
    
    c = np.random.normal(size=dim_state**(N + 1)).reshape(tuple([dim_state] * (N + 1)))
    
    V = state_vars.dot(state_vars)
    xi = (Matrix([V]).jacobian(state_vars) * 1.2).T
    g = sp.Matrix.zeros(dim_state, dim_inputs)
    
    side = np.array([sp.cos(state_var) for state_var in state_vars])

    for i in range(dim_state):
        res = c[i]
        for _ in range(N):
            res = np.tensordot(res, side, axes=1)
        g[i, i] = 3 * (1+0.5**2) * 2**(sum([-(state_var/10)**2 for state_var in state_vars])) + Per_mag * res[None][0]
    
    return g, state_vars, c, V, xi