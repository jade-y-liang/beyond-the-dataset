import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import sparse

def advance_time_matvecmul(A, u, epsilon):
    """Advances the simulation by one timestep, via matrix-vector multiplication
    Args:
        A: The 2d finite difference matrix, N^2 x N^2. 
        u: N x N grid state at timestep k.
        epsilon: stability constant.

    Returns:
        N x N Grid state at timestep k+1.
    """
    N = u.shape[0]
    u = u + epsilon * (A @ u.flatten()).reshape((N, N))
    return u

def get_A(N):
    """ Returns the corresponding matrix A according to N

    Returns:
        N^2 x N^2 matrix without all-zero rows or all-zero columns
    """

    n = N * N
    diagonals = [-4 * np.ones(n), 
                 np.ones(n-1), 
                 np.ones(n-1), 
                 np.ones(n-N), 
                 np.ones(n-N)]
    diagonals[1][(N-1)::N] = 0
    diagonals[2][(N-1)::N] = 0
    A = np.diag(diagonals[0]) + np.diag(diagonals[1], 1) + np.diag(diagonals[2], -1) + np.diag(diagonals[3], N) + np.diag(diagonals[4], -N)

    return A

def get_sparse_A(N):
    """Constructs the finite difference matrix for 2D heat diffusion in sparse format."""
    return sparse.BCOO.fromdense(jnp.array(get_A(N)))

def advance_time_numpy(u, epsilon):
    """Advances the heat diffusion solution by one timestep using numpy operations."""
    
    # Create a padded version of u with zeros around the border
    u_pad = np.pad(u, pad_width=1, mode='constant', constant_values=0)

    # Calculate the updated values by rolling the array
    u_new = (1 - 4 * epsilon) * u_pad[1:-1, 1:-1] + \
            epsilon * (np.roll(u_pad, shift=1, axis=0)[1:-1, 1:-1] + 
                       np.roll(u_pad, shift=-1, axis=0)[1:-1, 1:-1] +
                       np.roll(u_pad, shift=1, axis=1)[1:-1, 1:-1] + 
                       np.roll(u_pad, shift=-1, axis=1)[1:-1, 1:-1]) 
    return u_new

@jax.jit
def advance_time_jax(u, epsilon):
    """Advances the heat diffusion solution by one timestep using JAX operations."""
    u_pad = jnp.pad(u, pad_width=1, mode='constant', constant_values=0)
    
    u_new = (1 - 4 * epsilon) * u_pad[1:-1, 1:-1] + \
            epsilon * (jnp.roll(u_pad, shift=1, axis=0)[1:-1, 1:-1] + 
                       jnp.roll(u_pad, shift=-1, axis=0)[1:-1, 1:-1] +
                       jnp.roll(u_pad, shift=1, axis=1)[1:-1, 1:-1] +
                       jnp.roll(u_pad, shift=-1, axis=1)[1:-1, 1:-1]) 
    return u_new