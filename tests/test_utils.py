from config import *

from utils import GLXYZAW_from_file, GLXA_to_csv
from wyckoff import mult_table

def calc_n(G, W):
    @jax.vmap
    def lookup(G, W):
        return mult_table[G-1, W] # (n_max, )
    M = lookup(G, W) # (batchsize, n_max)
    N = M.sum(axis=-1)
    return N

def test_utils():

    atom_types = 119
    mult_types = 10
    n_max = 10
    dim = 3
    csv_file = os.path.join(datadir, 'mini.csv')

    G, L, X, A, W = GLXYZAW_from_file(csv_file, atom_types, mult_types, n_max, dim)
    
    assert G.ndim == 1
    assert L.ndim == 2 
    assert L.shape[-1] == 6

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    print ("A:\n", A)
    N = calc_n(G, W)

    assert jnp.all(N==5)

# def test_io():

#     atom_types = 119
#     wyck_types = 30
#     n_max = 24
#     dim = 3
#     num_test = 5

#     csv_file = os.path.join(datadir, 'mini.csv')
#     out_file = 'temp_out.csv'

#     G, L, X, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max, dim)

#     length, angle = jnp.split(L, 2, axis=-1)
#     num_atoms = calc_n(G, W) 
#     length = length*num_atoms[:, None]**(1/3)
#     angle = angle * (180.0 / jnp.pi) # to deg
#     L = jnp.concatenate([length, angle], axis=-1)

#     GLXA_to_csv(G[:num_test], L[:num_test], X[:num_test], A[:num_test], num_worker=1, filename=out_file)
#     G_io, L_io, X_io, A, W_io = GLXYZAW_from_file(out_file, atom_types, wyck_types, n_max, dim)
#     os.remove(out_file)

    # assert jnp.allclose(A[:num_test], A_io)
    # assert jnp.allclose(W[:num_test], W_io)
    # assert jnp.allclose(G[:num_test], G_io)
    # assert jnp.allclose(X[:num_test], X_io)
    # assert jnp.allclose(L[:num_test], L_io)


if __name__ == '__main__':

    test_utils()
    # test_io()

