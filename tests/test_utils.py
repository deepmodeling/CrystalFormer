from config import *

from crystalformer.src.utils import GLXYZAW_from_file
from crystalformer.src.wyckoff import mult_table

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
    csv_file = os.path.join(datadir, '../../data/mini.csv')

    G, L, X, A, W = GLXYZAW_from_file(csv_file, atom_types, mult_types, n_max, dim)
    
    assert G.ndim == 1
    assert L.ndim == 2 
    assert L.shape[-1] == 6

    import numpy as np 
    np.set_printoptions(threshold=np.inf)
    
    print ("A:\n", A)
    N = calc_n(G, W)

    assert jnp.all(N==5)

if __name__ == '__main__':

    test_utils()
