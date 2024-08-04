from config import * 

from crystalformer.src.utils import GLXYZAW_from_file
from crystalformer.src.wyckoff import mult_table
from crystalformer.src.transformer import make_transformer

def test_autoregressive():
    atom_types = 119
    wyck_types = 28
    Nf = 8
    n_max = 21
    Kx = 16 
    Kl = 8
    dim = 3
    dropout_rate = 0.0

    csv_file = os.path.join(datadir, '../../data/mini.csv')
    G, L, X, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max, dim)
        
    @jax.vmap
    def lookup(G, W):
        return mult_table[G-1, W] # (n_max, )
    M = lookup(G, W) # (batchsize, n_max)
    num_sites = jnp.sum(A!=0, axis=1)

    key = jax.random.PRNGKey(42)
    params, transformer = make_transformer(key, Nf, Kx, Kl, n_max, dim, 128, 4, 4, 8, 16,atom_types, wyck_types, dropout_rate) 

    def test_fn(X, M):
        output = transformer(params, None, G[0], X, A[0], W[0], M, False)
        print (output.shape)
        return output.sum(axis=-1)

    jac_x = jax.jacfwd(test_fn, argnums=0)(X[0], M[0])
    jac_m = jax.jacfwd(test_fn, argnums=1)(X[0], M[0].astype(jnp.float32))[:, :, None]

    print(jac_x.shape, jac_m.shape)

    def print_dependencey(jac):
        dependencey = jnp.linalg.norm(jac, axis=-1)
        for row in (dependencey != 0.).astype(int):
            print(" ".join(str(val) for val in row))

    print ("jac_a_x") 
    print_dependencey(jac_x[::2])
    print ("jac_x_x") 
    print_dependencey(jac_x[1::2])
    print ("jac_a_a") 
    print_dependencey(jac_m[::2])
    print ("jac_x_a") 
    print_dependencey(jac_m[1::2])


def test_perm():

    key = jax.random.PRNGKey(42)

    #W = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0])
    W = jnp.array([1,2, 2, 2, 5, 0,0, 0])
    n = len(W)
    key = jax.random.PRNGKey(42)

    temp = jnp.where(W>0, W, 9999)
    idx_perm = jax.random.permutation(key, jnp.arange(n))
    temp = temp[idx_perm]
    idx_sort = jnp.argsort(temp)
    idx = idx_perm[idx_sort]

    print (idx)
    print (W)
    assert jnp.allclose(W, W[idx])

test_autoregressive()
