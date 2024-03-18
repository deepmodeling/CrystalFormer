import jax
import jax.numpy as jnp

def make_lattice_mask():
    '''
    return mask for independent lattice params 
    '''
    # 1-2
    # 3-15 
    # 16-74
    # 75-142
    # 143-194
    # 195-230    
    mask = [1, 1, 1, 1, 1, 1] * 2 +\
           [1, 1, 1, 0, 1, 0] * 13+\
           [1, 1, 1, 0, 0, 0] * 59+\
           [1, 0, 1, 0, 0, 0] * 68+\
           [1, 0, 1, 0, 0, 0] * 52+\
           [1, 0, 0, 0, 0, 0] * 36

    return jnp.array(mask).reshape(230, 6)

def symmetrize_lattice(spacegroup, lattice):
    '''
    place lattice params into lattice according to the space group 
    '''

    a, b, c, alpha, beta, gamma = lattice

    L = lattice
    L = jnp.where(spacegroup <= 2,   L, jnp.array([a, b, c, 90., beta, 90.]))
    L = jnp.where(spacegroup <= 15,  L, jnp.array([a, b, c, 90., 90., 90.]))
    L = jnp.where(spacegroup <= 74,  L, jnp.array([a, a, c, 90., 90., 90.]))
    L = jnp.where(spacegroup <= 142, L, jnp.array([a, a, c, 90., 90., 120.]))
    L = jnp.where(spacegroup <= 194, L, jnp.array([a, a, a, 90., 90., 90.]))

    return L

if __name__ == '__main__':
    
    mask = make_lattice_mask()
    print (mask)

    key = jax.random.PRNGKey(42)
    lattice = jax.random.normal(key, (6,))
    lattice = lattice.reshape([1, 6]).repeat(3, axis=0)

    G = jnp.array([25, 99, 221])
    L = jax.vmap(symmetrize_lattice)(G, lattice)
    print (L)

