import jax
import jax.numpy as jnp

def make_lattice_mask_spacegroup():
    '''
    return mask for independent lattice params 
    '''
    # 1-2
    # 3-18 
    # 19-48
    # 49-64
    # 65-72
    # 73-80    
    mask = [1, 1, 1, 1, 1, 1] * 2 +\
           [1, 1, 1, 0, 1, 0] * 16+\
           [1, 1, 1, 0, 0, 0] * 30+\
           [1, 0, 1, 0, 0, 0] * 16+\
           [1, 0, 1, 0, 0, 0] * 16
        #    [1, 0, 0, 0, 0, 0] * 8

    return jnp.array(mask).reshape(80, 6)

def symmetrize_lattice_spacegroup(g, lattice):
    '''
    place lattice params into lattice according to the space group 
    '''

    a, b, c, alpha, beta, gamma = lattice

    L = lattice
    L = jnp.where(g <= 2,   L, jnp.array([a, b, c, 90., beta, 90.]))
    L = jnp.where(g <= 15,  L, jnp.array([a, b, c, 90., 90., 90.]))
    L = jnp.where(g <= 74,  L, jnp.array([a, a, c, 90., 90., 90.]))
    L = jnp.where(g <= 142, L, jnp.array([a, a, c, 90., 90., 120.]))
    L = jnp.where(g <= 194, L, jnp.array([a, a, a, 90., 90., 90.]))

    return L

def make_lattice_mask_layergroup():
    # 1-2
    # 3-7
    # 8-18
    # 19-48
    # 49-64
    # 65-72
    # 73-80

    mask =  [1, 1, 1, 1, 1, 1] * 2 +\
            [1, 1, 1, 0, 1, 0] * 16 +\
            [1, 1, 1, 0, 0, 0] * 30 +\
            [1, 0, 1, 0, 0, 0] * 16 +\
            [1, 0, 0, 1, 0, 0] * 8 +\
            [1, 0, 1, 0, 0, 0] * 8
    
    return jnp.array(mask).reshape(80, 6)

def symmetrize_lattice_layergroup(g, lattice):
    a, b, c, alpha, beta, gamma = lattice

    L = lattice
    L = jnp.where(g <= 2,   L, jnp.array([a, b, c, 90., beta, 90.]))
    L = jnp.where(g <= 18,  L, jnp.array([a, b, c, 90., 90., 90.]))
    L = jnp.where(g <= 48,  L, jnp.array([a, a, c, 90., 90., 90.]))
    L = jnp.where(g <= 64, L, jnp.array([a, a, a, alpha, alpha, alpha]))
    L = jnp.where(g <= 72, L, jnp.array([a, a, c, 90., 90., 120.]))

    return L

if __name__ == '__main__':
    
    mask = make_lattice_mask_layergroup()
    print (mask)

    key = jax.random.PRNGKey(42)
    lattice = jax.random.normal(key, (6,))
    lattice = lattice.reshape([1, 6]).repeat(3, axis=0)

    G = jnp.array([25, 99, 221])
    L = jax.vmap(symmetrize_lattice_layergroup)(G, lattice)
    print (L)

