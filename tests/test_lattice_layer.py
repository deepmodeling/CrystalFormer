from config import *

# from crystalformer.src.lattice import symmetrize_lattice, make_lattice_mask
from crystalformer.src.sym_group import *

sym_group = LayerGroup()

def test_symmetrize_lattice():
    key = jax.random.PRNGKey(42)

    G = jnp.arange(80) + 1
    L = jax.random.uniform(key, (6,))
    L = L.reshape([1, 6]).repeat(80, axis=0)

    lattice = jax.jit(jax.vmap(sym_group.symmetrize_lattice()))(G, L)
    print (lattice)    
    
    a, b, c, alpha, beta, gamma = lattice[49-1] 
    assert (alpha==beta==gamma==90)
    assert (a==b)

def test_make_mask():

    def make_spacegroup_mask(spacegroup):
        '''
        return mask for independent lattice params 
        '''

        mask = jnp.array([1, 1, 1, 1, 1, 1])

        mask = jnp.where(spacegroup <= 2,   mask, jnp.array([1, 1, 1, 0, 1, 0]))
        mask = jnp.where(spacegroup <= 18,  mask, jnp.array([1, 1, 1, 0, 0, 0]))
        mask = jnp.where(spacegroup <= 48,  mask, jnp.array([1, 0, 1, 0, 0, 0]))
        mask = jnp.where(spacegroup <= 64, mask, jnp.array([1, 0, 0, 1, 0, 0]))
        mask = jnp.where(spacegroup <= 72, mask, jnp.array([1, 0, 1, 0, 0, 0]))
        return mask
    
    mask = sym_group.make_lattice_mask()()

    for g in range(1, 81):
        assert jnp.allclose(mask[g-1] , make_spacegroup_mask(g))

test_symmetrize_lattice()
test_make_mask()

