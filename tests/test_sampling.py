from config import *
from crystalformer.src.wyckoff import symops

def test_symops():
    from crystalformer.src.wyckoff import wmax_table, mult_table
    def project_x(g, w, x, idx):
        '''
        One wants to project randomly sampled fc to the nearest Wyckoff point
        Alternately, we randomly select a Wyckoff point, and then project fc to that point
        To achieve that, we do the following 3 steps 
        '''
        w_max = wmax_table[g-1].item()
        m_max = mult_table[g-1, w_max].item()

        # (1) apply all space group symmetry op to the fc to get x
        ops = symops[g-1, w_max, :m_max] # (m_max, 3, 4)
        affine_point = jnp.array([*x, 1]) # (4, )
        coords = ops@affine_point # (m_max, 3) 
        coords -= jnp.floor(coords)

        # (2) search for the generator which satisfies op0(x) = x , i.e. the first Wyckoff position 
        # here we solve it in a jit friendly way by looking for the minimal distance solution for the lhs and rhs  
        #https://github.com/qzhu2017/PyXtal/blob/82e7d0eac1965c2713179eeda26a60cace06afc8/pyxtal/wyckoff_site.py#L115
        def dist_to_op0x(coord):
            diff = jnp.dot(symops[g-1, w, 0], jnp.array([*coord, 1])) - coord
            diff -= jnp.floor(diff)
            return jnp.sum(diff**2) 
        loc = jnp.argmin(jax.vmap(dist_to_op0x)(coords))
        x = coords[loc].reshape(3,)

        # (3) lastly, apply the given randomly sampled Wyckoff symmetry op to x
        op = symops[g-1, w, idx].reshape(3, 4)
        affine_point = jnp.array([*x, 1]) # (4, )
        x = jnp.dot(op, affine_point)  # (3, )
        x -= jnp.floor(x)
        return x 
    
    # these two tests shows that depending on the z coordinate (which is supposed to be rationals)
    # the WP can be recoginized differently, resulting different x
    # this motivate that we either predict idx in [1, m], or we predict all fc once there is a continuous dof
    g = 167 
    w = jnp.array(5)
    idx = jnp.array(5)
    x = jnp.array([0.123, 0.123, 0.75])
    y = project_x(g, w, x, idx)
    assert jnp.allclose(y, jnp.array([0.123, 0.123, 0.75]))

    x = jnp.array([0.123, 0.123, 0.25])
    y = project_x(g, w, x, idx)
    assert jnp.allclose(y, jnp.array([0.877, 0.877, 0.75]))

    g = 225
    w = jnp.array(5)
    x = jnp.array([0., 0., 0.7334])

    idx = jnp.array(0)
    y = project_x(g, w, x, idx)
    assert jnp.allclose(y, jnp.array([0.7334, 0., 0.]))

    idx = jnp.array(3)
    y = project_x(g, w, x, idx)
    assert jnp.allclose(y, jnp.array([0., 1.0-0.7334, 0.]))
    
    g = 166 
    w = jnp.array(8)
    x = jnp.array([0.1, 0.2, 0.3])

    idx = jnp.array(5)
    y = project_x(g, w, x, idx)
    assert jnp.allclose(y, jnp.array([1-0.1, 1-0.2, 1-0.3]))

def test_sample_top_p():
    from crystalformer.src.sample import sample_top_p
    key = jax.random.PRNGKey(42)
    logits = jnp.array([[1.0, 1.0, 2.0, 2.0, 3.0], 
                        [-1.0, 1.0, 4.0, 1.0, 0.0]
                        ]
                       )
    p = 0.8
    temperature = 1.0
    k = jax.jit(sample_top_p, static_argnums=2)(key, logits, p, temperature)
    print (k)

test_sample_top_p()
test_symops()
