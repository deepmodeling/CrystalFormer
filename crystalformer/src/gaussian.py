import jax
from jax import jit, lax, random
import jax.numpy as jnp
from functools import partial
from crystalformer.src.von_mises import sample_von_mises
   
def gaussian_centered(key, concentration, shape, dtype = jnp.float64):
    shape = shape or jnp.shape(concentration)
    dtype = jnp.result_type(dtype)
    concentration = lax.convert_element_type(concentration, dtype)
    concentration = jnp.broadcast_to(concentration, shape)
    return _gaussian_centered(key, concentration, shape, dtype)

@partial(jit, static_argnums = (2, 3))
def _gaussian_centered(key, concentration, shape, dtype):
    # what cutoff?
    s_cutoff_map = {
        jnp.dtype(jnp.float16): 1.8e-1,
        jnp.dtype(jnp.float32): 2e-2,
        jnp.dtype(jnp.float64): 1.2e-4,
    }
    s_cutoff = s_cutoff_map.get(dtype)

    r = 1.0 + jnp.sqrt(1.0 + 4.0 * concentration**2)
    rho = (r - jnp.sqrt(2.0 * r)) / (2.0 * concentration)
    s_exact = (1.0 + rho**2) / (2.0 * rho)

    s_approximate = 1.0 / concentration

    s = jnp.where(concentration > s_cutoff, s_exact, s_approximate)

    def cond_fn(*args):
        i, _, done, _, _ = args[0]
        return jnp.bitwise_and(i < 100, jnp.logical_not(jnp.all(done)))
    
    def body_fn(*args):
        i, key, done, _, w = args[0]
        uni_ukey, uni_vkey, key = random.split(key, 3)

        u = random.uniform(
            key = uni_ukey,
            shape = shape,
            dtype = concentration.dtype,
            minval = -1.0,
            maxval = 1.0,
        )
        z = jnp.cos(jnp.pi * u)
        w = jnp.where(done, w, (1.0 + s * z) / (s + z))

        y = concentration * (s - w)
        v = random.uniform(key = uni_vkey, shape = shape, dtype = concentration.dtype)

        accept = (y * (2.0 - y) >= v) | (jnp.log(y / v) + 1.0 >= y)

        return i + 1, key, accept | done, u, w
    
    init_done = jnp.zeros(shape, dtype=bool)
    init_u = jnp.zeros(shape)
    init_w = jnp.zeros(shape)

    _, _, done, u, w = lax.while_loop(
        cond_fun=cond_fn,
        body_fun=body_fn,
        init_val=(jnp.array(0), key, init_done, init_u, init_w),
    )

    return jnp.sign(u) * jnp.arccos(w)

def gaussian_logpdf(x, loc, concentration):
    '''
    concentration = kappa in von mises distribution = 1/variance
    concentration = 1/sigma^2 in gaussian distribution
    '''
    return -0.5 * (jnp.log(2 * jnp.pi) - jnp.log(concentration) + concentration * (x - loc) * (x - loc))



if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    num_samples = 1000
    loc = jnp.zeros(num_samples)
    kappa = jnp.ones(num_samples)
    x = sample_gaussian(key, loc, kappa, (num_samples, ))
    x_grid = jnp.linspace(-5.0, 5.0, 100)
    y = [0 for _ in range(100)]
    for i in range(num_samples):
        for j in range(100 - 1):
            if (x[i] >= x_grid[j]) & (x[i] <= x_grid[j+1]):
                y[j] += 1
                break

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1)
    ax.bar(x_grid, y)
    plt.show()