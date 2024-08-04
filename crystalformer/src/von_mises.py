# https://github.com/pyro-ppl/numpyro/blob/master/numpyro/distributions/util.py
# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

import jax
from jax import jit, lax, random
import jax.numpy as jnp
from functools import partial

def sample_von_mises(key, loc, concentration, shape):
    """Generate sample from von Mises distribution

    :param key: random number generator key
    :param sample_shape: shape of samples
    :return: samples from von Mises
    """
    samples = von_mises_centered(
        key, concentration, shape
    )
    samples = samples + loc  # VM(0, concentration) -> VM(loc,concentration)
    samples = (samples + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    return samples

def von_mises_centered(key, concentration, shape, dtype=jnp.float64):
    """Compute centered von Mises samples using rejection sampling from [1] with wrapped Cauchy proposal.
    *** References ***
    [1] Luc Devroye "Non-Uniform Random Variate Generation", Springer-Verlag, 1986;
        Chapter 9, p. 473-476. http://www.nrbook.com/devroye/Devroye_files/chapter_nine.pdf
    :param key: random number generator key
    :param concentration: concentration of distribution
    :param shape: shape of samples
    :param dtype: float precesions for choosing correct s cutfoff
    :return: centered samples from von Mises
    """
    shape = shape or jnp.shape(concentration)
    dtype = jnp.result_type(dtype)
    concentration = lax.convert_element_type(concentration, dtype)
    concentration = jnp.broadcast_to(concentration, shape)
    return _von_mises_centered(key, concentration, shape, dtype)


@partial(jit, static_argnums=(2, 3))
def _von_mises_centered(key, concentration, shape, dtype):
    # Cutoff from TensorFlow probability
    # (https://github.com/tensorflow/probability/blob/f051e03dd3cc847d31061803c2b31c564562a993/tensorflow_probability/python/distributions/von_mises.py#L567-L570)
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
        """check if all are done or reached max number of iterations"""
        i, _, done, _, _ = args[0]
        return jnp.bitwise_and(i < 100, jnp.logical_not(jnp.all(done)))

    def body_fn(*args):
        i, key, done, _, w = args[0]
        uni_ukey, uni_vkey, key = random.split(key, 3)

        u = random.uniform(
            key=uni_ukey,
            shape=shape,
            dtype=concentration.dtype,
            minval=-1.0,
            maxval=1.0,
        )
        z = jnp.cos(jnp.pi * u)
        w = jnp.where(done, w, (1.0 + s * z) / (s + z))  # Update where not done

        y = concentration * (s - w)
        v = random.uniform(key=uni_vkey, shape=shape, dtype=concentration.dtype)

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

def von_mises_logpdf(x, loc, concentration):
    '''
    kappa is the concentration. kappa = 0 means uniform distribution
    '''
    return -(jnp.log(2 * jnp.pi) + jnp.log(jax.scipy.special.i0e(concentration))
              ) + concentration * (jnp.cos((x - loc) % (2 * jnp.pi)) - 1)

if __name__=='__main__':
    key = jax.random.PRNGKey(42)
    loc = jnp.array([-1.0, 1.0, 0.0])
    kappa = jnp.array([10.0, 10.0, 100.0])
    x = sample_von_mises(key, loc, kappa, (3, ))
    print (x)

