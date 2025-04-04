import jax
import jax.numpy as jnp
import haiku as hk


def sine_table(features, length, min_timescale=1.0, max_timescale=10000.0):
    fraction = jnp.arange(0, features, 2, dtype=jnp.float32) / features
    timescale = min_timescale * (max_timescale / min_timescale) ** fraction
    rotational_frequency = 1.0 / timescale
    # Must use high precision einsum here, bfloat16 rounding is catastrophic.
    sinusoid_inp = jnp.einsum(
        'i,j->ij',
        jnp.arange(length),
        rotational_frequency,
        precision=jax.lax.Precision.HIGHEST,
    )
    sinusoid_inp = jnp.concatenate([sinusoid_inp, sinusoid_inp], axis=-1)
    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rotate_half(x):
    x1, x2 = jnp.split(x, 2, axis=-1)
    x = jnp.concatenate([-x2, x1], axis=-1)
    return x


# https://github.com/google/flax/blob/nnx/flax/experimental/nnx/examples/07_transformer.py#L131-L157
def apply_rotary_embedding(q, k, cos, sin, index=None):
    """
    Helper function to apply Rotary Embeddings.
    
    The implementation is different from the original Rotary position embeddings,
    more details can be found in F.2. section of https://arxiv.org/abs/2202.07765 
    """
    qlen, qheads, d = q.shape
    klen, kheads, kd = k.shape
    if index is not None:
        qcos = jax.lax.broadcast_in_dim(
        cos[index, :], (qlen, qheads, d), (2,)
        )
        qsin = jax.lax.broadcast_in_dim(
        sin[index, :], (qlen, qheads, d), (2,)
        )
    else:
        qcos = jax.lax.broadcast_in_dim(
        cos[:qlen, :], (qlen, qheads, d), (0, 2)
        )
        qsin = jax.lax.broadcast_in_dim(
        sin[:qlen, :], (qlen, qheads, d), (0, 2)
        )
    kcos = jax.lax.broadcast_in_dim(
        cos[:klen, :], (klen, kheads, d), (0, 2)
    )
    ksin = jax.lax.broadcast_in_dim(
        sin[:klen, :], (klen, kheads, d), (0, 2)
    )
    out_q = (q * qcos) + (rotate_half(q) * qsin)
    out_k = (k * kcos) + (rotate_half(k) * ksin)
    return out_q, out_k


class RelativePosition(hk.Module):
    """
    Relative Positional Embeddings
    
    e_ij = (x_i * W^Q) * (x_j * W^K)^T / sqrt(d) + d_ij
    d_ij is the relative position embedding
    """
    def __init__(self, max_relative_position):
        """
        max_relative_position: maximum relative position
        """
        
        super().__init__()
        self.max_relative_position = max_relative_position
        self.embeddings_table = hk.get_parameter(
            "embeddings_table",
            shape=(max_relative_position * 2 + 1, ),
            init=hk.initializers.TruncatedNormal(0.01)
        )

    def __call__(self, length_q, length_k):
        range_vec_q = jnp.arange(length_q)
        range_vec_k = jnp.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = jax.lax.clamp(-self.max_relative_position, distance_mat, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = final_mat.astype(int)
        embeddings = self.embeddings_table[final_mat]

        return embeddings
