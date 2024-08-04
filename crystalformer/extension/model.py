import jax
import jax.numpy as jnp
import haiku as hk


def make_classifier(key,
                    n_max = 21,
                    embed_size=32,
                    sequence_length=105,
                    outputs_size=64,
                    hidden_sizes=[128, 128],
                    num_classes=1,
                    dropout_rate=0.3):

    @hk.transform
    def network(g, l, w, h, is_training):
        """
        sequence_length = n_max * 5
        g : (embed_size, )
        l : (6, )
        w : (n_max,)
        h : (sequence_length, ouputs_size)
        """
        mask = jnp.where(w > 0, 1, 0)
        mask = jnp.repeat(mask, 5, axis=-1)
        # mask = hk.Reshape((sequence_length, ))(mask)
        h = h * mask[:, None]

        w = jnp.mean(h[0::5, :], axis=-2)
        a = jnp.mean(h[1::5, :], axis=-2)
        xyz = jnp.mean(h[2::5, :], axis=-2) + jnp.mean(h[3::5, :], axis=-2) + jnp.mean(h[4::5, :], axis=-2)

        h = jnp.concatenate([w, a, xyz], axis=0) 
        h = hk.Flatten()(h)

        h = jnp.concatenate([g, h, l], axis=0)

        h = jax.nn.relu(hk.Linear(hidden_sizes[0])(h))
        h = hk.dropout(hk.next_rng_key(), dropout_rate, h) if is_training else h  # Dropout after the first ReLU

        for hidden_size in hidden_sizes[1: -1]:
            h_dense = jax.nn.relu(hk.Linear(hidden_size)(h))
            h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense) if is_training else h_dense
            h = h + h_dense

        h = hk.Linear(hidden_sizes[-1])(h)
        h = jax.nn.relu(h)
        h = hk.Linear(num_classes)(h)
    
        return h
        
    g = jnp.ones(embed_size)
    w = jnp.ones(n_max)
    l = jnp.ones(6)
    h = jnp.zeros((sequence_length, outputs_size))

    params = network.init(key, g, l, w, h, True)
    return params, network.apply