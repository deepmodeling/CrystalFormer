'''
https://github.com/google-deepmind/dm-haiku/blob/main/examples/transformer/model.py
'''
import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np

from crystalformer.src.attention import MultiHeadAttention
from crystalformer.src.wyckoff import wmax_table, dof0_table

def make_transformer(key, Nf, Kx, Kl, n_max, h0_size, num_layers, num_heads, key_size, model_size, embed_size, atom_types, wyck_types, dropout_rate, widening_factor=4, sigmamin=1e-3):
    
    coord_types = 3*Kx
    lattice_types = Kl+2*6*Kl
    output_size = np.max(np.array([atom_types+lattice_types, coord_types, wyck_types]))

    def renormalize(h_x):
        n = h_x.shape[0]
        x_logit, x_loc, x_kappa = jnp.split(h_x[:, :coord_types], [Kx, 2*Kx], axis=-1)
        x_logit -= jax.scipy.special.logsumexp(x_logit, axis=1)[:, None] 
        x_kappa = jax.nn.softplus(x_kappa) 
        h_x = jnp.concatenate([x_logit, x_loc, x_kappa, jnp.zeros((n, output_size - coord_types))], axis=-1)  
        return h_x

    @hk.transform_with_state
    def network(G, XYZ, A, W, M, is_train):
        '''
        Args:
            G: scalar integer for space group id 1-230
            XYZ: (n, 3) fractional coordinates
            A: (n, )  element type 
            W: (n, )  wyckoff position index
            M: (n, )  multiplicities
            is_train: bool 
        Returns: 
            h: (5n+1, output_types)
        '''
        
        assert (XYZ.ndim == 2 )
        assert (XYZ.shape[0] == A.shape[0])
        assert (XYZ.shape[1] == 3)

        n = XYZ.shape[0]
        X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:,2]

        w_max = wmax_table[G-1]
        initializer = hk.initializers.TruncatedNormal(0.01)
        
        g_embeddings = hk.get_parameter('g_embeddings', [230, embed_size], init=initializer)[G-1]
        w_embeddings = hk.get_parameter('w_embeddings', [wyck_types, embed_size], init=initializer)[W]
        a_embeddings = hk.get_parameter('a_embeddings', [atom_types, embed_size], init=initializer)[A]

        _g_embeddings = hk.get_state("_g_embeddings", shape=g_embeddings.shape, dtype=float, init=jnp.ones)
        hk.set_state("_g_embeddings", g_embeddings)

        if h0_size >0:
            # compute w_logits depending on g 
            w_logit = hk.Sequential([hk.Linear(h0_size, w_init=initializer),
                                      jax.nn.gelu,
                                      hk.Linear(wyck_types, w_init=initializer)]
                                     )(g_embeddings)
        else:
            # w_logit of the first atom is simply a table
            w_params = hk.get_parameter('w_params', [230, wyck_types], init=initializer)
            w_logit = w_params[G-1]

        # (1) the first atom should not be the pad atom
        # (2) mask out unavaiable position for the given spacegroup
        w_mask = jnp.logical_and(jnp.arange(wyck_types)>0, jnp.arange(wyck_types)<=w_max)
        w_logit = jnp.where(w_mask, w_logit, w_logit-1e10)
        # normalization
        w_logit -= jax.scipy.special.logsumexp(w_logit) # (wyck_types, )
        
        h0 = jnp.concatenate([w_logit[None, :], 
                             jnp.zeros((1, output_size-wyck_types))], axis=-1)  # (1, output_size)
        if n == 0: return h0

        mask = jnp.tril(jnp.ones((1, 5*n, 5*n))) # mask for the attention matrix

        hW = jnp.concatenate([g_embeddings[None, :].repeat(n, axis=0),  # (n, embed_size)
                              w_embeddings,                             # (n, embed_size)
                              M.reshape(n, 1), # (n, 1)
                              ], axis=1) # (n, ...)
        hW = hk.Linear(model_size, w_init=initializer)(hW)  # (n, model_size)

        hA = jnp.concatenate([g_embeddings[None, :].repeat(n, axis=0),  # (n, embed_size)
                              a_embeddings,                             # (n, embed_size)
                             ], axis=1) # (n, ...)
        hA = hk.Linear(model_size, w_init=initializer)(hA)  # (n, model_size)

        hX = jnp.concatenate([g_embeddings[None, :].repeat(n, axis=0), 
                             ] + 
                             [fn(2*jnp.pi*X[:, None]*f) for f in range(1, Nf+1) for fn in (jnp.sin, jnp.cos)]
                             , axis=1) # (n, ...)
        hX = hk.Linear(model_size, w_init=initializer)(hX)  # (n, model_size)

        hY = jnp.concatenate([g_embeddings[None, :].repeat(n, axis=0), 
                             ] +
                             [fn(2*jnp.pi*Y[:, None]*f) for f in range(1, Nf+1) for fn in (jnp.sin, jnp.cos)]
                             , axis=1) # (n, ...)
        hY = hk.Linear(model_size, w_init=initializer)(hY)  # (n, model_size)

        hZ = jnp.concatenate([g_embeddings[None, :].repeat(n, axis=0), 
                             ]+
                             [fn(2*jnp.pi*Z[:, None]*f) for f in range(1, Nf+1) for fn in (jnp.sin, jnp.cos)]
                             , axis=1) # (n, ...)
        hZ = hk.Linear(model_size, w_init=initializer)(hZ)  # (n, model_size)

        # interleave the three matrices
        h = jnp.concatenate([hW[:, None, :], 
                             hA[:, None, :],
                             hX[:, None, :],
                             hY[:, None, :],
                             hZ[:, None, :]
                             ], axis=1) # (n, 5, model_size)
        h = h.reshape(5*n, -1)                                         # (5*n, model_size)

        positional_embeddings = hk.get_parameter(
                        'positional_embeddings', [5*n_max, model_size], init=initializer)
        h = h + positional_embeddings[:5*n, :]

        del hW
        del hA
        del hX
        del hY
        del hZ

        for _ in range(num_layers):
            attn_block = MultiHeadAttention(num_heads=num_heads,
                                               key_size=key_size,
                                               model_size=model_size,
                                               w_init =initializer, 
                                               dropout_rate =dropout_rate
                                              )
            h_norm = _layer_norm(h)
            h_attn = attn_block(h_norm, h_norm, h_norm, 
                                mask=mask, is_train=is_train)
            if is_train: 
                h_attn = hk.dropout(hk.next_rng_key(), dropout_rate, h_attn)
            h = h + h_attn

            dense_block = hk.Sequential([hk.Linear(widening_factor * model_size, w_init=initializer),
                                         jax.nn.gelu,
                                         hk.Linear(model_size, w_init=initializer)]
                                         )
            h_norm = _layer_norm(h)
            h_dense = dense_block(h_norm)
            if is_train:
                h_dense = hk.dropout(hk.next_rng_key(), dropout_rate, h_dense)
            h = h + h_dense

        h = _layer_norm(h)
        last_hidden_state = hk.get_state("last_hidden_state", shape=h.shape, dtype=float, init=jnp.ones)
        hk.set_state("last_hidden_state", h)

        h = hk.Linear(output_size, w_init=initializer)(h) # (5*n, output_size)
        
        h = h.reshape(n, 5, -1)
        h_al, h_x, h_y, h_z, w_logit = h[:, 0, :], h[:, 1, :], h[:, 2, :], h[:, 3, :], h[:, 4, :]
    
        # handle coordinate related params 
        h_x = renormalize(h_x)
        h_y = renormalize(h_y)
        h_z = renormalize(h_z)
        
        # we now do all kinds of masks to a_logit and w_logit
        
        a_logit = h_al[:, :atom_types]
        w_logit = w_logit[:, :wyck_types]
        
        # (1) impose the constrain that W_0 <= W_1 <= W_2 
        # while for Wyckoff points with zero dof it is even stronger W_0 < W_1 
        w_mask_less_equal = jnp.arange(1, wyck_types).reshape(1, wyck_types-1) < W[:, None]
        w_mask_less = jnp.arange(1, wyck_types).reshape(1, wyck_types-1) <= W[:, None]
        w_mask = jnp.where((dof0_table[G-1, W])[:, None], w_mask_less, w_mask_less_equal) # (n, wyck_types-1)

        w_mask = jnp.concatenate([jnp.zeros((n, 1)), w_mask], axis=1) # (n, wyck_types)
        w_logit = w_logit - jnp.where(w_mask, 1e10, 0.0)
        w_logit -= jax.scipy.special.logsumexp(w_logit, axis=1)[:, None] # normalization

        # (2) # enhance the probability of pad atoms if there is already a type 0 atom 
        w_mask = jnp.concatenate(
                [ jnp.where(W==0, jnp.ones((n)), jnp.zeros((n))).reshape(n, 1), 
                  jnp.zeros((n, wyck_types-1))
                ], axis = 1 )  # (n, wyck_types) mask = 1 for those locations to place pad atoms of type 0
        w_logit = w_logit + jnp.where(w_mask, 1e10, 0.0)
        w_logit -= jax.scipy.special.logsumexp(w_logit, axis=1)[:, None] # normalization

        # (3) mask out unavaiable position after w_max for the given spacegroup
        w_logit = jnp.where(jnp.arange(wyck_types)<=w_max, w_logit, w_logit-1e10)
        w_logit -= jax.scipy.special.logsumexp(w_logit, axis=1)[:, None] # normalization

        # (4) if w !=0 the mask out the pad atom, otherwise mask out true atoms
        a_mask = jnp.concatenate(
                 [(W>0).reshape(n, 1), 
                 (W==0).reshape(n, 1).repeat(atom_types-1, axis=1) 
                 ], axis = 1 )  # (n, atom_types) mask = 1 for those locations to be masked out
        a_logit = a_logit + jnp.where(a_mask, -1e10, 0.0)
        a_logit -= jax.scipy.special.logsumexp(a_logit, axis=1)[:, None] # normalization
            
        w_logit = jnp.concatenate([w_logit, 
                                   jnp.zeros((n, output_size - wyck_types))
                                   ], axis = -1) 
        
        # now move on to lattice part 
        l_logit, mu, sigma = jnp.split(h_al[:, atom_types:atom_types+lattice_types], 
                                                                 [Kl, 
                                                                  Kl+Kl*6, 
                                                                  ], axis=-1)

        # normalization
        l_logit -= jax.scipy.special.logsumexp(l_logit, axis=1)[:, None] 
        # ensure positivity
        sigma = jax.nn.softplus(sigma) + sigmamin

        h_al = jnp.concatenate([a_logit, l_logit, mu, sigma, 
                               jnp.zeros((n, output_size - atom_types - lattice_types))
                               ], axis=-1) # (n, output_size)
        
        # finally assemble everything together
        h = jnp.concatenate([h_al[:, None, :], 
                             h_x[:, None, :], 
                             h_y[:, None, :],
                             h_z[:, None, :],
                             w_logit[:, None, :]
                             ], axis=1) # (n, 5, output_size)
        h = h.reshape(5*n, output_size) # (5*n, output_size)

        h = jnp.concatenate( [h0, h], axis = 0) # (5*n+1, output_size)

        return h
 

    G = jnp.array(123)
    XYZ = jnp.zeros((n_max, 3), dtype=int) 
    A = jnp.zeros((n_max, ), dtype=int) 
    W = jnp.zeros((n_max, ), dtype=int) 
    M = jnp.zeros((n_max, ), dtype=int) 

    params, state = network.init(key, G, XYZ, A, W, M, True)
    return params, state, network.apply

def _layer_norm(x: jax.Array) -> jax.Array:
    """Applies a unique LayerNorm to `x` with default settings."""
    ln = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
    return ln(x)
