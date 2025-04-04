import jax
import jax.numpy as jnp
from functools import partial

from crystalformer.src.lattice import symmetrize_lattice
from crystalformer.src.wyckoff import mult_table
from crystalformer.src.sample import project_xyz, sample_top_p, sample_x


@partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0), out_axes=0) # batch 
def inference(model, params, G, W, A, X, Y, Z):
    XYZ = jnp.concatenate([X[:, None],
                           Y[:, None],
                           Z[:, None]
                           ], 
                           axis=-1)
    M = mult_table[G-1, W]
    return model(params, None, G, XYZ, A, W, M, False)


def make_sample_crystal(transformer, n_max, atom_types, wyck_types, Kx, Kl):
    """
    sample fucntion for different space group
    """

    @partial(jax.jit, static_argnums=(4, 5))
    def sample_crystal(key,  params,  G, atom_mask, top_p, temperature):
        
        def body_fn(i, state):
            key, W, A, X, Y, Z, L = state 

            # (1) W 
            w_logit = inference(transformer, params, G, W, A, X, Y, Z)[:, 5*i] # (batchsize, output_size)
            w_logit = w_logit[:, :wyck_types]
        
            key, subkey = jax.random.split(key)
            w = sample_top_p(subkey, w_logit, top_p, temperature)
            W = W.at[:, i].set(w)

            # (2) A
            h_al = inference(transformer, params, G, W, A, X, Y, Z)[:, 5*i+1] # (batchsize, output_size)
            a_logit = h_al[:, :atom_types]
        
            key, subkey = jax.random.split(key)
            a_logit = a_logit + jnp.where(atom_mask[i, :], 0.0, -1e10) # enhance the probability of masked atoms (do not need to normalize since we only use it for sampling, not computing logp)
            a = sample_top_p(subkey, a_logit, top_p, temperature)  # use T1 for the first atom type
            A = A.at[:, i].set(a)
        
            lattice_params = h_al[:, atom_types:atom_types+Kl+2*6*Kl]
            L = L.at[:, i].set(lattice_params)
        
            # (3) X
            h_x = inference(transformer, params, G, W, A, X, Y, Z)[:, 5*i+2] # (batchsize, output_size)
            key, x = sample_x(key, h_x, Kx, top_p, temperature, batchsize)
        
            # project to the first WP
            xyz = jnp.concatenate([x[:, None], 
                                    jnp.zeros((batchsize, 1)), 
                                    jnp.zeros((batchsize, 1)), 
                                    ], axis=-1) 
            xyz = jax.vmap(project_xyz, in_axes=(0, 0, 0, None), out_axes=0)(G, w, xyz, 0) 
            x = xyz[:, 0]
            X = X.at[:, i].set(x)
        
            # (4) Y
            h_y = inference(transformer, params, G, W, A, X, Y, Z)[:, 5*i+3] # (batchsize, output_size)
            key, y = sample_x(key, h_y, Kx, top_p, temperature, batchsize)
            
            # project to the first WP
            xyz = jnp.concatenate([X[:, i][:, None], 
                                   y[:, None], 
                                   jnp.zeros((batchsize, 1)), 
                                    ], axis=-1) 
            xyz = jax.vmap(project_xyz, in_axes=(0, 0, 0, None), out_axes=0)(G, w, xyz, 0) 
            y = xyz[:, 1]
            Y = Y.at[:, i].set(y)
        
            # (5) Z
            h_z = inference(transformer, params, G, W, A, X, Y, Z)[:, 5*i+4] # (batchsize, output_size)
            key, z = sample_x(key, h_z, Kx, top_p, temperature, batchsize)
            
            # project to the first WP
            xyz = jnp.concatenate([X[:, i][:, None], 
                                   Y[:, i][:, None], 
                                   z[:, None], 
                                    ], axis=-1) 
            xyz = jax.vmap(project_xyz, in_axes=(0, 0, 0, None), out_axes=0)(G, w, xyz, 0) 
            z = xyz[:, 2]
            Z = Z.at[:, i].set(z)

            return key, W, A, X, Y, Z, L
            
        # we waste computation time by always working with the maximum length sequence, but we save compilation time
        batchsize = G.shape[0]
        W = jnp.zeros((batchsize, n_max), dtype=int)
        A = jnp.zeros((batchsize, n_max), dtype=int)
        X = jnp.zeros((batchsize, n_max))
        Y = jnp.zeros((batchsize, n_max))
        Z = jnp.zeros((batchsize, n_max))
        L = jnp.zeros((batchsize, n_max, Kl+2*6*Kl)) # we accumulate lattice params and sample lattice after

        key, W, A, X, Y, Z, L = jax.lax.fori_loop(0, n_max, body_fn, (key, W, A, X, Y, Z, L))
    
        M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W)
        num_sites = jnp.sum(A!=0, axis=1)
        num_atoms = jnp.sum(M, axis=1)
        
        l_logit, mu, sigma = jnp.split(L[jnp.arange(batchsize), num_sites, :], [Kl, Kl+6*Kl], axis=-1)

        key, key_k, key_l = jax.random.split(key, 3)
        # k is (batchsize, ) integer array whose value in [0, Kl) 
        k = sample_top_p(key_k, l_logit, top_p, temperature)

        mu = mu.reshape(batchsize, Kl, 6)
        mu = mu[jnp.arange(batchsize), k]       # (batchsize, 6)
        sigma = sigma.reshape(batchsize, Kl, 6)
        sigma = sigma[jnp.arange(batchsize), k] # (batchsize, 6)
        L = jax.random.normal(key_l, (batchsize, 6)) * sigma*jnp.sqrt(temperature) + mu # (batchsize, 6)
        
        #scale length according to atom number since we did reverse of that when loading data
        length, angle = jnp.split(L, 2, axis=-1)
        length = length*num_atoms[:, None]**(1/3)
        angle = angle * (180.0 / jnp.pi) # to deg
        L = jnp.concatenate([length, angle], axis=-1)

        #impose space group constraint to lattice params
        L = jax.vmap(symmetrize_lattice, (0, 0))(G, L)  

        XYZ = jnp.concatenate([X[..., None], 
                               Y[..., None], 
                               Z[..., None]
                            ], 
                            axis=-1)

        return XYZ, A, W, M, L

    return sample_crystal


if __name__ == "__main__":
    from crystalformer.src.transformer import make_transformer
    atom_types = 119
    n_max = 21
    wyck_types = 28
    Nf = 5
    Kx = 16
    Kl  = 4
    dropout_rate = 0.1 

    key = jax.random.PRNGKey(42)
    params, transformer = make_transformer(key, Nf, Kx, Kl, n_max, 128, 4, 4, 8, 16, 16, atom_types, wyck_types, dropout_rate) 
    sample_crystal = make_sample_crystal(transformer, n_max, atom_types, wyck_types, Kx, Kl)
    atom_mask = jnp.zeros((n_max, atom_types))

    G = jnp.array([2, 12, 62, 139, 166, 194, 225])
    XYZ, A, W, M, L = sample_crystal(key, params, G, atom_mask, 1.0, 1.0)
    print(XYZ.shape, A.shape, W.shape, M.shape, L.shape)
    print ("G:\n", G)  # space group
    print ("XYZ:\n", XYZ)  # fractional coordinate 
    print ("A:\n", A)  # element type
    print ("W:\n", W)  # Wyckoff positions
    print ("M:\n", M)  # multiplicity 
    print ("N:\n", M.sum(axis=-1)) # total number of atoms
    print ("L:\n", L)  # lattice
 