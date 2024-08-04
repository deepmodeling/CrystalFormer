import jax
import jax.numpy as jnp
from functools import partial

from crystalformer.src.von_mises import sample_von_mises
from crystalformer.src.lattice import symmetrize_lattice
from crystalformer.src.wyckoff import mult_table, symops

def project_xyz(g, w, x, idx):
    '''
    apply the randomly sampled Wyckoff symmetry op to sampled fc, which 
    should be (or close to) the first WP
    '''
    op = symops[g-1, w, idx].reshape(3, 4)
    affine_point = jnp.array([*x, 1]) # (4, )
    x = jnp.dot(op, affine_point)  # (3, )
    x -= jnp.floor(x)
    return x 

@partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, 0), out_axes=0) # batch 
def inference(model, params, g, W, A, X, Y, Z):
    XYZ = jnp.concatenate([X[:, None],
                           Y[:, None],
                           Z[:, None]
                           ], 
                           axis=-1)
    M = mult_table[g-1, W]  
    return model(params, None, g, XYZ, A, W, M, False)

def sample_top_p(key, logits, p, temperature):
    '''
    drop remaining logits once the cumulative_probs is larger than p
    for very small p, we may drop everything excet the leading logit
    for very large p, we will keep everything
    '''
    assert (logits.ndim == 2)
    if p < 1.0:
        batchsize = logits.shape[0]
        batch_idx = jnp.arange(batchsize)[:, None]
        indices = jnp.argsort(logits, axis=1)[:, ::-1]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(logits[batch_idx, indices], axis=1), axis=1)
        mask =  jnp.concatenate([jnp.zeros((batchsize, 1)),  # at least keep the leading one
                                (cumulative_probs > p)[:, :-1]
                                ], axis=1)
        mask = mask.at[batch_idx, indices].set(mask) # logits to be dropped
        logits = logits + jnp.where(mask, -1e10, 0.0)
    
    samples = jax.random.categorical(key, logits/temperature, axis=1)
    return samples

def sample_x(key, h_x, Kx, top_p, temperature, batchsize):
    coord_types = 3*Kx 
    x_logit, loc, kappa = jnp.split(h_x[:, :coord_types], [Kx, 2*Kx], axis=-1)
    key, key_k, key_x = jax.random.split(key, 3)
    k = sample_top_p(key_k, x_logit, top_p, temperature)
    loc = loc.reshape(batchsize, Kx)[jnp.arange(batchsize), k]
    kappa = kappa.reshape(batchsize, Kx)[jnp.arange(batchsize), k]
    x = sample_von_mises(key_x, loc, kappa/temperature, (batchsize,))
    x = (x+ jnp.pi)/(2.0*jnp.pi) # wrap into [0, 1]
    return key, x 

@partial(jax.jit, static_argnums=(1, 3, 4, 5, 6, 7, 8, 9, 12, 14))
def sample_crystal(key, transformer, params, n_max, batchsize, atom_types, wyck_types, Kx, Kl, g, w_mask, atom_mask, top_p, temperature, T1, constraints):
       
    def body_fn(i, state):
        key, W, A, X, Y, Z, L = state 

        # (1) W 
        w_logit = inference(transformer, params, g, W, A, X, Y, Z)[:, 5*i] # (batchsize, output_size)
        w_logit = w_logit[:, :wyck_types]
    
        key, subkey = jax.random.split(key)
        if w_mask is not None:
            w_logit = w_logit.at[:, w_mask[i]].set(w_logit[:, w_mask[i]] + 1e10)
        w = sample_top_p(subkey, w_logit, top_p, temperature)
        W = W.at[:, i].set(w)

        # (2) A
        h_al = inference(transformer, params, g, W, A, X, Y, Z)[:, 5*i+1] # (batchsize, output_size)
        a_logit = h_al[:, :atom_types]
    
        key, subkey = jax.random.split(key)
        a_logit = a_logit + jnp.where(atom_mask[i, :], 1e10, 0.0) # enhance the probability of masked atoms (do not need to normalize since we only use it for sampling, not computing logp)
        _temp = jax.lax.cond(i==0,
                                true_fun=lambda x: jnp.array(T1, dtype=float),
                                false_fun=lambda x: temperature,
                                operand=None)
        _a = sample_top_p(subkey, a_logit, top_p, _temp)  # use T1 for the first atom type
        a = jax.lax.cond(constraints[i] < i,
                            lambda x, y: y,
                            lambda x, y: x,
                            _a, A[:, constraints[i]])
        A = A.at[:, i].set(a)
    
        lattice_params = h_al[:, atom_types:atom_types+Kl+2*6*Kl]
        L = L.at[:, i].set(lattice_params)
    
        # (3) X
        h_x = inference(transformer, params, g, W, A, X, Y, Z)[:, 5*i+2] # (batchsize, output_size)
        key, x = sample_x(key, h_x, Kx, top_p, temperature, batchsize)
    
        # project to the first WP
        xyz = jnp.concatenate([x[:, None], 
                                jnp.zeros((batchsize, 1)), 
                                jnp.zeros((batchsize, 1)), 
                                ], axis=-1) 
        xyz = jax.vmap(project_xyz, in_axes=(None, 0, 0, None), out_axes=0)(g, w, xyz, 0) 
        x = xyz[:, 0]
        X = X.at[:, i].set(x)
    
        # (4) Y
        h_y = inference(transformer, params, g, W, A, X, Y, Z)[:, 5*i+3] # (batchsize, output_size)
        key, y = sample_x(key, h_y, Kx, top_p, temperature, batchsize)
        
        # project to the first WP
        xyz = jnp.concatenate([X[:, i][:, None], 
                                y[:, None], 
                                jnp.zeros((batchsize, 1)), 
                                ], axis=-1) 
        xyz = jax.vmap(project_xyz, in_axes=(None, 0, 0, None), out_axes=0)(g, w, xyz, 0) 
        y = xyz[:, 1]
        Y = Y.at[:, i].set(y)
    
        # (5) Z
        h_z = inference(transformer, params, g, W, A, X, Y, Z)[:, 5*i+4] # (batchsize, output_size)
        key, z = sample_x(key, h_z, Kx, top_p, temperature, batchsize)
        
        # project to the first WP
        xyz = jnp.concatenate([X[:, i][:, None], 
                                Y[:, i][:, None], 
                                z[:, None], 
                                ], axis=-1) 
        xyz = jax.vmap(project_xyz, in_axes=(None, 0, 0, None), out_axes=0)(g, w, xyz, 0) 
        z = xyz[:, 2]
        Z = Z.at[:, i].set(z)

        return key, W, A, X, Y, Z, L
        
    # we waste computation time by always working with the maximum length sequence, but we save compilation time
    W = jnp.zeros((batchsize, n_max), dtype=int)
    A = jnp.zeros((batchsize, n_max), dtype=int)
    X = jnp.zeros((batchsize, n_max))
    Y = jnp.zeros((batchsize, n_max))
    Z = jnp.zeros((batchsize, n_max))
    L = jnp.zeros((batchsize, n_max, Kl+2*6*Kl)) # we accumulate lattice params and sample lattice after

    key, W, A, X, Y, Z, L = jax.lax.fori_loop(0, n_max, body_fn, (key, W, A, X, Y, Z, L))
   
    M = mult_table[g-1, W]
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
    L = jax.vmap(symmetrize_lattice, (None, 0))(g, L)  

    XYZ = jnp.concatenate([X[..., None], 
                           Y[..., None], 
                           Z[..., None]
                           ], 
                           axis=-1)

    return XYZ, A, W, M, L


def make_update_lattice(transformer, params, atom_types, Kl, top_p, temperature):

    @jax.jit
    def update_lattice(key, G, XYZ, A, W):

        num_sites = jnp.sum(A!=0, axis=1) # (batchsize, )
        M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) # (batchsize, n_max)
        #num_atoms = jnp.sum(M)
        batchsize = XYZ.shape[0]

        h = jax.vmap(transformer, in_axes=(None, None, 0, 0, 0, 0, 0, None))(params, key, G, XYZ, A, W, M, False)

        l_logit, mu, sigma = jax.vmap(lambda h, num_site: jnp.split(h[1::5][num_site,
                                                                    atom_types:atom_types+Kl+2*6*Kl], [Kl, Kl+Kl*6], axis=-1),
                                        in_axes=(0, 0))(h, num_sites)
        
        key, key_k, key_l = jax.random.split(key, 3)
        # k is (batchsize, ) integer array whose value in [0, Kl) 
        k = sample_top_p(key_k, l_logit, top_p, temperature)

        mu = mu.reshape(batchsize, Kl, 6)
        mu = mu[jnp.arange(batchsize), k]       # (batchsize, 6)
        sigma = sigma.reshape(batchsize, Kl, 6)
        sigma = sigma[jnp.arange(batchsize), k] # (batchsize, 6)
        L = jax.random.normal(key_l, (batchsize, 6)) * sigma*jnp.sqrt(temperature) + mu # (batchsize, 6)
        
        num_atoms = jnp.sum(M, axis=1)
        #scale length according to atom number since we did reverse of that when loading data
        length, angle = jnp.split(L, 2, axis=-1)
        length = length*num_atoms[:, None]**(1/3)
        angle = angle * (180.0 / jnp.pi) # to deg
        L = jnp.concatenate([length, angle], axis=-1)

        #impose space group constraint to lattice params
        L = jax.vmap(symmetrize_lattice, (0, 0))(G, L)  

        return L

    return update_lattice