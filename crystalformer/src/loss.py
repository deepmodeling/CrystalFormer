import jax
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

from crystalformer.src.von_mises import von_mises_logpdf
from crystalformer.src.lattice import make_lattice_mask
from crystalformer.src.wyckoff import mult_table, fc_mask_table


def make_loss_fn(n_max, atom_types, wyck_types, Kx, Kl, transformer, lamb_a=1.0, lamb_w=1.0, lamb_l=1.0):
    """
    Args:
      n_max: maximum number of atoms in the unit cell
      atom_types: number of atom types
      wyck_types: number of wyckoff types
      Kx: number of von mises components for x, y, z
      Kl: number of Guassian mixture components for lattice parameters
      transformer: model
      lamb_a: weight for atom type loss
      lamb_w: weight for wyckoff position loss
      lamb_l: weight for lattice parameter loss

    Returns:
      loss_fn: loss function
      logp_fn: log probability function
    """
    
    coord_types = 3*Kx
    lattice_mask = make_lattice_mask()

    def compute_logp_x(h_x, X, fc_mask_x):
        x_logit, loc, kappa = jnp.split(h_x, [Kx, 2*Kx], axis=-1)
        x_loc = loc.reshape(n_max, Kx)
        kappa = kappa.reshape(n_max, Kx)
        logp_x = jax.vmap(von_mises_logpdf, (None, 1, 1), 1)((X-0.5)*2*jnp.pi, loc, kappa) # (n_max, Kx)
        logp_x = jax.scipy.special.logsumexp(x_logit + logp_x, axis=1) # (n_max, )
        logp_x = jnp.sum(jnp.where(fc_mask_x, logp_x, jnp.zeros_like(logp_x)))

        return logp_x

    @partial(jax.vmap, in_axes=(None, None, 0, 0, 0, 0, 0, None), out_axes=0) # batch 
    def logp_fn(params, key, G, L, XYZ, A, W, is_train):
        '''
        G: scalar 
        L: (6,) [a, b, c, alpha, beta, gamma] 
        XYZ: (n_max, 3)
        A: (n_max,)
        W: (n_max,)
        '''

        num_sites = jnp.sum(A!=0)
        M = mult_table[G-1, W]  # (n_max,) multplicities
        #num_atoms = jnp.sum(M)

        h = transformer(params, key, G, XYZ, A, W, M, is_train) # (5*n_max+1, ...)
        w_logit = h[0::5, :wyck_types] # (n_max+1, wyck_types) 
        w_logit = w_logit[:-1] # (n_max, wyck_types)
        a_logit = h[1::5, :atom_types] 
        h_x = h[2::5, :coord_types]
        h_y = h[3::5, :coord_types]
        h_z = h[4::5, :coord_types]

        logp_w = jnp.sum(w_logit[jnp.arange(n_max), W.astype(int)])
        logp_a = jnp.sum(a_logit[jnp.arange(n_max), A.astype(int)])

        X, Y, Z = XYZ[:, 0], XYZ[:, 1], XYZ[:,2]

        fc_mask = jnp.logical_and((W>0)[:, None], fc_mask_table[G-1, W]) # (n_max, 3)
        logp_x = compute_logp_x(h_x, X, fc_mask[:, 0])
        logp_y = compute_logp_x(h_y, Y, fc_mask[:, 1])
        logp_z = compute_logp_x(h_z, Z, fc_mask[:, 2])

        logp_xyz = logp_x + logp_y + logp_z

        l_logit, mu, sigma = jnp.split(h[1::5][num_sites, 
                                               atom_types:atom_types+Kl+2*6*Kl], [Kl, Kl+Kl*6], axis=-1)
        mu = mu.reshape(Kl, 6)
        sigma = sigma.reshape(Kl, 6)
        logp_l = jax.vmap(jax.scipy.stats.norm.logpdf, (None, 0, 0))(L,mu,sigma) #(Kl, 6)
        logp_l = jax.scipy.special.logsumexp(l_logit[:, None] + logp_l, axis=0) # (6,)
        logp_l = jnp.sum(jnp.where((lattice_mask[G-1]>0), logp_l, jnp.zeros_like(logp_l)))
        
        return logp_w, logp_xyz, logp_a, logp_l

    def loss_fn(params, key, G, L, XYZ, A, W, is_train):
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, key, G, L, XYZ, A, W, is_train)
        loss_w = -jnp.mean(logp_w)
        loss_xyz = -jnp.mean(logp_xyz)
        loss_a = -jnp.mean(logp_a)
        loss_l = -jnp.mean(logp_l)

        return loss_xyz + lamb_a* loss_a + lamb_w*loss_w + lamb_l*loss_l, (loss_w, loss_a, loss_xyz, loss_l)
        
    return loss_fn, logp_fn

if __name__=='__main__':
    from utils import GLXYZAW_from_file
    from transformer import make_transformer
    atom_types = 119
    n_max = 20
    wyck_types = 20
    Nf = 5
    Kx = 16
    Kl  = 4
    dropout_rate = 0.1 

    csv_file = '../data/mini.csv'
    G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max)

    key = jax.random.PRNGKey(42)

    params, transformer = make_transformer(key, Nf, Kx, Kl, n_max, 128, 4, 4, 8, 16, 16, atom_types, wyck_types, dropout_rate) 
 
    loss_fn, _ = make_loss_fn(n_max, atom_types, wyck_types, Kx, Kl, transformer)
    
    value = jax.jit(loss_fn, static_argnums=7)(params, key, G[:1], L[:1], XYZ[:1], A[:1], W[:1], True)
    print (value)

    value = jax.jit(loss_fn, static_argnums=7)(params, key, G[:1], L[:1], XYZ[:1]+1.0, A[:1], W[:1], True)
    print (value)
