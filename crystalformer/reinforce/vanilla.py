import jax
import jax.numpy as jnp
import os
import optax

import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.lattice import norm_lattice


def make_reinforce_loss(batch_logp, batch_reward_fn):

    def loss(params, key, x, is_train):
        f = batch_reward_fn(x)
        f = jax.lax.stop_gradient(f)

        f_mean = jnp.mean(f)
        f_std = jnp.std(f)/jnp.sqrt(f.shape[0])

        G, L, XYZ, A, W = x
        L = norm_lattice(G, W, L)
        x = (G, L, XYZ, A, W)
        
        # TODO: now only support for crystalformer logp
        logp_w, logp_xyz, logp_a, logp_l = jax.jit(batch_logp, static_argnums=7)(params, key, *x, is_train)
        entropy = logp_w + logp_xyz + logp_a + logp_l

        return -jnp.mean((f - f_mean) * entropy), (-f_mean, f_std)

    return loss


def train(key, optimizer, opt_state, loss_fn, sample_crystal, params, epoch_finished, epochs, batchsize, path):
           
    def update(params, key, opt_state, spacegroup):
        @jax.jit
        def apply_update(grad, params, opt_state):
            grad = jax.tree_util.tree_map(lambda g_: g_ * -1.0, grad)  # invert gradient for maximization
            updates, opt_state = optimizer.update(grad, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state

        key, sample_key, loss_key = jax.random.split(key, 3)
        XYZ, A, W, M, L = sample_crystal(sample_key, params=params, g=spacegroup, batchsize=batchsize) 
        G = spacegroup * jnp.ones((batchsize), dtype=int)
        x = (G, L, XYZ, A, W)
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(params, loss_key, x, True)
        params, opt_state = apply_update(grad, params, opt_state)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch f_mean f_err\n")
 
    for epoch in range(epoch_finished+1, epochs):
        key, subkey = jax.random.split(key)
        params, opt_state, value = update(params, subkey, opt_state, spacegroup=1) # TODO: only for P1 for now
        _, (f_mean, f_err) = value

        f.write( ("%6d" + 2*"  %.6f" + "\n") % (epoch, f_mean, f_err))
        
        if epoch % 5 == 0:
            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return params, opt_state
