import jax
import jax.numpy as jnp
import os
import optax
import math
from functools import partial

import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.lattice import norm_lattice


def make_ppo_loss_fn(logp_fn, eps_clip, beta=0.1):

    """
    PPO clipped objective function with KL divergence regularization
    PPO_loss = PPO-clip + beta  * KL(P || P_pretrain)

    Note that we only consider the logp_xyz and logp_l in the logp_fn
    """

    def ppo_loss_fn(params, key, x, old_logp, pretrain_logp, advantages):

        logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, key, *x, False)
        logp = logp_w + logp_xyz + logp_a + logp_l

        kl_loss = logp - pretrain_logp
        advantages = advantages - beta * kl_loss

        # Finding the ratio (pi_theta / pi_theta__old)
        ratios = jnp.exp(logp - old_logp)

        # Finding Surrogate Loss  
        surr1 = ratios * advantages
        surr2 = jax.lax.clamp(1-eps_clip, ratios, 1+eps_clip) * advantages

        # Final loss of clipped objective PPO
        ppo_loss = jnp.mean(jnp.minimum(surr1, surr2))

        return ppo_loss, (jnp.mean(kl_loss))
    
    return ppo_loss_fn


def train(key, optimizer, opt_state, spg_mask, loss_fn, logp_fn, batch_reward_fn, ppo_loss_fn, sample_crystal, params, epoch_finished, epochs, ppo_epochs, batchsize, valid_data, path):

    num_devices = jax.local_device_count()
    batch_per_device = batchsize // num_devices
    shape_prefix = (num_devices, batch_per_device)
    print("num_devices: ", num_devices)
    print("batch_per_device: ", batch_per_device)
    print("shape_prefix: ", shape_prefix)

    @partial(jax.pmap, axis_name="p", in_axes=(None, None, None, 0, 0, 0, 0), out_axes=(None, None, 0),)
    def step(params, key, opt_state, x, old_logp, pretrain_logp, advantages):
        value, grad = jax.value_and_grad(ppo_loss_fn, has_aux=True)(params, key, x, old_logp, pretrain_logp, advantages)
        grad = jax.lax.pmean(grad, axis_name="p")
        value = jax.lax.pmean(value, axis_name="p")
        grad = jax.tree_util.tree_map(lambda g_: g_ * -1.0, grad)  # invert gradient for maximization
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch f_mean f_err v_loss v_loss_w v_loss_a v_loss_xyz v_loss_l\n")
    pretrain_params = params
    logp_fn = jax.jit(logp_fn, static_argnums=7)
    loss_fn = jax.jit(loss_fn, static_argnums=7)
    
    for epoch in range(epoch_finished+1, epochs+1):

        key, subkey1, subkey2 = jax.random.split(key, 3)
        G = jax.random.choice(subkey1,
                              a=jnp.arange(1, 231, 1),
                              p=spg_mask,
                              shape=(batchsize, ))
        XYZ, A, W, _, L = sample_crystal(subkey2, params, G)

        x = (G, L, XYZ, A, W)
        rewards = - batch_reward_fn(x)  # inverse reward
        f_mean = jnp.mean(rewards)
        f_err = jnp.std(rewards) / jnp.sqrt(batchsize)

        # running average baseline
        baseline = f_mean if epoch == epoch_finished+1 else 0.95 * baseline + 0.05 * f_mean
        advantages = rewards - baseline

        f.write( ("%6d" + 2*"  %.6f") % (epoch, f_mean, f_err))

        G, L, XYZ, A, W = x
        L = norm_lattice(G, W, L)
        x = (G, L, XYZ, A, W)

        key, subkey1, subkey2 = jax.random.split(key, 3)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, subkey1, *x, False)
        old_logp = logp_w + logp_xyz + logp_a + logp_l

        logp_w, logp_xyz, logp_a, logp_l = logp_fn(pretrain_params, subkey2, *x, False)
        pretrain_logp = logp_w + logp_xyz + logp_a + logp_l

        x = jax.tree_util.tree_map(lambda _x: _x.reshape(shape_prefix + _x.shape[1:]), x)
        old_logp = old_logp.reshape(shape_prefix + old_logp.shape[1:])
        pretrain_logp = pretrain_logp.reshape(shape_prefix + pretrain_logp.shape[1:])
        advantages = advantages.reshape(shape_prefix + advantages.shape[1:])

        for _ in range(ppo_epochs):
            key, subkey = jax.random.split(key)
            params, opt_state, value = step(params, subkey, opt_state, x, old_logp, pretrain_logp, advantages)
            ppo_loss, (kl_loss) = value
            print(f"epoch {epoch}, loss {jnp.mean(ppo_loss):.6f} {jnp.mean(kl_loss):.6f}")

        valid_loss = 0.0 
        valid_aux = 0.0, 0.0, 0.0, 0.0
        num_samples = len(valid_data[0])
        num_batches = math.ceil(num_samples / batchsize)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)
            batch_data = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], valid_data)

            key, subkey = jax.random.split(key)
            loss, aux = loss_fn(params, subkey, *batch_data, False)
            valid_loss, valid_aux = jax.tree_util.tree_map(
                    lambda acc, i: acc + i,
                    (valid_loss, valid_aux), 
                    (loss, aux)
                    )

        valid_loss, valid_aux = jax.tree_util.tree_map(
                    lambda x: x/num_batches, 
                    (valid_loss, valid_aux)
                    ) 
        valid_loss_w, valid_loss_a, valid_loss_xyz, valid_loss_l = valid_aux
        f.write( (5*"  %.6f" + "\n") % (valid_loss,
                                        valid_loss_w, 
                                        valid_loss_a, 
                                        valid_loss_xyz, 
                                        valid_loss_l))

        if epoch % 5 == 0:
            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()

    return params, opt_state
