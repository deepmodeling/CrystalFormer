import jax
import jax.numpy as jnp
import optax
import os
import math

from crystalformer.src.utils import shuffle
import crystalformer.src.checkpoint as checkpoint


def make_dpo_loss(logp_fn, beta, label_smoothing=0.0, gamma=0.0, ipo=False):
    
    # https://github.com/eric-mitchell/direct-preference-optimization/blob/f8b8c0f49dc92a430bae41585f9d467d3618fe2f/trainers.py#L45-L87
    def dpo_logp_fn(policy_chosen_logps,
                    policy_rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps):
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps

        logits = pi_logratios - ref_logratios

        if ipo:
            losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # label_smoothing=0 gives original DPO 
            losses = -jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing) - jax.nn.log_sigmoid(-beta * logits) * label_smoothing
        return jnp.mean(losses)
    
    def loss_fn(params, key, x_w, x_l, ref_chosen_logps, ref_rejected_logps):
        key, subkey = jax.random.split(key)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, subkey, *x_w, False)
        policy_chosen_logps = logp_w + logp_xyz + logp_a + logp_l

        key, subkey = jax.random.split(key)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, subkey, *x_l, False)
        policy_rejected_logps = logp_w + logp_xyz + logp_a + logp_l

        dpo_loss = dpo_logp_fn(policy_chosen_logps,
                               policy_rejected_logps,
                               ref_chosen_logps,
                               ref_rejected_logps)
        loss = dpo_loss - gamma * jnp.mean(policy_chosen_logps)

        return loss, (dpo_loss, jnp.mean(policy_chosen_logps), jnp.mean(policy_rejected_logps))

    return loss_fn


def train(key, optimizer, opt_state, dpo_loss_fn, logp_fn, params, epoch_finished, epochs, batchsize, chosen_data, rejected_data, path, val_ratio=0.2):

    @jax.jit
    def step(params, key, opt_state, x_w, x_l, ref_chosen_logps, ref_rejected_logps):
        value, grad = jax.value_and_grad(dpo_loss_fn, has_aux=True)(params, key, x_w, x_l, ref_chosen_logps, ref_rejected_logps)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch loss dpo_loss chosen_logp rejected_logp v_loss v_dpo_loss v_chosen_logp v_rejected_logp\n")
    ref_params = params
    logp_fn = jax.jit(logp_fn, static_argnums=7)

    ref_chosen_logps = jnp.array([])
    ref_rejected_logps = jnp.array([])
    _, chosen_L, _, _, _ = chosen_data
    num_samples = len(chosen_L)
    num_batches = math.ceil(num_samples / batchsize)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batchsize
        end_idx = min(start_idx + batchsize, num_samples)
        key, subkey1, subkey2 = jax.random.split(key, 3)

        data = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], chosen_data)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(ref_params, subkey1, *data, False)
        logp = logp_w + logp_xyz + logp_a + logp_l
        ref_chosen_logps = jnp.append(ref_chosen_logps, logp, axis=0)

        data = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], rejected_data)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(ref_params, subkey2, *data, False)
        logp = logp_w + logp_xyz + logp_a + logp_l
        ref_rejected_logps = jnp.append(ref_rejected_logps, logp, axis=0)

    print(ref_chosen_logps.shape, ref_rejected_logps.shape)
    print(f"ref_chosen_logps: {jnp.mean(ref_chosen_logps)}, ref_rejected_logps: {jnp.mean(ref_rejected_logps)}")
    print("Finished calculating reference logp")

    # Shuffle the data
    key, subkey = jax.random.split(key)
    idx = jax.random.permutation(subkey, jnp.arange(num_samples))
    chosen_data = jax.tree_util.tree_map(lambda x: x[idx], chosen_data)
    rejected_data = jax.tree_util.tree_map(lambda x: x[idx], rejected_data)
    ref_chosen_logps = ref_chosen_logps[idx]
    ref_rejected_logps = ref_rejected_logps[idx]

    # Split the data into training and validation
    num_val_samples = int(num_samples * val_ratio)
    num_train_samples = num_samples - num_val_samples
    print("num_train_samples: %d, num_val_samples: %d" % (num_train_samples, num_val_samples))

    train_chosen_data = jax.tree_util.tree_map(lambda x: x[:num_train_samples], chosen_data)
    train_rejected_data = jax.tree_util.tree_map(lambda x: x[:num_train_samples], rejected_data)
    train_ref_chosen_logps = ref_chosen_logps[:num_train_samples]
    train_ref_rejected_logps = ref_rejected_logps[:num_train_samples]

    val_chosen_data = jax.tree_util.tree_map(lambda x: x[num_train_samples:], chosen_data)
    val_rejected_data = jax.tree_util.tree_map(lambda x: x[num_train_samples:], rejected_data)
    val_ref_chosen_logps = ref_chosen_logps[num_train_samples:]
    val_ref_rejected_logps = ref_rejected_logps[num_train_samples:]

    
    for epoch in range(epoch_finished+1, epochs+1):
        key, subkey = jax.random.split(key)
        train_chosen_data = shuffle(subkey, train_chosen_data)
        train_rejected_data = shuffle(subkey, train_rejected_data)  

        idx = jax.random.permutation(subkey, jnp.arange(len(train_ref_chosen_logps)))
        train_ref_chosen_logps = train_ref_chosen_logps[idx]
        train_ref_rejected_logps = train_ref_rejected_logps[idx]

        train_loss = 0.0
        train_dpo_loss = 0.0
        train_policy_chosen_logps = 0.0
        train_policy_rejected_logps = 0.0
        _, chosen_L, _, _, _ = train_chosen_data
        num_samples = chosen_L.shape[0]
        num_batches = math.ceil(num_samples / batchsize)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)
            x_w = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx],  train_chosen_data)
            x_l = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], train_rejected_data)
            ref_chosen_logps_batch = train_ref_chosen_logps[start_idx:end_idx]
            ref_rejected_logps_batch = train_ref_rejected_logps[start_idx:end_idx]

            key, subkey = jax.random.split(key)
            params, opt_state, value = step(params, subkey, opt_state, x_w, x_l, ref_chosen_logps_batch, ref_rejected_logps_batch)
            loss, (dpo_loss, policy_chosen_logps, policy_rejected_logps) = value
            train_loss += loss
            train_dpo_loss += dpo_loss
            train_policy_chosen_logps += policy_chosen_logps
            train_policy_rejected_logps += policy_rejected_logps
        
        train_loss /= num_batches
        train_dpo_loss /= num_batches
        train_policy_chosen_logps /= num_batches
        train_policy_rejected_logps /= num_batches
        f.write( ("%6d" + 4*"  %.6f") % (epoch, train_loss, train_dpo_loss, train_policy_chosen_logps, train_policy_rejected_logps))

        # Validation
        val_loss = 0.0
        val_dpo_loss = 0.0
        val_policy_chosen_logps = 0.0
        val_policy_rejected_logps = 0.0
        num_val_samples = len(val_ref_chosen_logps)
        num_batches = math.ceil(num_val_samples / batchsize)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_val_samples)
            x_w = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx],  val_chosen_data)
            x_l = jax.tree_util.tree_map(lambda x: x[start_idx:end_idx], val_rejected_data)
            ref_chosen_logps_batch = val_ref_chosen_logps[start_idx:end_idx]
            ref_rejected_logps_batch = val_ref_rejected_logps[start_idx:end_idx]

            key, subkey = jax.random.split(key)
            loss, (dpo_loss, policy_chosen_logps, policy_rejected_logps) = jax.jit(dpo_loss_fn)(params, subkey, x_w, x_l, ref_chosen_logps_batch, ref_rejected_logps_batch)
            val_loss += loss
            val_dpo_loss += dpo_loss
            val_policy_chosen_logps += policy_chosen_logps
            val_policy_rejected_logps += policy_rejected_logps

        val_loss /= num_batches
        val_dpo_loss /= num_batches
        val_policy_chosen_logps /= num_batches
        val_policy_rejected_logps /= num_batches
        f.write( (4*"  %.6f" + "\n") % (val_loss, val_dpo_loss, val_policy_chosen_logps, val_policy_rejected_logps))


        if epoch % 1 == 0:
            ckpt = {"params": params,
                    "opt_state" : opt_state
                }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()

    return params, opt_state
