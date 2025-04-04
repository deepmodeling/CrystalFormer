import jax
import jax.numpy as jnp
from functools import partial
import os
import optax
import math

from crystalformer.src.utils import shuffle
import crystalformer.src.checkpoint as checkpoint


shard = jax.pmap(lambda x: x)
p_split = jax.pmap(lambda key: tuple(jax.random.split(key)))

    
def scatter(x: jnp.ndarray, retain_axis=False) -> jnp.ndarray:
    num_devices = jax.local_device_count()
    if x.shape[0] % num_devices != 0:
        raise ValueError("The first dimension of x must be divisible by the total number of GPU devices. "
                         "Got x.shape[0] = %d for %d devices now." % (x.shape[0], num_devices))
    dim_per_device = x.shape[0] // num_devices
    x = x.reshape(
        (num_devices,) +
        (() if dim_per_device == 1 and not retain_axis else (dim_per_device,)) +
        x.shape[1:]
    )
    return shard(x)


def train(key, optimizer, opt_state, loss_fn, params, epoch_finished, epochs, batchsize, train_data, valid_data, path, val_interval):
           
    num_devices = jax.local_device_count()
    batch_per_device = batchsize // num_devices
    shape_prefix = (num_devices, batch_per_device)
    print("num_devices: ", num_devices)
    print("batch_per_device: ", batch_per_device)
    print("shape_prefix: ", shape_prefix)

    key = jax.random.fold_in(key, jax.process_index())  # make different key for different process
    key, *keys = jax.random.split(key, num_devices + 1)
    keys = scatter(jnp.array(keys))

    @partial(jax.pmap, axis_name="p", in_axes=(None, 0, None, 0), out_axes=(None, None, 0),)
    def update(params, key, opt_state, data):
        G, L, X, A, W = data
        value, grad = jax.value_and_grad(loss_fn, has_aux=True)(params, key, G, L, X, A, W, True)
        grad = jax.lax.pmean(grad, axis_name="p")
        value = jax.lax.pmean(value, axis_name="p")
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch t_loss v_loss t_loss_w v_loss_w t_loss_a v_loss_a t_loss_xyz v_loss_xyz t_loss_l v_loss_l\n")
 
    for epoch in range(epoch_finished+1, epochs+1):
        key, subkey = jax.random.split(key)
        train_data = shuffle(subkey, train_data)

        _, train_L, _, _, _ = train_data

        train_loss = 0.0 
        train_aux = 0.0, 0.0, 0.0, 0.0
        num_samples = train_L.shape[0]
        if num_samples % batchsize == 0:
            num_batches = math.ceil(num_samples / batchsize)
        else:
            num_batches = math.ceil(num_samples / batchsize) - 1
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)
            data = jax.tree_map(lambda x: x[start_idx:end_idx], train_data)
            data = jax.tree_map(lambda x: x.reshape(shape_prefix + x.shape[1:]), data)
            
            keys, subkeys = p_split(keys)
            params, opt_state, (loss, aux) = update(params, subkeys, opt_state, data)
            train_loss, train_aux = jax.tree_map(   
                        lambda acc, i: acc + jnp.mean(i),
                        (train_loss, train_aux),  
                        (loss, aux)
                        )

        train_loss, train_aux = jax.tree_map(
                        lambda x: x/num_batches, 
                        (train_loss, train_aux)
                        ) 

        if epoch % val_interval == 0:
            _, valid_L, _, _, _ = valid_data 
            valid_loss = 0.0 
            valid_aux = 0.0, 0.0, 0.0, 0.0
            num_samples = valid_L.shape[0]
            if num_samples % batchsize == 0:
                num_batches = math.ceil(num_samples / batchsize)
            else:
                num_batches = math.ceil(num_samples / batchsize) - 1
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batchsize
                end_idx = min(start_idx + batchsize, num_samples)
                data = jax.tree_map(lambda x: x[start_idx:end_idx], valid_data)
                data = jax.tree_map(lambda x: x.reshape(shape_prefix + x.shape[1:]), data)

                keys, subkeys = p_split(keys)
                loss, aux = jax.pmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0, 0),
                                     static_broadcasted_argnums=7)(params, subkeys, *data, False)
                valid_loss, valid_aux = jax.tree_map(
                        lambda acc, i: acc + jnp.mean(i),
                        (valid_loss, valid_aux), 
                        (loss, aux)
                        )

            valid_loss, valid_aux = jax.tree_map(
                        lambda x: x/num_batches, 
                        (valid_loss, valid_aux)
                        ) 

            train_loss_w, train_loss_a, train_loss_xyz, train_loss_l = train_aux
            valid_loss_w, valid_loss_a, valid_loss_xyz, valid_loss_l = valid_aux

            f.write( ("%6d" + 10*"  %.6f" + "\n") % (epoch, 
                                                    train_loss,   valid_loss,
                                                    train_loss_w, valid_loss_w, 
                                                    train_loss_a, valid_loss_a, 
                                                    train_loss_xyz, valid_loss_xyz, 
                                                    train_loss_l, valid_loss_l
                                                    ))

            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            if jax.process_index() == 0:
                checkpoint.save_data(ckpt, ckpt_filename)
                print("Save checkpoint file: %s" % ckpt_filename)
                
    f.close()
    return params, opt_state
