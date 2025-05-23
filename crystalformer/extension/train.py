import os
import jax
import optax
import math

import crystalformer.src.checkpoint as checkpoint


def train(key, optimizer, opt_state, loss_fn, params, state, epoch_finished, epochs, batchsize, train_data, valid_data, path):
           
    @jax.jit
    def update(params, state, key, opt_state, data):
        G, L, X, A, W, labels = data
        value, grad = jax.value_and_grad(loss_fn)(params, state, key, G, L, X, A, W, labels, True)
        updates, opt_state = optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, value

    log_filename = os.path.join(path, "data.txt")
    f = open(log_filename, "w" if epoch_finished == 0 else "a", buffering=1, newline="\n")
    if os.path.getsize(log_filename) == 0:
        f.write("epoch t_loss v_loss\n")
 
    for epoch in range(epoch_finished+1, epochs):
        key, subkey = jax.random.split(key)
        train_data = jax.tree_util.tree_map(lambda x: jax.random.permutation(subkey, x), train_data)

        train_G, train_L, train_X, train_A, train_W, train_labels = train_data

        train_loss = 0.0 
        num_samples = len(train_labels)
        num_batches = math.ceil(num_samples / batchsize)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batchsize
            end_idx = min(start_idx + batchsize, num_samples)
            data = train_G[start_idx:end_idx], \
                   train_L[start_idx:end_idx], \
                   train_X[start_idx:end_idx], \
                   train_A[start_idx:end_idx], \
                   train_W[start_idx:end_idx], \
                   train_labels[start_idx:end_idx]

            key, subkey = jax.random.split(key)
            params, opt_state, loss = update(params, state, subkey, opt_state, data)
            train_loss = train_loss + loss

        train_loss = train_loss / num_batches

        if epoch % 10 == 0:
            valid_G, valid_L, valid_X, valid_A, valid_W, valid_labels = valid_data 
            valid_loss = 0.0 
            num_samples = len(valid_labels)
            num_batches = math.ceil(num_samples / batchsize)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batchsize
                end_idx = min(start_idx + batchsize, num_samples)
                G, L, X, A, W, labels = valid_G[start_idx:end_idx], \
                                        valid_L[start_idx:end_idx], \
                                        valid_X[start_idx:end_idx], \
                                        valid_A[start_idx:end_idx], \
                                        valid_W[start_idx:end_idx], \
                                        valid_labels[start_idx:end_idx]

                key, subkey = jax.random.split(key)
                loss = loss_fn(params, state, subkey, G, L, X, A, W, labels, False)
                valid_loss = valid_loss + loss

            valid_loss = valid_loss / num_batches

            f.write( ("%6d" + 2*"  %.6f" + "\n") % (epoch, 
                                                    train_loss,   valid_loss
                                                    ))
            
            ckpt = {"params": params,
                    "opt_state" : opt_state
                   }
            ckpt_filename = os.path.join(path, "epoch_%06d.pkl" %(epoch))
            checkpoint.save_data(ckpt, ckpt_filename)
            print("Save checkpoint file: %s" % ckpt_filename)

    f.close()
    return params, opt_state
