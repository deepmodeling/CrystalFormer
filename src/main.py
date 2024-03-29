import jax 
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp 
from jax.flatten_util import ravel_pytree
import optax
import os
import multiprocessing
import math

from utils import GLXYZAW_from_file, GLXA_to_csv
from elements import element_dict, element_list
from transformer import make_transformer  
from train import train
from sample import sample_crystal
from loss import make_loss_fn
import checkpoint
from wyckoff import mult_table

import argparse
parser = argparse.ArgumentParser(description='')

group = parser.add_argument_group('training parameters')
group.add_argument('--epochs', type=int, default=10000, help='')
group.add_argument('--batchsize', type=int, default=100, help='')
group.add_argument('--lr', type=float, default=1e-4, help='learning rate')
group.add_argument('--lr_decay', type=float, default=0.0, help='lr decay')
group.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
group.add_argument('--clip_grad', type=float, default=1.0, help='clip gradient')
group.add_argument("--optimizer", type=str, default="adam", choices=["none", "adam", "adamw"], help="optimizer type")

group.add_argument("--folder", default="../data/", help="the folder to save data")
group.add_argument("--restore_path", default=None, help="checkpoint path or file")

group = parser.add_argument_group('dataset')
group.add_argument('--train_path', default='/home/wanglei/cdvae/data/mp_20/train.csv', help='')
group.add_argument('--valid_path', default='/home/wanglei/cdvae/data/mp_20/val.csv', help='')
group.add_argument('--test_path', default='/home/wanglei/cdvae/data/mp_20/test.csv', help='')

group = parser.add_argument_group('transformer parameters')
group.add_argument('--Nf', type=int, default=5, help='number of frequencies for fc')
group.add_argument('--Kx', type=int, default=16, help='number of modes in x')
group.add_argument('--Kl', type=int, default=4, help='number of modes in lattice')
group.add_argument('--h0_size', type=int, default=256, help='hidden layer dimension for the first atom, 0 means we simply use a table for first aw_logit')
group.add_argument('--transformer_layers', type=int, default=16, help='The number of layers in transformer')
group.add_argument('--num_heads', type=int, default=16, help='The number of heads')
group.add_argument('--key_size', type=int, default=64, help='The key size')
group.add_argument('--model_size', type=int, default=64, help='The model size')
group.add_argument('--embed_size', type=int, default=32, help='The enbedding size')
group.add_argument('--dropout_rate', type=float, default=0.5, help='The dropout rate')

group = parser.add_argument_group('loss parameters')
group.add_argument("--lamb_a", type=float, default=1.0, help="weight for the a part relative to fc")
group.add_argument("--lamb_w", type=float, default=1.0, help="weight for the w part relative to fc")
group.add_argument("--lamb_l", type=float, default=1.0, help="weight for the lattice part relative to fc")

group = parser.add_argument_group('physics parameters')
group.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
group.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
group.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

group = parser.add_argument_group('sampling parameters')
group.add_argument('--spacegroup', type=int, help='The space group id to be sampled (1-230)')
group.add_argument('--elements', type=str, default=None, nargs='+', help='name of the chemical elemenets, e.g. Bi, Ti, O')
group.add_argument('--top_p', type=float, default=1.0, help='1.0 means un-modified logits, smaller value of p give give less diverse samples')
group.add_argument('--temperature', type=float, default=1.0, help='temperature used for sampling')
group.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')
group.add_argument('--num_samples', type=int, default=1000, help='number of test samples')
group.add_argument('--use_foriloop', action='store_true', help='use lax.fori_loop in sampling')
group.add_argument('--output_filename', type=str, default='output.csv', help='outfile to save sampled structures')

args = parser.parse_args()

key = jax.random.PRNGKey(42)

num_cpu = multiprocessing.cpu_count()
print('number of available cpu: ', num_cpu)
if args.num_io_process > num_cpu:
    print('num_io_process should not exceed number of available cpu, reset to ', num_cpu)
    args.num_io_process = num_cpu


################### Data #############################
if args.optimizer != "none":
    train_data = GLXYZAW_from_file(args.train_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)
    valid_data = GLXYZAW_from_file(args.valid_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)
else:
    assert (args.spacegroup is not None) # for inference we need to specify space group
    test_data = GLXYZAW_from_file(args.test_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)
    
    if args.elements is not None:
        idx = [element_dict[e] for e in args.elements]
        atom_mask = [1] + [1 if a in idx else 0 for a in range(1, args.atom_types)]
        atom_mask = jnp.array(atom_mask)
        print ('sampling structure formed by these elements:', args.elements)
        print (atom_mask)
    else:
        atom_mask = jnp.zeros((args.atom_types), dtype=int) # we will do nothing to a_logit in sampling

################### Model #############################
params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                      args.h0_size, 
                                      args.transformer_layers, args.num_heads, 
                                      args.key_size, args.model_size, args.embed_size, 
                                      args.atom_types, args.wyck_types,
                                      args.dropout_rate)
transformer_name = 'Nf_%d_Kx_%d_Kl_%d_h0_%d_l_%d_H_%d_k_%d_m_%d_e_%d_drop_%g'%(args.Nf, args.Kx, args.Kl, args.h0_size, args.transformer_layers, args.num_heads, args.key_size, args.model_size, args.embed_size, args.dropout_rate)

print ("# of transformer params", ravel_pytree(params)[0].size) 

################### Train #############################

loss_fn, logp_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, transformer, args.lamb_a, args.lamb_w, args.lamb_l)

print("\n========== Prepare logs ==========")
if args.optimizer != "none" or args.restore_path is None:
    output_path = args.folder + args.optimizer+"_bs_%d_lr_%g_decay_%g_clip_%g" % (args.batchsize, args.lr, args.lr_decay, args.clip_grad) \
                   + '_A_%g_W_%g_N_%g'%(args.atom_types, args.wyck_types, args.n_max) \
                   + ("_wd_%g"%(args.weight_decay) if args.optimizer == "adamw" else "") \
                   + ('_a_%g_w_%g_l_%g'%(args.lamb_a, args.lamb_w, args.lamb_l)) \
                   +  "_" + transformer_name 

    os.makedirs(output_path, exist_ok=True)
    print("Create directory for output: %s" % output_path)
else:
    output_path = os.path.dirname(args.restore_path)
    print("Will output samples to: %s" % output_path)


print("\n========== Load checkpoint==========")
ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path or output_path) 
if ckpt_filename is not None:
    print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
    ckpt = checkpoint.load_data(ckpt_filename)
    params = ckpt["params"]
else:
    print("No checkpoint file found. Start from scratch.")

if args.optimizer != "none":

    schedule = lambda t: args.lr/(1+args.lr_decay*t)

    if args.optimizer == "adam":
        optimizer = optax.chain(optax.clip_by_global_norm(args.clip_grad), 
                                optax.scale_by_adam(), 
                                optax.scale_by_schedule(schedule), 
                                optax.scale(-1.))
    elif args.optimizer == 'adamw':
        optimizer = optax.chain(optax.clip(args.clip_grad),
                                optax.adamw(learning_rate=schedule, weight_decay=args.weight_decay)
                               )

    opt_state = optimizer.init(params)
    try:
        opt_state.update(ckpt["opt_state"])
    except: 
        print ("failed to update opt_state from checkpoint")
        pass 
 
    print("\n========== Start training ==========")
    params, opt_state = train(key, optimizer, opt_state, loss_fn, params, epoch_finished, args.epochs, args.batchsize, train_data, valid_data, output_path)

else:
    pass

    print("\n========== Calculate the loss of test dataset ==========")
    import numpy as np 
    np.set_printoptions(threshold=np.inf)

    test_G, test_L, test_XYZ, test_A, test_W = test_data
    print (test_G.shape, test_L.shape, test_XYZ.shape, test_A.shape, test_W.shape)
    test_loss = 0
    num_samples = len(test_L)
    num_batches = math.ceil(num_samples / args.batchsize)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batchsize
        end_idx = min(start_idx + args.batchsize, num_samples)
        G, L, XYZ, A, W = test_G[start_idx:end_idx], \
                          test_L[start_idx:end_idx], \
                          test_XYZ[start_idx:end_idx], \
                          test_A[start_idx:end_idx], \
                          test_W[start_idx:end_idx]
        loss, _ = jax.jit(loss_fn, static_argnums=7)(params, key, G, L, XYZ, A, W, False)
        test_loss += loss
    test_loss = test_loss / num_batches
    print ("evaluating loss on test data:" , test_loss)

    print("\n========== Start sampling ==========")
    jax.config.update("jax_enable_x64", True) # to get off compilation warning, and to prevent sample nan lattice 
    #FYI, the error was [Compiling module extracted] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.

    num_batches = math.ceil(args.num_samples / args.batchsize)
    name, extension = args.output_filename.rsplit('.', 1)
    filename = os.path.join(output_path, 
                            f"{name}_{args.spacegroup}.{extension}")
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batchsize
        end_idx = min(start_idx + args.batchsize, args.num_samples)
        n_sample = end_idx - start_idx
        key, subkey = jax.random.split(key)
        XYZ, A, W, M, L = sample_crystal(subkey, transformer, params, args.n_max, n_sample, args.atom_types, args.wyck_types, args.Kx, args.Kl, args.spacegroup, atom_mask, args.top_p, args.temperature, args.use_foriloop)
        print ("XYZ:\n", XYZ)  # fractional coordinate 
        print ("A:\n", A)  # element type
        print ("W:\n", W)  # Wyckoff positions
        print ("M:\n", M)  # multiplicity 
        print ("N:\n", M.sum(axis=-1)) # total number of atoms
        print ("L:\n", L)  # lattice
        for a in A:
           print([element_list[i] for i in a])

        # output L, X, A, W, M, AW to csv file
        # output logp_w, logp_xyz, logp_a, logp_l to csv file
        import pandas as pd
        data = pd.DataFrame()
        data['L'] = np.array(L).tolist()
        data['X'] = np.array(XYZ).tolist()
        data['A'] = np.array(A).tolist()
        data['W'] = np.array(W).tolist()
        data['M'] = np.array(M).tolist()

        num_atoms = jnp.sum(M, axis=1)
        length, angle = jnp.split(L, 2, axis=-1)
        length = length/num_atoms[:, None]**(1/3)
        angle = angle * (jnp.pi / 180) # to rad
        L = jnp.concatenate([length, angle], axis=-1)

        G = args.spacegroup * jnp.ones((n_sample), dtype=int)
        logp_w, logp_xyz, logp_a, logp_l = jax.jit(logp_fn, static_argnums=7)(params, key, G, L, XYZ, A, W, False)

        data['logp_w'] = np.array(logp_w).tolist()
        data['logp_xyz'] = np.array(logp_xyz).tolist()
        data['logp_a'] = np.array(logp_a).tolist()
        data['logp_l'] = np.array(logp_l).tolist()
        data['logp'] = np.array(logp_xyz + args.lamb_w*logp_w + args.lamb_a*logp_a + args.lamb_l*logp_l).tolist()

        data = data.sort_values(by='logp', ascending=False) # sort by logp
        header = False if os.path.exists(filename) else True
        data.to_csv(filename, mode='a', index=False, header=header)

        # GLXA_to_csv(args.spacegroup, L, XYZ, A, num_worker=args.num_io_process, filename=filename)
        print ("Wrote samples to %s"%filename)
