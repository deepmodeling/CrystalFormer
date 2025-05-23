import jax 
import jax.numpy as jnp 
from jax.flatten_util import ravel_pytree
import numpy as np
import os
import multiprocessing
import math

from crystalformer.src.lattice import norm_lattice
from crystalformer.src.utils import GLXYZAW_from_file
from crystalformer.src.elements import element_dict, element_list
from crystalformer.src.transformer import make_transformer  
from crystalformer.src.loss import make_loss_fn
import crystalformer.src.checkpoint as checkpoint

from crystalformer.reinforce.sample import make_sample_crystal


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')

    group = parser.add_argument_group('dataset')
    group.add_argument("--restore_path", default=None, help="checkpoint path or file")
    group.add_argument('--test_path', default='/data/zdcao/crystal_gpt/dataset/alex/PBE/alex20/test.lmdb', help='')

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
    group.add_argument('--dropout_rate', type=float, default=0.1, help='The dropout rate for MLP')
    group.add_argument('--attn_dropout', type=float, default=0.1, help='The dropout rate for attention')

    group = parser.add_argument_group('loss parameters')
    group.add_argument("--lamb_a", type=float, default=1.0, help="weight for the a part relative to fc")
    group.add_argument("--lamb_w", type=float, default=1.0, help="weight for the w part relative to fc")
    group.add_argument("--lamb_l", type=float, default=1.0, help="weight for the lattice part relative to fc")

    group = parser.add_argument_group('physics parameters')
    group.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
    group.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
    group.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

    group = parser.add_argument_group('sampling parameters')
    group.add_argument('--seed', type=int, default=None, help='random seed to sample')
    group.add_argument('--batchsize', type=int, default=100, help='')
    group.add_argument('--elements', type=str, default=None, nargs='+', help='name of the chemical elemenets, e.g. Bi, Ti, O')
    group.add_argument('--remove_radioactive', action='store_true', help='remove radioactive elements and noble gas')
    group.add_argument('--top_p', type=float, default=1.0, help='1.0 means un-modified logits, smaller value of p give give less diverse samples')
    group.add_argument('--temperature', type=float, default=1.0, help='temperature used for sampling')
    group.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')
    group.add_argument('--num_samples', type=int, default=1000, help='number of test samples')
    group.add_argument('--output_filename', type=str, default='output.csv', help='outfile to save sampled structures')

    args = parser.parse_args()

    key = jax.random.PRNGKey(42)
    jnp.set_printoptions(threshold=jnp.inf)  # print full array

    num_cpu = multiprocessing.cpu_count()
    print('number of available cpu: ', num_cpu)
    if args.num_io_process > num_cpu:
        print('num_io_process should not exceed number of available cpu, reset to ', num_cpu)
        args.num_io_process = num_cpu

    ################### Data #############################
    try:
        print("\n========== Load dataset and get space group distribution =========")
        test_data = GLXYZAW_from_file(args.test_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)
        G = test_data[0]
        # convert space group to probability table
        spg_mask = jnp.bincount(G, minlength=231)
        spg_mask = spg_mask[1:] # remove 0
    except:
        print("\n====== failed to load dataset, back to uniform distribution ======")
        spg_mask = jnp.ones((230), dtype=int)

    spg_mask = spg_mask / jnp.sum(spg_mask)
    print(spg_mask)

    if args.elements is not None:
        idx = [element_dict[e] for e in args.elements]
        atom_mask = [1] + [1 if a in idx else 0 for a in range(1, args.atom_types)]
        atom_mask = jnp.array(atom_mask)
        atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)
        print ('sampling structure formed by these elements:', args.elements)
        print (atom_mask)

    else:
        if args.remove_radioactive:
            from crystalformer.src.elements import radioactive_elements_dict, noble_gas_dict
            # remove radioactive elements and noble gas
            atom_mask = [1] + [1 if i not in radioactive_elements_dict.values() and i not in noble_gas_dict.values() else 0 for i in range(1, args.atom_types)]
            atom_mask = jnp.array(atom_mask)
            atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)
            print('sampling structure formed by non-radioactive elements and non-noble gas')
            print(atom_mask)
            
        else:
            atom_mask = jnp.zeros((args.atom_types), dtype=int) # we will do nothing to a_logit in sampling
            atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)
            print(atom_mask)
    # print(f'there is total {jnp.sum(atom_mask)-1} elements')
    print(atom_mask.shape)      

    ################### Model #############################
    params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                           args.h0_size, 
                                           args.transformer_layers, args.num_heads, 
                                           args.key_size, args.model_size, args.embed_size, 
                                           args.atom_types, args.wyck_types,
                                           args.dropout_rate, args.attn_dropout)
    print ("# of transformer params", ravel_pytree(params)[0].size) 

    ################### Train #############################
    _, logp_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, transformer, args.lamb_a, args.lamb_w, args.lamb_l)

    print("\n========== Load checkpoint==========")
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        params = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")

    print("\n========== Start sampling ==========")
    jax.config.update("jax_enable_x64", True) # to get off compilation warning, and to prevent sample nan lattice 
    #FYI, the error was [Compiling module extracted] Very slow compile? If you want to file a bug, run with envvar XLA_FLAGS=--xla_dump_to=/tmp/foo and attach the results.

    if args.seed is not None:
        key = jax.random.PRNGKey(args.seed) # reset key for sampling if seed is provided

    sample_crystal = make_sample_crystal(transformer, args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl)

    num_batches = math.ceil(args.num_samples / args.batchsize)
    filename = os.path.join(args.restore_path, args.output_filename)
    for batch_idx in range(num_batches):
        start_idx = batch_idx * args.batchsize
        end_idx = min(start_idx + args.batchsize, args.num_samples)
        n_sample = end_idx - start_idx
        key, subkey1, subkey2 = jax.random.split(key, 3)
        G = jax.random.choice(subkey1,
                              a=jnp.arange(1, 231, 1),
                              p=spg_mask,
                              shape=(n_sample, ))
        XYZ, A, W, M, L = sample_crystal(subkey2, params, G, atom_mask, top_p=args.top_p, temperature=args.temperature)
        
        print ("G:\n", G)  # space group
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
        data['G'] = np.array(G).tolist()
        data['L'] = np.array(L).tolist()
        data['X'] = np.array(XYZ).tolist()
        data['A'] = np.array(A).tolist()
        data['W'] = np.array(W).tolist()
        data['M'] = np.array(M).tolist()

        L = norm_lattice(G, W, L)
        logp_w, logp_xyz, logp_a, logp_l = jax.jit(logp_fn, static_argnums=7)(params, key, G, L, XYZ, A, W, False)
        data['logp_w'] = np.array(logp_w).tolist()
        data['logp_xyz'] = np.array(logp_xyz).tolist()
        data['logp_a'] = np.array(logp_a).tolist()
        data['logp_l'] = np.array(logp_l).tolist()
        data['logp'] = np.array(logp_xyz + args.lamb_w*logp_w + args.lamb_a*logp_a + args.lamb_l*logp_l).tolist()

        data = data.sort_values(by='logp', ascending=False) # sort by logp
        header = False if os.path.exists(filename) else True
        data.to_csv(filename, mode='a', index=False, header=header)

        print ("Wrote samples to %s"%filename)


if __name__ == "__main__":
    main()
