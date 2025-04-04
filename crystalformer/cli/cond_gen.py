import os
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import pandas as pd
import numpy as np
from ast import literal_eval

from crystalformer.extension.loss import make_classifier_loss, make_cond_logp, make_multi_cond_logp
from crystalformer.extension.model import make_classifier
from crystalformer.extension.transformer import make_transformer as make_transformer_with_state
from crystalformer.extension.mcmc import make_mcmc_step

import crystalformer.src.checkpoint as checkpoint
from crystalformer.src.wyckoff import mult_table
from crystalformer.src.loss import make_loss_fn
from crystalformer.src.transformer import make_transformer


def main():

    import argparse
    parser = argparse.ArgumentParser(description='')

    group = parser.add_argument_group('physics parameters')
    group.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
    group.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
    group.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

    group = parser.add_argument_group('base transformer parameters')
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
    group.add_argument('--restore_path', default='./', help='The path to restore the base model')

    group = parser.add_argument_group('cond transformer parameters')
    group.add_argument('--cond_transformer_layers', type=int, default=4, help='The number of layers in transformer')
    group.add_argument('--cond_num_heads', type=int, default=8, help='The number of heads')
    group.add_argument('--cond_key_size', type=int, default=32, help='The key size')
    group.add_argument('--cond_dropout_rate', type=float, default=0.3, help='The dropout rate')

    group = parser.add_argument_group('classifier parameters')
    group.add_argument('--sequence_length', type=int, default=105, help='The sequence length')
    group.add_argument('--outputs_size', type=int, default=64, help='The outputs size')
    group.add_argument('--hidden_sizes', type=str, default='128,128,64' , help='The hidden sizes')
    group.add_argument('--num_classes', type=int, default=1, help='The number of classes')

    # restore_path = ("/data/zdcao/crystal_gpt/classifier/", 
    #                 "/data/zdcao/crystal_gpt/classifier/bandgap_mae/"
    # )
    group.add_argument('--cond_restore_path', help='The path to restore the conditional model')

    group = parser.add_argument_group('conditional generation parameters')
    group.add_argument('--spacegroup', type=int, help='The space group')
    group.add_argument('--input_path', default='./', help='The path to load the input data')
    group.add_argument('--mode', type=str, default="multi", help='single or multi')
    group.add_argument('--target', type=str, default="-3, 2", help='target value for formation energy and bandgap')
    group.add_argument('--alpha', type=str, default="10, 3", help='guidance strength')
    group.add_argument('--output_path', default='./', help='The path to output the generated data')

    group = parser.add_argument_group('MCMC parameters')
    group.add_argument('--mc_steps', type=int, default=1000, help='The number of MCMC steps')
    group.add_argument('--mc_width', type=float, default=0.1, help='The width of MCMC proposal')
    group.add_argument('--init_temp', type=float, default=10.0, help='The initial temperature')
    group.add_argument('--end_temp', type=float, default=1.0, help='The final temperature')
    group.add_argument('--decay_step', type=int, default=10, help='The number of decay steps')


    args = parser.parse_args()
    key = jax.random.PRNGKey(42)

    target = [float(x) for x in args.target.split(',')]
    alpha = [float(x) for x in args.alpha.split(',')]

    ################### Load Classifier Model #############################
    transformer_params, state, cond_transformer = make_transformer_with_state(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                                                              args.h0_size, 
                                                                              args.cond_transformer_layers,
                                                                              args.cond_num_heads, 
                                                                              args.cond_key_size,
                                                                              args.model_size, args.embed_size, 
                                                                              args.atom_types, args.wyck_types,
                                                                              args.cond_dropout_rate)

    print ("# of transformer params", ravel_pytree(transformer_params)[0].size) 
    
    key, subkey = jax.random.split(key)
    classifier_params, classifier = make_classifier(subkey,
                                                    n_max=args.n_max,
                                                    embed_size=args.embed_size,
                                                    sequence_length=args.sequence_length,
                                                    outputs_size=args.outputs_size,
                                                    hidden_sizes=[int(x) for x in args.hidden_sizes.split(',')],
                                                    num_classes=args.num_classes)

    print ("# of classifier params", ravel_pytree(classifier_params)[0].size) 

    cond_params = (transformer_params, classifier_params)

    print("\n========== Load checkpoint==========")
    restore_path = args.cond_restore_path.split(',')
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path[0]) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        cond_params1 = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")

    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(restore_path[1]) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        cond_params2 = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")

    if args.mode == "single":
        cond_params = cond_params1
    elif args.mode == "multi":
        cond_params = (cond_params1, cond_params2)
    else:
        raise ValueError("mode should be either single or multi")
    
    _, forward_fn = make_classifier_loss(cond_transformer, classifier)

    ################### Load BASE Model #############################
    base_params, base_transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                        args.h0_size, 
                                        args.transformer_layers, args.num_heads, 
                                        args.key_size, args.model_size, args.embed_size, 
                                        args.atom_types, args.wyck_types,
                                        args.dropout_rate)
    print ("# of transformer params", ravel_pytree(base_params)[0].size) 

    _, logp_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, base_transformer)

    print("\n========== Load checkpoint==========")
    ckpt_filename, epoch_finished = checkpoint.find_ckpt_filename(args.restore_path) 
    if ckpt_filename is not None:
        print("Load checkpoint file: %s, epoch finished: %g" %(ckpt_filename, epoch_finished))
        ckpt = checkpoint.load_data(ckpt_filename)
        base_params = ckpt["params"]
    else:
        print("No checkpoint file found. Start from scratch.")

    ################### Conditional Generation ############################
    forward = jax.vmap(forward_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, None))

    if args.mode == "single":
        cond_logp_fn = make_cond_logp(logp_fn, forward, 
                                      target=jnp.array(target[0]),
                                      alpha=alpha[0])
    else:
        cond_logp_fn = make_multi_cond_logp(logp_fn,
                                            forward_fns=(forward, forward),
                                            targets=jnp.array(target),
                                            alphas=alpha
                                            )

    print("\n========== Load sampled data ==========")
    csv_file = f"{args.input_path}/output_{args.spacegroup}.csv"
    origin_data = pd.read_csv(csv_file)
    L, XYZ, A, W = origin_data['L'], origin_data['X'], origin_data['A'], origin_data['W']
    L = L.apply(lambda x: literal_eval(x))
    XYZ = XYZ.apply(lambda x: literal_eval(x))
    A = A.apply(lambda x: literal_eval(x))
    W = W.apply(lambda x: literal_eval(x))

    # convert array of list to numpy ndarray
    G = jnp.array([args.spacegroup]*len(L))
    L = jnp.array(L.tolist())
    XYZ = jnp.array(XYZ.tolist())
    A = jnp.array(A.tolist())
    W = jnp.array(W.tolist())

    M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) # (batchsize, n_max)
    num_atoms = jnp.sum(M, axis=1)
    length, angle = jnp.split(L, 2, axis=-1)
    length = length/num_atoms[:, None]**(1/3)
    angle = angle * (jnp.pi / 180) # to rad
    L = jnp.concatenate([length, angle], axis=-1)

    print(G.shape, L.shape, XYZ.shape, A.shape, W.shape)

    print("\n========== Start MCMC ==========")
    mcmc = make_mcmc_step(base_params, cond_params, state, n_max=args.n_max, atom_types=args.atom_types)
    x = (G, L, XYZ, A, W)

    print("====== before mcmc =====")
    print ("XYZ:\n", XYZ)  # fractional coordinate 
    print ("A:\n", A)  # element type
    print ("W:\n", W)  # Wyckoff positions
    print ("L:\n", L)  # lattice

    temp = args.init_temp
    for i in range(args.decay_step):
        alpha = i/(args.decay_step-1)
        temp = 1/(alpha/args.end_temp + (1-alpha)/args.init_temp)
        # temp = init_temp - (init_temp - end_temp) * i / (decay_step-1)
        key, subkey = jax.random.split(key)
        x, acc = mcmc(cond_logp_fn, x_init=x, key=subkey,
                      mc_steps=args.mc_steps//args.decay_step,
                      mc_width=args.mc_width, temp=temp)
        print("i, temp, acc", i, temp, acc)

    G, L, XYZ, A, W = x

    key, subkey = jax.random.split(key)
    logp_w, logp_xyz, logp_a, logp_l = jax.jit(logp_fn, static_argnums=7)(base_params, subkey, G, L, XYZ, A, W, False)
    logp = logp_w + logp_xyz + logp_a + logp_l
    key, subkey = jax.random.split(key)
    logp_new = jax.jit(cond_logp_fn, static_argnums=9)(base_params, cond_params, state, subkey, G, L, XYZ, A, W, False)

    print("====== after mcmc =====")
    M = jax.vmap(lambda g, w: mult_table[g-1, w], in_axes=(0, 0))(G, W) 
    num_atoms = jnp.sum(M, axis=1)

    #scale length according to atom number since we did reverse of that when loading data
    length, angle = jnp.split(L, 2, axis=-1)
    length = length*num_atoms[:, None]**(1/3)
    angle = angle * (180.0 / jnp.pi) # to deg
    L = jnp.concatenate([length, angle], axis=-1)

    print ("XYZ:\n", XYZ)  # fractional coordinate 
    print ("A:\n", A)  # element type
    print ("W:\n", W)  # Wyckoff positions
    print ("L:\n", L)  # lattice

    data = pd.DataFrame()
    data['L'] = np.array(L).tolist()
    data['X'] = np.array(XYZ).tolist()
    data['A'] = np.array(A).tolist()
    data['W'] = np.array(W).tolist()
    data['M'] = np.array(M).tolist()
    data['logp'] = np.array(logp).tolist()
    data['logp_new'] = np.array(logp_new).tolist()

    filename = f'{args.output_path}/cond_output_{args.spacegroup}.csv'
    header = False if os.path.exists(filename) else True
    data.to_csv(filename, mode='a', index=False, header=header)

    print ("Wrote samples to %s"%filename)


if __name__  == "__main__":
    main()
