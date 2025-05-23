import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from functools import partial
import os
import optax
import warnings
warnings.filterwarnings("ignore")

from crystalformer.src.utils import GLXYZAW_from_file
from crystalformer.src.loss import make_loss_fn
from crystalformer.src.transformer import make_transformer
import crystalformer.src.checkpoint as checkpoint

from crystalformer.reinforce.ppo import train, make_ppo_loss_fn
from crystalformer.reinforce.sample import make_sample_crystal


def main():
    import argparse
    parser = argparse.ArgumentParser(description='')

    group = parser.add_argument_group('training parameters')
    group.add_argument('--epochs', type=int, default=100, help='')
    group.add_argument('--batchsize', type=int, default=100, help='')
    group.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    group.add_argument('--lr_decay', type=float, default=0.0, help='lr decay')
    group.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    group.add_argument('--clip_grad', type=float, default=1.0, help='clip gradient')
    group.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"], help="optimizer type")

    group.add_argument("--folder", default="./data/", help="the folder to save data")
    group.add_argument("--restore_path", default=None, help="checkpoint path or file")

    group = parser.add_argument_group('dataset')
    group.add_argument('--valid_path', default='/data/zdcao/crystal_gpt/dataset/mp_20/val.csv', help='')
    group.add_argument('--test_path', default=None, help='dataset to get the space group distribution')
    group.add_argument('--num_io_process', type=int, default=40, help='number of process used in multiprocessing io')

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

    group = parser.add_argument_group('physics parameters')
    group.add_argument('--n_max', type=int, default=21, help='The maximum number of atoms in the cell')
    group.add_argument('--atom_types', type=int, default=119, help='Atom types including the padded atoms')
    group.add_argument('--wyck_types', type=int, default=28, help='Number of possible multiplicites including 0')

    group = parser.add_argument_group('sampling parameters')
    group.add_argument('--remove_radioactive', action='store_true', help='remove radioactive elements and noble gas')
    group.add_argument('--top_p', type=float, default=1.0, help='1.0 means un-modified logits, smaller value of p give give less diverse samples')
    group.add_argument('--temperature', type=float, default=1.0, help='temperature used for sampling')

    group = parser.add_argument_group('reinforcement learning parameters')
    group.add_argument('--spacegroup', default=None, nargs='+', help='the number of spacegroups to sample from')
    group.add_argument('--reward', type=str, default='force', choices=['force', 'ehull', 'prop', 'dielectric'], help='reward function to use')
    group.add_argument('--convex_path', type=str, default='/data/zdcao/crystal_gpt/dataset/alex/PBE/convex_hull_pbe_2023.12.29.json.bz2')
    group.add_argument('--beta', type=float, default=0.1, help='weight for KL divergence')
    group.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
    group.add_argument('--ppo_epochs', type=int, default=5, help='number of PPO epochs')
    group.add_argument('--mlff_model', type=str, default='orb', choices=['mace', 'orb', 'matgl'], help='the model to use for RL reward')
    group.add_argument('--mlff_path', type=str, default='./data/orb-v2-20241011.ckpt', help='path to the MLFF model')

    group = parser.add_argument_group('property reward parameters')
    group.add_argument('--target', type=float, default=-3, help='target property value to optimize')
    group.add_argument('--dummy_value', type=float, default=0, help='dummy value for the property')
    group.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'mae'], help='loss type for the property reward')

    args = parser.parse_args()

    print("\n========== Load dataset ==========")
    valid_data = GLXYZAW_from_file(args.valid_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)

    print("================ parameters ================")
    # print all the parameters
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    ################### Data #############################
    try:
        print("\n========== Load dataset and get space group distribution =========")
        test_data = GLXYZAW_from_file(args.test_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)
        G = test_data[0]
        # convert space group to probability table
        spg_mask = jnp.bincount(G, minlength=231)
        spg_mask = spg_mask[1:] # remove 0
    except:
        print("\n====== failed to load dataset ======")
        if args.spacegroup is not None:
            print("Spacegroup specified, will sample from the specified spacegroups")
            spg_mask = jnp.zeros((230), dtype=int)
            for g in args.spacegroup:
                spg_mask = spg_mask.at[int(g)-1].set(1)
            
        else:
            print("No spacegroup specified, will sample from all spacegroups")
            spg_mask = jnp.ones((230), dtype=int)

    print("spacegroup mask", spg_mask)

    if args.remove_radioactive:
        from crystalformer.src.elements import radioactive_elements_dict, noble_gas_dict
        # remove radioactive elements and noble gas
        atom_mask = [1] + [1 if i not in radioactive_elements_dict.values() and i not in noble_gas_dict.values() else 0 for i in range(1, args.atom_types)]
        atom_mask = jnp.array(atom_mask)
        atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)
        print('sampling structure formed by non-radioactive elements and non-noble gas')
            
    else:
        atom_mask = jnp.zeros((args.atom_types), dtype=int) # we will do nothing to a_logit in sampling
        atom_mask = jnp.stack([atom_mask] * args.n_max, axis=0)
        print('sampling structure formed by all elements')

    print("atom_mask", atom_mask)

    print("\n========== Prepare transformer ==========")
    ################### Model #############################
    key = jax.random.PRNGKey(42)
    params, transformer = make_transformer(key, args.Nf, args.Kx, args.Kl, args.n_max, 
                                        args.h0_size, 
                                        args.transformer_layers, args.num_heads, 
                                        args.key_size, args.model_size, args.embed_size, 
                                        args.atom_types, args.wyck_types,
                                        args.dropout_rate, args.attn_dropout)

    transformer_name = 'Nf_%d_Kx_%d_Kl_%d_h0_%d_l_%d_H_%d_k_%d_m_%d_e_%d_drop_%g'%(args.Nf, args.Kx, args.Kl, args.h0_size, args.transformer_layers, args.num_heads, args.key_size, args.model_size, args.embed_size, args.dropout_rate)

    print ("# of transformer params", ravel_pytree(params)[0].size) 

    ################### Train #############################

    loss_fn, logp_fn = make_loss_fn(args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl, transformer)

    print("\n========== Prepare logs ==========")
    if args.optimizer != "none" or args.restore_path is None:
        output_path = args.folder + "ppo_%d_beta_%g_" % (args.ppo_epochs, args.beta) \
                    + args.optimizer+"_bs_%d_lr_%g_decay_%g_clip_%g" % (args.batchsize, args.lr, args.lr_decay, args.clip_grad) \
                    + '_A_%g_W_%g_N_%g'%(args.atom_types, args.wyck_types, args.n_max) \
                    + ("_wd_%g"%(args.weight_decay) if args.optimizer == "adamw" else "") \
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

    print("\n========== Load calculator and rl loss ==========")
    print(f"Using {args.mlff_model} model at {args.mlff_path}")
    if args.mlff_model == "mace":
        from mace.calculators import mace_mp
        calc = mace_mp(model=args.mlff_path,
                        dispersion=False,
                        default_dtype="float64",
                        device='cuda')
        
    elif args.mlff_model == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator

        # Load the ORB forcefield model
        orbff = pretrained.orb_v2(args.mlff_path, device='cuda') 
        calc = ORBCalculator(orbff, device='cuda')
    
    elif args.mlff_model == "matgl":
        # only available for property reward
        import matgl
        # try split the mlff_path according to the ',' symbol
        if ',' in args.mlff_path:
            import torch
            torch.set_default_dtype(torch.float32)
            model1 = matgl.load_model(args.mlff_path.split(',')[0])
            model1 = model1.predict_structure
            model2 = matgl.load_model(args.mlff_path.split(',')[1])
            model2 = partial(model2.predict_structure, state_attr=torch.tensor([0]))
            model = [model1, model2]
            
        else:
            model = matgl.load_model(args.mlff_path)
            model = model.predict_structure

    else:
        raise NotImplementedError

    if args.reward == "force":
        from crystalformer.reinforce.reward import make_force_reward_fn
        _, batch_reward_fn = make_force_reward_fn(calc)

    elif args.reward == "ehull":
        import json, bz2
        from crystalformer.reinforce.reward import make_ehull_reward_fn
        with bz2.open(args.convex_path) as fh:
            ref_data = json.loads(fh.read().decode('utf-8'))
            # remove 'structure' key in the 'entries' dictionary to reduce the size of the ref_data
            for entry in ref_data['entries']:
                entry.pop('structure')
                
        _, batch_reward_fn = make_ehull_reward_fn(calc, ref_data)
    
    elif args.reward == "prop":
        from crystalformer.reinforce.reward import make_prop_reward_fn
        _, batch_reward_fn = make_prop_reward_fn(model, args.target, args.dummy_value, args.loss_type)

    elif args.reward == "dielectric":
        assert len(model) == 2, "Two models are required for dielectric reward"
        from crystalformer.reinforce.reward import make_dielectric_reward_fn
        _, batch_reward_fn = make_dielectric_reward_fn(model, args.dummy_value)

    else:
        raise NotImplementedError

    print("\n========== Load partial sample function ==========")
    sample_crystal = make_sample_crystal(transformer, args.n_max, args.atom_types, args.wyck_types, args.Kx, args.Kl)
    partial_sample_crystal = partial(sample_crystal, atom_mask=atom_mask, top_p=args.top_p, temperature=args.temperature)

    print("\n========== Start RL training ==========")
    ppo_loss_fn = make_ppo_loss_fn(logp_fn, args.eps_clip, beta=args.beta)

    # PPO training
    params, opt_state = train(key, optimizer, opt_state, spg_mask, loss_fn, logp_fn, batch_reward_fn, ppo_loss_fn, partial_sample_crystal,
                                params, epoch_finished, args.epochs, args.ppo_epochs, args.batchsize, valid_data, output_path)


if __name__ == "__main__":
    main()
