import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import os
import optax
import warnings
warnings.filterwarnings("ignore")

from crystalformer.src.utils import GLXYZAW_from_file
from crystalformer.src.loss import make_loss_fn
from crystalformer.src.transformer import make_transformer
import crystalformer.src.checkpoint as checkpoint

from crystalformer.reinforce.dpo import make_dpo_loss, train


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
    group.add_argument("--optimizer", type=str, default="adam", choices=["none", "adam", "adamw"], help="optimizer type")

    group.add_argument("--folder", default="./data/", help="the folder to save data")
    group.add_argument("--restore_path", default=None, help="checkpoint path or file")

    group = parser.add_argument_group('dataset')
    group.add_argument('--chosen_path', default='/data/zdcao/crystal_gpt/dataset/mp_20/val.csv', help='')
    group.add_argument('--rejected_path', default='/data/zdcao/crystal_gpt/dataset/mp_20/val.csv', help='')
    group.add_argument("--val_ratio", type=float, default=0.2, help="validation ratio")
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

    group = parser.add_argument_group('reinforcement learning parameters')
    parser.add_argument('--beta', type=float, default=0.1, help='beta for DPO loss')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing for DPO loss')
    parser.add_argument('--gamma', type=float, default=0.0, help='logp regularization coefficient for DPO loss')
    parser.add_argument('--ipo', action='store_true', help='use IPO loss instead of DPO loss')

    args = parser.parse_args()

    print("\n========== Load dataset ==========")
    chosen_data = GLXYZAW_from_file(args.chosen_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)
    rejected_data = GLXYZAW_from_file(args.rejected_path, args.atom_types, args.wyck_types, args.n_max, args.num_io_process)

    print("================ parameters ================")
    # print all the parameters
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

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
        output_path = args.folder \
                    + "beta_%g_label_%g_gamma_%g_"%(args.beta, args.label_smoothing, args.gamma) \
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

        print("\n========== Start RL training ==========")
        dpo_loss_fn = make_dpo_loss(logp_fn,
                                    beta=args.beta,
                                    label_smoothing=args.label_smoothing,
                                    gamma=args.gamma,
                                    ipo=args.ipo)

        # PPO training
        params, opt_state = train(key, optimizer, opt_state, dpo_loss_fn, logp_fn, params, epoch_finished, 
                                  args.epochs, args.batchsize, chosen_data, rejected_data, output_path, args.val_ratio)

    else:
        raise NotImplementedError("No optimizer specified. Please specify an optimizer in the config file.")


if __name__ == "__main__":
    main()
