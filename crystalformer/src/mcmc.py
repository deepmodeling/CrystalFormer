import jax
import jax.numpy as jnp
from functools import partial

from crystalformer.src.von_mises import sample_von_mises
from crystalformer.src.sample import project_xyz


def make_mcmc_step(params, n_max, atom_types, atom_mask=None, constraints=None):

    if atom_mask is None or jnp.all(atom_mask == 0):
        atom_mask = jnp.ones((n_max, atom_types))

    if constraints is None:
        constraints = jnp.arange(0, n_max, 1)

    def update_A(i, A, a, constraints):
        def body_fn(j, A):
            A = jax.lax.cond(constraints[j] == constraints[i],
                            lambda _: A.at[:, j].set(a),
                            lambda _: A,
                            None)
            return A

        A = jax.lax.fori_loop(0, A.shape[1], body_fn, A)
        return A

    @partial(jax.jit, static_argnums=0)
    def mcmc(logp_fn, x_init, key, mc_steps, mc_width):
        """
            Markov Chain Monte Carlo sampling algorithm.

        INPUT:
            logp_fn: callable that evaluate log-probability of a batch of configuration x.
                The signature is logp_fn(x), where x has shape (batch, n, dim).
            x_init: initial value of x, with shape (batch, n, dim).
            key: initial PRNG key.
            mc_steps: total number of Monte Carlo steps.
            mc_width: size of the Monte Carlo proposal.

        OUTPUT:
            x: resulting batch samples, with the same shape as `x_init`.
        """
        def step(i, state):

            def true_func(i, state):
                x, logp, key, num_accepts = state
                G, L, XYZ, A, W = x
                key, key_proposal_A, key_proposal_XYZ, key_accept, key_logp = jax.random.split(key, 5)
                
                p_normalized = atom_mask[i%n_max] / jnp.sum(atom_mask[i%n_max])  # only propose atom types that are allowed
                _a = jax.random.choice(key_proposal_A, a=atom_types, p=p_normalized, shape=(A.shape[0], )) 
                # _A = A.at[:, i%n_max].set(_a)
                _A = update_A(i%n_max, A, _a, constraints)
                A_proposal = jnp.where(A == 0, A, _A)

                _xyz = XYZ[:, i%n_max] + sample_von_mises(key_proposal_XYZ, 0, 1/mc_width**2, XYZ[:, i%n_max].shape)
                _xyz = jax.vmap(project_xyz, in_axes=(0, 0, 0, None))(G, W[:, i%n_max], _xyz, 0)
                _XYZ = XYZ.at[:, i%n_max].set(_xyz)
                _XYZ -= jnp.floor(_XYZ)   # wrap to [0, 1)
                XYZ_proposal = _XYZ
                x_proposal = (G, L, XYZ_proposal, A_proposal, W)

                logp_w, logp_xyz, logp_a, _ = logp_fn(params, key_logp, *x_proposal, False)
                logp_proposal = logp_w + logp_xyz + logp_a

                ratio = jnp.exp((logp_proposal - logp))
                accept = jax.random.uniform(key_accept, ratio.shape) < ratio

                A_new = jnp.where(accept[:, None], A_proposal, A)  # update atom types
                XYZ_new = jnp.where(accept[:, None, None], XYZ_proposal, XYZ)  # update atom positions
                x_new = (G, L, XYZ_new, A_new, W)
                logp_new = jnp.where(accept, logp_proposal, logp)
                num_accepts += jnp.sum(accept*jnp.where(A[:, i%n_max]==0, 0, 1))
                jax.debug.print("logp {x} {y}", 
                                x=logp_new.mean(),
                                y=jnp.std(logp_new)/jnp.sqrt(logp_new.shape[0])
                                )
                return x_new, logp_new, key, num_accepts

            def false_func(i, state):
                x, logp, key, num_accepts = state
                return x, logp, key, num_accepts
            
            x, logp, key, num_accepts = state
            A = x[3]
            x, logp, key, num_accepts = jax.lax.cond(A[:, i%n_max].sum() != 0,
                                                     lambda _: true_func(i, state),
                                                     lambda _: false_func(i, state),
                                                     None)
            return x, logp, key, num_accepts

        key, subkey = jax.random.split(key)
        logp_w, logp_xyz, logp_a, _ = logp_fn(params, subkey, *x_init, False)
        logp_init = logp_w + logp_xyz + logp_a
        jax.debug.print("logp {x} {y}", 
                        x=logp_init.mean(),
                        y=jnp.std(logp_init)/jnp.sqrt(logp_init.shape[0])
                        )
        
        x, logp, key, num_accepts = jax.lax.fori_loop(0, mc_steps, step, (x_init, logp_init, key, 0.))
        # print("logp", logp)
        A = x[3]
        scale = jnp.sum(A != 0)/(A.shape[0]*n_max)
        accept_rate = num_accepts / (scale * mc_steps * x[0].shape[0])
        return x, accept_rate
    
    return mcmc


if __name__  == "__main__":
    from utils import GLXYZAW_from_file
    from loss import make_loss_fn
    from transformer import make_transformer
    atom_types = 119
    n_max = 21
    wyck_types = 28
    Nf = 5
    Kx = 16
    Kl  = 4
    dropout_rate = 0.3

    csv_file = '../../data/mini.csv'
    G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max)

    key = jax.random.PRNGKey(42)

    params, transformer = make_transformer(key, Nf, Kx, Kl, n_max, 128, 4, 4, 8, 16, 16, atom_types, wyck_types, dropout_rate) 
 
    loss_fn, logp_fn = make_loss_fn(n_max, atom_types, wyck_types, Kx, Kl, transformer)

    # MCMC sampling test
    mc_steps = 21
    mc_width = 0.1
    x_init = (G[:5], L[:5], XYZ[:5], A[:5], W[:5])

    value = jax.jit(logp_fn, static_argnums=7)(params, key, *x_init, False)

    jnp.set_printoptions(threshold=jnp.inf)
    mcmc = make_mcmc_step(params, n_max=n_max, atom_types=atom_types)

    for i in range(5):
        key, subkey = jax.random.split(key)
        x, acc = mcmc(logp_fn, x_init=x_init, key=subkey, mc_steps=mc_steps, mc_width=mc_width)
        print(i, acc)

    print("check if the atom type is changed")
    print(x_init[3])
    print(x[3])

    print("check if the atom position is changed")
    print(x_init[2])
    print(x[2])