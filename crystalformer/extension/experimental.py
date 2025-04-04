import jax
import jax.numpy as jnp
from functools import partial

from crystalformer.src.sample import project_xyz
from crystalformer.src.von_mises import sample_von_mises
from crystalformer.src.lattice import symmetrize_lattice


def make_cond_logp(logp_fn, forward_fn, target, alpha):
    '''
    logp_fn: function to calculate log p(x)
    forward_fn: function to calculate log p(y|x), x is G, L, XYZ, A, W
    target: target label
    alpha: hyperparameter to control the trade-off between log p(x) and log p(y|x)
    NOTE that the logp_fn and forward_fn should be vmapped before passing to this function
    '''

    def forward(G, L, XYZ, A, W, target):
        y = forward_fn(G, L, XYZ, A, W, target)
        return y

    def callback_forward(G, L, XYZ, A, W, target):
        result_shape = jax.ShapeDtypeStruct(G.shape, jnp.float32)
        return jax.experimental.io_callback(forward, result_shape, G, L, XYZ, A, W, target)

    def cond_logp_fn(params, key, G, L, XYZ, A, W, is_training):
        '''
        params: base model parameters
        '''
        # calculate log p(x)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(params, key, G, L, XYZ, A, W, is_training)
        logp_base = logp_xyz + logp_w + logp_a + logp_l

        # calculate p(y|x)
        logp_cond = callback_forward(G, L, XYZ, A, W, target)

        # trade-off between log p(x) and p(y|x)
        logp = logp_base - alpha * logp_cond.squeeze()
        return logp
    
    return cond_logp_fn


def make_mcmc_step(base_params, n_max, atom_types, atom_mask=None, constraints=None):

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
    def mcmc(logp_fn, x_init, key, mc_steps, mc_width, temp):
        """
            Markov Chain Monte Carlo sampling algorithm.

        INPUT:
            logp_fn: callable that evaluate log-probability of a batch of configuration x.
                The signature is logp_fn(x), where x has shape (batch, n, dim).
            x_init: initial value of x, with shape (batch, n, dim).
            key: initial PRNG key.
            mc_steps: total number of Monte Carlo steps.
            mc_width: size of the Monte Carlo proposal.
            temp: temperature in the smiulated annealing.

        OUTPUT:
            x: resulting batch samples, with the same shape as `x_init`.
        """

        def update_lattice(i, state):
            def update_length(key, L):
                length, angle = jnp.split(L, 2, axis=-1) 
                length += jax.random.normal(key, length.shape) * mc_width
                return jnp.concatenate([length, angle], axis=-1)

            def update_angle(key, L):
                length, angle = jnp.split(L, 2, axis=-1)
                angle += jax.random.normal(key, angle.shape) * mc_width
                return jnp.concatenate([length, angle], axis=-1)
            
            x, logp, key, num_accepts, temp = state
            G, L, XYZ, A, W = x
            key, key_proposal_L, key_accept, key_logp = jax.random.split(key, 4)

            L_proposal = jax.lax.cond(i % (n_max+2) % n_max == 0,
                                        lambda _: update_length(key_proposal_L, L),
                                        lambda _: update_angle(key_proposal_L, L),
                                        None)

            length, angle = jnp.split(L_proposal, 2, axis=-1)
            angle = jnp.rad2deg(angle)  # change the unit to degree
            L_proposal = jnp.concatenate([length, angle], axis=-1)
            L_proposal = jax.vmap(symmetrize_lattice, (0, 0))(G, L_proposal) 

            length, angle = jnp.split(L_proposal, 2, axis=-1)
            angle = jnp.deg2rad(angle)  # change the unit to rad
            L_proposal = jnp.concatenate([length, angle], axis=-1)
            
            x_proposal = (G, L_proposal, XYZ, A, W)
            logp_proposal = logp_fn(base_params, key_logp, *x_proposal, False)

            ratio = jnp.exp((logp_proposal - logp)/ temp)
            accept = jax.random.uniform(key_accept, ratio.shape) < ratio

            L_new = jnp.where(accept[:, None], L_proposal, L)  # update lattice
            x_new = (G, L_new, XYZ, A, W)
            logp_new = jnp.where(accept, logp_proposal, logp)
            num_accepts += jnp.sum(accept)

            jax.debug.print("logp {x} {y}", 
                            x=logp_new.mean(),
                            y=jnp.std(logp_new)/jnp.sqrt(logp_new.shape[0])
                            )
            return x_new, logp_new, key, num_accepts, temp
        

        def update_a_xyz(i, state):
            def true_func(i, state):
                x, logp, key, num_accepts, temp = state
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

                logp_proposal = logp_fn(base_params, key_logp, *x_proposal, False)

                ratio = jnp.exp((logp_proposal - logp)/ temp)
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
                return x_new, logp_new, key, num_accepts, temp

            def false_func(i, state):
                return state
            
            x, logp, key, num_accepts, temp = state
            A = x[3]
            x, logp, key, num_accepts, temp = jax.lax.cond(A[:, i%(n_max+2)%n_max].sum() != 0,
                                                            lambda _: true_func(i, state),
                                                            lambda _: false_func(i, state),
                                                            None)
            return x, logp, key, num_accepts, temp

        def step(i, state):
            x, logp, key, num_accepts, temp = jax.lax.cond(i % (n_max+2) < n_max,
                                                            lambda _: update_a_xyz(i, state),
                                                            lambda _: update_lattice(i, state),
                                                            None)
            return x, logp, key, num_accepts, temp

        key, subkey = jax.random.split(key)
        logp_init = logp_fn(base_params, subkey, *x_init, False)
        jax.debug.print("logp {x} {y}", 
                        x=logp_init.mean(),
                        y=jnp.std(logp_init)/jnp.sqrt(logp_init.shape[0]),
                        )
        
        x, logp, key, num_accepts, temp = jax.lax.fori_loop(0, mc_steps, step, (x_init, logp_init, key, 0., temp))
        A = x[3]
        scale = jnp.sum(A != 0)/(A.shape[0]*n_max)
        # accept_rate = num_accepts / (scale * mc_steps * x[0].shape[0])
        accept_rate = num_accepts / (scale*mc_steps*n_max/(n_max+2) + mc_steps*2/(n_max+2))
        accept_rate = accept_rate / x[0].shape[0]
        return x, accept_rate

    return mcmc
