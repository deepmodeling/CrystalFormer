import jax
import jax.numpy as jnp
from functools import partial
 
from crystalformer.src.wyckoff import mult_table


def make_classifier_loss(transformer, classifier):
    
    def forward_fn(params, state, key, G, L, XYZ, A, W, is_train):
        M = mult_table[G-1, W]  # (n_max,) multplicities
        transformer_params, classifier_params = params
        _, state = transformer(transformer_params, state, key, G, XYZ, A, W, M, is_train)

        h = state['~']['last_hidden_state']
        g = state['~']['_g_embeddings']

        key, subkey = jax.random.split(key)
        y = classifier(classifier_params, subkey, g, L, W, h, is_train)
        return y

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, None))
    def mae_loss(params, state, key, G, L, XYZ, A, W, labels, is_training):
        y = forward_fn(params, state, key, G, L, XYZ, A, W, is_training)
        return jnp.abs(y - labels)
    
    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0, None))
    def mse_loss(params, state, key, G, L, XYZ, A, W, labels, is_training):
        y = forward_fn(params, state, key, G, L, XYZ, A, W, is_training)
        return jnp.square(y - labels)
    
    def loss_fn(params, state, key, G, L, XYZ, A, W, labels, is_training):
        loss = jnp.mean(mae_loss(params, state, key, G, L, XYZ, A, W, labels, is_training))
        return loss
    
    return loss_fn, forward_fn


def make_cond_logp(logp_fn, forward_fn, target, alpha):
    '''
    logp_fn: function to calculate log p(x)
    forward_fn: function to calculate p(y|x)
    target: target label
    alpha: hyperparameter to control the trade-off between log p(x) and log p(y|x)
    NOTE that the logp_fn and forward_fn should be vmapped before passing to this function
    '''
    def cond_logp_fn(base_params, cond_params, state, key, G, L, XYZ, A, W, is_training):
        '''
        base_params: base model parameters
        cond_params: conditional model parameters
        '''
        # calculate log p(x)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(base_params, key, G, L, XYZ, A, W, is_training)
        logp_base = logp_xyz + logp_w + logp_a + logp_l

        # calculate p(y|x)
        y = forward_fn(cond_params, state, key, G, L, XYZ, A, W, is_training) # f(x)
        logp_cond = jnp.abs(target - y) # |y - f(x)|

        # trade-off between log p(x) and p(y|x)
        logp = logp_base - alpha * logp_cond.squeeze()
        return logp
    
    return cond_logp_fn


def make_multi_cond_logp(logp_fn, forward_fns, targets, alphas):
    '''
    logp_fn: function to calculate log p(x)
    forward_fns: functions to calculate p(y|x)
    targets: target labels
    alphas: hyperparameters to control the trade-off between log p(x) and log p(y|x)

    NOTE that the logp_fn and forward_fns should be vmapped before passing to this function
    '''

    num_conditions = len(forward_fns)
    assert len(forward_fns) == len(targets) == len(alphas), "The number of forward functions, targets, and alphas should be the same"
    print (num_conditions)

    def cond_logp_fn(base_params, cond_params, state, key, G, L, XYZ, A, W, is_training):
        '''
        base_params: base model parameters
        cond_params: conditional model parameters
        '''
        # calculate log p(x)
        logp_w, logp_xyz, logp_a, logp_l = logp_fn(base_params, key, G, L, XYZ, A, W, is_training)
        logp_base = logp_xyz + logp_w + logp_a + logp_l

        # calculate multiple p(y|x) 
        key, *subkeys = jax.random.split(key, num_conditions+1)
        logp_cond = 0
        for i in range(num_conditions):
            y = forward_fns[i](cond_params[i], state, subkeys[i], G, L, XYZ, A, W, is_training)
            logp_cond += jnp.abs(targets[i] - y).squeeze() * alphas[i]

        # trade-off between log p(x) and p(y|x)
        logp = logp_base - logp_cond

        return logp
    
    return cond_logp_fn


if __name__ == "__main__":
    from crystalformer.src.utils import GLXYZAW_from_file

    from model import make_classifier
    from transformer import make_transformer

    atom_types = 119
    n_max = 21
    wyck_types = 28
    Nf = 5
    Kx = 16
    Kl  = 4
    dropout_rate = 0.1 

    csv_file = '../data/mini.csv'
    G, L, XYZ, A, W = GLXYZAW_from_file(csv_file, atom_types, wyck_types, n_max)

    key = jax.random.PRNGKey(42)

    transformer_params, state, transformer = make_transformer(key, Nf, Kx, Kl, n_max, 128, 4, 4, 8, 16, 16, atom_types, wyck_types, dropout_rate) 
    classifier_params, classifier = make_classifier(key,
                                                    n_max=n_max,
                                                    embed_size=16,
                                                    sequence_length=105,
                                                    outputs_size=16,
                                                    hidden_sizes=[16, 16],
                                                    num_classes=1)

    params = (transformer_params, classifier_params)
    loss_fn, forward_fn = make_classifier_loss(transformer, classifier)

    # test loss_fn for classifier
    labels = jnp.ones(G.shape)
    value = jax.jit(loss_fn, static_argnums=9)(params, state, key, G[:1], L[:1], XYZ[:1], A[:1], W[:1], labels[:1], True)
    print (value)

    value = jax.jit(loss_fn, static_argnums=9)(params, state, key, G[:1], L[:1], XYZ[:1]+1.0, A[:1], W[:1], labels[:1], True)
    print (value)


    ############### test cond_loss_fn ################
    from loss import make_loss_fn
    from transformer import make_transformer as make_transformer_base

    base_params, base_transformer = make_transformer_base(key, Nf, Kx, Kl, n_max, 128, 4, 4, 8, 16, 16, atom_types, wyck_types, dropout_rate) 

    loss_fn, logp_fn = make_loss_fn(n_max, atom_types, wyck_types, Kx, Kl, base_transformer)

    # test_cond_loss
    forward = jax.vmap(forward_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, None))
    cond_logp_fn = make_cond_logp(logp_fn, forward, 
                                  target=1.0, 
                                  alpha=0.1)
    value = jax.jit(cond_logp_fn, static_argnums=9)(base_params, params, state, key, G, L, XYZ, A, W, False)
    print(value)
    print(value.shape)

    # test_multi_cond_loss
    forward_fns = (forward, forward)
    targets = (1.0, 1.0)
    alphas = (0.1, 0.1)
    multi_cond_logp_fn = make_multi_cond_logp(logp_fn, forward_fns, targets, alphas)
    value = jax.jit(multi_cond_logp_fn, static_argnums=9)(base_params, (params, params), state, key, G, L, XYZ, A, W, False)
    print(value)
    print(value.shape)
