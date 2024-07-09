import numpy as np
import tensorflow as tf

EPS = 1e-8
def clip_but_pass_gradient2(x, l=EPS):
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((l - x)*clip_low)
def new_relu(x, alpha_actv):
    r = tf.math.reciprocal(clip_but_pass_gradient2(x+alpha_actv,l=EPS))
    return r #+ part_3*0
def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=(None,dim) if dim else (None,))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def mlp(x, hidden_sizes=(32,), activation=tf.nn.leaky_relu, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)
def mlp_policy(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes:
        x = tf.layers.dense(x, units=h, activation=activation)
#        x = tf.layers.dropout(x,0.1,training=trainp)
    return x
def CNN(x, y,activation=tf.nn.relu, output_activation=None):
    x1 = tf.layers.conv1d(x, filters=32, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x11 = tf.layers.conv1d(x1, filters=64, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x0 = tf.layers.conv1d(x11, filters=128, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x3 = tf.layers.max_pooling1d(x0, pool_size=90, strides=1, padding='valid')
    x3_flatten = tf.layers.flatten(x3)
    return x3_flatten
def CNN_dense(x,activation=tf.nn.leaky_relu, output_activation=None):
    alpha_actv2= tf.Variable(initial_value=0.0, dtype='float32', trainable=True)
    x_input = x[:,0:6*90]
    x_input = tf.reshape(x_input,[-1,90,6])
    x00 = new_relu(x_input[:,:,2], alpha_actv2)
    x_input = tf.concat([x_input[:,:,0:2],tf.reshape(x00,[-1,90,1]),x_input[:,:,3:6]], axis=-1)
    x1 = tf.layers.conv1d(x_input, filters=32, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x11 = tf.layers.conv1d(x1, filters=64, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x0 = tf.layers.conv1d(x11, filters=128, kernel_size=1, strides=1, padding='valid',activation=tf.nn.leaky_relu)
    x3 = tf.layers.max_pooling1d(x0, pool_size=90, strides=1, padding='valid')
    x3_flatten = tf.layers.flatten(x3)
    return tf.concat([x3_flatten,x[:,6*90:]], axis=-1)
def CNN2(x, activation=tf.nn.relu, output_activation=None,):
    x1 = tf.layers.conv1d(x, filters=32, kernel_size=20, strides=10, padding='same',activation=tf.nn.relu)
    x2 = tf.layers.conv1d(x1, filters=16, kernel_size=10, strides=3, padding='same',activation=tf.nn.relu)
    x2_flatten = tf.layers.flatten(x2)
    x3 = tf.layers.dense(x2_flatten, units=128, activation=tf.nn.relu)
    return x3
def get_vars(scope):
    return [x for x in tf.global_variables() if scope in x.name]

def count_vars(scope):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = tf.cast(x > u, tf.float32)
    clip_low = tf.cast(x < l, tf.float32)
    return x + tf.stop_gradient((u - x)*clip_up + (l - x)*clip_low)


"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def create_log_gaussian(mu_t, log_sig_t, t):
    normalized_dist_t = (t - mu_t) * tf.exp(-log_sig_t)  # ... x D
    quadratic = - 0.5 * tf.reduce_sum(normalized_dist_t ** 2, axis=-1)
    # ... x (None)

    log_z = tf.reduce_sum(log_sig_t, axis=-1)  # ... x (None)
    D_t = tf.cast(tf.shape(mu_t)[-1], tf.float32)
    log_z += 0.5 * D_t * np.log(2 * np.pi)

    log_p = quadratic - log_z

    return log_p  # ... x (None)


def mlp_gaussian_policy(x, a,hidden_sizes, activation, output_activation,alpha_actv1):
    k=4
    act_dim = a.shape.as_list()[-1]
    x_input = x[:,0:6*90]
    x_input = tf.reshape(x_input,[-1,90,6])
    x0 = new_relu(x_input[:,:,2], alpha_actv1)
    x_input = tf.concat([x_input[:,:,0:2],tf.reshape(x0,[-1,90,1]),x_input[:,:,3:6]], axis=-1)
    w_input = x[:,6*90:6*90+8]
    w_input = tf.reshape(w_input,[-1,8])
    cnn_net = CNN(x_input,w_input)
    y = tf.concat([cnn_net,x[:,6*90:]], axis=-1)
    net = mlp_policy(y,list(hidden_sizes), activation, activation)
    w_and_mu_and_logsig_t = tf.layers.dense(net, (act_dim*2+1)*k, activation=output_activation)
    w_and_mu_and_logsig_t = tf.reshape( w_and_mu_and_logsig_t, shape=(-1, k, 2*act_dim+1))
    log_w_t = w_and_mu_and_logsig_t[..., 0]
    mu_t = w_and_mu_and_logsig_t[..., 1:1+act_dim]
    log_sig_t = w_and_mu_and_logsig_t[..., 1+act_dim:]

    """
    Because algorithm maximizes trade-off of reward and entropy,
    entropy must be unique to state---and therefore log_stds need
    to be a neural network output instead of a shared-across-states
    learnable parameter vector. But for deep Relu and other nets,
    simply sticking an activationless dense layer at the end would
    be quite bad---at the beginning of training, a randomly initialized
    net could produce extremely large values for the log_stds, which
    would result in some actions being either entirely deterministic
    or too random to come back to earth. Either of these introduces
    numerical instability which could break the algorithm. To 
    protect against that, we'll constrain the output range of the 
    log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
    slightly different from the trick used by the original authors of
    SAC---they used tf.clip_by_value instead of squashing and rescaling.
    I prefer this approach because it allows gradient propagation
    through log_std where clipping wouldn't, but I don't know if
    it makes much of a difference.
    """
#    log_std = tf.layers.dense(net, act_dim, activation=tf.tanh)
    log_sig_t = tf.tanh(log_sig_t)
    
    log_sig_t = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_sig_t + 1)
            # (N x K), (N x K x Dx), (N x K x Dx)
    xz_sigs_t = tf.exp(log_sig_t)

            # Sample the latent code.
    z_t = tf.multinomial(logits=log_w_t, num_samples=1)  # N x 1

            # Choose mixture component corresponding to the latent.
    mask_t = tf.one_hot( z_t[:, 0], depth=k, dtype=tf.bool,on_value=True, off_value=False )
    xz_mu_t = tf.boolean_mask(mu_t, mask_t)  # N x Dx
    xz_sig_t = tf.boolean_mask(xz_sigs_t, mask_t)  # N x Dx

            # Sample x.
    x_t = xz_mu_t + xz_sig_t * tf.random_normal((tf.shape(net)[0], act_dim))  # N x Dx
#    x_t = tf.stop_gradient(x_t)

            # log p(x|z)
    log_p_xz_t = create_log_gaussian(mu_t, log_sig_t, x_t[:, None, :])  # N x K

            # log p(x)
    log_p_x_t = tf.reduce_logsumexp(log_p_xz_t + log_w_t, axis=1)
    log_p_x_t -= tf.reduce_logsumexp(log_w_t, axis=1)  # N
    

    logp_pi = log_p_x_t
    return xz_mu_t,x_t, logp_pi

def apply_squashing_func(mu, pi, logp_pi):
    mu = tf.tanh(mu)
    pi = tf.tanh(pi)
#    pi_run = pi
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), axis=1)
    return mu, pi, logp_pi


"""
Actor-Critics
"""
def mlp_actor_critic(x, a,hidden_sizes=(128,128,128,128), activation=tf.nn.leaky_relu, 
                     output_activation=None, policy=mlp_gaussian_policy, action_space=None):
    # policy
    with tf.variable_scope('pi'):
        alpha_actv1 = tf.Variable(initial_value=0.0, dtype='float32', trainable=True)
        mu, pi, logp_pi = policy(x, a,[128,128,128,128], activation, output_activation,alpha_actv1)
        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

    # make sure actions are in correct range
#    action_scale = action_space.high[0]
#    mu *= action_scale
#    pi *= action_scale

    # vfs
    # the dim of q function and v function is 1, and hence it use +[1]
#    vf_cnn = lambda x : CNN(x)
#    vf_mlp = lambda y : tf.squeeze(mlp(y, list(hidden_sizes)+[1], activation, None), axis=1)
#    with tf.variable_scope('q1'):
#        q1 = vf_mlp(tf.concat([x,a], axis=-1))
#    with tf.variable_scope('q1', reuse=True):
#        q1_pi = vf_mlp(tf.concat([x,pi], axis=-1))
#    with tf.variable_scope('q2'):
#        q2 = vf_mlp(tf.concat([x,a], axis=-1))
#    with tf.variable_scope('q2', reuse=True):
#        q2_pi = vf_mlp(tf.concat([x,pi], axis=-1))
#    with tf.variable_scope('v'):
#        v = vf_mlp(x)        
    vf_mlp = lambda y : tf.squeeze(mlp(y, list(hidden_sizes)+[1], activation, None), axis=1)
    with tf.variable_scope('values'):   
        with tf.variable_scope('CNN'):
            y = CNN_dense(x,activation, None)
        with tf.variable_scope('q1'):
            q1 = vf_mlp(tf.concat([y,a], axis=-1))
        with tf.variable_scope('q1', reuse=True):
            q1_pi = vf_mlp(tf.concat([y,pi], axis=-1))
        with tf.variable_scope('q2'):
            q2 = vf_mlp(tf.concat([y,a], axis=-1))
        with tf.variable_scope('q2', reuse=True):
            q2_pi = vf_mlp(tf.concat([y,pi], axis=-1))
    return mu, pi, logp_pi, q1, q2, q1_pi, q2_pi
