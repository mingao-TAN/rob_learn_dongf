import numpy as np
import tensorflow as tf
import time
import core_config_GMM_ada_shape_vel_acc as core
from core_config_GMM_ada_shape_vel_acc import get_vars
# stage_dot_11 has larger linear velocity
from RealWorld1 import RealWorld
nets = 80
epi=499
import rospy
LENGTH = 0.45 # [m]
WIDTH1 = 0.50  # [m]
LENGTH2 = 0.55 # [m]
LENGTH1 = LENGTH+LENGTH2
BACKTOWHEEL1 = LENGTH2  # [m]
WHEEL_LEN = 0.36  # [m]
WHEEL_WIDTH = 0.24  # [m]
TREAD = 0.7  # [m]
WB = 0.5 # [m]
class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=128):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])

"""

Soft Actor-Critic

(With slight variations that bring it closer to TD3)

"""
def sac( actor_critic=core.mlp_actor_critic, seed=5, 
        steps_per_epoch=5000, epochs=10000, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr1=1e-4, lr2=1e-4,alpha=0.01, batch_size=100, start_epoch=100, 
        max_ep_len=1000,MAX_EPISODE=10000):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``mu``       (batch, act_dim)  | Computes mean actions from policy
                                           | given states.
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``. Critical: must be differentiable
                                           | with respect to policy parameters all
                                           | the way through action sampling.
            ``q1``       (batch,)          | Gives one estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q2``       (batch,)          | Gives another estimate of Q* for 
                                           | states in ``x_ph`` and actions in
                                           | ``a_ph``.
            ``q1_pi``    (batch,)          | Gives the composition of ``q1`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q1(x, pi(x)).
            ``q2_pi``    (batch,)          | Gives the composition of ``q2`` and 
                                           | ``pi`` for states in ``x_ph``: 
                                           | q2(x, pi(x)).
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. 
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

#    logger = EpochLogger(**logger_kwargs)
#    logger.save_config(locals())
    sac=1
    obs_dim = 540+8
    act_dim = 2

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
#    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
#    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi = actor_critic(x_ph, a_ph)
    
    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _,_, q1_pi_targ, q2_pi_targ  = actor_critic(x2_ph, a_ph)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main/values', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi_targ, q2_pi_targ)
    min_q = tf.minimum(q1_pi, q2_pi)
#    min_q_pi = tf.maximum(min_q_pi, 10.0)
#    min_q_pi = tf.minimum(min_q_pi, (24-0.28+tf.log(4.0)))
#    min_q_pi = tf.minimum(min_q_pi, 24.0)

    # Targets for Q and V regression
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_backup)
    

    regularizerpi = tf.contrib.layers.l2_regularizer(scale=0.001, scope='main/pi')
    all_trainable_weights_pi = tf.trainable_variables(scope='main/pi')
    regularization_penalty_pi = tf.contrib.layers.apply_regularization(regularizerpi, all_trainable_weights_pi)
#        policy_loss = (policy_kl_loss
#                       + policy_regularization_loss + regularization_penalty_pi - self.ent_coef * policy_entropy)

    # Soft actor-critic losses
    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - min_q)+ regularization_penalty_pi
    q1_loss = tf.reduce_mean((q1-backup)**2)
    q2_loss = tf.reduce_mean((q2-backup)**2)
    q_loss =  q2_loss + q1_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr2)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr1)
    value_params = get_vars('main/values')
#    value_params1 = get_vars('main/q2')
#    with tf.control_dependencies([train_pi_op]):
    train_q_op = value_optimizer.minimize(q_loss, var_list=value_params)
#    with tf.control_dependencies([train_value_op1]):    
#        train_value_op2 = value_optimizer.minimize(q2_loss, var_list=value_params1)
    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
#    with tf.control_dependencies([train_value_op2]):
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops1 = [q_loss, q1, q2, train_q_op]
    step_ops2 = [pi_loss, train_pi_op, target_update]

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    trainables = tf.trainable_variables()
    trainable_saver = tf.train.Saver(trainables,max_to_keep=None)
    sess.run(tf.global_variables_initializer())
    trainable_saver.restore(sess,"/home/zhw/rect/Realworld/real_world_tro/rectangualr/configback540GMMdense1lambda0-100")
#    trainable_saver.restore(sess,"/home/zhw1993/Monocular-Obstacle-Avoidance/D3QN/sac_multi_constrain/retrain/-2899")
    # Setup model saving
#    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
#                                outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

    # Main loop: collect experience in env and update/log each epoch
    env = RealWorld()
    rate = rospy.Rate(10)
    episode=0
    traj = np.zeros((10,1000,2))
    dtime = np.zeros((10))
    env.reset()
#    for i in range(100):
#        env.show_text_in_rviz(i,i)
    while episode < MAX_EPISODE:
            env.publish_goal()
            start_time = time.time()
            if episode==0:
                o, r, d,goal_reach,position =env.step()
            T=0
#                traj[episode,0,1]=
            for T in range(1000):
                print(T)
                traj[episode,T,0]=position[0]
                traj[episode,T,1]=position[1]
                env.show_text_in_rviz(position[0],position[1])
#                T = T+1
                a = get_action(o)
                print(a)
        
                # Step the env
                env.Control(a)
                rate.sleep()
                env.Control(a)
                rate.sleep()
                o2, r, d,goal_reach,position= env.step()
                o = o2
                np.save(str(sac)+".npy",traj)
                if d:
                    end_time = time.time()
                    dtime[episode] = end_time-start_time
                    print(dtime[episode])
                    np.save(str(sac)+"dtime.npy",dtime)
                    for k in range(T,T+20):
                        o, r, d,goal_reach,position =env.step()
                        traj[episode,k,0]=position[0]
                        traj[episode,k,1]=position[1] 
                        rate.sleep()    
                    np.save(str(sac)+".npy",traj)                                    
                    break
            episode = episode+1
def main():
    sac(actor_critic=core.mlp_actor_critic)
if __name__ == '__main__':
	time.sleep(1)
	main()
