import numpy as np
import tensorflow as tf
import time
import core
from core import get_vars
from RealWorld1 import RealWorld
import random
import rospy

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
        polyak=0.995, lr1=1e-4, lr2=1e-4,alpha=0.1, batch_size=128, start_epoch=100, 
        max_ep_len=800,MAX_EPISODE=10000):

#    logger = EpochLogger(**logger_kwargs)
#    logger.save_config(locals())
    sac=1
    obs_dim = 2200
    act_dim = 2

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
#    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
#    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
#    act_limit = env.action_space.high[0]

    # Share information about action space with policy architecture
#    ac_kwargs['action_space'] = env.action_space

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)
    train = tf.placeholder(dtype=tf.bool, shape=None)

    # Main outputs from computation graph
    with tf.variable_scope('main'):
        mu, pi, logp_pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph,train)
    
    # Target value network
    with tf.variable_scope('target'):
        _, _, _, _, _, _, _, v_targ  = actor_critic(x2_ph, a_ph,train)

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in 
                       ['main/pi', 'main/q1', 'main/q2', 'main/v', 'main'])
    print(('\nNumber of parameters: \t pi: %d, \t' + \
           'q1: %d, \t q2: %d, \t v: %d, \t total: %d\n')%var_counts)

    # Min Double-Q:
    min_q_pi = tf.minimum(q1_pi, q2_pi)

    # Targets for Q and V regression
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
    v_backup = tf.stop_gradient(min_q_pi - alpha * logp_pi)

    # Soft actor-critic losses
    pi_loss = tf.reduce_mean(alpha * logp_pi - q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q_backup - q1)**2)
    q2_loss = 0.5 * tf.reduce_mean((q_backup - q2)**2)
    v_loss = 0.5 * tf.reduce_mean((v_backup - v)**2)
    q_loss = q2_loss + q1_loss

    # Policy train op 
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    # (has to be separate from value train op, because q1_pi appears in pi_loss)
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr2)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))

    # Value train op
    # (control dep of train_pi_op because sess.run otherwise evaluates in nondeterministic order)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=lr1)
    q_params = get_vars('main/q')
#    value_params1 = get_vars('main/q2')
#    with tf.control_dependencies([train_pi_op]):
    train_q_op = q_optimizer.minimize(q_loss, var_list=q_params)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr1)
    value_params = get_vars('main/v')
#    value_params1 = get_vars('main/q2')
#    with tf.control_dependencies([train_pi_op]):
    train_v_op = value_optimizer.minimize(v_loss, var_list=value_params)
    
#    with tf.control_dependencies([train_value_op1]):    
#        train_value_op2 = value_optimizer.minimize(q2_loss, var_list=value_params1)
    # Polyak averaging for target variables
    # (control flow because sess.run otherwise evaluates in nondeterministic order)
#    with tf.control_dependencies([train_value_op2]):
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # All ops to call during one training step
    step_ops1 = [v_loss,q1_loss,q2_loss,q_loss, q1, q2, train_q_op,v,train_v_op]
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
    trainable_saver.restore(sess,"/home/zhw/attention_dot/upload/sac77lambda0-119")
    # Setup model saving
#    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
#                                outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    def get_action(o, deterministic=True,istrain= False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1),train:istrain})[0]

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
                traj[episode,T,0]=position[0]
                traj[episode,T,1]=position[1]
                env.show_text_in_rviz(position[0],position[1])
#                T = T+1
                a = get_action(o)
                print(a)
        
                # Step the env
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
	time.sleep(2)
	main()
