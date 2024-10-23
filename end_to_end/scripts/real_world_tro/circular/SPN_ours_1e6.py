'''
width is shorter than length
'''
import numpy as np
import tensorflow as tf
import gym
import time
import core_config_GMM_ada_shape_vel_acc as core
from core_config_GMM_ada_shape_vel_acc import get_vars
from stage_obs_ada_shape_vel_acc_tested import StageWorld
import random
import rospy
import os
import signal
import subprocess
import sys
from collections import deque
import scipy.stats as stats
from scipy.stats import rankdata
from scipy.stats import truncnorm

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
#Boosting Soft Actor-Critic: Emphasizing Recent Experience without Forgetting the Past
    def sample_batch(self, batch_size=128,start=0):
        idxs = np.random.randint(int(start), self.size, size=batch_size)
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
        steps_per_epoch=5000, epochs=10000, replay_size=int(2e6), gamma=0.99, 
        polyak=0.995, lr1=1e-4, lr2=1e-4,alpha=0.01, batch_size=100, start_epoch=100, 
        max_ep_len=400,MAX_EPISODE=10000):
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
    sac=2
    obs_dim = 4*90+8
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


    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    sess = tf.Session(config=config)
    reward_var = tf.Variable(0., trainable=False)
    robot_size_var = tf.Variable(0., trainable=False)
    average_speed_var = tf.Variable(0., trainable=False)
    goal_reach_var = tf.Variable(0., trainable=False)
    reward_epi = tf.summary.scalar('reward', reward_var)
    robot_size_epi = tf.summary.scalar('robot_size', robot_size_var)
    average_speed_epi = tf.summary.scalar('average_speed', average_speed_var)
    goal_reach_epi = tf.summary.scalar('goal_reach', goal_reach_var)
    # define summary
    merged_summary = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter('./logssac'+str(sac), sess.graph)

    sess.run(tf.global_variables_initializer())
    sess.run(target_init)
    trainables = tf.trainable_variables()
    trainable_saver = tf.train.Saver(trainables,max_to_keep=None)
    sess.run(tf.global_variables_initializer())
#    trainable_saver.restore(sess,"/home/zhw1993/Monocular-Obstacle-Avoidance/D3QN/sac_multi_constrain/retrain/-2899")
    # Setup model saving
#    logger.setup_tf_saver(sess, inputs={'x': x_ph, 'a': a_ph}, 
#                                outputs={'mu': mu, 'pi': pi, 'q1': q1, 'q2': q2, 'v': v})

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]


    # Main loop: collect experience in env and update/log each epoch
    episode=0
    T = 0
    env = StageWorld(540)
    rate = rospy.Rate(10)
    epi_thr = 0

        
#    test_result_plot = np.load('test_result_plot90888.npy')
    suc_record_all = np.zeros((5,150,9))
    suc_record_all_new = np.zeros((5,150,9))
    test_result_plot = np.zeros((5,5,50,200,5))
    env_record = np.zeros((5,4))
    train_result = np.zeros((5,12000))
#    np.save('suc_record_all'+str(sac)+'.npy',suc_record_all) 
#    np.save('suc_record_all_new'+str(sac)+'.npy',suc_record_all_new) 
#    np.save('test_result_plot'+str(sac)+'.npy',test_result_plot)
    test_time = 0
    for hyper_exp in range(5):
        goal_reach=0
        past_env = [0]
        current_env=[0]
        suc_record = np.zeros((5,50))
        suc_record1 = np.zeros((5,50))
        suc_record2 = np.zeros((5,50))
        suc_pointer = np.zeros(5)
        mean_rate = np.zeros(5)
        env_list = np.zeros(5)
        p = np.zeros(9)
        p[0]=1.0
        mean_rate[0]=0.0
        best_value = 0
        seed = hyper_exp
        tf.set_random_seed(seed)
        np.random.seed(seed)
        replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
        sess.run(tf.global_variables_initializer())
        sess.run(target_init)
        trainables = tf.trainable_variables()
        trainable_saver = tf.train.Saver(trainables,max_to_keep=None)
        sess.run(tf.global_variables_initializer())
#        trainable_saver.restore(sess,"/home/zhw1993/Monocular-Obstacle-Avoidance/D3QN/paper3_dot/networks/configback540GMMdense124lambda0-16")
        episode=0
        T = 0
        epi_thr = 0
        goal_reach=0
        test_time = 0
        new_env = 0
        succ_rate_test = 0
        b_test = True
        length_index=0
        while test_time<51:       
#                    current_env.remove(env_t)
            if new_env==0:
                env_no = 0
            else:
#                if np.random.uniform(0,1)<0.5:
                env_no = new_env
            print(new_env)
#            reverse_rate = 6.0-rankdata(env_list)
#            sum_rate = np.sum(reverse_rate)
#            p=reverse_rate/sum_rate
#            length_index=np.random.choice(range(5),p=p)  
            length_index=np.random.choice(range(5))  
            radius=np.random.uniform(length_index*0.1+0.1, 0.2+length_index*0.1)
            env_no=int(env_list[length_index])
         
#                else:
#                    reverse_range = rankdata(mean_rate[0:new_env], method='min')
#                    reverse_rate=1.0/(reverse_range+0.0)
#                    sum_rate = np.sum(reverse_rate)
#                    p=reverse_rate/sum_rate
#                    env_no=np.random.choice(range(new_env),p=p)    
            T_step =0
            goal_reach=0
#                if new_env==0:
#                    env_no = 0
            if goal_reach==1 and b_test:
                robot_size = env.Reset(env_no)
            else:
                robot_size = env.ResetWorld(env_no,radius)
                b_test = True
            env.GenerateTargetPoint(mean_rate[length_index])
            o, r, d,goal_reach,r2gd,robot_pose =env.step()
            rate.sleep()
            try_time=0
            while r2gd<0.3 and try_time<100:
                reset = False
                try_time = try_time+1
#                    robot_size = env.ResetWorld(env_no)
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d,goal_reach,r2gd,robot_pose =env.step()
                #env.Control([-1,0.1])
                rate.sleep()
            try_time = 0
            while d and try_time<100:
                try_time = try_time+1
                reset = False
                robot_size = env.ResetWorld(env_no,radius)
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d,goal_reach,r2gd,robot_pose =env.step()
                #env.Control([-1,0.1])
                rate.sleep()
            try_time=0
            while r2gd<0.3 and try_time<1000:
                reset = False
                try_time = try_time+1
#                    robot_size = env.ResetWorld(env_no)
                env.GenerateTargetPoint(mean_rate[length_index])
                o, r, d,goal_reach,r2gd,robot_pose =env.step()
                #env.Control([-1,0.1])
                rate.sleep()
            max_ep_len = int(40*0.8**env_no/robot_size*5)
            print("train env no. is",env_no,'target distance',round(r2gd,1),'succ rate',mean_rate[length_index],"radius",round(radius,2),"max_ep_len",max_ep_len)
            reset = False
#            power= max(4.0*(0.995**episode),0.5)
            return_epoch =0
            total_vel = 0
            ep_len = 0
    #        o = np.reshape(o,[1,56])
            d = False
            last_d = 0  
            while not reset:
                if episode > start_epoch:
                    a= get_action(o, deterministic=False)
                else:
                    a = env.PIDController()
        
                # Step the env
                env.Control(a)
                rate.sleep()
                env.Control(a)
                rate.sleep()
                past_a = a
                o2, r, d,goal_reach,r2gd,robot_pose2= env.step()
                return_epoch = return_epoch +r
                total_vel = total_vel +a[0]     
        
                # Ignore the "done" signal if it comes from hitting the time
                # horizon (that is, when it's an artificial terminal signal
                # that isn't based on the agent's state)
    #            d = False if ep_len==max_ep_len else d
        
                # Store experience to replay buffer
                replay_buffer.store(o, a, r, o2, d)
                ep_len += 1                    
        
                # Super critical, easy to overlook step: make sure to update 
                # most recent observation!
                o = o2
                last_d=d
#                if ep_len >= max_ep_len and :
#                    scores_out.append(1.0)
#                else:
#                    scores_out.append(0.0)
#                    print("Time out")
                if d:
                    if episode > start_epoch:
                        suc_record[length_index,int(suc_pointer[length_index])] = goal_reach
                        if env.stop_counter < 1.0:
                            suc_record1[length_index,int(suc_pointer[length_index])] = goal_reach
                            suc_record2[length_index,int(suc_pointer[length_index])] = 0
                        else:
                            suc_record1[length_index,int(suc_pointer[length_index])] = 0.0
                            suc_record2[length_index,int(suc_pointer[length_index])] = 0
                        suc_pointer[length_index]=(suc_pointer[length_index]+1)%50
                    reset = True
                else:
                    if ep_len >= max_ep_len: 
                        print("time out")
                        reset = True
                        if episode > start_epoch:
                            suc_record[length_index,int(suc_pointer[length_index])] = goal_reach
                            suc_record1[length_index,int(suc_pointer[length_index])] = goal_reach
                            suc_record2[length_index,int(suc_pointer[length_index])] = 1.0
                            suc_pointer[length_index]=(suc_pointer[length_index]+1)%50
                        
                        
                if  episode > start_epoch:
                    if goal_reach==1 or env.crash_stop or (ep_len >= max_ep_len):
                        if ep_len==0:
                            ep_len=1
                        reset = True
                        average_vel = total_vel/ep_len
                        """
                        Perform all SAC updates at the end of the trajectory.
                        This is a slight difference from the SAC specified in the
                        original paper.
                        """
                        for j in range(ep_len):
                            total_vel_test = 0
                            return_epoch_test = 0
                            T =T+1
                            ep_len_test = 0
                            start = np.minimum(replay_buffer.size*(1.0-(0.996**(j*1.0/ep_len*1000.0))),np.maximum(replay_buffer.size-10000,0))
                            batch = replay_buffer.sample_batch(batch_size,start=start)
                            feed_dict = {x_ph: batch['obs1'],
                                     x2_ph: batch['obs2'],
                                     a_ph: batch['acts'],
                                     r_ph: batch['rews'],
                                     d_ph: batch['done'],
                                    }
#                            print(batch['obs1'])
                            outs = sess.run(step_ops1, feed_dict) 
                            outs = sess.run(step_ops2, feed_dict)
                                # testing the agent every 5000 training steps 
                            if T%10000==0:
                                for shape_no in range(5):
                                    for k in range(50):
                                        total_vel_test = 0
                                        return_epoch_test = 0
                                        ep_len_test = 0
                                        rospy.sleep(2.0)
                                        velcity = env.set_robot_pose_test(k,int(env_list[shape_no]),shape_no)
                                        env.GenerateTargetPoint_test(k,int(env_list[shape_no]),shape_no)
                                        max_ep_len = int(40*0.8**env_no/velcity*5)
                                        o, r, d,goal_reach,r2gd,robot_pose =env.step()
                                        for i in range(1000):
                                            a = get_action(o, deterministic=True)  
                                        # Step the env
                                            env.Control(a)
                                            rate.sleep()
                                            env.Control(a)
                                            rate.sleep()
                                            o2, r, d,goal_reach,r2gd,robot_pose2= env.step()
                                            return_epoch_test = return_epoch_test +r
                                            total_vel_test = total_vel_test +a[0]           
                                        #                                    replay_buffer.store(o, a, r, o2, d)
                                            ep_len_test += 1
                                            o = o2
                                            if d or (ep_len_test >= max_ep_len):
                                                test_result_plot[hyper_exp,shape_no,k,test_time,0] = return_epoch_test
                                                test_result_plot[hyper_exp,shape_no,k,test_time,1] = goal_reach
                                                test_result_plot[hyper_exp,shape_no,k,test_time,2] = ep_len_test*1.0/max_ep_len  
                                                test_result_plot[hyper_exp,shape_no,k,test_time,3] = 1.0*goal_reach - ep_len_test*2.0/max_ep_len
                                                test_result_plot[hyper_exp,shape_no,k,test_time,4] = env.crash_stop
                                                if k==49:
                                                    mean_rate[shape_no] = np.mean(test_result_plot[hyper_exp,shape_no,:,test_time,1])
                                                    print("success rate in"+str(50)+"is:",mean_rate[shape_no])
                                                    if mean_rate[shape_no]>=0.90 and new_env<8:
                                                        env_list[shape_no]=env_list[shape_no]+1
                                                        succ_rate_test = 0
                                                        mean_rate[shape_no] = 0.0
                                                        suc_record[shape_no,:]=0.0
                                                        suc_pointer[shape_no]=0.0
                                                    np.save('test_result_plot'+str(sac)+'.npy',test_result_plot)

                    #                                            if np.sum(test_result_plot[hyper_exp,:,test_time,4])>best_value:
                    #                                                best_value = np.sum(test_result_plot[hyper_exp,:,test_time,4])
                    #                                                trainable_saver.save(sess, '/home/zhw1993/Monocular-Obstacle-Avoidance/D3QN/input_compare/ETH/networks/sac'+str(sac)+'lambda'+str(hyper_exp), global_step=test_time)
                                                    print("tested 49 times")
                                                    if shape_no==4:
                                                        test_time = test_time+1
                                                        if (test_time) % 1== 0:
                                                            trainable_saver.save(sess, '/home/zhw1993/Monocular-Obstacle-Avoidance/D3QN/paper3_dot/Good_with_backward/networks/circular'+str(sac)+'lambda'+str(hyper_exp), global_step=test_time)
                                                            rospy.sleep(3.0)
                                                    b_test = False
                                                break                                
                           
                           
                    
#            if episode > start_epoch:
#                summary_str = sess.run(merged_summary, feed_dict={reward_var: return_epoch,robot_size_var:robot_size,goal_reach_var:np.mean(scores_window),average_speed_var:average_vel})
#                summary_writer.add_summary(summary_str, episode - start_epoch)        

        # save progress every 500 episodes
            

            if episode%5==0:
                print("EPISODE"+str(sac)+str(hyper_exp), episode, "/ REWARD", return_epoch, "/ steps ", T,"env list",env_list,"succ rate each env",mean_rate)
            episode = episode + 1
            epi_thr = epi_thr+1
                

        # End of epoch wrap-up
#        if t > 0 and t % steps_per_epoch == 0:
#            epoch = t // steps_per_epoch

            # Save model

            # Test the performance of the deterministic version of the agent.
def main():
    sac(actor_critic=core.mlp_actor_critic)
if __name__ == '__main__':
	random_number = random.randint(10000, 15000)
	port = str(random_number) #os.environ["ROS_PORT_SIM"]
	os.environ["ROS_MASTER_URI"] = "http://localhost:"+port
	# os.environ["GAZEBO_MASTER_URI"] = "http://localhost:"+self.port_gazebo
	#
	# self.ros_master_uri = os.environ["ROS_MASTER_URI"];


	#start roscore
	subprocess.Popen(["roscore", "-p", port])
	time.sleep(2)
	print ("Roscore launched!")

	# Launch the simulation with the given launchfile name



	subprocess.Popen(["rosrun","stage_ros1", "stageros", "/home/zhw1993/Monocular-Obstacle-Avoidance/D3QN/paper3_dot/worlds/d8888153.world"])
	print ("environment launched!")
	time.sleep(2)
	main()
