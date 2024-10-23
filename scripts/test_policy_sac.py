# -*- coding: utf-8 -*-
# import os
from os.path import dirname, abspath, join
import argparse
import sys
import numpy as np
import torch
import rospy
import time
from geometry_msgs.msg import PointStamped
from waypoint import GlobalPlanSampler
sys.path.append(dirname(dirname(abspath(__file__))))
from os.path import dirname, abspath, join
import gym
#  import tensorrt as trt
# import tensorflow as tf
import tensorflow._api.v2.compat.v1 as tf
tf.disable_v2_behavior()
tf.get_logger().setLevel('ERROR')
# add the gpu root
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
# add the gpu root
import yaml
import real_world_tro.rectangualr.core_config_GMM_ada_shape_vel_acc as core

from envs.wrappers import ShapingRewardWrapper, StackFrame
from td3.train import initialize_policy
# 与训练代码类似的模型定义和初始化部分
obs_dim = 548
act_dim = 2

x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

with tf.variable_scope('main'):
    mu, pi, logp_pi, q1, q2, q1_pi, q2_pi = core.mlp_actor_critic(x_ph, a_ph)

sess = tf.Session()
saver = tf.train.Saver()
# saver.restore(sess,"/home/eias/20240118/Monocular-Obstacle-Avoidance/D3QN/paper3_dot/Good_with_backward/network/configback540GMMdense11lambda0-100")
saver.restore(sess,"/home/eias/20240118/Monocular-Obstacle-Avoidance/D3QN/paper3_dot/Good_with_backward/network11/configback540GMMdense101lambda0-100")
#saver.restore(sess,"/home/eias/end2end/jackal_ws/src/barn-EIT-NUS/end_to_end/scripts/network3/configback540GMMdense11lambda0-99")
#saver.restore(sess,"/home/eias/end2end/jackal_ws/src/barn-EIT-NUS/end_to_end/scripts/network4/configback540GMMdense1111lambda4-98")
#saver.restore(sess,"/home/eias/Desktop/network/networks/configback540GMMdense111lambda0-100")
#saver.restore(sess,"/home/eias/end2end/jackal_ws/src/barn-EIT-NUS/end_to_end/scripts/real_world_tro/rectangualr/configback540GMMdense1lambda0-100")
sys.path.append(dirname(dirname(abspath(__file__))))

def get_action(o, deterministic= True):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1,-1)})[0]

def get_world_name(config, id):
    assert 0 <= id < 300, "BARN dataset world index ranges from 0-299"
    world_name = "BARN/world_%d.world" %(id)
    return world_name
def target_points_callback(msgs):
    return msgs

def load_policy(policy, policy_path):
    print("-------------------------------------------------------------------------------------------------------------------------------")
    print(policy_path)
    policy.load(policy_path, "last_policy")
    policy.exploration_noise = 0
    return policy

def _debug_print_robot_status(env, count, rew, actions):
    Y = env.move_base.robot_config.Y
    X = env.move_base.robot_config.X
    p = env.gazebo_sim.get_model_state().pose.position
    print(actions)
    print('current step: %d, X position: %f(world_frame), %f(odem_frame), Y position: %f(world_frame), %f(odom_frame), rew: %f' %(count, p.x, X, p.y, Y , rew))

def main(args):
    with open(join(args.policy_path, "config.yaml"), "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    env_config = config['env_config']
    world_name = get_world_name(config, args.id)

    env_config["kwargs"]["world_name"] = world_name
    if args.gui:
        env_config["kwargs"]["gui"] = True
    env_config["kwargs"]["init_sim"] = False
    env = gym.make(env_config["env_id"], **env_config["kwargs"])
    env = StackFrame(env, stack_frame=env_config["stack_frame"])
    # policy, _ = initialize_policy(config, env)
    # policy = load_policy(policy, args.policy_path)
    print(">>>>>>>>>>>>>> Running on %s <<<<<<<<<<<<<<<<" %(world_name))
    ep = 0
    while ep < args.repeats:
        obs = env.reset()
        ep += 1
        print(">>>>>>>>>>>>>> Running on the step number %s <<<<<<<<<<<<<<<<" % ep)
        step = 0
        done = False
        while True:
            if not args.default_dwa:
                gpu_devices = tf.config.list_physical_devices()
                if gpu_devices:
                    print("GPU")
                actions = get_action(obs)
            else:
                actions = get_action(obs)
            x1 = time.time()
            obs_new, rew, done, info = env.step(actions)
            x2 = time.time()
            x3 = x1 - x2
            print("time11111111111111111111111111:",x3)
            info["world"] = world_name
            obs = obs_new
            step += 1

            if args.verbose:
                _debug_print_robot_status(env, step, rew, actions)
    env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'start an tester')
    parser.add_argument('--world_id', dest='id', type=int, default=0)
    parser.add_argument('--policy_path', type=str, default="end_to_end/data_sac")
    parser.add_argument('--default_dwa', action="store_true")
    parser.add_argument('--verbose', action="store_true")
    parser.add_argument('--gui', action="store_true")
    parser.add_argument('--repeats', type=int, default=1)
    args = parser.parse_args()
    main(args)

