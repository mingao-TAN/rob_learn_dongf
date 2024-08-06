# -*- coding: utf-8 -*-
#!/usr/bin/env python
import numpy as np
import tensorflow as tf
import time
import core_config_GMM_ada_shape_vel_acc as core
from RealWorld1 import RealWorld
import rospy

# 与训练代码类似的模型定义和初始化部分
obs_dim = 540 + 8
act_dim = 2

x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

with tf.variable_scope('main'):
    mu, pi, logp_pi, q1, q2, q1_pi, q2_pi = core.mlp_actor_critic(x_ph, a_ph)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,"/home/wheeltec-client/Desktop/rob_learn_dongf-real_world_test/real_world_tro/rectangualr/configback540GMMdense1lambda0-100")

def get_action(o, deterministic=True):
    act_op = mu if deterministic else pi
    return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})[0]

env = RealWorld()
env.reset()

rate = rospy.Rate(10)
o, r, d, goal_reach, position = env.step()

while not goal_reach:
    a = get_action(o)
    env.Control(a)
    rate.sleep()
    o, r, d, goal_reach, position = env.step()

# 添加停止模块
while goal_reach:
    env.stop()
    

