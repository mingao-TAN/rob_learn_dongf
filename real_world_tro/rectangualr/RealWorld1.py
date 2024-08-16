# -*- coding: utf-8 -*-
#!/usr/bin/env python
import rospy
import roslib
import roslaunch
import numpy as np
import tf
import cv2
# import sys
import os
import random
import math
import traceback  
import subprocess
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
import copy
import tf
#  from kobuki_msgs.msg import BumperEvent
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
import sys
sys.path.append('/home/zhw/usb_4_mic_array')
# sys.path.append('/home/zhw/usb_4_mic_array')
# from tuning import Tuning
# import usb.core
# import usb.util
import time
#  from kobuki_msgs.msg import BumperEvent
np.set_printoptions(threshold=np.inf)
class RealWorld():
# lidar + 0.01
    def __init__(self):
        # Launch the simulation with the given launchfile name
        rospy.init_node('RealWorld', anonymous=False)
#        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size = 10)

        # 创建发布者，用于发送速度命令 测试 test
        self.cmd_vel = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
#        self.action_table = [[-1.,-1.],[-1.,0.],[-1.,1.],[0.,-1.],[0.,0.],[0.,1.],[1.,-1.],[1.,0.],[1.,1.]]

        #  -------------- update 724  by mingao :  设置动作和加速度的限制 ------------------------  #
        #  -------------- update 724  by mingao : 快的情况 ------------------------  #
        # self.max_action = [1.0, np.pi/2]  # 最大线速度和角速度
        # self.min_action = [0.0, 0.0]  # 最小线速度和角速度
        # self.max_acc = [1.0, np.pi]  # 最大线加速度和角加速度
        # self.self_speed = [0.3, 0.3]  # 初始速度
        #  -------------- update 724  by mingao : 慢的情况 ------------------------  #
        self.max_action = [0.5, np.pi/5]  # 最大线速度和角速度
        self.min_action = [0.0, 0.0]  # 最小线速度和角速度
        self.max_acc = [2.0, np.pi]  # 最大线加速度和角加速度
        self.self_speed = [0.3, 0.3]  # 初始速度
        
        #  -------------- update 724  by mingao : 订阅机器人位姿和速度信息 1. 自适应蒙特卡洛方法 2.里程计信息 ------------------------  #
        self.robot_pose_sub = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped,self.PoseCallback) 
        self.robot_twist_sub = rospy.Subscriber("/odom", Odometry,self.TwistCallback) 
        # update
        self.marker_pub = rospy.Publisher('pool_state_marker', Marker, queue_size=10)
    
    #   ------------------------  #
        # self.target_position = [1.56, -4.03]
        # self.targets = [[1.56, -4.03], [1.92, 0.5]]
        
        # corridor0815
        self.target_position = [3.06, 0.08]
        self.targets = [[3.06, 0.08], [6.72, 0.139], [4.81, 2.33], [6.01, 0.193], [3.06, 0.08], [-0.38, 0.03]]
        self.count = 0

        # print("action bound is", self.max_action,"acc bound is", self.max_acc)

    #  -------------- update 724  by mingao : set the robot length and width------------------------  #
        self.length1=0.18 # front length : action core -> base  -> 0.15 (core <-> camera)
        self.length2=0.51 # back length : action core -> base -> 0.51 (core <-> back)   0815 : 0.25 0.56 0.36   1.0 pi/3     |    1.0 0.8  pi/3
        self.width=0.3  # half width  -> 0.375
        self.control_period=0.2  # 控制周期
#        rospy.sleep(2.)
        # self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=0)
        self.countm = 1
#        rospy.sleep(2.0)                                                             
#        self.show_text_in_rviz(marker_publisher, 'Goal',self.target_position[0],self.target_position[1])

    #  -------------- update 724  by mingao : add the pool state in rviz  ------------------------  #
    def publish_pool_state(self, pool_state):
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "pool_state"
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        # print(pool_state.shape[0])
        for i in range(pool_state.shape[0]):
            # print(pool_state)
            
            y = -pool_state[i, 0] * pool_state[i, 2]
            x =  pool_state[i, 1] * pool_state[i, 2]
            z = 0
            p = Point()
            p.x = x
            p.y = y
            p.z = z
            marker.points.append(p)

        self.marker_pub.publish(marker) 

    #  -------------- update by mingao :  发布相关状态 ------------------------  #
    
    #  -------------- 回调函数，用于更新机器人位姿 ------------------------  #
    def PoseCallback(self, pose):
        self.robot_pose = pose

    #  -------------- 回调函数，用于更新机器人速度 ------------------------  #
    def TwistCallback(self, twist):
        self.robot_twist = twist
    
   #  -------------- update by mingao :   将激光扫描数据离散化 ------------------------  #
    def discretize_observation(self,data):
         discretized_ranges = []
         state = []
         min_range = 0.2
         done = False
            
         for i, item in enumerate(data.ranges):
             if data.ranges[i] == float ('Inf'):
                 discretized_ranges.append(10)
             elif np.isnan(data.ranges[i]):
                 print("no data found")
                #  discretized_ranges.append(0)
            #  elif data.ranges[i]>3.5:
            #      discretized_ranges.append(3.5) 
             # elif data.ranges[i]<0.1:
             #       discretized_ranges.append(3.5)                   
             else:
                 discretized_ranges.append((data.ranges[i]))
         state = discretized_ranges;
         if min_range > np.min(state):
#            print(data.ranges[i])
             done = True
 #            print(np.min(state))
        #  print(np.min(state))
         return state,done

# ------------------------ change by mingao, 0813 minpool 代码: 将1667个点转为90个，采用分段转换的方法 ------------------------  #
    def min_pool(self, state, num_pools=90):
        # 确保输入是一维数组
        # print("state_state:", state)
        state = np.array(state).flatten()
        pooled_state = np.zeros((num_pools, 6))
        for i in range(45):
            # # 计算角度（与原代码保持一致）
            #angle = i * np.pi / 45.0 - np.pi * 3/2 + np.pi/90
            angle1 = (i * 18.0/1667)*2*np.pi + (9.0/1667)*2*np.pi
            pooled_state[i,0] = np.cos(angle1)
            pooled_state[i,1] = np.sin(angle1)
            min_value1 = np.min(state[18*i : (18*i+18)])
            pooled_state[i, 2] = min_value1
            pooled_state[i, 3] = self.length1
            pooled_state[i, 4] = self.length2
            pooled_state[i, 5] = self.width        
               
        for i in range(45,89):
            angle2 = (44.0*18.0/1667)*2*np.pi +  ( (i - 44.0) * (19.0/1667)*2.0*np.pi) +(19.0/3334)*2.0*np.pi
            pooled_state[i,0] = np.cos(angle2)
            pooled_state[i,1] = np.sin(angle2)
            min_value2 = np.min(state[45*18 + 19*(i-44) :  45*18 +(19*(i-44)+19)])
            pooled_state[i, 2] = min_value2
            pooled_state[i, 3] = self.length1
            pooled_state[i, 4] = self.length2
            pooled_state[i, 5] = self.width 
            
        angle3 = (45.0 *18.0 + 44.0*19.0 /1667)  *2.0*np.pi + (10.5/1667)*2*np.pi 
        pooled_state[89, 0] = np.cos(angle3)
        pooled_state[89, 1] = np.sin(angle3)  
        pooled_state[89, 2] =  np.min(state[1646: len(state)])
        pooled_state[89, 3] = self.length1  
        pooled_state[89, 4] = self.length2                          
        pooled_state[89, 5] = self.width
        
        for i in range(90): 
            x_dis = pooled_state[i, 0] * pooled_state[i, 2] 
            y_dis = pooled_state[i, 1] * pooled_state[i, 2]  
            # print("1111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111")
            min_value3 = math.sqrt ( y_dis*y_dis + (x_dis  + 0.05) *(x_dis  + 0.05) )
            # print(min_value3)
            # min_value3 = 100
            
            # use teachers axis, x = -y; y = x + 0.05
            pooled_state[i, 0]  =  (-1* y_dis  / min_value3)
            
            pooled_state[i, 1]  =  ((x_dis + 0.05) / min_value3)
           
            pooled_state[i, 2] =  min_value3 

            # print((pooled_state[i, 0])* min_value3, pooled_state[i, 1] * min_value3, pooled_state[i, 2])
            # if abs(x_dis)<=self.width and y_dis<=self.length1 and y_dis>=-self.length2:
            #     self.stop_counter += 1.0    

            
        return pooled_state

# ------------------------ change by mingao： 执行一步动作并返回新的状态 ------------------------  #
    def step(self):
        data = None
        terminate= False
        reset=False

        # 获取激光扫描数据
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=10)  # /scan -> /scan_fusion
                # # # # # # # # # # # # # 更新LaserScan消息
                data_length = len(data_ranges)

# ------------------------ change by mingao： 计算四分之一长度 ------------------------  #
                quarter_length = data_length // 4

# ------------------------ change by mingao： 重新排列数据，从右侧开始逆时针扫描 ------------------------  #
                data_ranges1 = np.concatenate( (data_ranges[data_length-quarter_length:data_length], data_ranges[0:data_length-quarter_length]) )
                data_ranges = data_ranges1
                # print("data range:", data_ranges)
                # data.ranges = rotated_ranges.tolist()

                # 调整角度相关的参数
                data.angle_min = (data.angle_min - np.pi/2) % (2*np.pi) - np.pi
                data.angle_max = (data.angle_max - np.pi/2) % (2*np.pi) - np.pi
                if data.angle_max < data.angle_min:
                    data.angle_max += 2*np.pi
            except:
                pass
# ------------------------ change by mingao： 处理激光扫描数据 ------------------------  #
        state,done = self.discretize_observation(data)
        pool_state = self.min_pool(state)
        data1 = pool_state[0:90, 1]
        data2 = pool_state[0:90, 2]
        
        # # 打印数据
        # print("pool_state[:,1]:", data1)
        # print("\npool_state[:,2]:",data2)
        
# ------------------------ change by mingao, 0724:  发布 pool_state 数据到 Rviz ------------------------  #
        self.publish_pool_state(pool_state)       
        pool_state = np.reshape(pool_state,(540))
        reward = 1
        in_pose = self.robot_pose.pose.pose
        position = [in_pose.position.x,in_pose.position.y]
        abs_x = self.target_position[0] - in_pose.position.x
        abs_y = self.target_position[1] - in_pose.position.y
        (roll, pitch, yaw) = euler_from_quaternion ([in_pose.orientation.x,in_pose.orientation.y,in_pose.orientation.z,in_pose.orientation.w])
        
        # ... (计算相对距离和角度)
        trans_matrix = np.matrix([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        rela = np.matmul(trans_matrix,np.array([[abs_x],[abs_y]]))
        rela_x = rela[0,0]
        rela_y = rela[1,0]
        done = False
        rela_distance = np.sqrt(rela_x** 2 + rela_y ** 2)
        rela_angle = np.arctan2(rela_y,rela_x)
        min_range = 0.2
        # if min_range > np.min(state):
        #     done = True 
# ------------------------ change by mingao:  检查是否到达目标 ------------------------  #   0815: from 0.1  -> 0.5
        if rela_distance<=0.5:
            terminate=True
            reset = True
            #self.stop
        # if done:
        #    terminate=True
        # if np.abs(rela_angle)>np.pi-0.1:
        #     terminate=True
        #     reset = True
#        print(rela_distance)
#        print(rela_angle)

        # 构建状态向量
        target_pose = [rela_distance,rela_angle]
        in_twist = self.robot_twist.twist.twist
        v = in_twist.linear.x
        w = in_twist.angular.z
        cur_act = [v,w]
        state = np.concatenate([pool_state,target_pose,cur_act,[self.max_action[0],self.max_action[1], self.max_acc[0], self.max_acc[1]]], axis=0)
#        cur_act[0] = cur_act[0]
        return state,reward, terminate,reset,position
    

# ------------------------ 控制机器人移动------------------------  #
    def Control(self,action):
        in_twist = self.robot_twist.twist.twist
        v = in_twist.linear.x
        w = in_twist.angular.z

# ----------------------- change by shanze 0815 comment clip-------
# ------------------------ 计算新的速度，考虑加速度限制------------------------  #
        # self.self_speed[0] = np.clip(action[0]*self.max_action[0],v-self.max_acc[0]*self.control_period,v+self.max_acc[0]*self.control_period)
        self.self_speed[0] = action[0]*self.max_action[0]
#        print(self.self_speed[0])        
        # self.self_speed[1] =  np.clip(action[1]*self.max_action[1],w-self.max_acc[1]*self.control_period,w+self.max_acc[1]*self.control_period)
        self.self_speed[1] =  action[1]*self.max_action[1]

# ------------------------ 发布速度命令------------------------  #
        move_cmd = Twist()
        move_cmd.linear.x = self.self_speed[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = self.self_speed[1]
        self.cmd_vel.publish(move_cmd)
        
 # ------------------------ 发布速度命令------------------------  #   
    def Control1(self,action):
        move_cmd = Twist()
        move_cmd.linear.x = 0
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = 0
        self.cmd_vel.publish(move_cmd)    
    
# ------------------------ 发布新的目标点------------------------  #
    def publish_goal(self):
        self.target_position[0] =  self.targets[self.count][0]
        self.target_position[1] =  self.targets[self.count][1]
        self.count = self.count+1

# ------------------------ 停止机器人------------------------  #
    def stop(self):
        move_cmd = Twist()
        move_cmd.linear.x = 0
        move_cmd.angular.z = 0
        self.cmd_vel.publish(move_cmd)
        
# ------------------------ 重置环境------------------------  #
    def reset(self):
        # Resets the state of the environment and returns an initial observation.
#        rospy.sleep(4.0)
#        self.show_text_in_rviz(self.targets[0][0],self.targets[0][1])
#        self.show_text_in_rviz(self.targets[1][0],self.targets[1][1])
#        self.show_text_in_rviz(self.targets[2][0],self.targets[2][1])
        rospy.sleep(3.0)
#        self.show_text_in_rviz1(self.targets[3][0],self.targets[3][1])
