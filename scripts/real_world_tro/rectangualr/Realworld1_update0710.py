# -*- coding: UTF-8 -*-
#!/usr/bin/env python
import rospy
import roslib
import roslaunch
import time
import numpy as np
import tf
import cv2
import sys
import os
import random
import traceback 
import subprocess
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from std_srvs.srv import Empty
from sensor_msgs.msg import LaserScan
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Path
from nav_msgs.msg import Odometry
import time
import copy
import tf
from kobuki_msgs.msg import BumperEvent
from geometry_msgs.msg import PoseWithCovarianceStamped, Point, Quaternion, Vector3
from visualization_msgs.msg import Marker
from std_msgs.msg import Header, ColorRGBA
import sys
sys.path.append('/home/zhw/usb_4_mic_array')
from tuning import Tuning
import usb.core
import usb.util
import time
from kobuki_msgs.msg import BumperEvent

class RealWorld():
# lidar + 0.01
    def __init__(self):
        # Launch the simulation with the given launchfile name
        rospy.init_node('RealWorld', anonymous=False)
#        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size = 10)

        # 创建发布者，用于发送速度命令
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size = 10)
#        self.action_table = [[-1.,-1.],[-1.,0.],[-1.,1.],[0.,-1.],[0.,0.],[0.,1.],[1.,-1.],[1.,0.],[1.,1.]]

        # 设置动作和加速度的限制
        self.max_action = [0.5, np.pi/2]  # 最大线速度和角速度
        self.min_action = [0.0, -np.pi/2]  # 最小线速度和角速度
        self.max_acc = [0.5, np.pi/2]  # 最大线加速度和角加速度
        self.self_speed = [0.3, 0.0]  # 初始速度

        # 订阅机器人位姿和速度信息
        self.robot_pose_sub = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped,self.PoseCallback) 
        self.robot_twist_sub = rospy.Subscriber("/odom", Odometry,self.TwistCallback) 

        # 设置目标位置和多个目标点
        self.target_position = [-0.10, 1.53]
        self.targets = [[-0.10, 1.53], [-2.4, -0.05], [-4.1, -0.739], [0.0, -0.663]]
        self.count = 0

        #the size of the real robot
        self.max_action[0] = 0.5
        self.max_action[1] = np.pi/2
        self.max_acc[0] = 1.0
        self.max_acc[1] = np.pi
        #print("action bound is", self.max_action,"acc bound is", self.max_acc)

        # 设置机器人的物理参数
        self.length1=0.15  # front length : action core -> base  -> 0.15 (core <-> camera)
        self.length2=0.51  # back length : action core -> base -> 0.51 (core <-> back)
        self.width=0.375  # half width  -> 0.375
        self.control_period=0.2  # 控制周期
#        rospy.sleep(2.)
        # self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=0)
        self.countm = 1
#        rospy.sleep(2.0)                                                             
#        self.show_text_in_rviz(marker_publisher, 'Goal',self.target_position[0],self.target_position[1])

    # 回调函数，用于更新机器人速度
    def PoseCallback(self, pose):
        self.robot_pose = pose

    # 回调函数，用于更新机器人速度
    def TwistCallback(self, twist):
        self.robot_twist = twist

    # def show_text_in_rviz(self, goal_x,goal_y):
    #     if self.countm<=3:
    #         scale1 = 0.5
    #         color1 = ColorRGBA(1.0, 0.0, 0.0, 1.0)
    #     else:
    #         scale1 = 0.3
    #         color1 = ColorRGBA(1.0, 0.0, 0.5, 0.5)           
    #     marker = Marker(
    #                 type=Marker.CYLINDER,
    #                 id=self.countm,
    #                 lifetime=rospy.Duration(1000),
    #                 pose=Pose(Point(goal_x, goal_y, 0.2), Quaternion(0, 0, 0, 1)),
    #                 scale=Vector3(scale1, scale1, scale1),
    #                 header=Header(frame_id='map'),
    #                 color=color1)
    #     self.marker_publisher.publish(marker)
    #     self.countm = self.countm+1
    # def show_text_in_rviz1(self, goal_x,goal_y):
    #     marker1 = Marker(
    #                 type=Marker.SPHERE,
    #                 id=self.countm,
    #                 lifetime=rospy.Duration(1000),
    #                 pose=Pose(Point(goal_x, goal_y, 0.2), Quaternion(0, 0, 0, 1)),
    #                 scale=Vector3(0.1, 0.1, 0.1),
    #                 header=Header(frame_id='map'),
    #                 color=ColorRGBA(1.0, 0.0, 0.8, 0.0))
    #     self.marker_publisher.publish(marker1)
    #     self.countm = self.countm+1

    # 将激光扫描数据离散化
    def discretize_observation(self,data,new_ranges):
        discretized_ranges = []
        state = []
        min_range = 0.2
        done = False

#        mod = 720/new_ranges
#        for i in range(1080):
#            if data.ranges[i]<0.2:
#                print(i)
#                print(data.ranges[i])
#            print(1111111111111111111111111111111111111111111111)
        for i, item in enumerate(data.ranges):
            if data.intensities[i]<100:
                discretized_ranges.append(3.5)
            elif data.ranges[i] == float ('Inf'):
                discretized_ranges.append(3.5)
            elif np.isnan(data.ranges[i]):
                discretized_ranges.append(0)
            elif data.ranges[i]>3.5:
                discretized_ranges.append(3.5) 
            elif data.ranges[i]<0.1:
                discretized_ranges.append(3.5)                   
            else:
                discretized_ranges.append((data.ranges[i]))
        for i in range(new_ranges):
            state.append(np.min(discretized_ranges[i]))
#        print(state)
        if min_range > np.min(state):
#            print(data.ranges[i])
            done = True
#            print(np.min(state))
        return state,done


    def min_pool(self, state, num_pools=90):
        # 确保输入是一维数组
        state = np.array(state).flatten()
        
        # 计算每组的大小
        group_size = len(state) // num_pools
        remainder = len(state) % num_pools
        
        # 初始化结果数组
        pooled_state = np.zeros((num_pools, 6))
        
        for i in range(num_pools):
            # 计算当前组的起始和结束索引
            start = i * group_size + min(i, remainder)
            end = (i + 1) * group_size + min(i + 1, remainder)
            
            # 取当前组的最小值
            min_value = np.min(state[start:end])
            
            # 计算角度（与原代码保持一致）
            angle = i * np.pi / 45.0 - np.pi * 3/2 + np.pi/90
            
            # 填充池化结果
            pooled_state[i, 0] = np.cos(angle)
            pooled_state[i, 1] = np.sin(angle)
            pooled_state[i, 2] = min_value
            pooled_state[i, 3] = self.length1
            pooled_state[i, 4] = self.length2
            pooled_state[i, 5] = self.width
        
        return pooled_state.flatten()


    # 执行一步动作并返回新的状态
    def step(self):
        data = None
        terminate= False
        reset=False

        # 获取激光扫描数据
        while data is None:
            try:
                data = rospy.wait_for_message('/scan_fusion', LaserScan, timeout=10)  # /scan -> /scan_fusion
                # new code : add offset to the data.ranges(laser_link -> base_link)
                # 添加偏移量
                data_ranges = np.array(data.ranges) + 0.05

                # new code : turn data 90 degrees to fit the teacher's tf 
                # 旋转数据90度（顺时针）
                rotated_ranges = np.zeros_like(data_ranges)
                rotation_index = int(len(data_ranges) * 0.25)  # 90度对应四分之一的数组长度
                rotated_ranges = np.roll(data_ranges, rotation_index)

                # 更新LaserScan消息
                data.ranges = rotated_ranges.tolist()

                # 调整角度相关的参数
                data.angle_min = (data.angle_min - np.pi/2) % (2*np.pi) - np.pi
                data.angle_max = (data.angle_max - np.pi/2) % (2*np.pi) - np.pi
                if data.angle_max < data.angle_min:
                    data.angle_max += 2*np.pi

            except:
                pass
        state,done = self.discretize_observation(data,1667)
        state = np.reshape(state,(1667))  # 360 points 


        # 处理激光扫描数据
        # pool_state = np.zeros((90,6))  # min_pooling - > 90
        # for i in range(90):
        #     pool_state[i,0] = np.cos(i*np.pi/45.0-np.pi*3/2+np.pi/90)  # direction angle 
        #     pool_state[i,1] = np.sin(i*np.pi/45.0-np.pi*3/2+np.pi/90)
        #     dis = np.min(state[4*i:(4*i+4)])
        #     x_dis = pool_state[i,0]*dis
        #     y_dis = pool_state[i,1]*dis
        #     pool_state[i,2] = dis
        #     pool_state[i,3] = self.length1
        #     pool_state[i,4] = self.length2
        #     pool_state[i,5] = self.width
            #if abs(x_dis)<=self.width and y_dis<=self.length1 and y_dis>=-self.length2:
                #self.stop_counter += 1.0

#            if abs( pool_state[i,0])<=0.2 and abs(pool_state[i,1])<=0.2:
#                self.stop_counter =1 

        pool_state = self.min_pool(state)
        pool_state = np.reshape(pool_state,(540))
#        print(state)
        reward = 1

        # 计算相对于目标的位置和角度
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
        rela_distance = np.sqrt(rela_x** 2 + rela_y ** 2)
        rela_angle = np.arctan2(rela_y,rela_x)

        # 检查是否到达目标
        if rela_distance<=0.2:
            terminate=True
            reset = True
#        if done:
#            terminate=True
#        if np.abs(rela_angle)>np.pi-0.1:
#            terminate=True
#            reset = True
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
    

    # 控制机器人移动
    def Control(self,action):
        in_twist = self.robot_twist.twist.twist
        v = in_twist.linear.x
        w = in_twist.angular.z

        # 计算新的速度，考虑加速度限制
        self.self_speed[0] = np.clip(action[0]*self.max_action[0],v-self.max_acc[0]*self.control_period,v+self.max_acc[0]*self.control_period)
#        print(self.self_speed[0])        
        self.self_speed[1] =  np.clip(action[1]*self.max_action[1],w-self.max_acc[1]*self.control_period,w+self.max_acc[1]*self.control_period)

        # 发布速度命令
        move_cmd = Twist()
        move_cmd.linear.x = self.self_speed[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = self.self_speed[1]
        self.cmd_vel.publish(move_cmd)

    # 发布新的目标点
    def publish_goal(self):
        self.target_position[0] =  self.targets[self.count][0]
        self.target_position[1] =  self.targets[self.count][1]
        self.count = self.count+1

    # 停止机器人
    def stop(self):
        move_cmd = Twist()
        move_cmd.linear.x = 0
        move_cmd.angular.z = 0
        self.cmd_vel.publish(move_cmd)
        
    # 重置环境
    def reset(self):
        # Resets the state of the environment and returns an initial observation.
#        rospy.sleep(4.0)
#        self.show_text_in_rviz(self.targets[0][0],self.targets[0][1])
#        self.show_text_in_rviz(self.targets[1][0],self.targets[1][1])
#        self.show_text_in_rviz(self.targets[2][0],self.targets[2][1])
        rospy.sleep(3.0)
#        self.show_text_in_rviz1(self.targets[3][0],self.targets[3][1])
