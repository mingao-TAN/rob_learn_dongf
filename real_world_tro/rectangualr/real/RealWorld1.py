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

    def __init__(self):
        # Launch the simulation with the given launchfile name
        rospy.init_node('RealWorld', anonymous=False)
#        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size = 10)
        self.cmd_vel = rospy.Publisher('cmd_vel_mux/input/navi', Twist, queue_size = 10)
#        self.action_table = [[-1.,-1.],[-1.,0.],[-1.,1.],[0.,-1.],[0.,0.],[0.,1.],[1.,-1.],[1.,0.],[1.,1.]]
        self.max_action = [0.5,np.pi/2]
        self.min_action = [0.0,-np.pi/2]
        self.max_acc = [0.5,np.pi/2]
        self.self_speed = [0.3, 0.0]
        self.robot_pose_sub = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped,self.PoseCallback) 
        self.target_position=[0.03,-0.256]
        self.targets=[[0.03,-0.256],[3.0,-3.0]]
        self.count = 0
        #the size of the real robot
        self.max_action[0] = 0.5
        self.max_action[1] = np.pi/2
        self.max_acc[0] = 0.5
        self.max_acc[1] = np.pi/2
        #print("action bound is", self.max_action,"acc bound is", self.max_acc)
        self.length1=0.4
        self.length2=0.2
        self.width=0.2
#        rospy.sleep(2.)
        self.marker_publisher = rospy.Publisher('visualization_marker', Marker, queue_size=0)
        self.countm = 1
#        rospy.sleep(2.0)                                                             
#        self.show_text_in_rviz(marker_publisher, 'Goal',self.target_position[0],self.target_position[1])
    def PoseCallback(self, pose):
        self.robot_pose = pose
    def show_text_in_rviz(self, goal_x,goal_y):
        if self.countm<=3:
            scale1 = 0.5
            color1 = ColorRGBA(1.0, 0.0, 0.0, 1.0)
        else:
            scale1 = 0.3
            color1 = ColorRGBA(1.0, 0.0, 0.5, 0.5)           
        marker = Marker(
                    type=Marker.CYLINDER,
                    id=self.countm,
                    lifetime=rospy.Duration(1000),
                    pose=Pose(Point(goal_x, goal_y, 0.2), Quaternion(0, 0, 0, 1)),
                    scale=Vector3(scale1, scale1, scale1),
                    header=Header(frame_id='map'),
                    color=color1)
        self.marker_publisher.publish(marker)
        self.countm = self.countm+1
    def show_text_in_rviz1(self, goal_x,goal_y):
        marker1 = Marker(
                    type=Marker.SPHERE,
                    id=self.countm,
                    lifetime=rospy.Duration(1000),
                    pose=Pose(Point(goal_x, goal_y, 0.2), Quaternion(0, 0, 0, 1)),
                    scale=Vector3(0.1, 0.1, 0.1),
                    header=Header(frame_id='map'),
                    color=ColorRGBA(1.0, 0.0, 0.8, 0.0))
        self.marker_publisher.publish(marker1)
        self.countm = self.countm+1

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


    def step(self):
        data = None
        terminate= False
        reset=False
        while data is None:
            try:
                data = rospy.wait_for_message('/scan', LaserScan, timeout=10)
            except:
                pass
        state,done = self.discretize_observation(data,360)
        state = np.reshape(state,(360))
        
        pool_state = np.zeros((90,6))
        for i in range(90):
            pool_state[i,0] = np.cos(i*np.pi/45.0-np.pi/2+np.pi/90)
            pool_state[i,1] = np.sin(i*np.pi/45.0-np.pi/2+np.pi/90)
            dis = np.min(state[4*i:(4*i+4)])
            x_dis = pool_state[i,0]*dis
            y_dis = pool_state[i,1]*dis
            pool_state[i,2] = dis
            pool_state[i,3] = self.length1
            pool_state[i,4] = self.length2
            pool_state[i,5] = self.width
            if abs(x_dis)<=self.width and y_dis<=self.length1 and y_dis>=-self.length2:
                self.stop_counter += 1.0

#            if abs( pool_state[i,0])<=0.2 and abs(pool_state[i,1])<=0.2:
#                self.stop_counter =1 
        pool_state = np.reshape(pool_state,(540))
#        print(state)
        reward = 1
        in_pose = self.robot_pose.pose.pose
        position = [in_pose.position.x,in_pose.position.y]
        abs_x = self.target_position[0] - in_pose.position.x
        abs_y = self.target_position[1] - in_pose.position.y
        (roll, pitch, yaw) = euler_from_quaternion ([in_pose.orientation.x,in_pose.orientation.y,in_pose.orientation.z,in_pose.orientation.w])
        trans_matrix = np.matrix([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
        rela = np.matmul(trans_matrix,np.array([[abs_x],[abs_y]]))
        rela_x = rela[0,0]
        rela_y = rela[1,0]
        rela_distance = np.sqrt(rela_x** 2 + rela_y ** 2)
        rela_angle = np.arctan2(rela_y,rela_x)
        if rela_distance<0.2:
            terminate=True
            reset = True
#        if done:
#            terminate=True
#        if np.abs(rela_angle)>np.pi-0.1:
#            terminate=True
#            reset = True
#        print(rela_distance)
#        print(rela_angle)
        target_pose = [rela_distance,rela_angle]
        cur_act = self.self_speed
        state = np.concatenate([pool_state,target_pose,cur_act,[self.max_action[0],self.max_action[1], self.max_acc[0], self.max_acc[1]]], axis=0)
#        cur_act[0] = cur_act[0]
        return state,reward, terminate,reset,position
    def Control(self,action):
        '''
        v_m, w_m, the velocities of the meta agent
        '''
        v_m = action[0]*self.max_action[0]
        w_m = action[1]*self.max_action[1]
        '''
        calculation of the velocity of the tested robot based on the equations in our paper
        '''
        self.self_speed[0] = v_m
        #		print(self.self_speed[0])        
        self.self_speed[1] =  w_m
        move_cmd = Twist()
        move_cmd.linear.x = self.self_speed[0]
        move_cmd.linear.y = 0.
        move_cmd.linear.z = 0.
        move_cmd.angular.x = 0.
        move_cmd.angular.y = 0.
        move_cmd.angular.z = self.self_speed[1]
        self.cmd_vel.publish(move_cmd)
    def publish_goal(self):
        self.target_position[0] =  self.targets[self.count][0]
        self.target_position[1] =  self.targets[self.count][1]
        self.count = self.count+1
    def stop(self):
        move_cmd = Twist()
        move_cmd.linear.x = 0
        move_cmd.angular.z = 0
        self.cmd_vel.publish(move_cmd)
        
    def reset(self):
        # Resets the state of the environment and returns an initial observation.
#        rospy.sleep(4.0)
#        self.show_text_in_rviz(self.targets[0][0],self.targets[0][1])
#        self.show_text_in_rviz(self.targets[1][0],self.targets[1][1])
#        self.show_text_in_rviz(self.targets[2][0],self.targets[2][1])
        rospy.sleep(3.0)
#        self.show_text_in_rviz1(self.targets[3][0],self.targets[3][1])
