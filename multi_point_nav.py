#!/usr/bin/env python
# encoding: utf-8
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Point, Quaternion, PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped, Vector3
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
import tf
import numpy as np
from tf.transformations import euler_from_quaternion

class MultiPointNav:
    def __init__(self):
        rospy.init_node('multi_point_nav', anonymous=True)
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

        self.robot_pose = None
        self.target_position = None

        # 设置目标点发布器
        self.goal_pub = rospy.Publisher('/goal', PoseStamped, queue_size=10)

        self.robot_pose_sub = rospy.Subscriber("amcl_pose", PoseWithCovarianceStamped, self.PoseCallback)
        self.robot_twist_sub = rospy.Subscriber("/odom", Odometry, self.TwistCallback)

    def PoseCallback(self, pose):
        self.robot_pose = pose
        
    def TwistCallback(self, twist):
        self.robot_twist = twist
        
    def move_to_goal(self, x, y):
        self.target_position = [x, y]
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position = Point(x, y, 0)
        goal.target_pose.pose.orientation = Quaternion(0, 0, 0, 1)

        self.client.send_goal(goal)
    
        rate = rospy.Rate(10)  # 10Hz
        while not rospy.is_shutdown():
            if self.robot_pose is None:
                rospy.loginfo("Waiting for robot pose...")
                rate.sleep()
                continue

            in_pose = self.robot_pose.pose.pose
            position = [in_pose.position.x, in_pose.position.y]
            abs_x = self.target_position[0] - in_pose.position.x
            abs_y = self.target_position[1] - in_pose.position.y
            
            quaternion = (
                in_pose.orientation.x,
                in_pose.orientation.y,
                in_pose.orientation.z,
                in_pose.orientation.w)
            euler = euler_from_quaternion(quaternion)
            yaw = euler[2]

            trans_matrix = np.array([[np.cos(yaw), np.sin(yaw)], 
                                     [-np.sin(yaw), np.cos(yaw)]])
            rela = np.dot(trans_matrix, np.array([[abs_x], [abs_y]]))
            rela_x = rela[0, 0]
            rela_y = rela[1, 0]
            rela_distance = np.sqrt(rela_x** 2 + rela_y ** 2)

            if rela_distance < 0.5:
                self.client.cancel_goal()
                rospy.loginfo("Reached target ({}, {}) successfully".format(x, y))
                return True

            rate.sleep()

        self.client.cancel_goal()
        return False

    def navigate(self):
        waypoints = [
            (1.5, 0),
            (0,0),
            # 可以添加更多的目标点
        ]

        for i, point in enumerate(waypoints):
            rospy.loginfo("Moving to target point {}: ({}, {})".format(i + 1, point[0], point[1]))
            result = self.move_to_goal(point[0], point[1])
            if result:
                rospy.sleep(2)  # 在每个目标点停留2秒
                
                # 发布下一个目标点
                if i + 1 < len(waypoints):
                    next_point = waypoints[i + 1]
                    pose = PoseStamped()
                    pose.header.frame_id = "map"
                    pose.header.stamp = rospy.Time.now()
                    pose.pose.position = Point(next_point[0], next_point[1], 0)
                    pose.pose.orientation = Quaternion(0, 0, 0, 1)
                    self.goal_pub.publish(pose)
                    rospy.loginfo("Published next target point: ({}, {})".format(next_point[0], next_point[1]))
            else:
                rospy.loginfo("Failed to reach target {}".format(i + 1))

        rospy.loginfo("Finished navigating all points")

if __name__ == '__main__':
    try:
        nav = MultiPointNav()
        nav.navigate()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation test finished.")
