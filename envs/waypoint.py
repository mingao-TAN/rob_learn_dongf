import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
import math

class GlobalPlanSampler:
    def __init__(self):
        rospy.init_node('global_plan_sampler')

        # 发布目标点
        self.target_pub = rospy.Publisher('/target_points', PointStamped, queue_size=10)

        # 订阅全局路径
        self.global_plan_sub = rospy.Subscriber('/move_base/TrajectoryPlannerROS/global_plan', Path, self.global_plan_callback)

    def global_plan_callback(self, msg):
        """处理接收到的全局路径消息"""
        path_length = len(msg.poses)

        if path_length == 0:
            rospy.logwarn("Received empty global plan!")
            return

        # 获取机器人的当前位置（路径的第一个点）
        robot_position = msg.poses[0].pose.position

        # 遍历路径的点，寻找距离机器人的位置大于或等于2米的第一个点
        for pose in msg.poses:
            point = pose.pose.position
            distance = math.sqrt((point.x - robot_position.x) ** 2 + (point.y - robot_position.y) ** 2)

            if distance >=6.00:
                target_point = PointStamped()
                target_point.header = msg.header  # 复制时间戳和坐标系信息
                #target_point.header.frame_id = "base_link"  # 复制时间戳和坐标系信息
                target_point.point = point
                print(target_point.header)
                # 发布目标点
                self.target_pub.publish(target_point)

                # 打印目标点
                rospy.loginfo(f"Target Point published: ({target_point.point.x}, {target_point.point.y}, {target_point.point.z})")
                break  # 找到第一个符合条件的点后退出循环

if __name__ == '__main__':
    try:
        sampler = GlobalPlanSampler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass