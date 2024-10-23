import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
import math
import sys
sys.path.insert(0, '/home/eias/catkin_ws_2024/devel/lib/python3/dist-packages')
import tf

class GlobalPlanSampler:
    def __init__(self):
        rospy.init_node('global_plan_sampler')
       
        # 发布目标点
        self.target_pub = rospy.Publisher('/target_points', PointStamped, queue_size=10)
       
        # 订阅全局路径
        self.global_plan_sub = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.global_plan_callback, queue_size=10)
       
        # 创建tf监听器
        self.listener = tf.TransformListener()

    def global_plan_callback(self, msg):
        """处理接收到的全局路径消息"""
        path_length = len(msg.poses)
        print (path_length)
        if path_length == 0:
            rospy.logwarn("Received empty global plan!")
            return

        # 获取机器人的当前位置（路径的第一个点）
        robot_position = msg.poses[0].pose.position

        # 遍历路径的点，寻找距离机器人的位置大于或等于2米的第一个点
        for pose in msg.poses:
            point = pose.pose.position
            distance = math.sqrt((point.x - robot_position.x) ** 2 + (point.y - robot_position.y) ** 2)
            # print(distance)
            if distance >= 2.00:
                target_point = PointStamped()
                target_point.header = msg.header
               
                # 转换坐标到 base_link
                target_point.point = point
                # self.target_pub.publish(target_point)
                # print(target_point)
                try:
                    self.listener.waitForTransform('base_link', 'map', rospy.Time(0), rospy.Duration(4.0))
                    transformed_point = self.listener.transformPoint('base_link', target_point)
                    self.target_pub.publish(transformed_point)
                    #print(transformed_point)
                    # 打印目标点
                    rospy.loginfo(f"Target Point published: ({transformed_point.point.x}, {transformed_point.point.y}, {transformed_point.point.z})")
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                    rospy.logerr(f"Transform error: {e}")
                break  # 找到第一个符合条件的点后退出循环

if __name__ == '__main__':
    try:
        sampler = GlobalPlanSampler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass