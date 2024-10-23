import rospy
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped

class GlobalPlanSampler:
    def __init__(self):
        rospy.init_node('global_plan_sampler')

        # 订阅全局路径
        self.global_plan_sub = rospy.Subscriber('/move_base/NavfnROS/plan', Path, self.global_plan_callback)

        # 存储目标点的数组
        self.target_points = []

    def global_plan_callback(self, msg):
        """处理接收到的全局路径消息"""
        path_length = len(msg.poses)

        if path_length == 0:
            rospy.logwarn("Received empty global plan!")
            return

        # 计算每个目标点的间隔
        interval = max(1, path_length // 4)  # 至少选取4个点

        # 均匀截取目标点
        self.target_points = []
        for i in range(0, path_length, interval):
            if len(self.target_points) < 4:  # 只存储四个目标点
                self.target_points.append(msg.poses[i].pose)

        # 打印目标点
        rospy.loginfo("Target Points:")
        for point in self.target_points:
            rospy.loginfo(f"Target Point: ({point.position.x}, {point.position.y}, {point.position.z})")

    def get_target_points(self):
        """获取当前的目标点数组"""
        return self.target_points

if __name__ == '__main__':
    try:
        sampler = GlobalPlanSampler()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass