#! /usr/bin/env python
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math

scan_camera = None
publisher_scan_fusion = None


def callback_scan_camera(data):
    """
    realsense订阅者的回调函数。保存激光扫描数据以便与hokuyo激光扫描数据融合。
    :param data: 激光扫描数据 (LaserScan)。
    :return: None
    """
    global scan_camera
    scan_camera = data

def callback_scan(scan):
    """
    hokuyo订阅者的回调函数。将来自realsense的激光扫描数据与hokuyo激光扫描数据融合。当realsense激光扫描的距离较小时，将其插入到hokuyo激光扫描数据中。发布融合后的激光扫描数据。
    :param scan: hokuyo激光扫描数据 (LaserScan)。
    :return: None
    """


    if scan_camera is not None:

        scan_ranges = list(scan.ranges)
        
        # 计算realsense扫描数据的一半长度
        half_length_camera = int(len(scan_camera.ranges) / 2)
        
        # 计算对齐的起始位置

        index_hokuyo_start = 0

        for index_hokuyo in range(half_length_camera):
            range_hokuyo = scan.ranges[index_hokuyo]
            range_realsense = scan_camera.ranges[half_length_camera + index_hokuyo]
            
            # 如果realsense的激光扫描距离比hokuyo的距离小，并且在有效范围内，则使用realsense的激光扫描数据
            if range_realsense < scan_camera.range_max and range_realsense < range_hokuyo:
                scan_ranges[index_hokuyo] = range_realsense


        # 计算另一半
        # 计算对齐的起始位置
        index_hokuyo_end = len(scan.ranges) - 1

        for index_realsense in range(half_length_camera):
            index_hokuyo = index_hokuyo_end - index_realsense
            range_hokuyo = scan.ranges[int(index_hokuyo)]
            range_realsense = scan_camera.ranges[half_length_camera - index_realsense]
            
            # 如果realsense的激光扫描距离比hokuyo的距离小，并且在有效范围内，则使用realsense的激光扫描数据
            if range_realsense < scan_camera.range_max and range_realsense < range_hokuyo:
                scan_ranges[int(index_hokuyo)] = range_realsense




   # 更新扫描数据
    scan.ranges = scan_ranges

    # 发布融合后的扫描数据
    publisher_scan_fusion.publish(scan)


def start():
    """
    启动扫描融合节点，将hokuyo和realsense的激光扫描数据融合。
    :return: None
    """
    rospy.init_node('scan_fusion')

    global publisher_scan_fusion
    publisher_scan_fusion = rospy.Publisher('scan_fusion', LaserScan, queue_size=1)
    rospy.Subscriber("scan", LaserScan, callback_scan)
    rospy.Subscriber("scan_camera", LaserScan, callback_scan_camera)

    rospy.spin()

if __name__ == '__main__':
    start()


