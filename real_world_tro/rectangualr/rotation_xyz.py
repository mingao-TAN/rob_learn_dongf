import rospy
import numpy as np
from sensor_msgs.msg import LaserScan

def rotate_laserscan(scan_msg, axis, angle_degrees):
    """
     axis 可以为 x y z轴
     默认为顺时针旋转，数值为角度值
    """
    # 将角度转换为弧度
    angle_radians = np.radians(angle_degrees)
    
    # 计算长度
    num_increments = len(scan_msg.ranges)
    
    # 创建LaserScan消息的副本
    rotated_scan = LaserScan()
    rotated_scan.header = scan_msg.header
    rotated_scan.angle_min = scan_msg.angle_min
    rotated_scan.angle_max = scan_msg.angle_max
    rotated_scan.angle_increment = scan_msg.angle_increment
    rotated_scan.time_increment = scan_msg.time_increment
    rotated_scan.scan_time = scan_msg.scan_time
    rotated_scan.range_min = scan_msg.range_min
    rotated_scan.range_max = scan_msg.range_max
    rotated_scan.intensities = list(scan_msg.intensities)

    # 根据指定的轴旋转范围
    if axis == 'x' or axis == 'y':  # 我们在2D中对待x和y旋转是一样的
        rotation_index = int(num_increments * (angle_degrees / 360.0))
        rotated_scan.ranges = np.roll(scan_msg.ranges, rotation_index).tolist()
    elif axis == 'z':  #z轴旋转影响角度值
        rotation_index = int(num_increments * (angle_degrees / 360.0))
        rotated_scan.ranges = np.roll(scan_msg.ranges, rotation_index).tolist()
        rotated_scan.angle_min += angle_radians
        rotated_scan.angle_max += angle_radians
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    return rotated_scan