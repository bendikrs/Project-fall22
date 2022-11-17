import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist

class AutoDrive(Node):
    def __init__(self):
        super().__init__('auto_drive')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 0.1
        msg.angular.z = 0.1333
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)

    auto_drive = AutoDrive()

    rclpy.spin(auto_drive)

    auto_drive.destroy_node()
    rclpy.shutdown()