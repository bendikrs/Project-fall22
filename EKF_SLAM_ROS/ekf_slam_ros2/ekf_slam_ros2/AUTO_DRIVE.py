import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist

class AutoDrive(Node):
    def __init__(self):
        super().__init__('auto_drive')
        timer_period = 5 # seconds
        self.linear_velocity = 0.1
        self.angular_velocity = self.radius2angular_velocity(0.7)

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = self.linear_velocity
        msg.angular.z = self.angular_velocity
        self.publisher_.publish(msg)

    def radius2angular_velocity(self, radius):
        return self.linear_velocity / radius

def main(args=None):
    rclpy.init(args=args)

    auto_drive = AutoDrive()

    rclpy.spin(auto_drive)

    auto_drive.destroy_node()
    rclpy.shutdown()
