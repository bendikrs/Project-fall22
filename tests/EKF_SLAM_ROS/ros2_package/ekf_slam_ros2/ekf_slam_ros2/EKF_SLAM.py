import numpy as np
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

class EKF_SLAM(Node):

    def __init__(self):
        super().__init__('EKF_SLAM')

        # subscriber
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # publisher
        # self.publisher_ = self.create_publisher(String, 'topic', 10)
        # timer_period = 0.5  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0

    def listener_callback(self, msg):
        # self.get_logger().info('I heard: "%s"' % msg.ranges[0])

        # get data
        ranges = msg.ranges # list of ranges
        angle_min = msg.angle_min # start angle
        angle_max = msg.angle_max # end angle
        angle_increment = msg.angle_increment # angle increment between scans (rad)
        time_increment = msg.time_increment # time increment between measurements (sec)
        scan_time = msg.scan_time # time between scans (sec)
        range_min = msg.range_min # minimum range value (m)
        range_max = msg.range_max # maximum range value (m)

        # remove inf and nan
        ranges = [x for x in ranges if not np.isinf(x) and not np.isnan(x)]
        ranges = np.array(ranges)

        # make cartesian coordinates
        theta = np.linspace(angle_min, angle_max, len(ranges))
        x = ranges * np.cos(theta)
        y = ranges * np.sin(theta)

        # clustering with DBSCAN
        X = np.array([x, y]).T
        db = DBSCAN(eps=0.5, min_samples=10).fit(X)
        labels = db.labels_


        self.get_logger().info('Clusters: "%s"' % labels)



    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    ekf_slam = EKF_SLAM()


    rclpy.spin(ekf_slam)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    ekf_slam.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
