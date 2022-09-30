import numpy as np
from sklearn import cluster
from sklearn.cluster import DBSCAN

# not in setup.py
from collections import Counter
import circlehough.hough as hough
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# ---

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

class EKF_SLAM(Node):

    def __init__(self):
        super().__init__('EKF_SLAM')
        self.fig, self.ax = plt.subplots()

        # EKF
        self.x = np.array([0, 0, 0])
        self.P = np.eye(3)

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

        # make cartesian coordinates
        theta = np.linspace(angle_min, angle_max, len(ranges))
        x = ranges * np.cos(theta)
        y = ranges * np.sin(theta)

        # remove inf and nan
        x = [i for i in x if not np.isinf(i) and not np.isnan(i)]
        y = [i for i in y if not np.isinf(i) and not np.isnan(i)]
        point_cloud = np.array([x, y]).T

        # clustering with DBSCAN
        db = DBSCAN().fit(point_cloud)

        # make array of clusters
        clusters = []
        for i in range(len(db.components_)):
            clusters.append(point_cloud[db.labels_ == i])

        landmarks = []
        for cluster in clusters:
            self.ax.scatter(cluster[:,0], cluster[:,1])

            if len(cluster) > 3 and len(cluster) < 100:
                guessed_cx = np.mean(cluster[:,0])
                guessed_cy = np.mean(cluster[:,1])
                self.ax.add_patch(patches.Circle((guessed_cx, guessed_cy), 0.125, fill=False, color='red'))         
                landmarks.append(np.array([guessed_cx, guessed_cy]))

        plt.pause(0.5)
        plt.cla()


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
