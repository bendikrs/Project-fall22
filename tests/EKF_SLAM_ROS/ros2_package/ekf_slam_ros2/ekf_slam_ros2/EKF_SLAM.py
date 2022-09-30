import numpy as np
from sklearn import cluster
from sklearn.cluster import DBSCAN

# not in setup.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# ---

import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist


class EKF_SLAM(Node):

    def __init__(self):
        super().__init__('EKF_SLAM')
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # Robot motion
        self.u = np.array([0, 0]) # [v, omega]

        # I got the magic in me
        self.landmark_threshhold = 0.2
        self.landmark_init_cov = 10.0

        # EKF
        self.timeStep = 1
        self.rangeLimit = 6
        self.Rt = np.diag([0.1, 0.1, 0.01]) ** 2
        self.Qt = np.diag([0.1, 0.1]) ** 2
        self.x = np.zeros((3, 1))
        self.P = np.eye(3)


        # subscribers
        self.twistSubscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.twist_callback,
            10)
        self.twistSubscription

        self.scanSubscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10)
        self.scanSubscription  # prevent unused variable warning


    def twist_callback(self, msg):
        self.u[0] = msg.linear.x
        self.u[1] = msg.angular.z
        self.get_logger().info('v: "%f" omega: "%f"' % (msg.linear.x, msg.angular.z))

    def scan_callback(self, msg):
        point_cloud = self.get_laser_scan(msg)

        # clustering with DBSCAN
        db = DBSCAN().fit(point_cloud)

        # make array of clusters
        clusters = [point_cloud[db.labels_ == i] for i in range(db.labels_.max() + 1)]

        landmarks = self.get_landmarks(clusters)
        self.compare_and_add_landmarks(landmarks)
        print(landmarks.shape)
        print(self.x.shape)
        print(self.x)

        plt.pause(1)
        plt.cla()
        plt.axis('equal')


    def timer_callback(self):
        msg = String()
        msg.data = 'Hello World: %d' % self.i
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg.data)
        self.i += 1

    def get_laser_scan(self, msg):
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
        return point_cloud

    def get_landmarks(self, clusters):
        landmarks = []
        for cluster in clusters:
            self.ax.scatter(cluster[:,0], cluster[:,1])

            if len(cluster) > 3 and len(cluster) < 20:
                guessed_cx = np.mean(cluster[:,0])
                guessed_cy = np.mean(cluster[:,1])
                self.ax.add_patch(patches.Circle((guessed_cx, guessed_cy), 0.125, fill=False, color='red'))         
                landmarks.append(guessed_cx)
                landmarks.append(guessed_cy)
        return np.array(landmarks).reshape(-1, 1)

    def compare_and_add_landmarks(self, landmarks):
        # if not exist, add all
        if len(self.x) == 3 and len(landmarks) > 0:
            self.x = np.vstack((self.x, landmarks))
            self.P = np.zeros((len(self.x), len(self.x)))
            self.P[:3, :3] = np.eye(3)
            self.P[3:, 3:] = np.eye(len(self.x) - 3) * self.landmark_init_cov
            
        # compare new landmarks with old landmarks
        elif len(self.x) > 3 and len(landmarks) > 0:
            for i in range(0, len(landmarks), 2):
                for j in range(3, len(self.x), 2):
                    print(np.linalg.norm(landmarks[i:i+2] - self.x[j:j+2])
                    if np.linalg.norm(landmarks[i:i+2] - self.x[j:j+2]) < self.landmark_threshhold:
                        self.x[j:j+2] = landmarks[i:i+2] 
                    else:
                        self.x = np.vstack((self.x, landmarks[i:i+2]))
                        self.P = np.block([[self.P, np.zeros((len(self.P), 2))], 
                                           [np.zeros((2, len(self.P))), np.eye(2) * self.landmark_init_cov]])
        
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
