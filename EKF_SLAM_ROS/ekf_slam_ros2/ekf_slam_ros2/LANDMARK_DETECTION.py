import numpy as np
from sklearn.cluster import DBSCAN

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseArray, Pose

def circle_fitting(x, y):
    """Fit a circle to a set of points using the least squares method.
    This implementation is heavily 
    based on the PythonRobotics implementation: https://arxiv.org/abs/1808.10703

    parameters:
        x (1xn) numpy array): x coordinates of the points [m]
        y (1xn) numpy array): y coordinates of the points [m]
    output: 
        cxe:   x coordinate of the center of the circle, [m]
        cye:   y coordinate of the center of the circle, [m]
        re:    radius of the circle, [m]
        error: prediction error
    """

    # calculate the different sums needed
    sumx = sum(x)
    sumy = sum(y)
    sumx2 = sum([ix ** 2 for ix in x])
    sumy2 = sum([iy ** 2 for iy in y])
    sumxy = sum([ix * iy for (ix, iy) in zip(x, y)])

    F = np.array([[sumx2, sumxy, sumx],
                  [sumxy, sumy2, sumy],
                  [sumx, sumy, len(x)]]) 

    G = np.array([[-sum([ix ** 3 + ix * iy ** 2 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 * iy + iy ** 3 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 + iy ** 2 for (ix, iy) in zip(x, y)])]])

    # solve the linear system
    T = np.linalg.inv(F).dot(G)

    cxe = float(T[0] / -2)
    cye = float(T[1] / -2)
    re = np.sqrt(cxe**2 + cye**2 - T[2])

    error = sum([np.hypot(cxe - ix, cye - iy) - re for (ix, iy) in zip(x, y)])

    return (cxe, cye, re, error)

class LANDMARK_DETECTION(Node):
    '''Class to detect landmarks from the laser scan data
    '''
    def __init__(self):
        '''Constructor for the LANDMARK_DETECTION node
        '''
        super().__init__('LANDMARK_DETECTION')

        # Detection parameters
        self.distance_threshold = 0.025
        self.landmark_threshhold = 0.2
        # self.landmark_radius = 0.08
        self.landmark_radius = 0.15

        self.landmarks = None

        # subscribe to the laser scan data
        self.scanSubscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.scanSubscription  # prevent unused variable warning

        # publish the landmarks
        self.landmarkPublisher = self.create_publisher(
            PoseArray,
            '/new_landmarks',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))
        self.landmarkPublisher
        self.timer = self.create_timer(0.5, self.timer_callback)

    def timer_callback(self):
        '''Timer callback function
        '''
        self.publish_landmarks()

    def publish_landmarks(self):
        '''Publish the landmarks
        '''
        if self.landmarks is not None:
            landmark_msg = PoseArray()
            for i in range(int(self.landmarks.shape[0]/2)):
                pose = Pose()
                pose.position.x = self.landmarks[2*i,0]
                pose.position.y = self.landmarks[2*i+1,0]
                landmark_msg.poses.append(pose)
            self.landmarkPublisher.publish(landmark_msg)

    def scan_callback(self, msg):
        '''Callback function for the laser scan subscriber
        '''
        point_cloud = self.get_laser_scan(msg) # Robot frame

        # clustering with DBSCAN
        db = DBSCAN(eps=0.1, min_samples=12).fit(point_cloud)

        # make array of clusters
        clusters = [point_cloud[db.labels_ == i] for i in range(db.labels_.max() + 1)]
        
        self.landmarks = self.get_landmarks(clusters) # Robot frame
        
    def get_laser_scan(self, msg):
        '''Converts the laser scan message to a point cloud in the world frame

        Parameters:
            msg (LaserScan): ROS2 LaserScan message
        Returns:
            point_cloud (2*n x 1 np.array): Point cloud in the world frame, where n is the number of points
        '''
        # get data
        ranges = msg.ranges # list of ranges
        angle_min = msg.angle_min # start angle
        angle_max = msg.angle_max # end angle

        # set inf and nan to 0
        ranges = np.array(ranges)
        ranges[np.isinf(ranges)] = 0
        ranges[np.isnan(ranges)] = 0

        # make cartesian coordinates
        theta = np.linspace(angle_min, angle_max, len(ranges))
        x = ranges * np.cos(theta)
        y = ranges * np.sin(theta)

        # remove points at origin
        x = x[ranges != 0]
        y = y[ranges != 0]

        return np.vstack((x, y)).T

    def get_landmarks(self, clusters):
        '''Finds circular shapes in the measured point cloud clusters and returns the landmark positions

        Parameters:
            clusters (m x n_i x 2 numpy array): Point cloud clusters, where m is the number of clusters and n_i is the number of points in cluster i
        Returns:
            landmarks (p x 2 numpy array): Landmark positions, where p is the number of landmarks
        '''
        landmarks = []
        for cluster in clusters:
            cxe, cye, re, error = circle_fitting(cluster[:,0], cluster[:,1])
            if abs(error) < 0.005 and re <= self.landmark_radius + self.distance_threshold and re >= self.landmark_radius - self.distance_threshold:
                landmarks.append(cxe)
                landmarks.append(cye)
        if landmarks == []:
            return None
        else :
            return np.array(landmarks).reshape(-1,1)



def main(args=None):
    rclpy.init(args=args)
    landmark_detection = LANDMARK_DETECTION()

    print('LANDMARK_DETECTION node started')
    rclpy.spin(landmark_detection)

    landmark_detection.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()