import numpy as np
from sklearn import cluster
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from geometry_msgs.msg import Twist, Quaternion
from tf2_ros import TransformStamped



def wrapToPi(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

def rot(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

def circle_fitting(x, y):
    """Fit a circle to a set of points using the least squares method.

    input:
        x, y: coordinates of the points [x1, x2, ..., xn], [y1, y2, ..., yn]
    output: 
        cxe:   x coordinate of the center
        cye:   y coordinate of the center
        re:    radius of the circle
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

def ransac_circle(points, x_guess, y_guess, r, iterations, threshold):
    best_inliers = []
    best_params = None
    for i in range(iterations):
        x = x_guess + np.random.uniform(-0.2, 0.2)
        y = y_guess + np.random.uniform(-0.2, 0.2)
        # x = x_guess
        # y = y_guess

        # Calculate inliers
        inliers = []
        for point in points:
            if np.sqrt((point[0] - x)**2 + (point[1] - y)**2) < r + threshold:
                inliers.append(point)

        # Update best inliers
        if len(inliers) + 10 > len(best_inliers):
            best_inliers = inliers
            best_params = (x, y, r)
    
    return best_inliers, best_params

class EKF:
    def __init__(self, timeStep=1.0):
        self.timeStep = timeStep

    def g(self, x, u, Fx): 
        '''
        Motion model
        u: control input (v, omega)
        x: state [x, y, theta, x1, y1, x2, y2, ...] (it's x_(t-1) )
        '''
        theta =  x[2,0]
        v, omega = u[0], u[1]
        if omega == 0:
            omega = 1e-9
        T = np.array([[-(v/omega)*np.sin(theta) + (v/omega)*np.sin(theta + omega*self.timeStep)],
                    [(v/omega)*np.cos(theta) - (v/omega)*np.cos(theta + omega*self.timeStep)],
                    [omega*self.timeStep]])

        return x + Fx.T @ T

    def jacobian(self, x, u, Fx):
        '''
        Jacobian of motion model
        u: control input (v, omega)
        x: state [x, y, theta, x1, y1, x2, y2, ...].T
        '''
        theta =  x[2,0]
        v, omega = u[0], u[1]  
        if omega == 0:
            omega = 1e-9     
        T = np.array([[0, 0, -(v/omega)*np.cos(theta) + (v/omega)*np.cos(theta + omega*self.timeStep)],
                    [0, 0, -(v/omega)*np.sin(theta) + (v/omega)*np.sin(theta + omega*self.timeStep)],
                    [0, 0 , 0]])

        return np.eye(x.shape[0]) + Fx.T @ T @ Fx

    def cov(self, Gt, P, Rt, Fx):
        '''
        Covariance update
        '''
        return Gt @ P @ Gt.T + Fx.T @ Rt @ Fx

    def predict(self, x, u, P, Rt):
        '''
        Predict step
        '''
        Fx = np.zeros((3, x.shape[0]))
        Fx[:3, :3] = np.eye(3)
        x_hat = self.g(x, u, Fx)
        Gt = self.jacobian(x, u, Fx)
        P_hat = self.cov(Gt, P, Rt, Fx)

        return x_hat, P_hat

    def update(self, x_hat, P_hat, Qt, z):
        '''
        Update step
        x_hat: state [x, y, theta, x1, y1, x2, y2, ...],  shape (3 + 2 * num_landmarks, 1)
        P_hat: covariance matrix, shape (3 + 2 * num_landmarks, 3 + 2 * num_landmarks)
        z: processed landmark locations [range r, bearing theta, j landmark index], shape: (number of currently observed landmarks*3, 1)
        Qt: measurement noise, shape: (2, 2)
        Fx: Jacobian of motion model, shape: (3, 3 + 2 * num_landmarks)
        '''

        if z.shape[0] == 0:
            # print('No measurement')
            return x_hat, P_hat


        for i in range(0, z.shape[0], 3): # for each landmark
            # print(i)
            z_r, z_theta, j = z[i,0], z[i+1,0], int(z[i+2,0]) # range, bearing, landmark index

            # Distance between robot and landmark
            delta = np.array([x_hat[3 + 2*j,0] - x_hat[0,0], x_hat[4 + 2*j,0] - x_hat[1,0]])

            # Measurement estimate from robot to landmark
            q = delta.T @ delta
            z_hat = np.array([[np.sqrt(q)],[wrapToPi(np.arctan2(delta[1], delta[0]) - x_hat[2, 0])]])

            # Jacobian of measurement model
            Fx = np.zeros((5,x_hat.shape[0]))
            Fx[:3,:3] = np.eye(3)
            Fx[3,2*j+3] = 1
            Fx[4,2*j+4] = 1

            H = np.array([[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0, np.sqrt(q)*delta[0], np.sqrt(q)*delta[1]],
                            [delta[1], -delta[0], -q, -delta[1], delta[0]]], dtype='float')
            H = (1/q)*H @ Fx

            # Kalman gain
            K = P_hat @ H.T @ np.linalg.inv(H @ P_hat @ H.T + Qt)
            
            # Calculate difference between expected and real observation
            z_dif = np.array([[z_r], [z_theta]]) - z_hat

            # Update state and covariance
            x_hat = x_hat + K @ z_dif
            x_hat[2,0] = wrapToPi(x_hat[2,0])
            P_hat = (np.eye(x_hat.shape[0]) - K @ H) @ P_hat

        return x_hat, P_hat

class Map():
    def __init__(self):
        self.map = np.array([[0.0, 0.0]]) # Pointcloud for map [[x, y], [x, y], ...]

    def add_point(self, x, y):
        '''Directly add point to map without transformation'''
        self.map = np.append(self.map, [[x, y]], axis=0) # Add new measurement to map
    
    def add_pointcloud(self, pointCloud):
        '''Takes a pointcloud and aligns it with the current map and add it to the map
        input:
        pointCloud: [[x, y], [x, y], ...]
        robotPose: [x, y, theta].T
        '''
        self.map = np.vstack((self.map, pointCloud))
        self.optimize_map()

    def optimize_map(self):
        '''Optimizes the map by removing duplicates, outliers and too dense areas'''
        # Remove duplicates
        self.map = np.unique(self.map, axis=0)

        # Remove outliers
        # TODO: e ditte nødvendig?

        # Remove too dense areas using nearest neighbor
        neigh = NearestNeighbors(n_neighbors=2)
        neigh.fit(self.map)
        distances, indices = neigh.kneighbors(self.map)
        self.map = np.delete(self.map, np.where(distances[:,1] < 0.02)[0], axis=0)
        # self.map = self.map[neigh.kneighbors(self.map, return_distance=False)[:,1:].flatten()]




class EKF_SLAM(Node):

    def __init__(self):
        super().__init__('EKF_SLAM')
        
        # EKF
        self.timeStep = 0.2
        self.Rt = np.diag([0.1, 0.1, 0.01]) ** 2 
        self.Qt = np.diag([0.1, 0.1]) ** 2
        self.x = np.zeros((3, 1))
        self.P = np.eye(3)
        self.ekf = EKF(timeStep=self.timeStep)


        # Map
        self.map = Map()
        
        # RANSAC
        self.iterations = 20
        self.distance_threshold = 0.025
        self.landmark_radius = 0.15

        # Robot motion
        self.u = np.array([0.0, 0.0]) # [v, omega]

        # I got the magic in me
        self.landmark_threshhold = 0.2
        self.landmark_init_cov = self.P[0,0]

        # subscribers
        self.twistSubscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.twist_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.twistSubscription

        self.scanSubscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.scanSubscription  # prevent unused variable warning

        # publishers
        self.robotPublisher = self.create_publisher(
            TransformStamped,
            '/robot',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.robotPublisher

        self.landmarkPublisher = self.create_publisher(
            TransformStamped,
            '/landmarks',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.landmarkPublisher

        self.mapPublisher = self.create_publisher(
            LaserScan,
            '/map',
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.mapPublisher


        self.timer = self.create_timer(self.timeStep, self.timer_callback)

    def timer_callback(self):
        # Publish robot position
        self.publish_robot()

        # Publish landmarks
        self.publish_landmarks()

        # Publish map
        self.publish_map()


    
    def publish_robot(self):
        '''Publishes the robot position'''
        robotPose = TransformStamped()
        robotPose.header.stamp = self.get_clock().now().to_msg()
        robotPose.header.frame_id = 'map'
        robotPose.child_frame_id = 'robot'
        robotPose.transform.translation.x = self.x[0,0]
        robotPose.transform.translation.y = self.x[1,0]
        robotPose.transform.translation.z = 0.0
        robotPose.transform.rotation = Quaternion(
            x=0.0,
            y=0.0,
            z=np.sin(self.x[2,0]/2),
            w=np.cos(self.x[2,0]/2))
        self.robotPublisher.publish(robotPose)        # Publish landmarks
        self.publish_landmarks()

    def publish_landmarks(self):
        '''Publishes the landmarks'''
        landmarks = TransformStamped()
        landmarks.header.stamp = self.get_clock().now().to_msg()
        landmarks.header.frame_id = 'map'
        landmarks.child_frame_id = 'landmarks'
        landmarks.transform.translation.x = 0.0
        landmarks.transform.translation.y = 0.0
        landmarks.transform.translation.z = 0.0
        landmarks.transform.rotation = Quaternion(
            x=0.0,
            y=0.0,
            z=0.0,
            w=1.0)
        self.landmarkPublisher.publish(landmarks)

    def publish_map(self):
        '''Publishes the map'''
        mapMsg = LaserScan()
        mapMsg.header.stamp = self.get_clock().now().to_msg()
        mapMsg.header.frame_id = 'map'
        mapMsg.angle_min = 0.0
        mapMsg.angle_max = 2*np.pi
        mapMsg.angle_increment = 0.01
        mapMsg.range_min = 0.0
        mapMsg.range_max = 10.0
        mapMsg.ranges = self.map.map[:,0]
        mapMsg.intensities = self.map.map[:,1]
        self.mapPublisher.publish(mapMsg)
        


    def twist_callback(self, msg):
        # self.get_logger().info('v: "%f" omega: "%f"' % (msg.linear.x, msg.angular.z))
        self.u[0] = msg.linear.x
        self.u[1] = msg.angular.z

    def scan_callback(self, msg):
        point_cloud = self.get_laser_scan(msg) # Robot frame

        # self.map.add_pointcloud(point_cloud)
        # print(self.map.map[0:10])

        # clustering with DBSCAN
        db = DBSCAN().fit(point_cloud)

        # make array of clusters
        clusters = [point_cloud[db.labels_ == i] for i in range(db.labels_.max() + 1)]
        
        landmarks = self.get_landmarks(clusters) # World frame

        z = self.compare_and_add_landmarks(landmarks)
        print('Total number of landmarks', (self.x.shape[0]-3)//2)

        x_hat, P_hat = self.ekf.predict(self.x, self.u, self.P, self.Rt)
        self.x, self.P = self.ekf.update(x_hat, P_hat, self.Qt, z)
        
        self.map.add_pointcloud(point_cloud)
        print("points in map:" ,len(self.map.map) , self.map.map[0:10])


    def get_laser_scan(self, msg):
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

        point_cloud = rot(self.x[2,0]) @ np.vstack((x, y))
        
        return point_cloud.T + self.x[0:2,0]

    def get_landmarks(self, clusters):
        landmarks = []
        for cluster in clusters:
            if len(cluster) > 15:
                guessed_cx = np.mean(cluster[:,0])
                guessed_cy = np.mean(cluster[:,1])
                # inliers, parameters = ransac_circle(cluster, guessed_cx, guessed_cy, self.landmark_radius, self.iterations, self.distance_threshold)
                # if len(inliers) > 0 and len(inliers) == len(cluster):
                #     # landmarks.append(parameters[0])
                #     # landmarks.append(parameters[1])
                #     self.plotter.plotRansacCircle(parameters[0], parameters[1], parameters[2], self.distance_threshold)
                #     landmarks.append(guessed_cx)
                #     landmarks.append(guessed_cy)
                
                cxe, cye, re, error = circle_fitting(cluster[:,0], cluster[:,1])
                if abs(error) < 0.005 and re <= self.landmark_radius + self.distance_threshold and re >= self.landmark_radius - self.distance_threshold:
                    landmarks.append(cxe)
                    landmarks.append(cye)

        return np.array(landmarks).reshape(-1, 1)

    def compare_and_add_landmarks(self, landmarks):
        '''
        Compare landmarks with current landmarks and add new ones
        
        input:
            landmarks: array of currently observed landmarks [[x1], [y1], [x2], [y2], ...] in world frame
            
        output:
                    z: array of landmarks [r, theta, j, r, theta, j, ...] in robot frame
        '''
        z = np.zeros(((landmarks.shape[0]//2)*3, 1))
        # print(z)
        # if not exist, add all
        if len(self.x) == 3 and len(landmarks) > 0:
            self.x = np.vstack((self.x, landmarks))
            self.P = np.zeros((len(self.x), len(self.x)))
            self.P[:3, :3] = np.eye(3)
            self.P[3:, 3:] = np.eye(len(self.x) - 3) * self.landmark_init_cov

            z[::3,0] = np.sqrt((self.x[0,0] - landmarks[::2,0])**2 + (self.x[1,0] - landmarks[1::2,0])**2) # r
            z[1::3,0] = wrapToPi(np.arctan2(landmarks[1::2,0] - self.x[1,0], landmarks[::2,0] - self.x[0,0]) - self.x[2,0]) # theta
            z[2::3,0] = np.arange(0, len(landmarks)//2)#.reshape(-1, 1) # j

        # compare new landmarks with old landmarks
        elif len(self.x) > 3 and len(landmarks) > 0:
            for i in range(0, len(landmarks), 2):
                meas_x = landmarks[i,0] 
                meas_y = landmarks[i+1,0]
                dists = np.sqrt((self.x[3::2,0] - meas_x)**2 + (self.x[4::2,0] - meas_y)**2) 
                
                i = int(i * 3/2)
                z[i,0] = np.sqrt((meas_x - self.x[0,0])**2 + (meas_y - self.x[1,0])**2)
                z[i+1,0] = wrapToPi(np.arctan2(meas_y - self.x[1,0], meas_x - self.x[0,0]) - self.x[2,0])

                if np.min(dists) < self.landmark_threshhold: # if landmark already exists
                    z[i+2,0] = int(np.argmin(dists))
                    
                else: # if landmark does not exist
                    self.x = np.vstack((self.x, np.array([[meas_x], [meas_y]])))
                    self.P = np.block([[self.P, np.zeros((len(self.P), 2))], 
                                       [np.zeros((2, len(self.P))), np.eye(2)*self.landmark_init_cov]])
                    z[i+2,0] = int(((len(self.x) - 3)//2 - 1))
        return z  

        

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
