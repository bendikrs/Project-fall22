import numpy as np
from sklearn import cluster
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from std_msgs.msg import String, Header
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from geometry_msgs.msg import Twist, Quaternion, Point, Pose
from tf2_ros import TFMessage, TransformStamped, TransformBroadcaster
from nav_msgs.msg import OccupancyGrid, MapMetaData

def quaternion2euler(x, y, z, w):
    '''
    Convert quaternion to euler angles
    
    parameters:
        x, y, z, w (float): quaternion
    output:
        roll, pitch, yaw (float): euler angles [rad]
    '''
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = np.arctan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = np.arcsin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = np.arctan2(t3, t4)

    return X, Y, Z

def euler2quaternion(roll, pitch, yaw):
    """Convert euler angles to quaternion

    parameters:
        roll (float): roll angle [rad]
        pitch (float): pitch angle [rad]
        yaw (float): yaw angle [rad]
    output:
        q (,x4 numpy array): quaternion [x, y, z, w]
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    q = np.zeros((4))
    q[0] = cy * sr * cp - sy * cr * sp # x
    q[1] = cy * cr * sp + sy * sr * cp # y
    q[2] = sy * cr * cp - cy * sr * sp # z
    q[3] = cy * cr * cp + sy * sr * sp # w

    return q

def wrapToPi(theta):
    '''
    Wrap angle to [-pi, pi]
    
    parameters:
        theta (float): angle [rad] 
    output:
        (float): angle [rad]
    '''
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

def rot(theta):
    '''
    Rotation matrix
    
    parameters:
        theta (float): angle [rad]
    output:
        (2x2 numpy array): rotation matrix
    '''
    return np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])

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
    N = x.shape[0]
    points = np.hstack((x, y))

    # Set up matrices
    A = np.hstack((points, np.ones((N,1))))
    B = points[:,0]*points[:,0] + points[:,1]*points[:,1]

    # Least square approximation
    X = np.linalg.pinv(A) @ B

    # Calculate circle parameter
    cxe = X[0]/2
    cye = X[1]/2
    re = np.sqrt(4*X[2] + X[0]**2 + X[1]**2 )/2

    error = np.sum(np.hypot(cxe - x, cye - y) - re)
    return (cxe, cye, re, error)

class EKF:
    '''
    A class for the Extended Kalman Filter

    Parameters:
        timeStep (float): time step [s]
    '''
    def __init__(self, timeStep=1.0):
        '''
        Initialize the EKF class
        
        Parameters:
            timeStep (float): time step [s]
        '''
        self.timeStep = timeStep

    def g(self, x, u, Fx): 
        '''
        Move robot one step using a motion model.

        Parameters:
            u (1x2 numpy array): control input [v, omega] [m/s, rad/s]
            x (3+2*numLandmarks x 1 numpy array): state [x, y, theta, x1, y1, x2, y2, ...] [m, m, rad, m, m, ...]
            Fx (3+2*numLandmarks x 3+2*numLandmarks numpy array): Masking matrix
        Returns:
            (3+2*numLandmarks x 1 numpy array): new state [x, y, theta, x1, y1, x2, y2, ...] [m, m, rad, m, m, ...]
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
        Jacobian

        Parameters:
            u (1x2 numpy array): control input [v, omega] [m/s, rad/s]
            x (3+2*numLandmarks x 1 numpy array): state [x, y, theta, x1, y1, x2, y2, ...].T [m, m, rad, m, m, ...]
            Fx (3+2*numLandmarks x 3+2*numLandmarks numpy array): Masking matrix
        Returns:
            (3+2*numLandmarks x 3+2*numLandmarks numpy array): Jacobian matrix
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

        Parameters:
            Gt (3+2*numLandmarks x 3+2*numLandmarks numpy array): Jacobian matrix
            P (3+2*numLandmarks x 3+2*numLandmarks numpy array): Covariance matrix
            Rt (2x2): Covariance matrix
            Fx (3+2*numLandmarks x 3+2*numLandmarks numpy array): Masking matrix
        Returns:
            (3+2*numLandmarks x 3+2*numLandmarks numpy array): Covariance matrix
        '''
        return Gt @ P @ Gt.T + Fx.T @ Rt @ Fx

    def predict(self, x, u, P, Rt):
        '''
        EKF predict step

        Parameters:
            u (1x2 numpy array): control input [v, omega] [m/s, rad/s]
            x (3+2*numLandmarks x 1 numpy array): state [x, y, theta, x1, y1, x2, y2, ...].T [m, m, rad, m, m, ...]
            P (3+2*numLandmarks x 3+2*numLandmarks numpy array): Covariance matrix
            Rt (2x2): Covariance matrix
        Returns:
            x_hat (3+2*numLandmarks x 1 numpy array): new estimated state [x, y, theta, x1, y1, x2, y2, ...].T [m, m, rad, m, m, ...]
            P_hat (3+2*numLandmarks x 3+2*numLandmarks numpy array): new covariance matrix
        '''
        Fx = np.zeros((3, x.shape[0]))
        Fx[:3, :3] = np.eye(3)
        x_hat = self.g(x, u, Fx)
        Gt = self.jacobian(x, u, Fx)
        P_hat = self.cov(Gt, P, Rt, Fx)
        return x_hat, P_hat

    def update(self, x_hat, P_hat, Qt, z):
        '''
        EKF update step

        Parameters:
            x_hat (3+2*numLandmarks x 1 numpy array): estimated state [x, y, theta, x1, y1, x2, y2, ...].T [m, m, rad, m, m, ...]
            P_hat (3+2*numLandmarks x 3+2*numLandmarks numpy array): Covariance matrix
            Qt (2x2 numpy array): measurement noise covariance matrix
            z (num Currently Observed Landmarks x 1 numpy array): measurement [range, bearing, j landmark index] [m, rad, index]
        Returns:
            x (3+2*numLandmarks x 1 numpy array): new estimated state [x, y, theta, x1, y1, x2, y2, ...].T [m, m, rad, m, m, ...]
            P (3+2*numLandmarks x 3+2*numLandmarks numpy array): new covariance matrix
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
    '''
    Map class
    '''
    def __init__(self):
        '''
        Initialize a map
        
        Parameters:
            None
        '''
        self.map = np.array([[0.0, 0.0]]) # Pointcloud for map [[x, y], [x, y], ...]
        self.occ_map = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.xy_resolution = None
        self.EXTEND_AREA = 5.0
        self.xy_resolution = 0.05

        self.robot_x = None
        self.robot_y = None

    def update_occ_grid(self, pointCloud):
        '''Updates the occupancy grid with a new point cloud with n points
        
        Parameters:
            pointCloud (n x 2 numpy array): point cloud [[x, y], [x, y], ...]
        Returns:
            None
        '''
        ox, oy = pointCloud[:,0], pointCloud[:,1]
        new_occ_map, min_x, max_x, min_y, max_y, xy_resolution = \
        self.generate_ray_casting_grid_map(ox, oy, self.xy_resolution, breshen=True)
        if self.occ_map is None:
            self.occ_map = new_occ_map
            self.min_x = min_x
            self.max_x = max_x
            self.min_y = min_y
            self.max_y = max_y
            self.xy_resolution = xy_resolution
        else:
            self.min_x = min_x
            self.max_x = max_x
            self.min_y = min_y
            self.max_y = max_y
            self.xy_resolution = xy_resolution
            # self.occ_map = new_occ_map # TODO: merge maps
            temp_map = np.logical_and(self.occ_map, new_occ_map)
            self.occ_map =  np.logical_and(self.occ_map, temp_map)     
            # merge new map with old map, based on min and max values and resolution
    
    def bresenham(self, start, end):
        """
        Bresenham's line drawing algorithm
        This implementation is heavily 
        based on the PythonRobotics implementation: https://arxiv.org/abs/1808.10703

        Parameters:
            start (,x2 numpy array): start point of line [x, y] [m, m]
            end (,x2 numpy array): end point of line [x, y] [m, m]
        Returns:
            line (n x 2 numpy array): line points [[x, y], [x, y], ...] [m, m]
        """
        # setup initial conditions
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x1, y1 = y1, x1
            x2, y2 = y2, x2
        # swap start and end points if necessary and store swap state
        swapped = False
        if x1 > x2:
            x1, x2 = x2, x1
            y1, y2 = y2, y1
            swapped = True
        dx = x2 - x1  # recalculate differentials
        dy = y2 - y1  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y1 < y2 else -1
        # iterate over bounding box generating points between start and end
        y = y1
        points = []
        for x in range(x1, x2 + 1):
            coord = [y, x] if is_steep else (x, y)
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        if swapped:  # reverse the list if the coordinates were swapped
            points.reverse()
        points = np.array(points)
        return points

    def calc_grid_map_config(self, ox, oy, xy_resolution):
        """
        Calculates the size, and the maximum distances according to the the
        measurement center. This implementation is heavily 
        based on the PythonRobotics implementation: https://arxiv.org/abs/1808.10703

        Parameters:
            ox (n x 1 numpy array): x coordinates of the measurement center [m]
            oy (n x 1 numpy array): y coordinates of the measurement center [m]
            xy_resolution (float): resolution of the grid map [m]
        Returns:
            min_x (float): minimum x coordinate of the grid map [m]
        """
        # set the map to be of size 2*EXTEND_AREA x 2*EXTEND_AREA
        min_x = -self.EXTEND_AREA
        min_y = -self.EXTEND_AREA
        max_x = self.EXTEND_AREA
        max_y = self.EXTEND_AREA

        xw = int(round((max_x - min_x) / xy_resolution))
        yw = int(round((max_y - min_y) / xy_resolution))
        return min_x, min_y, max_x, max_y, xw, yw

    def atan_zero_to_twopi(y, x):
        """
        Calculates the angle of a vector with the x-axis. This implementation is
        from the PythonRobotics implementation: https://arxiv.org/abs/1808.10703

        Parameters:
            y (float): y coordinate of the vector [m]
            x (float): x coordinate of the vector [m]
        Returns:
            angle (float): angle of the vector with the x-axis [rad]
        """
        angle = np.atan2(y, x)
        if angle < 0.0:
            angle += np.pi * 2.0
        return angle

    def init_flood_fill(self, center_point, obstacle_points, xy_points, min_coord,
                        xy_resolution):
        """ ----------- fjerne denne funksjonen? -------------
        This implementation is
        from the PythonRobotics implementation: https://arxiv.org/abs/1808.10703
        
        center_point: center point
        obstacle_points: detected obstacles points (x,y)
        xy_points: (x,y) point pairs
        """
        center_x, center_y = center_point
        prev_ix, prev_iy = center_x - 1, center_y
        ox, oy = obstacle_points
        xw, yw = xy_points
        min_x, min_y = min_coord
        occupancy_map = (np.ones((xw, yw))) * 0.5
        for (x, y) in zip(ox, oy):
            # x coordinate of the the occupied area
            ix = int(round((x - min_x) / xy_resolution))
            # y coordinate of the the occupied area
            iy = int(round((y - min_y) / xy_resolution))
            free_area = self.bresenham((prev_ix, prev_iy), (ix, iy))
            for fa in free_area:
                occupancy_map[fa[0]][fa[1]] = 0  # free area 0.0
            prev_ix = ix
            prev_iy = iy
        return occupancy_map

    def flood_fill(self, center_point, occupancy_map):
        """----------- fjerne denne funksjonen? -------------
        This implementation is
        from the PythonRobotics implementation: https://arxiv.org/abs/1808.10703

        center_point: starting point (x,y) of fill
        occupancy_map: occupancy map generated from Bresenham ray-tracing
        """
        # Fill empty areas with queue method
        sx, sy = occupancy_map.shape
        fringe = deque()
        fringe.appendleft(center_point)
        while fringe:
            n = fringe.pop()
            nx, ny = n
            # West
            if nx > 0:
                if occupancy_map[nx - 1, ny] == 0.5:
                    occupancy_map[nx - 1, ny] = 0.0
                    fringe.appendleft((nx - 1, ny))
            # East
            if nx < sx - 1:
                if occupancy_map[nx + 1, ny] == 0.5:
                    occupancy_map[nx + 1, ny] = 0.0
                    fringe.appendleft((nx + 1, ny))
            # North
            if ny > 0:
                if occupancy_map[nx, ny - 1] == 0.5:
                    occupancy_map[nx, ny - 1] = 0.0
                    fringe.appendleft((nx, ny - 1))
            # South
            if ny < sy - 1:
                if occupancy_map[nx, ny + 1] == 0.5:
                    occupancy_map[nx, ny + 1] = 0.0
                    fringe.appendleft((nx, ny + 1))

    def generate_ray_casting_grid_map(self, ox, oy, xy_resolution, breshen=True):
        """
        This implementation is
        from the PythonRobotics implementation: https://arxiv.org/abs/1808.10703

        The breshen boolean tells if it's computed with bresenham ray casting
        (True) or with flood fill (False)
        """
        min_x, min_y, max_x, max_y, x_w, y_w = self.calc_grid_map_config(
            ox, oy, xy_resolution)
        # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]
        occupancy_map = np.ones((x_w, y_w)) / 2
        # center_x = int(
        #     round(-min_x / xy_resolution))  # center x coordinate of the grid map
        # center_y = int(
        #     round(-min_y / xy_resolution))  # center y coordinate of the grid map
        center_x = int(round(self.robot_x/xy_resolution)) + int(
            round(-min_x / xy_resolution))
        center_y = int(round(self.robot_y/xy_resolution)) + int(
            round(-min_y / xy_resolution))
        # occupancy grid computed with bresenham ray casting
        if breshen:
            for (x, y) in zip(ox, oy):
                # x coordinate of the the occupied area
                ix = int(round((x - min_x) / xy_resolution))
                # y coordinate of the the occupied area
                iy = int(round((y - min_y) / xy_resolution))
                laser_beams = self.bresenham((center_x, center_y), (
                    ix, iy))  # line form the lidar to the occupied point
                for laser_beam in laser_beams:
                    occupancy_map[laser_beam[0]][
                        laser_beam[1]] = 0.0  # free area 0.0
                occupancy_map[ix][iy] = 1.0  # occupied area 1.0
                occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
                occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
                occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
        # occupancy grid computed with with flood fill
        else:
            occupancy_map = self.init_flood_fill((center_x, center_y), (ox, oy),
                                            (x_w, y_w),
                                            (min_x, min_y), xy_resolution)
            self.flood_fill((center_x, center_y), occupancy_map)
            occupancy_map = np.array(occupancy_map, dtype=float)
            for (x, y) in zip(ox, oy):
                ix = int(round((x - min_x) / xy_resolution))
                iy = int(round((y - min_y) / xy_resolution))
                occupancy_map[ix][iy] = 1.0  # occupied area 1.0
                occupancy_map[ix + 1][iy] = 1.0  # extend the occupied area
                occupancy_map[ix][iy + 1] = 1.0  # extend the occupied area
                occupancy_map[ix + 1][iy + 1] = 1.0  # extend the occupied area
        return occupancy_map, min_x, max_x, min_y, max_y, xy_resolution

class EKF_SLAM(Node):
    '''Class for the EKF SLAM ROS2 node
    '''
    def __init__(self):
        '''Constructor for the EKF SLAM ROS2 node
        
        Parameters:
            None
        '''
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
        # self.landmark_radius = 0.08
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

        # broadcasters
        self.robot_tf_broadcaster = TransformBroadcaster(self, qos=QoSProfile(depth=10))
        self.landmark_tf_broadcaster = TransformBroadcaster(self, qos=QoSProfile(depth=10))
        
        # publishers topics list
        self.mapPublisher = self.create_publisher(
            OccupancyGrid,
            '/map', 
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))

        self.timer = self.create_timer(self.timeStep, self.timer_callback)

    def publish_robot(self):
        '''Publishes the robot position as a TransformStamped ROS2 message
        '''
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'robot'
        t.transform.translation.x = self.x[0,0] 
        t.transform.translation.y = self.x[1,0]
        t.transform.translation.z = 0.14
        q_robot_array = euler2quaternion(0, 0, self.x[2,0])
        q_robot = Quaternion(x=q_robot_array[0], y=q_robot_array[1], z=q_robot_array[2], w=q_robot_array[3])
        t.transform.rotation = q_robot
        self.robot_tf_broadcaster.sendTransform(t)

    def publish_landmarks(self):
        '''Publishes the landmarks as a TransformStamped ROS2 message
        '''
        if self.x.shape[0] > 3:
            for i in range(int((self.x.shape[0]-3)/2)):
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = 'odom'
                t.child_frame_id = 'landmark_' + str(i + 1)
                t.transform.translation.x = self.x[2*i+3,0]
                t.transform.translation.y = self.x[2*i+4,0]
                t.transform.translation.z = 0.14
                t.transform.rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                self.landmark_tf_broadcaster.sendTransform(t)

    def publish_map(self):
        '''Publishes the occupancy map as a ROS2 OccupancyGrid message
        '''
        if self.map.occ_map is not None:
            map_msg = OccupancyGrid()
            map_msg.header.stamp = self.get_clock().now().to_msg()
            map_msg.header.frame_id = 'odom'
            map_msg.info.map_load_time = self.get_clock().now().to_msg()
            map_msg.info.resolution = self.map.xy_resolution
            map_msg.info.width = self.map.occ_map.shape[0]
            map_msg.info.height = self.map.occ_map.shape[1]
            map_msg.info.origin.position.x = - self.map.occ_map.shape[0] * self.map.xy_resolution / 2
            map_msg.info.origin.position.y = - self.map.occ_map.shape[1] * self.map.xy_resolution / 2 
            map_msg.info.origin.position.z = 0.14 # 14 cm above ground
            map_msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            map_msg.data = (np.int8(self.map.occ_map*100).T).flatten().tolist()
            self.mapPublisher.publish(map_msg)
            
    def twist_callback(self, msg):    
        '''Callback function for the robot input twist subscriber
        '''
        self.u[0] = msg.linear.x
        self.u[1] = msg.angular.z

    def scan_callback(self, msg):
        '''Callback function for the laser scan subscriber
        '''
        point_cloud = self.get_laser_scan(msg) # World frame

        # clustering with DBSCAN
        db = DBSCAN(eps=0.1).fit(point_cloud)

        # make array of clusters
        clusters = [point_cloud[db.labels_ == i] for i in range(db.labels_.max() + 1)]
        
        landmarks = self.get_landmarks(clusters) # World frame

        z = self.compare_and_add_landmarks(landmarks)
        x_hat, P_hat = self.ekf.predict(self.x, self.u, self.P, self.Rt)
        self.x, self.P = self.ekf.update(x_hat, P_hat, self.Qt, z)
        
        self.map.robot_x = self.x[0,0]
        self.map.robot_y = self.x[1,0]
        self.map.update_occ_grid(point_cloud)

    def timer_callback(self):
        '''Callback function for the publishers
        '''
        # Publish robot position
        self.publish_robot()

        # Publish landmarks
        self.publish_landmarks()

        # Publish map
        self.publish_map()

    def get_laser_scan(self, msg):
        '''Converts the laser scan message to a point cloud in the world frame

        Parameters:
            msg (LaserScan): ROS2 LaserScan message
        Returns:
            point_cloud (n x 2 np.array): Point cloud in the world frame, where n is the number of points
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

        point_cloud = rot(self.x[2,0]) @ np.vstack((x, y))
        
        return point_cloud.T + self.x[0:2,0]

    def get_landmarks(self, clusters):
        '''Finds circular shapes in the measured point cloud clusters and returns the landmark positions

        Parameters:
            clusters (m x n_i x 2 numpy array): Point cloud clusters, where m is the number of clusters and n_i is the number of points in cluster i
        Returns:
            landmarks (p x 2 numpy array): Landmark positions, where p is the number of landmarks
        '''
        landmarks = []
        for cluster in clusters:
            if len(cluster) > 15:
                cxe, cye, re, error = circle_fitting(cluster[:,0], cluster[:,1])
                if abs(error) < 0.005 and re <= self.landmark_radius + self.distance_threshold and re >= self.landmark_radius - self.distance_threshold:
                    landmarks.append(cxe)
                    landmarks.append(cye)

        return np.array(landmarks).reshape(-1, 1)

    def compare_and_add_landmarks(self, landmarks):
        '''
        Compare landmarks with current landmarks and add new ones
        
        Parameters:
            landmarks (2n x 1 numpy array): array of currently observed landmarks [[x1], [y1], [x2], [y2], ...] in world frame  
        Returns:
            z (3n numpy array): array of landmarks [r, theta, j, r, theta, j, ...] in robot frame
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
    
    print('EKF SLAM node started')
    rclpy.spin(ekf_slam)

    ekf_slam.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()