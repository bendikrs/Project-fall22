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

def quaternion_to_euler(x, y, z, w):
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
        self.occ_map = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.xy_resolution = None
        self.EXTEND_AREA = 10.0
        self.xy_resolution = 0.05

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

    def quaternion_from_euler(self, roll, pitch, yaw):
        """Convert euler angles to quaternion.
        roll, pitch, yaw: Euler angles in radians.
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

    def update_occ_grid(self, pointCloud):
        '''Updates the occupancy grid with the new point cloud'''
        ox, oy = pointCloud[:,0], pointCloud[:,1]
        new_occ_map, min_x, max_x, min_y, max_y, xy_resolution = \
        self.generate_ray_casting_grid_map(ox, oy, self.xy_resolution, True)
        if self.occ_map is None:
            self.occ_map = new_occ_map
            self.min_x = min_x
            self.max_x = max_x
            self.min_y = min_y
            self.max_y = max_y
            self.xy_resolution = xy_resolution
        else:
            self.occ_map = new_occ_map # TODO: merge maps
            
            # merge new map with old map, based on min and max values and resolution
            
    def bresenham(self, start, end):
        """
        Implementation of Bresenham's line drawing algorithm
        See en.wikipedia.org/wiki/Bresenham's_line_algorithm
        Bresenham's Line Algorithm
        Produces a np.array from start and end (original from roguebasin.com)
        >>> points1 = bresenham((4, 4), (6, 10))
        >>> print(points1)
        np.array([[4,4], [4,5], [5,6], [5,7], [5,8], [6,9], [6,10]])
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
        measurement center
        """
        min_x = round(min(ox) - self.EXTEND_AREA / 2.0)
        min_y = round(min(oy) - self.EXTEND_AREA / 2.0)
        max_x = round(max(ox) + self.EXTEND_AREA / 2.0)
        max_y = round(max(oy) + self.EXTEND_AREA / 2.0)
        # min_x = -5
        # min_y = -5
        # max_x = 5
        # max_y = 5
        xw = int(round((max_x - min_x) / xy_resolution))
        yw = int(round((max_y - min_y) / xy_resolution))
        print("The grid map is ", xw, "x", yw, ".")
        return min_x, min_y, max_x, max_y, xw, yw

    def atan_zero_to_twopi(y, x):
        angle = np.atan2(y, x)
        if angle < 0.0:
            angle += np.pi * 2.0
        return angle

    def init_flood_fill(self, center_point, obstacle_points, xy_points, min_coord,
                        xy_resolution):
        """
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
        """
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
        The breshen boolean tells if it's computed with bresenham ray casting
        (True) or with flood fill (False)
        """
        min_x, min_y, max_x, max_y, x_w, y_w = self.calc_grid_map_config(
            ox, oy, xy_resolution)
        # default 0.5 -- [[0.5 for i in range(y_w)] for i in range(x_w)]
        occupancy_map = np.ones((x_w, y_w)) / 2
        center_x = int(
            round(-min_x / xy_resolution))  # center x coordinate of the grid map
        center_y = int(
            round(-min_y / xy_resolution))  # center y coordinate of the grid map
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

    def __init__(self):
        super().__init__('EKF_SLAM')

        # Visualization
        self.x_origin = 0.0
        self.y_origin = 0.0
        self.z_origin = 0.0
        self.rot_origin = 0.0
        
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

        self.originSubscription = self.create_subscription(
            TFMessage,
            '/tf',
            self.origin_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.originSubscription

        # broadcasters
        self.robot_tf_broadcaster = TransformBroadcaster(self, qos=QoSProfile(depth=10))
        self.landmark_tf_broadcaster = TransformBroadcaster(self, qos=QoSProfile(depth=10))
        
        # # publishers topics list
        self.mapPublisher = self.create_publisher(
            OccupancyGrid,
            '/map', 
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))


        self.timer = self.create_timer(self.timeStep, self.timer_callback)



    def publish_robot(self):
        '''Publishes the robot position'''
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'odom'
        t.child_frame_id = 'robot'
        t.transform.translation.x = self.x[0,0] 
        t.transform.translation.y = self.x[1,0]
        t.transform.translation.z = 0.14
        q_robot_array = self.map.quaternion_from_euler(0, 0, self.x[2,0])
        q_robot = Quaternion(x=q_robot_array[0], y=q_robot_array[1], z=q_robot_array[2], w=q_robot_array[3])
        t.transform.rotation = q_robot
        self.robot_tf_broadcaster.sendTransform(t)


    def publish_landmarks(self):
        '''Publishes the landmarks'''
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
        '''Publishes the map'''
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




    def origin_callback(self, msg):
        '''Callback for the tf message'''
        # self.get_logger().info(msg.transforms[0].child_frame_id)
        if not self.x_origin and msg.transforms[0].child_frame_id == 'base_footprint':
            self.x_origin = msg.transforms[0].transform.translation.x
            self.y_origin = msg.transforms[0].transform.translation.y
            self.z_origin = msg.transforms[0].transform.translation.z
            self.q_origin = msg.transforms[0].transform.rotation

            print(self.x_origin, self.y_origin, self.z_origin, self.rot_origin)
            
    def twist_callback(self, msg):    
        # self.get_logger().info('v: "%f" omega: "%f"' % (msg.linear.x, msg.angular.z))
        self.u[0] = msg.linear.x
        self.u[1] = msg.angular.z

    def scan_callback(self, msg):
        point_cloud = self.get_laser_scan(msg) # Robot frame

        # self.map.add_pointcloud(point_cloud)
        # print(self.map.map[0:10])

        # clustering with DBSCAN
        db = DBSCAN(eps=0.1).fit(point_cloud)

        # make array of clusters
        clusters = [point_cloud[db.labels_ == i] for i in range(db.labels_.max() + 1)]
        
        landmarks = self.get_landmarks(clusters) # World frame

        z = self.compare_and_add_landmarks(landmarks)
        print('Total number of landmarks', (self.x.shape[0]-3)//2)
        x_hat, P_hat = self.ekf.predict(self.x, self.u, self.P, self.Rt)
        self.x, self.P = self.ekf.update(x_hat, P_hat, self.Qt, z)
        
        # self.map.add_pointcloud(point_cloud)
        # point_cloud[:,0] = point_cloud[:,0] + self.x_origin
        # point_cloud[:,1] = point_cloud[:,1] + self.y_origin
        self.map.update_occ_grid(point_cloud)

        # if self.map.occ_map is not None:
        #     self.mapPublisher.publish(self.map.occ_map)

    def timer_callback(self):
        # Publish robot position
        self.publish_robot()

        # Publish landmarks
        self.publish_landmarks()

        # Publish map
        self.publish_map()

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
