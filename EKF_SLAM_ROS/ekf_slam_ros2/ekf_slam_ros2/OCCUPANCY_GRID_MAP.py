import numpy as np
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import LaserScan
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Quaternion

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

class OCCUPANCY_GRID_MAP(Node):
    '''
    Map class
    '''
    def __init__(self):
        '''
        Initialize a map
        
        Parameters:
            None
        '''
        super().__init__('OCCUPANCY_GRID_MAP')

        self.map = np.array([[0.0, 0.0]]) # Pointcloud for map [[x, y], [x, y], ...]
        self.occ_map = None
        self.min_x = None
        self.max_x = None
        self.min_y = None
        self.max_y = None
        self.xy_resolution = None
        self.EXTEND_AREA = 5.0
        self.xy_resolution = 0.02

        self.x = None

        # Subscribe to the laser scan topic
        self.scanSubscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.scanSubscription

        # Create a transform listener
        self.target_frame = self.declare_parameter('odom', 'robot').get_parameter_value().string_value
        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer, self)

        # Publish the map
        self.mapPublisher = self.create_publisher(
            OccupancyGrid,
            '/map', 
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE))

        self.timer = self.create_timer(1.0, self.timer_callback)

    def scan_callback(self, msg):
        '''Callback function for the laser scan subscriber
        '''
        t = self.tfBuffer.lookup_transform(self.target_frame, 'odom', rclpy.time.Time())

        roll, pitch, yaw = quaternion2euler(t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w)
        self.x = np.array([[t.transform.translation.x], [t.transform.translation.y], [yaw]])
        print(self.x, "   ", self.target_frame)
        self.update_occ_grid(self.get_laser_scan(msg))



    def publish_map(self):
        '''Publishes the occupancy map as a ROS2 OccupancyGrid message
        '''
        if self.occ_map is not None:
            map_msg = OccupancyGrid()
            map_msg.header.stamp = self.get_clock().now().to_msg()
            map_msg.header.frame_id = 'odom'
            map_msg.info.map_load_time = self.get_clock().now().to_msg()
            map_msg.info.resolution = self.xy_resolution
            map_msg.info.width = self.occ_map.shape[0]
            map_msg.info.height = self.occ_map.shape[1]
            map_msg.info.origin.position.x = - self.occ_map.shape[0] * self.xy_resolution / 2
            map_msg.info.origin.position.y = - self.occ_map.shape[1] * self.xy_resolution / 2 
            map_msg.info.origin.position.z = 0.14 # 14 cm above ground
            map_msg.info.origin.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            map_msg.data = (np.int8(self.occ_map*100).T).flatten().tolist()
            self.mapPublisher.publish(map_msg)

    def timer_callback(self):
        '''Callback function for the publishers
        '''
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
            # temp_map = np.logical_and(self.occ_map, new_occ_map)
            # self.occ_map =  np.logical_and(self.occ_map, temp_map)     
            # merge new map with old map, based on min and max values and resolution
            self.occ_map = np.where(new_occ_map == 0.5, self.occ_map, new_occ_map)
    
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
        center_x = int(round(self.x[0,0]/xy_resolution)) + int(
            round(-min_x / xy_resolution))
        center_y = int(round(self.x[1,0]/xy_resolution)) + int(
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

def main(args=None):
    rclpy.init(args=args)
    occupancy_grid_map = OCCUPANCY_GRID_MAP()

    print('OCCUPANCY_GRID_MAP node started')
    rclpy.spin(occupancy_grid_map)

    occupancy_grid_map.destroy_node()
    rclpy.shutdown()
