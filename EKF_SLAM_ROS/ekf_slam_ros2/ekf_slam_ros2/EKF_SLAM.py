import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Twist, Quaternion, PoseArray, Pose
from tf2_ros import TransformStamped, TransformBroadcaster
import time


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
    
    def setTimeStep(self, timeStep):
        '''
        Set time step

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
            return x_hat, P_hat


        for i in range(0, z.shape[0], 3): # for each landmark
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
        self.t0 = time.time()
        self.xTrue = np.zeros((3, 1))

        # Robot motion
        self.u = np.array([0.0, 0.0]) # [v, omega]
        self.new_landmark = np.array([])

        # Mahalanobis distance threshold
        self.landmark_threshold = 0.2

        # subscribers
        self.twistSubscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.twist_callback,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
        self.twistSubscription

        self.newLandmarkSubscription = self.create_subscription(
            PoseArray,
            '/new_landmarks',
            self.new_landmark_callback,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
        self.newLandmarkSubscription

        # publishers
        self.odomPublisher = self.create_publisher(
            Pose,
            '/robot_pose',
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
        self.odomPublisher 

        # self.NEESPublisher = self.create_publisher(
        #     Float64,
        #     '/NEES',
        #     QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
        # self.NEESPublisher
        self.RMSEPublisher = self.create_publisher(
            Float64MultiArray,
            '/RMSE',
            QoSProfile(depth=5, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST))
        self.RMSEPublisher

        # broadcasters
        self.robot_tf_broadcaster = TransformBroadcaster(self, qos=QoSProfile(depth=10))
        self.landmark_tf_broadcaster = TransformBroadcaster(self, qos=QoSProfile(depth=10))

        self.timer = self.create_timer(self.timeStep, self.timer_callback)

    def publish_pose(self):
        '''Publishes the robot odometry
        '''
        pose = Pose()
        pose.position.x = self.x[0,0]
        pose.position.y = self.x[1,0]
        pose.position.z = 0.0
        q = euler2quaternion(0.0, 0.0, self.x[2,0])
        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]
        self.odomPublisher.publish(pose)
        

    def publish_state(self):
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

    # def publish_NEES(self):
    #     '''Publishes the NEES
    #     '''
    #     NEES = Float64()
    #     data = self.x.T @ np.linalg.inv(self.P) @ self.x
    #     NEES.data = data[0,0] / self.x.shape[0]
    #     self.NEESPublisher.publish(NEES)

    def publish_RMSE(self):
        '''Publishes the RMSE
        '''
        RMSE = Float64MultiArray()

        pose_data = np.sqrt(((self.x[0,0] - self.xTrue[0,0])**2 + (self.x[1,0] - self.xTrue[1,0])**2) / 2)
        # if self.x[2,0] - self.xTrue[2,0] >= np.pi:
        #     if self.x[2,0] < 0:
        #         heading_data = np.sqrt((self.x[2,0]+np.pi - self.xTrue[2,0])**2)
        #     elif self.xTrue[2,0] < 0:
        #         heading_data = np.sqrt((self.x[2,0] - self.xTrue[2,0]+np.pi)**2)
        # else:
        #     heading_data = np.sqrt((self.x[2,0] - self.xTrue[2,0])**2)
        # if self.x[2,0] - self.xTrue[2,0] >= np.pi:
        #     heading_data = np.sqrt((self.x[2,0] - self.xTrue[2,0])**2)
        # else:
        #     heading_data = np.sqrt((self.x[2,0] - self.xTrue[2,0])**2)

        ang = (self.x[2,0] - self.xTrue[2,0] + np.pi) % (2*np.pi) - (np.pi)
        heading_data = np.sqrt(ang**2)

        RMSE.data = [pose_data, heading_data]
        self.RMSEPublisher.publish(RMSE)

    def twist_callback(self, msg):    
        '''Callback function for the robot input twist subscriber
        '''
        self.u[0] = msg.linear.x
        self.u[1] = msg.angular.z

    def new_landmark_callback(self, msg):
        '''Callback function for the new landmark subscriber
        '''
        # Get the position of the new landmark in world coordinates
        if len(msg.poses) > 0:
            self.new_landmark = np.zeros((2*len(msg.poses), 1))
            for i in range(len(msg.poses)):
                x = msg.poses[i].position.x
                y = msg.poses[i].position.y
                temp = rot(self.x[2,0]) @ np.array([[x], [y]]) 
                self.new_landmark[2*i  , 0] = temp[0,0] + self.x[0,0]
                self.new_landmark[2*i+1, 0] = temp[1,0] + self.x[1,0]
        else:
            self.new_landmark = np.zeros((0, 1))
        
        z = self.compare_and_add_landmarks(self.new_landmark)
        self.ekf.setTimeStep(time.time() - self.t0)
        # self.get_logger().info("Time: " + str(time.time() - self.t0))
        self.xTrue = self.ekf.g(self.xTrue, self.u, np.eye(3))
        self.xTrue[2,0] = wrapToPi(self.xTrue[2,0])
        x_hat, P_hat = self.ekf.predict(self.x, self.u, self.P, self.Rt)
        self.t0 = time.time()
        self.x, self.P = self.ekf.update(x_hat, P_hat, self.Qt, z)

    def timer_callback(self):
        '''Callback function for the publishers
        '''
        # # Publish NEES
        # self.publish_NEES()

        # Publish RMSE
        self.publish_RMSE()

        # Publish odometry
        self.publish_pose()

        # Publish combined state vector
        self.publish_state()


    def compare_and_add_landmarks(self, landmarks):
        '''
        Compare landmarks with current landmarks and add new ones
        
        Parameters:
            landmarks (2n x 1 numpy array): array of currently observed landmarks [[x1], [y1], [x2], [y2], ...] in world frame  
        Returns:
            z (3n numpy array): array of landmarks [r, theta, j, r, theta, j, ...] in robot frame
        '''
        z = np.zeros(((landmarks.shape[0]//2)*3, 1))

        # if not exist, add all
        if len(self.x) == 3 and len(landmarks) > 0:
            self.x = np.vstack((self.x, landmarks))
            self.P = np.zeros((len(self.x), len(self.x)))
            self.P[:3, :3] = np.eye(3)
            self.P[3:, 3:] = np.eye(len(self.x) - 3)

            z[::3,0] = np.sqrt((self.x[0,0] - landmarks[::2,0])**2 + (self.x[1,0] - landmarks[1::2,0])**2) # r
            z[1::3,0] = wrapToPi(np.arctan2(landmarks[1::2,0] - self.x[1,0], landmarks[::2,0] - self.x[0,0]) - self.x[2,0]) # theta
            z[2::3,0] = np.arange(0, len(landmarks)//2)#.reshape(-1, 1) # j

        # compare new landmarks with old landmarks
        elif len(self.x) > 3 and len(landmarks) > 0:
            for i in range(0, len(landmarks), 2):
                meas_x = landmarks[i,0] 
                meas_y = landmarks[i+1,0]

                dists = self.get_mahalanobis_distances(np.array([[meas_x], [meas_y]]))
                i = int(i * 3/2)
                z[i,0] = np.sqrt((meas_x - self.x[0,0])**2 + (meas_y - self.x[1,0])**2)
                z[i+1,0] = wrapToPi(np.arctan2(meas_y - self.x[1,0], meas_x - self.x[0,0]) - self.x[2,0])

                if np.min(dists) < self.landmark_threshold and np.min(dists) != 0.0: # if landmark already exists
                    z[i+2,0] = int(np.argmin(dists))
                    
                else: # if landmark does not exist
                    self.x = np.vstack((self.x, np.array([[meas_x], [meas_y]])))
                    self.P = np.block([[self.P, np.zeros((len(self.P), 2))], 
                                       [np.zeros((2, len(self.P))), np.eye(2)]])
                    z[i+2,0] = int(((len(self.x) - 3)//2 - 1))
        return z  
    
    def get_mahalanobis_distances(self, z):
        '''
        Calculate the Mahalanobis distance between measured landmark z and all landmarks in the map

        Parameters
            z (2x1, numpy array): measured landmark position
        Returns:
            d (2nx1, numpy array): Mahalanobis distance [m]
        '''
        d = np.zeros(((len(self.x)-3)//2, 1))
        for i in range(3, len(self.x), 2):
            temp = (z - np.array([[self.x[i,0]], [self.x[i+1,0]]]))
            p_temp = np.linalg.pinv(self.P[0:2, i:i+2])
            dist = temp.T @ p_temp @ temp
            d[(i-3)//2, 0] = dist[0,0]
        return d


def main(args=None):
    rclpy.init(args=args)
    ekf_slam = EKF_SLAM()
    
    print('EKF SLAM node started')
    rclpy.spin(ekf_slam)

    ekf_slam.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()