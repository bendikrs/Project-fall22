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

class Plotter:
    def __init__(self):
        self.NEES = []

    def plot(self, x_hat, P_hat, landmarks):
        self.plotLandmarks(landmarks)
        self.plotEstimatedLandmarks(x_hat)
        # self.plotRobot(robot)
        self.plotEstimatedRobot(x_hat)
        # self.plotMeasurement(x_hat, z, num_landmarks)
        # self.plotCov(x_hat, P_hat, num_landmarks, ax)
        # self.plotMeasurementDistance(robot.xTrue, robot.range)
        # self.NEES.append(self.calculateNEES(x_hat, robot.xTrue, P_hat, landmarks))
        plt.pause(0.01)

    def plotLandmarks(self, landmarks):
        plt.plot(landmarks[0::2], landmarks[1::2], 'g+')


    def plotEstimatedLandmarks(self, x_hat):
        estimatedLandmarks = x_hat[3:]
        plt.plot(estimatedLandmarks[0::2], estimatedLandmarks[1::2], 'ro', markersize=2)


    def plotRobot(self, robot):
        plt.arrow(robot.xTrue[0,0], robot.xTrue[1,0],0.5*np.cos(robot.xTrue[2,0]), 0.5*np.sin(robot.xTrue[2,0]), head_width=0.5, color='g')


    def plotEstimatedRobot(self, x_hat):
        plt.arrow(x_hat[0,0], x_hat[1,0], 0.5*np.cos(x_hat[2,0]), 0.5*np.sin(x_hat[2,0]), head_width=0.5, color='r')


    def plotMeasurement(self, x_hat, z, num_landmarks):
        z_xy = np.zeros((2*num_landmarks, 1))
        for j in range(num_landmarks):
            if z[2*j] < 1e4:
                plt.plot([x_hat[0,0], x_hat[2*j+3,0]], [x_hat[1,0], x_hat[2*j+4,0]], color=(0,0,1), linewidth=.5)


    def plotCov(self, x_hat, P_hat, num_landmarks, ax):
        for j in range(num_landmarks):
            if P_hat[2*j+3, 2*j+3] < 1e6:
                P_hat_x = np.sqrt(P_hat[2*j+3, 2*j+3])
                P_hat_y = np.sqrt(P_hat[2*j+4, 2*j+4])

                xLandmark = x_hat[2*j + 3]
                yLandmark = x_hat[2*j + 4]
                ax.add_patch(patches.Ellipse((xLandmark, yLandmark), \
                P_hat_x, P_hat_y, color=(0,0,1), fill=False))


    def calculateNEES(self, x, xTrue, P, landmarks):
        '''Calculate the Normalized Estimation Error Squared'''
        xTrue = np.vstack((xTrue, landmarks))
        e = x - xTrue
        NEES = np.dot(e.T, np.dot(np.linalg.inv(P), e))
        # NEES = e.T @ np.linalg.inv(P) @ e
        return NEES[0][0]


    def plotMeasurementDistance(self, xTrue, rangeLimit):
        # Plot the range of the measurements as a circle
        circle = plt.Circle((xTrue[0,0], xTrue[1,0]), rangeLimit, color='0.8', fill=False)
        plt.gcf().gca().add_artist(circle)

def wrapToPi(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

class EKF:
    def __init__(self, timeStep=0.1):
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

    def update(self, x_hat, P_hat, Qt, threshold=1e6):
        '''
        Update step
        x_hat: state [x, y, theta, x1, y1, x2, y2, ...],  shape (3 + 2 * num_landmarks, 1)
        P_hat: covariance matrix, shape (3 + 2 * num_landmarks, 3 + 2 * num_landmarks)
        z: processed landmark locations [range r, bearing theta, j landmark index], shape: (number of currently observed landmarks*3, 1)
        Qt: measurement noise, shape: (2, 2)
        Fx: Jacobian of motion model, shape: (3, 3 + 2 * num_landmarks)
        '''
        num_landmarks = (len(x_hat)-3)//2
        z = self.get_polar_coordinates(x_hat[3:], x_hat)
        for j in range(num_landmarks):

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
            z_dif = np.array([[z[2*j,0]], [z[2*j+1,0]]]) - z_hat

            # Update state and covariance
            x_hat = x_hat + K @ z_dif
            x_hat[2,0] = wrapToPi(x_hat[2,0])
            P_hat = (np.eye(x_hat.shape[0]) - K @ H) @ P_hat

        return x_hat, P_hat

    def get_polar_coordinates(self, landmarks, x):
        z = []
        for i in range(0, len(landmarks), 2):
            dx = landmarks[i] - x[0,0]
            dy = landmarks[i+1] - x[1,0]
            q = dx ** 2 + dy ** 2
            z.append(np.sqrt(q))
            z.append(np.arctan2(dy, dx) - x[2,0])
        return np.array(z).reshape(-1, 1)


class EKF_SLAM(Node):

    def __init__(self):
        super().__init__('EKF_SLAM')
        self.fig, self.ax = plt.subplots(figsize=(10, 10))

        # Robot motion
        self.u = np.array([0, 0]) # [v, omega]

        # I got the magic in me
        self.landmark_threshhold = 0.25
        self.landmark_init_cov = 10.0

        # EKF
        self.timeStep = 1.0
        self.Rt = np.diag([0.1, 0.1, 0.01]) ** 2
        self.Qt = np.diag([0.1, 0.1]) ** 2
        self.x = np.zeros((3, 1))
        self.P = np.eye(3)
        self.ekf = EKF(timeStep=self.timeStep)
        self.plotter = Plotter()

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


        x_hat, P_hat = self.ekf.predict(self.x, self.u, self.P, self.Rt)
        self.x, self.P = self.ekf.update(x_hat, P_hat, self.Qt)
        self.plotter.plot(self.x, self.P, landmarks)
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
            for i in range(3, len(self.x), 2):
                x = np.allclose(self.x[i], landmarks[::2], atol=self.landmark_threshhold)
                y = np.allclose(self.x[i+1], landmarks[1::2], atol=self.landmark_threshhold)
                # when x and y are false, add new landmark
                # TODO: check if this is correct
                arr = np.bitwise_xor(x, y)
                print(arr)
                if not arr:
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
