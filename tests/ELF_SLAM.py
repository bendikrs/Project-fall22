import numpy as np
import matplotlib.pyplot as plt
from ekf import EKF
from robot import Robot
ekf = EKF()
robot = Robot(10.0)

def createLandmarks(v, omega, delta_r, num_landmarks):
    '''Create map of landmarks, for use on the bonnabel test'''
    landmarks = np.zeros((2*num_landmarks, 1))
    # print(landmarks)
    r = v/omega
    for i in range(num_landmarks):
        rho = r + delta_r
        theta=(2*np.pi*i)/num_landmarks
        landmarks[2*i  ] = rho*np.cos(theta)
        landmarks[2*i+1] = rho*np.sin(theta)
    return landmarks

def plotLandmarks(landmarks):
    plt.plot(landmarks[0::2], landmarks[1::2], 'go')

def plotEstimatedLandmarks(x_hat):
    estimatedLandmarks = x_hat[3:]
    plt.plot(estimatedLandmarks[0::2], estimatedLandmarks[1::2], 'ro')

def plotRobot(robot):
    plt.arrow(robot.xTrue[0], robot.xTrue[1],0.5*np.cos(robot.xTrue[2]), 0.5*np.sin(robot.xTrue[2]), head_width=0.5, color='g')

def plotEstimatedRobot(x_hat):
    plt.arrow(x_hat[0,0], x_hat[1,0], 0.5*np.cos(x_hat[2,0]), 0.5*np.sin(x_hat[2,0]), head_width=0.5, color='r')

# Map
num_landmarks = 8
landmarks = createLandmarks(1, 2*np.pi/40, 1, num_landmarks)

Rt = np.array([[0.1, 0.0, 0.0], 
               [0.0, 0.1, 0.0],
               [0.0, 0.0, 0.1]]) # Robot motion noise

Qt = np.array([[0.1, 0.0],
               [0.0, 0.1]]) # Landmark measurement noise

x = np.zeros((3,1)) # Initial robot pose
x_hat = np.zeros((3 + 2 * num_landmarks, 1)) # Initial state x, y, theta, x1, y1, x2, y2, ...
x_hat[:3] = x

P_hat = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks))
P_hat[3:, 3:] = np.eye(2*num_landmarks)*1e6 # set intial covariance for landmarks to large value

u = np.array([0.1, 0.001]) # control input (v, omega)

for i in range(25):
    x_hat, P_hat = ekf.predict(x_hat, u, P_hat, Rt)
    z = robot.sense(landmarks, num_landmarks, x_hat, Qt)
    x_hat, P_hat = ekf.update(x_hat, P_hat, z, Qt, num_landmarks)
    x = robot.move(x, u, Rt)
    print("Robot pose: ", x_hat[:3].T)
    print("xTrue     : ", robot.xTrue)

    # Plot
    plt.cla()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plotLandmarks(landmarks)
    plotEstimatedLandmarks(x_hat)
    plotRobot(robot)
    plotEstimatedRobot(x_hat)
    plt.pause(0.1)
