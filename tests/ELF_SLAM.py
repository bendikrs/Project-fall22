import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from ekf import EKF
from robot import Robot


def createLandmarks(v, omega, delta_r, num_landmarks):
    '''Create map of landmarks, for use on the bonnabel test'''
    landmarks = np.zeros((2*num_landmarks, 1))
    r = v/omega
    for i in range(num_landmarks):
        rho = r + delta_r
        theta=(2*np.pi*i)/num_landmarks
        landmarks[2*i  ] = rho*np.cos(theta)
        landmarks[2*i+1] = rho*np.sin(theta) + 5
    return landmarks

def plotLandmarks(landmarks):
    plt.plot(landmarks[0::2], landmarks[1::2], 'g+')

def plotEstimatedLandmarks(x_hat):
    estimatedLandmarks = x_hat[3:]
    plt.plot(estimatedLandmarks[0::2], estimatedLandmarks[1::2], 'ro', markersize=2)

def plotRobot(robot):
    plt.arrow(robot.xTrue[0,0], robot.xTrue[1,0],0.5*np.cos(robot.xTrue[2,0]), 0.5*np.sin(robot.xTrue[2,0]), head_width=0.5, color='g')

def plotEstimatedRobot(x_hat):
    plt.arrow(x_hat[0,0], x_hat[1,0], 0.5*np.cos(x_hat[2,0]), 0.5*np.sin(x_hat[2,0]), head_width=0.5, color='r')

def plotMeasurement(x_hat, P_hat, z, num_landmarks):
    z_xy = np.zeros((2*num_landmarks, 1))
    for j in range(num_landmarks):
        z_xy[2*j] = x_hat[0,0] + z[2*j]*np.cos(z[2*j+1])
        z_xy[2*j+1] = x_hat[1,0] + z[2*j]*np.sin(z[2*j+1])
        # plt.plot([x_hat[0,0], z_xy[2*j,0]], [x_hat[1,0], z_xy[2*j+1,0]], color=(1,0,1))

        plt.plot([x_hat[0,0], x_hat[2*j+3,0]], [x_hat[1,0], x_hat[2*j+4,0]], color=(0,0,1), linewidth=.5)

def plotCov(x_hat, P_hat, z, num_landmarks, ax):
    for j in range(num_landmarks):
        if P_hat[2*j+3, 2*j+3] < 1e4:
            P_hat_x = P_hat[2*j+3, 2*j+3]
            P_hat_y = P_hat[2*j+4, 2*j+4]
            
            r, theta = z[j], z[j+1]
            xLandmark = x_hat[2*j + 3]
            yLandmark = x_hat[2*j + 4]
            ax.add_patch(patches.Ellipse((xLandmark, yLandmark), 
            P_hat_x*10000, P_hat_y*10000, color=(0,0,1), fill=False))


timeStep = 0.1
robot = Robot(range=50)
ekf = EKF(timeStep=timeStep)

# Map
num_landmarks = 20
landmarks = createLandmarks(1, 2*np.pi/40, 1, num_landmarks)

Rt = np.array([[0.1, 0.0], 
               [0.0,0.0001]])**2 # Robot motion noise

Qt = np.array([[0.01, 0.0],
               [0.0, 0.01]])**2 # Landmark measurement noise

x = np.zeros((3,1)) # Initial robot pose

x_hat = np.zeros((3 + 2 * num_landmarks, 1)) # Initial state x, y, theta, x1, y1, x2, y2, ...
x_hat[:3] = x

P_hat = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks))
P_hat[3:, 3:] = np.eye(2*num_landmarks)*1e7 # set intial covariance for landmarks to large value

u = np.array([1, np.deg2rad(9)]) # control input (v, omega)
x = robot.move(x, u, Rt)
fig, ax = plt.subplots()
for i in range(100):
    x_hat, P_hat = ekf.predict(x_hat, u, P_hat, Rt)
    z = robot.sense(landmarks, num_landmarks, x_hat, Qt)
    x_hat, P_hat = ekf.update(x_hat, P_hat, z, Qt, num_landmarks)
    x = robot.move(x, u, Rt, timeStep=timeStep)

    # Plot
    plt.cla()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 15)
    plotLandmarks(landmarks)
    plotEstimatedLandmarks(x_hat)
    plotRobot(robot)
    plotEstimatedRobot(x_hat)
    plotMeasurement(x_hat, P_hat, z, num_landmarks)
    plotCov(x_hat, P_hat, z, num_landmarks, ax)
    plt.pause(0.1)
