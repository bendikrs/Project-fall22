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
        landmarks[2*i+1] = rho*np.sin(theta) + 6.25
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

def plotMeasurement(x_hat, z, num_landmarks):
    z_xy = np.zeros((2*num_landmarks, 1))
    for j in range(num_landmarks):
        if z[2*j] < 1e4:
            plt.plot([x_hat[0,0], x_hat[2*j+3,0]], [x_hat[1,0], x_hat[2*j+4,0]], color=(0,0,1), linewidth=.5)

def plotCov(x_hat, P_hat, num_landmarks, ax):
    for j in range(num_landmarks):
        if P_hat[2*j+3, 2*j+3] < 1e6:
            P_hat_x = np.sqrt(P_hat[2*j+3, 2*j+3])
            P_hat_y = np.sqrt(P_hat[2*j+4, 2*j+4])

            xLandmark = x_hat[2*j + 3]
            yLandmark = x_hat[2*j + 4]
            ax.add_patch(patches.Ellipse((xLandmark, yLandmark), \
            P_hat_x, P_hat_y, color=(0,0,1), fill=False))


def calculateNEES(x, xTrue, P, landmarks):
    '''Calculate the Normalized Estimation Error Squared'''
    xTrue = np.vstack((xTrue, landmarks))
    e = x - xTrue
    NEES = np.dot(e.T, np.dot(np.linalg.inv(P), e))
    # NEES = e.T @ np.linalg.inv(P) @ e
    return NEES[0][0]

def plotMeasurementDistance(xTrue, rangeLimit):
    # Plot the range of the measurements as a circle
    circle = plt.Circle((xTrue[0,0], xTrue[1,0]), rangeLimit, color='0.8', fill=False)
    plt.gcf().gca().add_artist(circle)


timeStep = 0.1
rangeLimit = 6
robot = Robot(range=rangeLimit, timeStep=timeStep)
ekf = EKF(range=rangeLimit,timeStep=timeStep)

# Map
num_landmarks = 20
landmarks = createLandmarks(1, 2*np.pi/50, 1, num_landmarks)

Rt = np.diag([0.1, 0.1, 0.01]) ** 2
Qt = np.diag([0.1, 0.1]) ** 2

x = np.zeros((3 + 2 * num_landmarks, 1)) # Initial state x, y, theta, x1, y1, x2, y2, ...
x[:3] = np.zeros((3,1)) # Initial robot pose
x[3:] = np.ones((2*num_landmarks,1))*1e6  # Initial landmark positions

P = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks))
P[:3, :3] = np.diag([1.0, 1.0, 1.0])
P[3:, 3:] = np.eye(2*num_landmarks)*1e6 # set intial covariance for landmarks to large value

fig, ax = plt.subplots()
u = np.array([1.0, np.deg2rad(9.0)]) # control input (v, omega)
NEES = []

for i in range(400):
    z = robot.sense(landmarks, num_landmarks, Qt)
    robot.move(u)
    x_hat, P_hat = ekf.predict(x, u, P, Rt)
    x, P = ekf.update(x_hat, P_hat, z, Qt, num_landmarks)
    NEES.append(calculateNEES(x, robot.xTrue, P, landmarks))

    # Plot
    plt.cla()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 18)
    plotLandmarks(landmarks)
    plotEstimatedLandmarks(x)
    plotRobot(robot)
    plotEstimatedRobot(x)
    plotMeasurement(x, z, num_landmarks)
    plotCov(x, P, num_landmarks, ax)
    plotMeasurementDistance(robot.xTrue, rangeLimit)
    plt.pause(0.01)

# plt.cla()
# plt.plot(NEES)
# plt.show()