import numpy as np
import matplotlib.pyplot as plt

from ekf import EKF
from robot import Robot
from plotting import Plotter
from simulator import Simulator

simulator = Simulator()
landmarks, num_landmarks = simulator.createCircularLandmarks(1, 2*np.pi/50, 1)

timeStep = 0.1
rangeLimit = 6
robot = Robot(range=rangeLimit, timeStep=timeStep)
ekf = EKF(range=rangeLimit,timeStep=timeStep)
plotter = Plotter()

Rt = np.diag([0.1, 0.1, 0.01]) ** 2
Qt = np.diag([0.1, 0.1]) ** 2

x = np.zeros((3 + 2 * num_landmarks, 1)) # Initial state x, y, theta, x1, y1, x2, y2, ...
x[:3] = np.zeros((3,1)) # Initial robot pose
x[3:] = np.ones((2*num_landmarks,1))*1e6  # Initial landmark positions

P = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks))
P[:3, :3] = np.diag([1.0, 1.0, 1.0])
P[3:, 3:] = np.eye(2*num_landmarks)*1e6 # set intial covariance for landmarks to large value

u = np.array([1.0, np.deg2rad(9.0)]) # control input (v, omega)

fig, ax = plt.subplots(figsize=(10,10))
for i in range(40):
    z = robot.sense(landmarks, num_landmarks, Qt)
    robot.move(u)
    x_hat, P_hat = ekf.predict(x, u, P, Rt)
    x, P = ekf.update(x_hat, P_hat, z, Qt, num_landmarks)

    # Plot
    plt.cla()
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 18)
    plotter.plot(x, P, z, num_landmarks, landmarks, robot, ax)
