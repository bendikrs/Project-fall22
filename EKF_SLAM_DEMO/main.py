import numpy as np
import matplotlib.pyplot as plt

from ekf import EKF
from robot import Robot
from plotting import Plotter
from map import Map

# Set up constant
u = np.array([1.0, np.deg2rad(9.0)]) # constant control input (v, omega)
num_landmarks = 20 # number of landmarks
timeStep = 1 # time step
numLoops = 10 # number of loops to drive
simTime = ((2*np.pi)/u[1])*numLoops # simulation time
rangeLimit = 5 # range limit of sensor
Rt = np.diag([0.1, 0.1, 0.1])
Qt = np.diag([0.01, 0.01])

# Set up initial state and covariance matrix
x = np.zeros((3 + 2 * num_landmarks, 1)) # state vector
x[:3] = np.zeros((3,1)) # Initial robot pose
x[3:] = np.ones((2*num_landmarks,1))*1e6  # Initial landmark positions
P = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks)) # covariance matrix
P[:3, :3] = np.diag([1.0, 1.0, 1.0])*0.001 # Initial robot pose uncertainty
P[3:, 3:] = np.eye(2*num_landmarks)*1e6 # set intial covariance for landmarks to large value

# Initialize robot, map, and plot
fig, ax = plt.subplots(figsize=(9,9))
map = Map(num_landmarks=num_landmarks)
robot = Robot(range=rangeLimit, x=x, timeStep=timeStep)
ekf = EKF(range=rangeLimit,timeStep=timeStep)
plotter = Plotter(fig, ax)


for i in range(int(simTime/timeStep)):
    z = robot.sense(map.landmarks, Qt)
    robot.move(u)
    x_hat, P_hat = ekf.predict(x, u, P, Rt)
    plotter.updateTrajectory(robot, x)
    x, P = ekf.update(x_hat, P_hat, z, Qt)

plt.cla()
ax.set_xlim(-10, 10)
ax.set_ylim(-5, 18)
plotter.plot(x, P, z, map.num_landmarks, map.landmarks, robot, ax)
plt.pause(100)