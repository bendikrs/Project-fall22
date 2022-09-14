import numpy as np
import matplotlib.pyplot as plt

# Map
seed = 1338
x_dim, y_dim = 100, 100
num_landmarks = 10
landmarks = np.hstack((np.random.RandomState(seed).rand(num_landmarks, 2) * np.array([x_dim, y_dim]), \
    np.arange(num_landmarks).reshape(-1, 1)))
fov = 360 # field of view

plt.xlim(0, x_dim)
plt.ylim(0, y_dim)
plt.plot(landmarks[:, 0], landmarks[:, 1], 'x')

Rt = np.array([[0.1,   0,   0], 
                [  0, 0.1,   0],
                [  0,   0, 0.1]]) # Robot motion noise
Qt = np.array([[0.1,   0],
                [  0, 0.1]]) # Landmark measurement noise
x = np.array([50, 20, 0]) # Initial robot pose

timestep = 1
x_hat = np.zeros((3 + 2 * num_landmarks, 1)) # mu, Initial state x, y, theta, x1, y1, x2, y2, ...
x_hat[:3] = x.reshape(-1, 1)
P_hat = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks)) # sigma0
P_hat[3:, 3:] = np.eye(2*num_landmarks)*1e6 # set intial covariance for landmarks to large value

from ekf import EKF
from robot import Robot
ekf = EKF()
robot = Robot(fov, 100)

# simulate
u = np.array([1, 0])
while True:
    # move robot
    # predict
    # sense
    # update
    # plot
    # 

    # move
    # x = robot.move(x, u, Rt)
    # predict
    x_hat, P_hat = ekf.predict(x_hat, u, P_hat, Rt)
    # sense
    z = robot.sense(landmarks, x_hat, Qt)

    # update
    x_hat, P_hat = ekf.update(x_hat, P_hat, z, Qt)
    
    # plot
    plt.cla()
    plt.xlim(0, x_dim)
    plt.ylim(0, y_dim)
    plt.plot(landmarks[:, 0], landmarks[:, 1], 'x')
    plt.plot(x_hat[0], x_hat[1], 'o')
    plt.plot(x_hat[3::2], x_hat[4::2], 'o')
    plt.pause(0.1)