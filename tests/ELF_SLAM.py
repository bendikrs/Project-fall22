import numpy as np
import matplotlib.pyplot as plt
from ekf import EKF
from robot import Robot
ekf = EKF()
robot = Robot(10.0)

def createLandmarks(v, omega, delta_r, num_landmarks):
    '''Create map of landmarks, for use on the bonnabel test'''
    landmarks = np.zeros((num_landmarks, 2))
    # print(landmarks)
    r = v/omega
    for i in range(num_landmarks):
        rho = r + delta_r
        theta=(2*np.pi*i)/num_landmarks
        landmarks[i, 0] = rho*np.cos(theta)
        landmarks[i, 1] = rho*np.sin(theta)

    return landmarks



# Map
num_landmarks = 10

landmarks = createLandmarks(1, 2*np.pi/40, 1, num_landmarks)
plt.plot(landmarks[:, 0], landmarks[:, 1], 'x')

# Rt = np.array([[0.1, 0.0], 
#                [0.0, 0.1]]).astype("float64") # Robot motion noise
Rt = np.array([[0.1, 0.0, 0.0], 
               [0.0, 0.1, 0.0],
               [0.0, 0.0, 0.1]]).astype("float64") # Robot motion noise

Qt = np.array([[0.1, 0.0],
               [0.0, 0.1]]) # Landmark measurement noise

x = np.array([0.0, 0.0, 0.0]) # Initial robot pose
x_hat = np.zeros((3 + 2 * num_landmarks, 3)) # mu, Initial state x, y, theta, x1, y1, x2, y2, ...
x_hat[0] = x

P_hat = np.zeros((3 + 2 * num_landmarks, 3 + 2 * num_landmarks)) # sigma0
P_hat[3:, 3:] = np.eye(2*num_landmarks)*1e6 # set intial covariance for landmarks to large value

u = np.array([1.0, 0.4]) # control input (v, omega)

for i in range(100):
    x = robot.move(x, u, Rt)
    x_hat, P_hat = ekf.predict(x_hat, u, P_hat, Rt)
    z = robot.sense(landmarks, x_hat, Qt)
    x_hat, P_hat = ekf.update(x_hat, P_hat, z, Qt)

    plt.cla()
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.plot(landmarks[:, 0], landmarks[:, 1], 'x')
    plt.plot(x_hat[:,0], x_hat[:,1], 'o')
    plt.plot(x[0], x[1], 'o')
    plt.pause(0.1)

