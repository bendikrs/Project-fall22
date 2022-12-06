# Script to visualize the map used in the experiments

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create the outline of the 2x2.4m map
map_outline = np.array([[0, 0, 2.4, 2.4, 0], [0, 2, 2, 0, 0]])

# center point
center = np.array([1.2, 1])

# Create the landmarks
landmarks = np.array([[0, 0], [0.25, 0.25], [0.25, -0.25], [-0.25, 0.25], [-0.25, -0.25]]) + center

# pose of the robot
X = np.array([center[0] - 0, center[1] - 0.65, 0])

# create robot shape as a polygon, size 0.28x0.3m
robot_shape = np.array([[0.15, 0.15, -0.15, -0.15, 0.15], [0.15, -0.15, -0.15, 0.15, 0.15]])

# create a rotation matrix
R = np.array([[np.cos(X[2]), -np.sin(X[2])], [np.sin(X[2]), np.cos(X[2])]])

# rotate the robot shape
robot_shape = R @ robot_shape

# translate the robot shape
robot_shape[0, :] = robot_shape[0, :] + X[0]
robot_shape[1, :] = robot_shape[1, :] + X[1]

# plot the robot shape
plt.plot(robot_shape[0, :], robot_shape[1, :], 'k-')

# plot the robot center
plt.plot(X[0], X[1], 'ro')

# mark the origin of the robot with stippled lines and x, y, theta labels
plt.plot([X[0], X[0]], [X[1], 0], 'k--', linewidth=0.8)
plt.plot([X[0], 0], [X[1], X[1]], 'k--', linewidth=0.8)

# plot the robot heading as an arrow
plt.arrow(X[0], X[1], 0.15*np.cos(X[2]), 0.15*np.sin(X[2]), head_width=0.1, head_length=0.1, fc='k', ec='k')

# plot the map
plt.plot(map_outline[0], map_outline[1], 'k')
plt.plot(landmarks[:, 0], landmarks[:, 1], 'kx', label='landmarks')

# mark the landmarks with dashed line from the axis
plt.plot([landmarks[0, 0], landmarks[0, 0]], [landmarks[0, 1], 0], 'k--', linewidth=0.8)
plt.plot([landmarks[0, 0], 0], [landmarks[0, 1], landmarks[0, 1]], 'k--', linewidth=0.8)

plt.plot([landmarks[1, 0], landmarks[1, 0]], [landmarks[1, 1], 0], 'k--', linewidth=0.8)
plt.plot([landmarks[3, 0], landmarks[3, 0]], [landmarks[3, 1], 0], 'k--', linewidth=0.8)

plt.plot([landmarks[1, 0], 0], [landmarks[1, 1], landmarks[1, 1]], 'k--', linewidth=0.8)
plt.plot([landmarks[2, 0], 0], [landmarks[2, 1], landmarks[2, 1]], 'k--', linewidth=0.8)

# plt.plot([landmarks[4, 0], 0], [landmarks[4, 1], landmarks[3, 1]], 'k--', linewidth=0.8)



# plot the trajectory as a circle with radius 0.65m using a dashed line, and mark the radius with  an arrow and a label
plt.plot(center[0] + 0.65*np.cos(np.linspace(0, 2*np.pi, 100)), center[1] + 0.65*np.sin(np.linspace(0, 2*np.pi, 100)), 'k--', label='Robot trajectory')
plt.arrow(center[0], center[1], 0.65*np.cos(np.deg2rad(10)), 0.65*np.sin(np.deg2rad(10)), head_width=0.05, head_length=0.05, fc='k', ec='k', length_includes_head=True)
plt.text(center[0] + 0.65, center[1] + 0.1, '$r = 0.65m$', fontsize=10)


# set ticks
plt.xticks([0, 1.2, 2.4, center[0]-0.25, center[0]+0.25])
plt.yticks([0, 1-0.65, 1, 2, center[1]-0.25, center[1]+0.25])
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.xlim([0, 2.4])
plt.ylim([0, 2])
plt.legend()
plt.show()
