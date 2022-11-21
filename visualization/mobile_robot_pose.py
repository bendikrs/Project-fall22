# create a visualization of the x, y, and theta of a mobile robot in a 2D plane

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{{amsmath}}'



# pose of the robot
X = np.array([2, 1.5, 2*np.pi/3])

# create robot shape as a polygon
robot_shape = np.array([[0.5, 0.5, -0.5, -0.5, 0.5], [0.5, -0.5, -0.5, 0.5, 0.5]])

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

# create a visualization of the theta angle using a sector of a circle
sector = Wedge((X[0], X[1]), 0.5, 0, 360*X[2]/(2*np.pi), linewidth=1, fc='lightgray', ec='silver')
plt.gca().add_patch(sector)
# add a label for the theta angle
plt.text(X[0]+0.3, X[1]+0.1, '$\\theta$', fontsize=12)


# plot the robot heading as an arrow
plt.arrow(X[0], X[1], 0.5*np.cos(X[2]), 0.5*np.sin(X[2]), head_width=0.1, head_length=0.1, fc='k', ec='k')


plt.axis('equal')
# add axis arrows
plt.arrow(0, 0, 3, 0, head_width=0.1, head_length=0.1, fc='k', ec='k')
plt.arrow(0, 0, 0, 3, head_width=0.1, head_length=0.1, fc='k', ec='k')

# add label to the x and y position
plt.text(X[0]-0.05, -0.15, '$x$', fontsize=12)
plt.text(-0.15, X[1], '$y$', fontsize=12)

# remove outer box
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
# remove ticks
plt.gca().set_xticks([])
plt.gca().set_yticks([])


# add the robot pose vector to the plot
plt.text(X[0]+0.5, X[1]+0.5, r'$\textbf{x}_t = \begin{bmatrix} x \\ y \\ \theta \end{bmatrix}$') 




plt.show()

