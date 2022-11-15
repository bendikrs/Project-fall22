# colors 
red = (0.6196078431372549, 0.00392156862745098, 0.25882352941176473, 1.0)
yellow = (0.998077662437524, 0.9992310649750096, 0.7460207612456747, 1.0)
blue = 'darkcyan'

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Wedge, Rectangle


# make a vizualization of a laser scan and a occupancy grid map
fig, ax = plt.subplots()

# pose of the robot
X = np.array([1, 2, 0])

# create robot shape as a polygon
robot_shape = np.array([[0.5, 0.5, -0.5, -0.5, 0.5], [0.5, -0.5, -0.5, 0.5, 0.5]])*0.6

# create a rotation matrix
R = np.array([[np.cos(X[2]), -np.sin(X[2])], [np.sin(X[2]), np.cos(X[2])]])

# translate the robot shape
robot_shape[0, :] = robot_shape[0, :] + X[0]
robot_shape[1, :] = robot_shape[1, :] + X[1]

# plot the robot shape
ax.plot(robot_shape[0, :], robot_shape[1, :], 'k-')
# fill the robot shape with white color
ax.fill(robot_shape[0, :], robot_shape[1, :], 'w')


# plot the robot heading as an arrow
plt.arrow(X[0], X[1], 0.5*np.cos(X[2]), 0.5*np.sin(X[2]), head_width=0.1, head_length=0.1, fc='k', ec='k')


# plot the laser scan as a wedge
ax.add_patch(Wedge((X[0], X[1]), 4.05, -20, 20, fc='none', ec=blue, lw=2))

# add a grid map
# create a grid map with 0.2m resolution and gray color
res = 0.5 # resolution of the grid map in meters 
c = int(1/res) # number of cells per meter
grid_map = np.zeros((c*4, c*6)) + 0.5

# add a wall
grid_map[2:6, 10] = 0
grid_map[1, 9] = 0
grid_map[6, 9] = 0

# add the unoccupied space as white
grid_map[2:6, 9] = 1
grid_map[1:7, 8] = 1
grid_map[2:6, 7] = 1
grid_map[2:6, 6] = 1
grid_map[3:5, 5] = 1
grid_map[3:5, 4] = 1
grid_map[3:5, 3] = 1
grid_map[3:4, 2] = 1



# add grid lines
for i in range(grid_map.shape[0]*c):
    for j in range(grid_map.shape[1]*c):
        ax.add_patch(Rectangle((j*res, i*res), res, res, fc='none', ec='dimgray', lw=0.5))



# plot the grid map
plt.imshow(grid_map, cmap=plt.get_cmap('gray'), origin='lower', extent=[0, 6, 0, 4], vmin=0, vmax=1)




# remove the axis
plt.axis('off')


# plt.axis('equal')
plt.show()
