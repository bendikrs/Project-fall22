import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
# Parameters of the system
# This is the parameters we need to estimate
# Center
C = np.array([0 , 0])
# Radius
R = 1
# Number of points
N = 500
# Noise
k = 0.15

## Create noisy circle
alpha = 2*np.pi*np.random.rand(N)
noise = k*2*np.random.rand(N)-1
points = C + np.array([ R*noise*np.cos(alpha) , R*noise*np.sin(alpha) ]).T

A = np.hstack((points, np.ones((N,1))))
B = points[:,0]*points[:,0] + points[:,1]*points[:,1]

# Least square approximation
X = np.linalg.pinv(A) @ B

# Calculate circle parameter
xc = X[0]/2
yc = X[1]/2
r = np.sqrt(4*X[2] + X[0]**2 + X[1]**2 )/2


fig, ax = plt.subplots()
ax.plot(points[:,0],points[:,1], '.', markersize=3, color='blue')
ax.add_patch(Circle((xc,yc),r,fill=False, linewidth=4, color='red'))
plt.legend(['Measured points', 'Fitted circle'], loc='upper right')
plt.grid()
plt.show()