import numpy as np
import matplotlib.pyplot as plt

# Create line
x = np.linspace(-10, 10, 10)
y = 0.5*x + 1
xy = np.vstack([x, y]) 

print(xy)
# Translate and rotate line
R = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4)], [np.sin(np.pi/4), np.cos(np.pi/4)]])
t = np.array([1, 1])
xy = R @ xy + t[:, np.newaxis]

# Plot
plt.plot(x, y, 'r')
plt.plot(xy[0, :], xy[1, :], 'b')
plt.show()