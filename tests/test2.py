import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets


# create points on a half-circle
def create_points(n):
    r = 10
    theta = np.linspace(0, np.pi, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.array([x, y]).T
    return points

# plot points
def plot_points(points):
    plt.scatter(points[:, 0], points[:, 1], c='b')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.show()



points = create_points(100)

# find the mean of the points
mean = np.mean(points, axis=0)
plt.plot(mean[0], mean[1], 'ro')
print(f'{mean[1]:.2f}')
plot_points(points)


guessed_cx = guessed_cx + 0.4*0.15 * np.sin(self.x[2,0])
guessed_cy = guessed_cy + 0.4*0.15 * np.cos(self.x[2,0])

