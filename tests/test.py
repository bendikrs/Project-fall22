import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# Generate noisy circle
points = datasets.make_circles(n_samples=1000, noise=0.05, factor=0.3)[0]


# Generate random parameters for a cirle
def generate_random_circle(r):
    """
    Generate random circle parameters
    params: r - radius of the circle
    return: x, y, r
    """
    x = np.random.uniform(-1, 1)
    y = np.random.uniform(-1, 1)
    r = r
    return x, y, r


# RANSAC algorithm
def ransac(points, iterations=1000, threshold=0.1):
    best_inliers = []
    best_params = None
    for i in range(iterations):
        # Generate random circle
        x, y, r = generate_random_circle(r=0.3)

        # Calculate inliers
        inliers = []
        for point in points:
            if (point[0] - x)**2 + (point[1] - y)**2 < r**2 + threshold:
                inliers.append(point)

        # Update best inliers
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_params = (x, y, r)
    return best_inliers, best_params


inliers, parameters = ransac(points, iterations=1000, threshold=0.05)

# Plot
plt.scatter(points[:, 0], points[:, 1], c='b', s=1)
plt.scatter(np.array(inliers)[:, 0], np.array(inliers)[:, 1], c='r', s=1)
cirle = plt.Circle((parameters[0], parameters[1]), parameters[2], color='g', fill=False)
plt.gca().add_patch(cirle)
plt.show()