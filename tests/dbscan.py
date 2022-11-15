from sklearn.cluster import DBSCAN 
from sklearn.datasets import make_blobs
import sklearn as sk
import matplotlib.pyplot as plt
import numpy as np


# Making a visualization of the DBSCAN algorithm

# make a dataset
sk.random.seed(10)
X, y = make_blobs(n_samples=15, centers=2, n_features=2, random_state=0, cluster_std=1, center_box=(-4.0, 4.0))

# run DBSCAN algorithm and 
db = DBSCAN(eps=0.7, min_samples=3).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# plot the results with different colors for core, border, and outlier points and add them to legend
# and plot the eps circle in gray around each point
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
                for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
        if k == -1:
                # Gray used for noise.
                col = [0, 0, 0, 0.5]
        
        class_member_mask = (labels == k)
        # if the point is a core point, plot it as a blue point, add it to the legend
        if k != -1:
                xy = X[class_member_mask & core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                                markeredgecolor='k', markersize=14, label="Core Point")
        
        # if the point is a border point, plot it as a red point, add it to the legend
        if k!=-1:
                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                                markeredgecolor='k', markersize=6, label="Border Point")

        # if the point is an outlier, plot it as a black point, add it to the legend
        if k == -1:
                xy = X[class_member_mask & ~core_samples_mask]
                plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                                markeredgecolor='k', markersize=6, label="Outlier")

        # plot the eps circle in gray around each core point
        if k != -1:
                for i in range(len(X[class_member_mask & core_samples_mask])):
                        circle = plt.Circle((X[class_member_mask& core_samples_mask][i][0], X[class_member_mask& core_samples_mask][i][1]), 0.7, color='gray', fill=False)
                        plt.gca().add_artist(circle)

# remove axis labels
plt.xticks([])
plt.yticks([])

plt.axis('equal')
plt.legend()
plt.show()
