import numpy as np
from scipy.spatial.distance import mahalanobis

def mahalanobis_distance(self, z, P, x):
    '''
    Calculate the Mahalanobis distance between the predicted and measured state

    Parameters
        P (2x2, numpy array): predicted state covariance matrix
        z (2x1, numpy array): measured landmark position
    Returns:
        d (float): Mahalanobis distance [m]
    '''
    d = np.sqrt((x[0:2] - z) @ np.linalg.inv(P) @ (x[0:2] - z).T)
    return d


# test the mahalanobis distance function

P = np.array([[0.1, 0.0],
                [0.0, 0.1]])
z = np.array([[2.0],
                [0.0]])
x = np.array([[0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0],
                [0.0]])

d = mahalanobis_distance(x, z, P, x[3::2,0])
print(d)
# print(x[0:2,0].shape)
# print((z - x[0:2]).shape)

d0 = mahalanobis(z[0:2,0], x[0:2,0], np.linalg.inv(P))
print(d0)

print(np.sqrt((x[0,0] - z[0,0])**2 + (x[1,0] - z[1,0])**2))