import numpy as np

class Robot:
    def __init__(self, x0, fov, Qt, Rt):
        '''
        x0: initial state (x, y, theta)
        fov: field of view int degrees
        Qt: process noise covariance matrix 3x3
        Rt: measurement noise covariance matrix 2x2
        '''
        self.x = x0
        self.x[2] = wrapToPi(self.x[2])
        self.fov = fov
        self.Qt = Qt
        self.Rt = Rt

    
    def move(self, u):
        '''
        u: control input (v, w)
        
        return:
        x: new state (x, y, theta)
        '''
        
        self.x[0] += u[0] * np.cos(self.x[2])
        self.x[1] += u[0] * np.sin(self.x[2])
        self.x[2] += u[1]

        return self.x


    def sense(self, z):
        '''
        z: measurement (x, y)'''
        pass


def wrapToPi(theta):
    return (theta + np.pi) % (2 * np.pi) - np.pi