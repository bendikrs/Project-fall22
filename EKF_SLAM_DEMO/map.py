import numpy as np

class Map:
    def __init__(self, num_landmarks=20):
        self.num_landmarks = num_landmarks
        self.landmarks = self.createCircularLandmarks() # get landmarks with default parameters

    def createCircularLandmarks(self, v=1, omega=2*np.pi/50, delta_r=1):
        '''Create map of landmarks, for use on the bonnabel test'''
        landmarks = np.zeros((2*self.num_landmarks, 1))
        r = v/omega
        for i in range(self.num_landmarks):
            rho = r + delta_r
            theta=(2*np.pi*i)/self.num_landmarks
            landmarks[2*i  ] = rho*np.cos(theta)
            landmarks[2*i+1] = rho*np.sin(theta) + 6.25
        return landmarks