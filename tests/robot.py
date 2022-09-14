import numpy as np

class Robot:
    def __init__(self, fov, range):
        '''
        fov: field of view int degrees
        range: sensing range int meters
        '''
        self.fov = fov
        self.range = range

    
    def move(self, x, u, Rt):
        '''
        Move robot one step, including noise
        u: control input (v, w)
        x: state (x, y, theta)
        
        return:
        x: new state (x, y, theta)
        '''
        
        x[0] += u[0] * np.cos(x[2])
        x[1] += u[0] * np.sin(x[2])
        x[2] += u[1]
        x[2] = self.wrapToPi(x[2])

        return x


    def sense(self, landmarks, x, Qt):
        '''
        Sense landmarks, including noise
        landmarks: list of landmarks [(x, y, j), ...]

        return:
        z: measurement (x, y)'''
        z = np.zeros((2, len(landmarks)))
        for i, landmark in enumerate(landmarks):
            r = np.sqrt((landmark[0] - x[0])**2 + (landmark[1] - x[1])**2)
            theta =  np.arctan2(landmark[1] - x[1], landmark[0] - x[0])
            if r < self.range and abs(self.wrapToPi(theta - x[2])) < self.fov/2:
                z[0, i] = r + Qt[0][0]*np.random.randn(1)
                z[1, i] = self.wrapToPi(theta + Qt[1][1]*np.random.randn(1))
                z[2, i] = landmark[2]

        return z


    def wrapToPi(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi