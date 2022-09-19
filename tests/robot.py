import numpy as np

class Robot:
    def __init__(self, range, x = np.zeros((3,1))):
        '''
        range: sensing range int meters
        x0: initial state (x, y, theta)
        '''
        self.range = range
        self.xTrue = x # true state of robot (no noise)

    
    # @property
    # def xTrue(self):
    #     return self._xTrue

    # @xTrue.setter
    # def xTrue(self, x):
    #     self._xTrue = x

    def move(self, x, u, Rt, timeStep = 1):
        '''
        Move robot one step, including noise
        u: control input (v, omega)
        x: state (x, y, theta)
        
        return:
        x: new state (x, y, theta)
        '''
        u = timeStep*u
        self.xTrue[0,0] += u[0]*np.cos(self.xTrue[2,0] + u[1]) 
        self.xTrue[1,0] += u[0]*np.sin(self.xTrue[2,0] + u[1]) 
        self.xTrue[2,0] += u[1]
        self.xTrue[2,0] = self.wrapToPi(self.xTrue[2,0])

        randMat = np.random.randn(1,2)
        u_noise = u + (randMat@Rt[:2,:2])[0]
        x[0] += u_noise[0] * np.cos(x[2] + u_noise[1]) # x
        x[1] += u_noise[0] * np.sin(x[2] + u_noise[1]) # y 
        x[2] += self.wrapToPi(x[2] + u_noise[1]) # theta
        return x


    def sense(self, landmarks, num_landmarks, x, Qt):
        '''
        Sense landmarks, including noise
        landmarks: list of landmarks [x1,
                                      y1,
                                      x2,
                                      y2,...]

        return:
        z: measurement (r, theta)
        '''
        z = np.zeros((len(landmarks), 1))
        for i in range(num_landmarks):
            r = np.sqrt((landmarks[2*i,0] - x[0,0])**2 + (landmarks[2*i+1,0] - x[1,0])**2)
            theta =  np.arctan2(landmarks[2*i+1,0] - x[1,0], landmarks[2*i,0] - x[0,0])
            if r < self.range:
                z[2*i]   = r + Qt[0,0]*np.random.randn(1)
                z[2*i+1] = self.wrapToPi(theta + Qt[1,1]*np.random.randn(1))
        return z


    def wrapToPi(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi