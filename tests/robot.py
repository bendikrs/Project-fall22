import numpy as np

class Robot:
    def __init__(self, range, x = np.zeros(3)):
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

    def move(self, x, u, Rt):
        '''
        Move robot one step, including noise
        u: control input (v, omega)
        x: state (x, y, theta)
        
        return:
        x: new state (x, y, theta)
        '''
        u = u.astype("float64")
        self.xTrue[0] += u[0]*np.cos(self.xTrue[2] + u[1]) 
        self.xTrue[1] += u[0]*np.sin(self.xTrue[2] + u[1]) 
        self.xTrue[2] += u[1]
        self.xTrue[2] = self.wrapToPi(self.xTrue[2])

        randMat = np.random.randn(1,2).astype("float64")
        u += (randMat@Rt[:2,:2])[0]
        x[0] += u[0] * np.cos(x[2] + u[1]) # x
        x[1] += u[0] * np.sin(x[2] + u[1]) # y 
        x[2] += u[1] # theta
        x[2] = self.wrapToPi(x[2]) 
        return x


    def sense(self, landmarks, x, Qt):
        '''
        Sense landmarks, including noise
        landmarks: list of landmarks [[x,y],
                                      [x,y],
                                      ...]

        return:
        z: measurement (r, theta)
        '''
        z = np.zeros((len(landmarks), 1))
        for i in range(1,len(landmarks),2):
            r = np.sqrt((landmarks[i-1,0] - x[0,0])**2 + (landmarks[i,0] - x[1,0])**2)
            theta =  np.arctan2(landmarks[i,0] - x[1,0], landmarks[i-1,0] - x[0,0])
            if r < self.range:
                z[i-1] = r + Qt[0,0]*np.random.randn(1)
                z[i]   = self.wrapToPi(theta + Qt[1,1]*np.random.randn(1))
        return z


    def wrapToPi(self, theta):
        return (theta + np.pi) % (2 * np.pi) - np.pi