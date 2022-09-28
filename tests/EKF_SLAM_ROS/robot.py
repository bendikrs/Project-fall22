import numpy as np

class Robot:
    def __init__(self, range, x = np.zeros((3,1)), timeStep = 0.2):
        '''
        range: sensing range int meters
        x0: initial state (x, y, theta)
        '''
        self.range = range
        self.xTrue = x # true state of robot (no noise)
        self.timeStep = timeStep
        self.Rt = np.diag([0.001, 0.001])  # Robot motion noise


    def move(self, u):
        '''
        Move robot one step, including noise
        u: control input (v, omega)
        x: state (x, y, theta)
        
        return:
        x: new state (x, y, theta)
        '''
        randMat = np.random.randn(1,2)
        u = (u + (randMat@self.Rt)[0]) * self.timeStep
        self.xTrue[0,0] += u[0]*np.cos(self.xTrue[2,0] + u[1]) 
        self.xTrue[1,0] += u[0]*np.sin(self.xTrue[2,0] + u[1]) 
        self.xTrue[2,0] += u[1]
        self.xTrue[2,0] = self.wrapToPi(self.xTrue[2,0])
        self.xTrue[2,0] = self.xTrue[2,0]


    def sense(self, landmarks, num_landmarks, Qt):
        '''
        Sense landmarks, including noise
        landmarks: list of landmarks [x1,
                                      y1,
                                      x2,
                                      y2,...]

        return:
        z: measurement (r, theta)
        '''
        x = self.xTrue
        z = np.ones((len(landmarks), 1)) * 1e6 # Initial measurement
        for i in range(num_landmarks):
            r = np.sqrt((landmarks[2*i,0] - x[0,0])**2 + (landmarks[2*i+1,0] - x[1,0])**2)
            theta =  np.arctan2(landmarks[2*i+1,0] - x[1,0], landmarks[2*i,0] - x[0,0])
            if r <= self.range:
                # z[2*i+1] = theta - x[2,0] + Qt[1,1]*np.random.randn(1)
                z[2*i+1] = self.wrapToPi(theta - x[2,0] + Qt[1,1]*np.random.randn(1))
                z[2*i]   = r + Qt[0,0]*np.random.randn(1)

        return z


    def wrapToPi(self, theta):
        return (theta + np.pi) % (2.0 * np.pi) - np.pi
