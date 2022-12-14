import numpy as np

class Robot:
    def __init__(self, range, x = np.zeros((3,1)), timeStep = 0.2):
        '''
        Initialize the robot class

        Parameters:
            range (float): sensing range
            x (3x2 numpy array): initial state
            timeStep (float): time step
        '''
        self.range = range
        self.xTrue = x # true state of robot (no noise)
        self.timeStep = timeStep
        self.Rt = np.diag([0.1, 0.01])  # Robot motion noise


    def move(self, u):
        '''
        Move robot one step. Updates self.xTrue

        Parameters:
            u (1x2 numpy array): control input [v, omega] [m/s, rad/s]
        '''
        u = u * self.timeStep
        self.xTrue[0,0] += u[0]*np.cos(self.xTrue[2,0] + u[1]) 
        self.xTrue[1,0] += u[0]*np.sin(self.xTrue[2,0] + u[1]) 
        self.xTrue[2,0] += u[1]
        self.xTrue[2,0] = self.wrapToPi(self.xTrue[2,0])
        self.xTrue[2,0] = self.xTrue[2,0]


    def sense(self, landmarks, Qt):
        '''
        Sense landmarks, including noise

        Parameters:
            landmarks (2*N numpy array): landmark positions [x1, y1, x2, y2, ...] [m, m, m, m, ...]
        
        return:
            z (2xN numpy array): measurement [[r, theta], [r, theta], ...] [[m, rad], [m, rad], ...]
        '''
        x = self.xTrue
        z = np.ones((len(landmarks), 1)) * 1e6 # Initial measurement
        for i in range(len(landmarks)//2):
            r = np.sqrt((landmarks[2*i,0] - x[0,0])**2 + (landmarks[2*i+1,0] - x[1,0])**2)
            theta =  np.arctan2(landmarks[2*i+1,0] - x[1,0], landmarks[2*i,0] - x[0,0])
            if r <= self.range:
                # z[2*i+1] = theta - x[2,0] + Qt[1,1]*np.random.randn(1)
                z[2*i+1] = self.wrapToPi(theta - x[2,0] + Qt[1,1]*np.random.randn(1))
                z[2*i]   = r + Qt[0,0]*np.random.randn(1)

        return z


    def wrapToPi(self, theta):
        '''
        Wrap angle to [-pi, pi]

        Parameters:
            theta (float): angle [rad]

        return:
            theta (float): angle in [-pi, pi] [rad]
        '''
        return (theta + np.pi) % (2.0 * np.pi) - np.pi
