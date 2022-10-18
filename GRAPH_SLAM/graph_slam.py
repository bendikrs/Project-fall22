# from tests.EKF_SLAM.ELF_SLAM import createLandmarks
# from tests.EKF_SLAM.robot import Robot
# from ..EKF_SLAM.robot import Robot
from robot import Robot
import numpy as np
import scipy.optimize as opt



def createLandmarks(v, omega, delta_r, num_landmarks):
    '''Create map of landmarks, for use on the bonnabel test'''
    landmarks = np.zeros((2*num_landmarks, 1))
    r = v/omega
    for i in range(num_landmarks):
        rho = r + delta_r
        theta=(2*np.pi*i)/num_landmarks
        landmarks[2*i  ] = rho*np.cos(theta)
        landmarks[2*i+1] = rho*np.sin(theta) + 6.25
    return landmarks



class Graph():
    '''class for handling graph based Slam algorithms
    '''
    def __init__(self, landMarks, robot):
        self.landMarks = landMarks
        self.graph = np.zeros((3, 3))
        self.robot = robot

    def moveRobot(self):
        pass

    def updateGraph(self):
        pass

    def runOptimization(self):
        pass

    def plot(self):
        pass


num_landmarks = 20
landMarks = createLandmarks(1, 2*np.pi/50, 1, num_landmarks)

graph = Graph(landMarks, 
                Robot(range=5, 
                x=np.array([[0], [0], [0]]), 
                timeStep=0.1))


for i in range(100):
    graph.robot.move()
    graph.updateGraph()
    graph.runOptimization()
    graph.plot()



'''Notat

State vector: x = [x, y, theta]
Measurement vector: z = [rho, phi, rho, phi, ...]

information matrix: Is the inverse of the covariance matrix. 
Contains the links poses/features and the uncertainty of the links
omega = [[L11 , L12 , L13],[L21 , L22 , L23],[L31 , L32 , L33]]]] ...

Information vector: Contains the same poses/features as the information matrix 
ksi = [x0, x1, x2, x3,  ..., m0, m1, m2, m3,  ...]

mu = omega^-1 * ksi 



- Create landmarks
- Create a robot object
- Create a graph object    
-  

loop:
    1. Move robot
    2. Update graph (add new nodes and edges)
    3. Optimize graph
    4. Plot results



'''