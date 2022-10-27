from cProfile import label
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class Plotter:
    def __init__(self, fig, ax):
        self.NEES = []
        self.estimatedRobotTrajectory = []
        self.trueRobotTrajectory = []
        self.fig = fig
        self.ax = ax

    def plot(self, x_hat, P_hat, z, num_landmarks, landmarks, robot, ax):
        self.plotLandmarks(landmarks)
        self.plotEstimatedLandmarks(x_hat)
        self.plotRobot(robot)
        self.plotEstimatedRobot(x_hat)
        self.plotMeasurement(x_hat, z, num_landmarks)
        self.plotCov(x_hat, P_hat, num_landmarks, ax)
        self.plotMeasurementDistance(robot.xTrue, robot.range)
        # self.NEES.append(self.calculateNEES(x_hat, robot.xTrue, P_hat, landmarks))
        self.updateTrajectory(robot, x_hat)
        self.plotTrajectory()
        self.addLegend()
        plt.pause(0.01)

    def addLegend(self):
        self.ax.legend(loc='upper right')
    
    def plotTrajectory(self):
        self.ax.plot(np.array(self.estimatedRobotTrajectory)[:,0], np.array(self.estimatedRobotTrajectory)[:,1], 'r', label='Estimated Robot Trajectory')
        self.ax.plot(np.array(self.trueRobotTrajectory)[:,0], np.array(self.trueRobotTrajectory)[:,1], 'g', label='True Robot Trajectory')

    def updateTrajectory(self, robot, x_hat):
        self.estimatedRobotTrajectory.append(x_hat[0:2])
        self.trueRobotTrajectory.append([robot.xTrue[0,0], robot.xTrue[1,0]])

    def plotLandmarks(self, landmarks):
        self.ax.plot(landmarks[0::2], landmarks[1::2], 'g+', label='True Landmarks')


    def plotEstimatedLandmarks(self, x_hat):
        estimatedLandmarks = x_hat[3:]
        self.ax.plot(estimatedLandmarks[0::2], estimatedLandmarks[1::2], 'ro', markersize=2, label='Estimated Landmarks')


    def plotRobot(self, robot):
        self.ax.arrow(robot.xTrue[0,0], robot.xTrue[1,0],0.5*np.cos(robot.xTrue[2,0]), 0.5*np.sin(robot.xTrue[2,0]),
         head_width=0.5, color='g')


    def plotEstimatedRobot(self, x_hat):
        self.ax.arrow(x_hat[0,0], x_hat[1,0], 0.5*np.cos(x_hat[2,0]), 0.5*np.sin(x_hat[2,0]), head_width=0.5, color='r')


    def plotMeasurement(self, x_hat, z, num_landmarks):
        z_xy = np.zeros((2*num_landmarks, 1))
        for j in range(num_landmarks):
            if z[2*j] < 1e4:
                plt.plot([x_hat[0,0], x_hat[2*j+3,0]], [x_hat[1,0], x_hat[2*j+4,0]], color=(0,0,1), linewidth=.5)


    def plotCov(self, x_hat, P_hat, num_landmarks, ax):
        # Plot the covariance of the robot
        P_hat_x = np.sqrt(P_hat[0, 0])
        P_hat_y = np.sqrt(P_hat[1, 1])

        ellipse = patches.Ellipse((x_hat[0,0], x_hat[1,0]), P_hat_x, P_hat_y, color='r', fill=False)
        ax.add_patch(ellipse)

        for j in range(num_landmarks):
            if P_hat[2*j+3, 2*j+3] < 1e6:
                P_hat_x = np.sqrt(P_hat[2*j+3, 2*j+3])
                P_hat_y = np.sqrt(P_hat[2*j+4, 2*j+4])

                xLandmark = x_hat[2*j + 3]
                yLandmark = x_hat[2*j + 4]
                self.ax.add_patch(patches.Ellipse((xLandmark, yLandmark), \
                P_hat_x, P_hat_y, color=(0,0,1), fill=False))


    def calculateNEES(self, x, xTrue, P, landmarks):
        '''Calculate the Normalized Estimation Error Squared'''
        xTrue = np.vstack((xTrue, landmarks))
        e = x - xTrue
        NEES = np.dot(e.T, np.dot(np.linalg.inv(P), e))
        # NEES = e.T @ np.linalg.inv(P) @ e
        return NEES[0][0]


    def plotMeasurementDistance(self, xTrue, rangeLimit):
        # Plot the range of the measurements as a circle
        circle = plt.Circle((xTrue[0,0], xTrue[1,0]), rangeLimit, color='0.8', fill=False)
        plt.gcf().gca().add_artist(circle)