from matplotlib import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class Plotter:
    def __init__(self, fig, ax):
        '''
        Initialize the plotter
        
        Parameters:
            fig: The figure to plot on
            ax: The axis to plot on
        '''
        self.NEES = []
        self.estimatedRobotTrajectory = []
        self.trueRobotTrajectory = []
        self.fig = fig
        self.ax = ax

    def plot(self, x_hat, P_hat, z, num_landmarks, landmarks, robot, ax):
        '''
        Plots all the relevant information
        
        Parameters:
            x_hat: The estimated state
            P_hat: The estimated covariance
            z: The measurement
            num_landmarks: The number of landmarks
            landmarks: The true landmark positions
            robot: The robot object
            ax: The axis to plot on
        '''
        self.plotLandmarks(landmarks)
        self.plotEstimatedLandmarks(x_hat)
        self.plotRobot(robot)
        self.plotEstimatedRobot(x_hat)
        self.plotMeasurement(x_hat, z, num_landmarks)
        self.plotCov(x_hat, P_hat, num_landmarks)
        self.plotMeasurementDistance(robot.xTrue, robot.range)
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

    def plotCov(self, x_hat, P_hat, num_landmarks):
        # Plot the covariance of the robot
        P_hat_x = np.sqrt(P_hat[0, 0])
        P_hat_y = np.sqrt(P_hat[1, 1])

        ellipse = patches.Ellipse((x_hat[0,0], x_hat[1,0]), P_hat_x, P_hat_y, color='r', fill=False)
        self.ax.add_patch(ellipse)

        for j in range(num_landmarks):
            if P_hat[2*j+3, 2*j+3] < 1e6:
                P_hat_x = np.sqrt(P_hat[2*j+3, 2*j+3])
                P_hat_y = np.sqrt(P_hat[2*j+4, 2*j+4])

                xLandmark = x_hat[2*j + 3]
                yLandmark = x_hat[2*j + 4]
                self.ax.add_patch(patches.Ellipse((xLandmark, yLandmark), \
                P_hat_x, P_hat_y, color=(0,0,1), fill=False))

    def plotMeasurementDistance(self, xTrue, rangeLimit):
        '''Plot the range of the sensor as a circle around the robot
        '''
        circle = plt.Circle((xTrue[0,0], xTrue[1,0]), rangeLimit, color='0.8', fill=False)
        self.ax.add_artist(circle)

    def addConfidenceEllipse(self, x, y, cov, n_std=3.0, facecolor="none", **kwargs):
        """
        Create a plot of the covariance confidence ellipse of *x* and *y*.

        Parameters
        ----------
        x, y : array_like, shape (n, )
            Input data.

        cov : array_like, shape (2, 2)
            Covariance matrix of the data.

        ax : matplotlib.axes.Axes
            The axes object to draw the ellipse into.

        n_std : float
            The number of standard deviations to determine the ellipse's radiuses.

        Returns
        -------
        matplotlib.patches.Ellipse

        Other parameters
        ----------------
        kwargs : `~matplotlib.patches.Patch` properties
        """

        pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
        # Using a special case to obtain the eigenvalues of this
        # two-dimensionl dataset.
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = patches.Ellipse((0, 0),
                        width=ell_radius_x * 2,
                        height=ell_radius_y * 2,
                        facecolor=facecolor,
                        **kwargs)

        # Calculating the stdandard deviation of x from
        # the squareroot of the variance and multiplying
        # with the given number of standard deviations.
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)

        # calculating the stdandard deviation of y ...
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)

        transf = transforms.Affine2D() \
            .rotate_deg(45) \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)

        ellipse.set_transform(transf + self.ax.transData)
        return self.ax.add_patch(ellipse)