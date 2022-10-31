import numpy as np

def wrapToPi(theta):
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

class EKF:
    '''
    A class for the Extended Kalman Filter

    Parameters:
        timeStep (float): time step [s]
        range (float): range of the sensor [m]
    '''
    def __init__(self, range=10, timeStep=0.1):
        '''
        Initialize the EKF class
        
        Parameters:
            timeStep (float): time step [s]
            range (float): range of the sensor [m]
        '''
        self.range = range
        self.timeStep = timeStep

    def g(self, x, u, Fx): 
        '''
        Move robot one step using a motion model.

        Parameters:
            u (1x2 numpy array): control input [v, omega] [m/s, rad/s]
            x (3+2*numLandmarks x 1 numpy array): state [x, y, theta, x1, y1, x2, y2, ...] [m, m, rad, m, m, ...]
            Fx (3+2*numLandmarks x 3+2*numLandmarks numpy array): Masking matrix
        Returns:
            (3+2*numLandmarks x 1 numpy array): new state [x, y, theta, x1, y1, x2, y2, ...] [m, m, rad, m, m, ...]
        '''
        theta =  x[2,0]
        v, omega = u[0], u[1]
        if omega == 0:
            omega = 1e-9
        T = np.array([[-(v/omega)*np.sin(theta) + (v/omega)*np.sin(theta + omega*self.timeStep)],
                    [(v/omega)*np.cos(theta) - (v/omega)*np.cos(theta + omega*self.timeStep)],
                    [omega*self.timeStep]])

        return x + Fx.T @ T

    def jacobian(self, x, u, Fx):
        '''
        Jacobian

        Parameters:
            u (1x2 numpy array): control input [v, omega] [m/s, rad/s]
            x (3+2*numLandmarks x 1 numpy array): state [x, y, theta, x1, y1, x2, y2, ...].T [m, m, rad, m, m, ...]
            Fx (3+2*numLandmarks x 3+2*numLandmarks numpy array): Masking matrix
        Returns:
            (3+2*numLandmarks x 3+2*numLandmarks numpy array): Jacobian matrix
        '''
        theta =  x[2,0]
        v, omega = u[0], u[1]  
        if omega == 0:
            omega = 1e-9     
        T = np.array([[0, 0, -(v/omega)*np.cos(theta) + (v/omega)*np.cos(theta + omega*self.timeStep)],
                    [0, 0, -(v/omega)*np.sin(theta) + (v/omega)*np.sin(theta + omega*self.timeStep)],
                    [0, 0 , 0]])

        return np.eye(x.shape[0]) + Fx.T @ T @ Fx

    def cov(self, Gt, P, Rt, Fx):
        '''
        Covariance update

        Parameters:
            Gt (3+2*numLandmarks x 3+2*numLandmarks numpy array): Jacobian matrix
            P (3+2*numLandmarks x 3+2*numLandmarks numpy array): Covariance matrix
            Rt (2x2): Covariance matrix
            Fx (3+2*numLandmarks x 3+2*numLandmarks numpy array): Masking matrix
        Returns:
            (3+2*numLandmarks x 3+2*numLandmarks numpy array): Covariance matrix
        '''
        return Gt @ P @ Gt.T + Fx.T @ Rt @ Fx

    def predict(self, x, u, P, Rt):
        '''
        EKF predict step

        Parameters:
            u (1x2 numpy array): control input [v, omega] [m/s, rad/s]
            x (3+2*numLandmarks x 1 numpy array): state [x, y, theta, x1, y1, x2, y2, ...].T [m, m, rad, m, m, ...]
            P (3+2*numLandmarks x 3+2*numLandmarks numpy array): Covariance matrix
            Rt (2x2): Covariance matrix
        Returns:
            x_hat (3+2*numLandmarks x 1 numpy array): new estimated state [x, y, theta, x1, y1, x2, y2, ...].T [m, m, rad, m, m, ...]
            P_hat (3+2*numLandmarks x 3+2*numLandmarks numpy array): new covariance matrix
        '''
        Fx = np.zeros((3, x.shape[0]))
        Fx[:3, :3] = np.eye(3)
        x_hat = self.g(x, u, Fx)
        Gt = self.jacobian(x, u, Fx)
        P_hat = self.cov(Gt, P, Rt, Fx)

        return x_hat, P_hat

    def update(self, x_hat, P_hat, z, Qt, threshold=1e6):
        '''
        EKF update step

        Parameters:
            x_hat (3+2*numLandmarks x 1 numpy array): estimated state [x, y, theta, x1, y1, x2, y2, ...].T [m, m, rad, m, m, ...]
            P_hat (3+2*numLandmarks x 3+2*numLandmarks numpy array): Covariance matrix
            Qt (2x2 numpy array): measurement noise covariance matrix
            threshold (float): threshold for the covariance matrix
        Returns:
            x (3+2*numLandmarks x 1 numpy array): new estimated state [x, y, theta, x1, y1, x2, y2, ...].T [m, m, rad, m, m, ...]
            P (3+2*numLandmarks x 3+2*numLandmarks numpy array): new covariance matrix
        '''
        num_landmarks = (len(x_hat)-3)//2
        
        for j in range(num_landmarks):
            if z[2*j,0] <= self.range:
                if P_hat[3 + 2*j, 3 + 2*j] >= threshold and P_hat[4 + 2*j, 4 + 2*j] >= threshold:
                    # initialize landmark
                    x_hat[3 + 2*j,0] = x_hat[0,0] + z[2*j,0] * np.cos(x_hat[2,0] + z[2*j+1,0])
                    x_hat[4 + 2*j,0] = x_hat[1,0] + z[2*j,0] * np.sin(x_hat[2,0] + z[2*j+1,0])
    
                # Distance between robot and landmark
                delta = np.array([x_hat[3 + 2*j,0] - x_hat[0,0], x_hat[4 + 2*j,0] - x_hat[1,0]])

                # Measurement estimate from robot to landmark
                q = delta.T @ delta
                z_hat = np.array([[np.sqrt(q)],[wrapToPi(np.arctan2(delta[1], delta[0]) - x_hat[2, 0])]])

                # Jacobian of measurement model
                Fx = np.zeros((5,x_hat.shape[0]))
                Fx[:3,:3] = np.eye(3)
                Fx[3,2*j+3] = 1
                Fx[4,2*j+4] = 1

                H = np.array([[-np.sqrt(q)*delta[0], -np.sqrt(q)*delta[1], 0, np.sqrt(q)*delta[0], np.sqrt(q)*delta[1]],
                                [delta[1], -delta[0], -q, -delta[1], delta[0]]], dtype='float')
                H = (1/q)*H @ Fx

                # Kalman gain
                K = P_hat @ H.T @ np.linalg.inv(H @ P_hat @ H.T + Qt)
                
                # Calculate difference between expected and real observation
                z_dif = np.array([[z[2*j,0]], [z[2*j+1,0]]]) - z_hat

                # Update state and covariance
                x_hat = x_hat + K @ z_dif
                x_hat[2,0] = wrapToPi(x_hat[2,0])
                P_hat = (np.eye(x_hat.shape[0]) - K @ H) @ P_hat

        return x_hat, P_hat
