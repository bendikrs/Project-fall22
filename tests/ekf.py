import numpy as np

class EKF:
    def __init__(self, u, num_landmarks):
        self.u = u
        self.num_landmarks = num_landmarks
        self.timestep = 1
        self.Rt = np.array([[0.1,   0,   0], 
                       [  0, 0.1,   0],
                       [  0,   0, 0.1]]) # Robot motion noise
        self.Qt = np.array([[0.1,   0],
                       [  0, 0.1]]) # Landmark measurement noise
        self.x0 = np.array([50, 20, 0]) # Initial robot pose

        self.x_hat = np.zeros((3 + 2 * self.num_landmarks, 1)) # mu, Initial state x, y, theta, x1, y1, x2, y2, ...
        self.P_hat = np.zeros((3 + 2 * self.num_landmarks, 3 + 2 * self.num_landmarks)) # sigma0
        self.P_hat[3:, 3:] = np.eye(20)*1000 # set intial covariance for landmarks to large value
        self.Fx = np.zeros((3, self.x_hat.shape[0]))
        self.Fx[:3, :3] = np.eye(3)

    def g(self, x): 
        '''
        Motion model
        u: control input [v, w]
        x: state [x, y, theta, x1, y1, x2, y2, ...] (it's x_(t-1) )
        '''
        theta =  x[2]
        v, omega = self.u[0], self.u[1]

        T = np.array([[-v/omega*np.sin(theta) + v/omega*np.sin(theta + omega*self.timestep)],
                    [v/omega*np.cos(theta) - v/omega*np.cos(theta + omega*self.timestep)],
                    [omega*self.timestep]])

        return x + self.Fx.T @ T

    def jacobian(self, x):
        '''
        Jacobian of motion model
        u: control input [v, w]
        x: state [x, y, theta, x1, y1, x2, y2, ...]
        '''
        theta = x[2]
        v, omega = self.u[0], self.u[1]

        T = np.array([[0, 0, -v/omega*np.cos(theta) + v/omega*np.cos(theta + omega*self.timestep)],
                    [0, 0, -v/omega*np.sin(theta) + v/omega*np.sin(theta + omega*self.timestep)],
                    [0, 0 , 0]])

        return np.eye(3) + self.Fx.T @ T @ self.Fx

    def cov(self, Gt, P):
        '''
        Covariance update
        '''
        return Gt @ P @ Gt.T + self.Fx.T @ self.Rt @ self.Fx

    def predict(self, x, P):
        '''
        Predict step
        '''
        x_hat = self.g(self.u, x, self.Fx, self.timestep)
        Gt = self.jacobian(self.u, x, self.Fx, self.timestep)
        P_hat = self.cov(Gt, self.Rt, P, self.Fx)
        return x_hat, P_hat

    def update(self, z, threshold=1e6):
        '''
        Update step
        x_hat: state [x, y, theta, x1, y1, x2, y2, ...],  shape (3 + 2 * num_landmarks, 1)
        P_hat: covariance matrix, shape (3 + 2 * num_landmarks, 3 + 2 * num_landmarks)
        z: measurement [range r, bearing theta, landmark index j], shape: (3, num_landmarks)
        Qt: measurement noise, shape: (2, 2)
        Fx: Jacobian of motion model, shape: (3, 3 + 2 * num_landmarks)
        '''
        for r, theta, j in z:
            j = int(j)
            if self.P_hat[3 + 2*j, 3 + 2*j] >= threshold and self.P_hat[3 + 2*j + 1, 3 + 2*j + 1] >= threshold:
                # initialize landmark
                self.x_hat[3 + 2*j, 0] = self.x_hat[0, 0] + r * np.cos(self.x_hat[2, 0] + theta)
                self.x_hat[3 + 2*j + 1, 0] = self.x_hat[1, 0] + r * np.sin(self.x_hat[2, 0] + theta)
        
            # Distance between robot and landmark
            delta = np.array([[self.x_hat[3 + 2*j, 0] - self.x_hat[0, 0]],
                                [self.x_hat[3 + 2*j + 1, 0] - self.x_hat[1, 0]]])

            # Measurement estimate from robot to landmark
            q = delta.T @ delta
            z_hat = np.array([[np.sqrt(q)],
                                [np.arctan2(delta[1, 0], delta[0, 0]) - self.x_hat[2, 0]]])

            # Jacobian of measurement model
            H = (np.array([[-np.sqrt(q)*delta[0, 0], -np.sqrt(q)*delta[1, 0], 0, np.sqrt(q)*delta[0, 0], np.sqrt(q)*delta[1, 0]],
                            [delta[1, 0], -delta[0, 0], -q, -delta[1, 0], delta[0, 0]]]) / q) @ self.Fx

            # Kalman gain
            K = self.P_hat @ H.T @ np.linalg.inv(H @ self.P_hat @ H.T + self.Qt)

            # Update state and covariance
            self.x_hat = self.x_hat + K @ (np.array([[r], [theta]]) - z_hat)
            self.P_hat = (np.eye(3 + 2 * self.num_landmarks) - K @ H) @ self.P_hat
        return self.x_hat, self.P_hat