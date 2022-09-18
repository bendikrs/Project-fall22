import numpy as np

class EKF:
    def __init__(self, timestep=1):
        self.timestep = timestep
    def g(self, x, u, Fx): 
        '''
        Motion model
        u: control input (v, omega)
        x: state [x, y, theta, x1, y1, x2, y2, ...] (it's x_(t-1) )
        '''
        theta =  x[2,0]
        v, omega = u[0], u[1]
        # if omega == 0:
        #     omega = 1e-9
        # T = np.array([[-(v/omega)*np.sin(theta) + (v/omega)*np.sin(theta + omega*self.timestep)],
        #             [(v/omega)*np.cos(theta) - (v/omega)*np.cos(theta + omega*self.timestep)],
        #             [omega*self.timestep]])
        T = np.array([[v*np.cos(x[2,0] + omega)],
                      [v*np.sin(x[2,0] + omega)],
                      [omega]])*self.timestep

        return x + Fx.T @ T

    def jacobian(self, x, u, Fx):
        '''
        Jacobian of motion model
        u: control input (v, omega)
        x: state [x, y, theta, x1, y1, x2, y2, ...]
        '''
        theta =  x[2,0]
        v, omega = u[0], u[1]  
        if omega == 0:
            omega = 1e-9     
        # T = np.array([[0, 0, -(v/omega)*np.cos(theta) + (v/omega)*np.cos(theta + omega*self.timestep)],
        #             [0, 0, -(v/omega)*np.sin(theta) + (v/omega)*np.sin(theta + omega*self.timestep)],
        #             [0, 0 , 0]])
        T = np.array([[0, 0, -v*np.sin(x[2,0])],
                    [0, 0, v*np.cos(x[2,0])],
                    [0, 0 , 0]])
        return np.eye(x.shape[0]) + Fx.T @ T @ Fx

    def cov(self, Gt, P, Rt, Fx):
        '''
        Covariance update
        '''
        return Gt @ P @ Gt.T + Fx.T @ Rt @ Fx

    def predict(self, x, u, P, Rt):
        '''
        Predict step
        '''
        Fx = np.zeros((3, x.shape[0]))
        Fx[:3, :3] = np.eye(3)

        x_hat = self.g(x, u, Fx)
        Gt = self.jacobian(x, u, Fx)

        # EKF state covariance
        Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2
        
        P_hat = self.cov(Gt, P, Cx, Fx)
        # print('Predicted location\t x: {0:.4f} \t y: {1:.4f} \t theta: {2:.4f}'.format(x_hat[0,0],x_hat[1,0],x_hat[2,0]))
        return x_hat, P_hat

    def update(self, x_hat, P_hat, z, Qt, num_landmarks, threshold=1e6):
        '''
        Update step
        x_hat: state [x, y, theta, x1, y1, x2, y2, ...],  shape (3 + 2 * num_landmarks, 1)
        P_hat: covariance matrix, shape (3 + 2 * num_landmarks, 3 + 2 * num_landmarks)
        z: measurement [range r, bearing theta, landmark index j], shape: (3, num_landmarks)
        Qt: measurement noise, shape: (2, 2)
        Fx: Jacobian of motion model, shape: (3, 3 + 2 * num_landmarks)
        '''
        for j in range(num_landmarks):
            if P_hat[3 + 2*j, 3 + 2*j] >= threshold and P_hat[4 + 2*j, 4 + 2*j] >= threshold:
                # initialize landmark
                x_hat[3 + 2*j,0] = x_hat[0,0] + z[2*j,0] * np.cos(x_hat[2,0] + z[2*j+1,0])
                x_hat[4 + 2*j,0] = x_hat[1,0] + z[2*j,0] * np.sin(x_hat[2,0] + z[2*j+1,0])
        
            # Distance between robot and landmark
            delta = np.array([x_hat[3 + 2*j,0] - x_hat[0,0], x_hat[4 + 2*j,0] - x_hat[1,0]])

            # Measurement estimate from robot to landmark
            q = delta.T @ delta

            z_hat = np.array([[np.sqrt(q)],[np.arctan2(delta[1], delta[0]) - x_hat[2, 0]]])

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
            z_dif = (z_dif + np.pi) % (2*np.pi) - np.pi

            # Update state and covariance
            x_hat = x_hat + K @ z_dif
            P_hat = (np.eye(x_hat.shape[0]) - K @ H) @ P_hat
        return x_hat, P_hat