import  numpy as np
import  matplotlib  as mpl
import  matplotlib.pyplot  as plt

from  matplotlib.patches  import  Ellipse
import  matplotlib.transforms  as  transform

# A function  to plot  the  confidence  interval  of a landmark  is defined
def confidence_ellipse(x, y , cov, ax, n_std=3.0, facecolor="none", **kwargs):
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
    ellipse = Ellipse((0, 0),
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

    transf = transform.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


# Define some functions needed to calculate the necessary equations
def skew2(a):
    return a * np.array([[0, -1], [1, 0]])

def exp2(theta):
    theta = theta.item()
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])
            
def Ese2(theta):
    a = np.sinc(theta / np.pi)
    b = (theta/2) * (np.square(np.sinc(theta/(2*np.pi))))
    return np.array([[a, -b], [b, a]])

def integrate_se2(h, x, v, om):
    vv = np.array([v, 0])
    x_int = np.block([x[0] + h*om, x[1:3] + exp2(x[0]) @ Ese2(om * h) @ (vv * h)])
    return x_int


# Define  the  function  that  creates  the  real  trajectory  and  real landmark  positions
def generate_map(v, omega, delta_r, n_L):
    p_L = np.zeros((2,n_L)); r = v/omega
    for i in range(0, n_L):
        rho = r + delta_r
        theta = 2 * np.pi * i / n_L
        p_L[:,i] = np.array([rho * np.cos(theta), rho * np.sin(theta) + r])
    return p_L

def real_world_dynamics(n_time , h, v, om , p_L):
    n_L = np.size(p_L,1)
    X = np.zeros((3,n_time))
    Y = np.zeros((2,n_L,n_time))
    for i in range(0, n_L):
        Y[:,i,0] = p_L[:,i] - X[1:3,0]
    for k in range(1, n_time):
        X[0:3,k] = integrate_se2(h, X[0:3,k-1], v, om)
        for i in range(0, n_L):
            Y[:,i,k] = exp2(X[0,k]).T @ (p_L[:,i] - X[1:3,k])
    return X, Y


# Algorithm for the EKF SLAM
def propagation_ekf(h, Xh, v, om, P, Q):
    n_state = np.size(Xh)
    vv = np.array([v,0])
    Rh = exp2(Xh[0])
    A = np.eye(n_state)
    A[1:3 ,0] = Rh @ skew2 (1) @ (h*vv)
    G = np.zeros((n_state ,n_state))
    G[0,0] = 1
    G[1:3 ,1:3] = Rh
    Xh [0:3] = integrate_se2(h, Xh[0:3], v, om)
    Pp = A @ P @ A.T + G @ Q @ G.T
    return Xh, Pp

def update_ekf(Xh, Pp, Yk, p_L, Rn):
    Rp = exp2(Xh[0])
    xh = Xh[1:3]
    M = - skew2(1) @ Rp.T
    C = np.zeros((2*n_L , 3 + 2*n_L))
    for i in  range(0,n_L):
        C[2*i:2*i+2, 0] = M @ (Xh[3+2*i: 3+2*i+2] - xh)
        C[2*i:2*i+2, 1:3] = -Rp.T
        C[2*i:2*i+2, 2*i+3:2*i+3+2] = Rp.T
    S = C @ Pp @ C.T + Rn
    K = Pp @ C.T @ np.linalg.inv(S)
    y = np.zeros (2*n_L)
    for i in  range(0,n_L):
        y[2*i:2*i+2] = Yk[:,i]
        yh = np.zeros (2* n_L)
    for i in  range(0,n_L):
        yh[2*i:2*i+2] = Rp.T @ (Xh[3+2*i: 3+2*i+2] - xh)
        Xhu = Xh + K @ (y - yh)
        Pu = (np.eye (3+2* n_L) - K @ C) @ Pp
    return Xhu , Pu , K


# Define  the  system  initialization  parameters
v = 1
t_circ = 40
om = 2*np.pi/t_circ
h = 1
n_L = 20
n_time = 80
n_state = 3 + 2*n_L
t = h*np.arange(n_time)


# Create  the  true  trajectory  and  true  landmark  positions
p_L = generate_map(v, om, 1, n_L)
X, Y = real_world_dynamics(n_time, h, v, om, p_L)

# Define the noise initialization parameters
Q = np.zeros ((n_state ,n_state))
sigmaq = np.array ([0.1, 0.1,  0.1]); Q[0:3 ,0:3] = np.outer(sigmaq, sigmaq)
for i in range(3, 3+2*n_L):
    Q[i,i] = 100
Q = Q
P = Q
Rn = 0.1*np.diag(np.diag(np.ones (((2*n_L ,2* n_L)))))


Xh0 = np.zeros((n_state))
for i in  range(0,n_L):
    # Apply  an  offset  for the  landmarks  to  confirm  convergence
    Xh0[3+2*i:3+2*i+2] = p_L[:,i] + [2,5]

ekf_Pu = np.zeros((n_state , n_state , n_time))

ekf_Xhp = np.zeros((n_state , n_time))
ekf_Xhu = np.zeros((n_state , n_time))
ekf_Xhp[:,0] = Xh0.copy()
ekf_Xhu[:,0] = Xh0.copy()

for k in range(1, n_time):
    ekf_Xhp[:,k], Pp = propagation_ekf(h, ekf_Xhu[:,k-1].copy(), v, om , P, Q)
    ekf_Xhu[:,k], ekf_Pu[:,:,k], K = update_ekf(ekf_Xhp[:,k].copy(), Pp, Y[:,:,k], p_L , Rn)

# Plot the EKF SLAM
plt.figure()
ax = plt.gca()
plt.plot(X[1,:], X[2,:], 'b', ekf_Xhu[1,:], ekf_Xhu[2,:], 'r--', p_L[0,:], p_L[1,:], 'bo')

for l in range(n_L):
    plt.plot(ekf_Xhu [3+2*l,:][-1],  ekf_Xhu [3+2*l+1,:][-1], "rx")
    confidence_ellipse(ekf_Xhu[3+2*l,:][-1],  ekf_Xhu[3+2*l+1,:][-1], ekf_Pu [3+2*l:5+2*l ,3+2*l:5+2*l,  -1], ax, edgecolor='r', n_std=0.5)

plt.title("EKF SLAM")
plt.legend(["Trajectory of robot", "Estimated trajectory of robot", "Landmarks", "Estimated landmarks"])
plt.xlabel("x [m]")
plt.ylabel("y [m]")
plt.show()