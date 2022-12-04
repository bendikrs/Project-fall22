import numpy as np
import matplotlib.pyplot as plt

def wrapToPi(theta):
    '''
    Wrap angle to [-pi, pi]
    
    parameters:
        theta (float): angle [rad] 
    output:
        (float): angle [rad]
    '''
    return (theta + np.pi) % (2.0 * np.pi) - np.pi

n = 100

x = np.linspace(-np.pi, np.pi, n)
xTrue = np.linspace(-np.pi, np.pi, n)

# print(xTrue)
# print(np.min(x - xTrue))
heading_data = []

for i in range(len(x)):
    for j in range(len(xTrue)):
        a = x[i] - xTrue[j]
        heading_data.append((a + np.pi) % (2*np.pi) - (np.pi))

        # if x[i] - xTrue[j] >= np.pi:
        #     heading_data.append(np.sqrt((x[i] - xTrue[j])**2))
        # else:
        #     heading_data.append(np.sqrt((x[i] - xTrue[j])**2))
    
print(np.min(heading_data))
print(np.argmin(heading_data))
print(np.max(heading_data))
print(np.argmax(heading_data))

for i in range(len(heading_data[::n])):
    plt.plot(x, heading_data[i:i+n])
plt.show()


a = (-np.pi + 0.1) - (np.pi )
print((a + np.pi) % (2*np.pi) - (np.pi))