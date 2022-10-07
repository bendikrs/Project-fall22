import numpy as np
import matplotlib.pyplot as plt

# Create a cross made of points
x1 = np.linspace( 1,-1, 10)
y1 = np.linspace(-1, 1, 10)

x2 = np.linspace(-1, 1, 10)
y2 = np.linspace(-1, 1, 10)

# Rotate the cross
x1r = x1 * np.cos(np.pi/6) - y1 * np.sin(np.pi/6)
y1r = x1 * np.sin(np.pi/6) + y1 * np.cos(np.pi/6)

x2r = x2 * np.cos(np.pi/6) - y2 * np.sin(np.pi/6)
y2r = x2 * np.sin(np.pi/6) + y2 * np.cos(np.pi/6)


# Plot the cross
fig, ax = plt.subplots(1,2)
ax[0].plot(x1, y1, 'o')
ax[0].plot(x2, y2, 'o')
ax[0].set_title('Not rotated')

ax[1].plot(x1r, y1r, 'o')
ax[1].plot(x2r, y2r, 'o')
ax[1].set_title('Rotated')

plt.show()
